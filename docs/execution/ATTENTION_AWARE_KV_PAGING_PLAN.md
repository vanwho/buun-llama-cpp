# Qwen3.8 256K attention-aware Turbo4 KV paging plan

Status: execution plan, revised 2026-09-03
Canonical worktree: `/srv/repos/vanwho/buun-llama-cpp`
Pinned Buun base: `cb703be37e3628dadb71912f3b3b25b82090555b`
Compared llama.cpp base: `67a17c17caa95742186f8b1ecadd1b5abd6d5ebb`
Execution state: `docs/execution/WORK_STATE.json`

## 1. Outcome

Build an opt-in Qwen3.8 path with a 262,144-token logical target context in which:

- all target and MTP K/V are TurboQuant Turbo4;
- approximately 77,824 target-attention tokens are physically resident in VRAM at a time;
- every sealed target-attention page has a canonical opaque Turbo4 copy in CPU RAM; hot pages also have
  a GPU mapping, while transfers use either measured pinned backing or a bounded pinned staging ring;
- the full MTP draft K/V remains resident in VRAM and is never selected as a paging victim;
- recurrent/linear-attention state remains exact and GPU-resident;
- a small all-page routing index can recall relevant cold pages;
- selected target pages are compacted into a bounded physical working set before Turbo4 flash attention;
- transfers are page-batched and asynchronous, so a focused workload is much faster than ordinary
  `--no-kv-offload` execution.

The 77,824 figure is a measured target, not a hardcoded allocation promise. Admission must derive the
largest safe multiple of 256 tokens after model weights, graphs, Turbo4 dequant scratch, the full MTP
cache, recurrent state, transfer rings, routing summaries, and a configurable VRAM headroom have been
reserved. The acceptance campaign should land close to 77K on the reference machine. If it does not,
the ledger must explain every byte.

## 2. Required semantics

The fast mode is intentionally selective, not bit-equivalent dense attention. Calling it ordinary KV
offload would be misleading. It has two different decisions:

1. **Retrieval:** choose candidate logical pages from all 1,024 pages, including pages not currently
   resident. This must use an all-page routing index plus structural anchors. Resident attention weights
   alone cannot discover a cold page.
2. **Retention:** after real attention executes, aggregate observed attention mass and reuse data to
   decide which resident pages should remain hot.

A correct implementation keeps those decisions separate. A page that is not retrieved receives no
attention weight; treating that missing observation as zero importance creates an irreversible cache
trap.

Four operating modes are part of the contract from day one, although `exact` is implemented last:

- `off`: unchanged upstream behavior and zero pager work;
- `observe`: dense/current execution plus routing and retention telemetry, with no page movement;
- `selective`: bounded resident set with host Turbo4 backing and selected attention;
- `exact`: all logical pages participate and partial online-softmax states are merged across hot and
  cold page partitions.

At small contexts, dense Turbo4 and selected-all-pages execution are the early correctness oracles. The
full-context exact implementation comes only after selective mode meets its quality/speed gates. Its
minimum implementation streams cold Turbo4 page waves through bounded GPU staging and merges `(max,
sum_exp, weighted_value)` states. A CPU/GPU split may replace the cold branch once a correct CPU Turbo4
attention primitive exists. Exact mode is expected to be slow; it proves storage/page-table correctness
and quantifies selective-attention quality rather than serving as the primary performance mode.

## 3. Reference-model geometry and memory budget

The current Qwen3.8-27B/qwen35 model metadata observed in the previous work is:

- deployed model: `/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf` via `current.gguf`;
- profile: `/srv/ai/config/profiles/qwen38-fast.env`;
- native context: 262,144 tokens;
- 65 total blocks: 64 target trunk blocks plus one native MTP block;
- 16 full-attention target layers (`full_attention_interval = 4`);
- four K/V heads;
- K width 256 and V width 256 per full-attention layer;
- 4.125 effective bits/value for Turbo4 rows, including block scales.

The plan must re-read the deployed GGUF metadata and fail closed if this geometry differs. No generic
API should silently assume it.

### 3.1 Exact payload arithmetic

For the 16 target full-attention layers:

```text
values/token = 16 layers * 4 KV heads * (256 K + 256 V) = 32,768
bytes/token  = 32,768 * 4.125 / 8 = 16,896
```

For the single MTP layer:

```text
values/token = 1 layer * 4 KV heads * (256 K + 256 V) = 2,048
bytes/token  = 2,048 * 4.125 / 8 = 1,056
```

| Allocation | Tokens/pages | Turbo4 payload |
| --- | ---: | ---: |
| One target logical page | 256 / 1 | 4.125 MiB |
| Target VRAM working set | 77,824 / 304 | 1.224609375 GiB |
| Target host cold set | 184,320 / 720 | 2.900390625 GiB |
| Full target logical context | 262,144 / 1,024 | 4.125 GiB |
| Full MTP VRAM cache | 262,144 / 1,024 | 264 MiB |
| Target + MTP Turbo4 payload | 262,144 each | 4.3828125 GiB |

These are encoded payloads only. The budget must separately report allocator/VMM granularity,
alignment, page tables, positions, masks, routing summaries, pinned staging, CUDA graph reserves,
recurrent state, and dequant scratch.

### 3.2 Initial 304-page residency partition

Use an explicit, measured initial partition rather than an undifferentiated LRU pool. Counts are exact
256-token pages and are disjoint after deduplication:

| Pool | Pages | Tokens | Initial purpose |
| --- | ---: | ---: | --- |
| Recent | 96 | 24,576 | Always keep the newest committed pages |
| Persistent/structural | 32 | 8,192 | System/tool/task anchors and attention sinks |
| Attention-selected history | 140 | 35,840 | Router plus observed-attention retention |
| Transient/prefetch | 36 | 9,216 | Promotions and focus changes without immediate thrash |
| **Total** | **304** | **77,824** | Reference hot-set target |

This is a starting policy, not a fixed ABI. Structural pages can be identified from server token/turn
metadata or explicitly pinned by the application; the pager must not pretend it understands “active
objective” semantics from attention scores alone. Overflow of mandatory/pinned pages is a typed refusal,
not silent eviction. Benchmarks may change the partition while keeping the total budget authoritative.

### 3.3 Why MTP must be Turbo4 and pinned

The previous September plan used F16 for MTP, which would consume 1 GiB at 256K. Turbo4 consumes
264 MiB, saving 760 MiB. The current profile and benchmark already prove that Buun accepts Turbo4 for
draft K/V and that acceptance/throughput were indistinguishable within the small sample:

- F16 result: `/srv/ai/paged-kv/results/profile-benchmark-default-fast-20260902-233315`;
- Turbo4 result: `/srv/ai/paged-kv/results/profile-benchmark-default-fast-20260902-233429`.

Therefore “add Turbo4 draft support” is complete before this plan starts. The remaining work is to
make the MTP allocation independently budgeted, full-length, GPU-pinned, and excluded from the target
pager/VBR controller.

The current parser confirms that `get_all_kv_cache_types(false)` includes `GGML_TYPE_TURBO4_0` and
accepts `t4`, `turbo4`, `turbo4_0`, or `4`. Generated CLI/server tables still show only upstream cache
types, so their generation/update path should be corrected when the MTP placement work is documented.

### 3.4 Independent MTP placement is the first code milestone

The recommendation to fix MTP placement first is correct and is now explicit. Today
`common_context_params_to_llama()` derives `cparams.offload_kqv` from the base
`common_params::no_kv_offload`; `common_base_params_to_speculative()` begins with a wholesale base copy;
and the native server MTP path builds its context from `params_base`. Consequently a conventional
target `--no-kv-offload` configuration can also move draft K/V to RAM.

Add a tri-state draft residency policy, with the exact public spelling confirmed through upstream
discussion. The preferred semantic shape is `--spec-draft-kv-device auto|gpu|cpu`; an
`--spec-draft-kv-offload auto|on|off` spelling is acceptable if maintainers prefer existing terminology.
`auto` preserves legacy placement, `gpu` requires a usable GPU backend and independently sets the draft
context’s KQV/KV placement, and `cpu` explicitly disables it. Model weight placement remains controlled
by existing draft-device/layer options.

Implement this as a small generic change before pager storage. Test external draft and native MTP
context construction, fit/accounting, and explicit failure. Keep the implementation separable for
`vanwho/llama.cpp`/upstream #28115, while Buun additionally documents and verifies Turbo4 draft types.

## 4. Findings from the previous work

### 4.1 Reusable ideas

The dirty historical worktree at `/srv/ai/paged-kv/repos/buun-llama-cpp` is reference-only. It contains
a useful prototype with:

- typed logical page identity including a session generation;
- deterministic policy and hot-set delta calculation;
- transaction-style page-table publication/replacement;
- logical-position run validation;
- focused tests for page-table and replacement invariants.

Reimplement those ideas against current Buun APIs. Do not copy unreviewed code wholesale, do not reset
or clean that worktree, and do not make it the execution root.

### 4.2 Prototype limitations that must not survive

- fixed 512-token pages, while current VBR generation ownership uses 256 cells;
- `physical_cells == context_size`, so no VRAM capacity is saved;
- whole-cache/state-serializer capture through synchronous host vectors;
- no automatic faults, prefetch, eviction, host catalog, or CUDA residency changes;
- no paged Turbo4 flash-attention kernel;
- no recurrent/MTP/speculative transaction;
- public context-parameter surface added before the internal contract was proven;
- new test binaries without maintainer approval.

The old probe demonstrated metadata behavior, not paging performance.

## 5. What current Buun already provides

Buun master at `cb703be37` is a substantially better foundation than the old prototype base.

### 5.1 VBR/VMM and representation safety

Current files under `src/llama-vbr-*` and `src/llama-kv-cache.*` provide:

- stable virtual-address VMM pools per layer/side;
- mapped-physical accounting and recoverable map projection;
- asynchronous retiering, per-device streams, fences, and deferred unmaps;
- process-wide bounded pinned capture/adoption rings;
- authenticated host artifacts and projected capture manifests;
- explicit capture/adopt transactions and occupied replacement;
- representation, checkpoint, sequence, and page generations;
- capture leases and fail-closed validation;
- composite operations for hybrid/recurrent companions;
- automatic server prompt-cache lifecycle and retention/accounting catalogs;
- multi-GPU artifact bindings and batched HIP VMM maps.

This machinery solves most of the hard ownership, failure, and lifecycle problems, but it must not
collapse two independent axes into one abstraction:

- **KV representation / VBR:** which encoding and precision owns a row;
- **KV residency / pager:** which logical token page has a GPU mapping and where its host backing lives.

The pager should therefore be a distinct residency controller and selected-page data path. It shares
VMM map/unmap/event primitives, representation descriptors, artifact segments, generations, and the
pinned ring with VBR, but does not become another VBR tier or a parallel state serializer. One composite
KV coordinator orders representation changes and residency changes atomically so neither subsystem can
publish against a stale epoch.

Important boundary: VBR’s existing 256-cell generation pages describe mutation dependencies. Existing
live VMM mappings are principally tensor extents/watermarks and representation bands. They do not yet
provide a logical-page-to-arbitrary-physical-slot attention table. Sharing the 256-cell constant and
generation records is correct; claiming the existing subsystem already implements demand paging is not.

### 5.2 Prompt artifacts and MTP checkpoints

Relevant landed commits include:

- `34941d33b`: automatic VBR prompt-cache lifecycle;
- `2714303b5`: speculative restore repair;
- `2174ad63`: multi-GPU prompt capture;
- `283ba19ed`: configurable VBR entry tiers;
- `65eb44ebc`: batched HIP VMM maps;
- `87b37eac9`: align MTP context sizing and fit;
- `764f3a044`: move artifact stream buffers off stack;
- `cb703be37`: preserve MTP carry state across prompt checkpoints.

Host prompt artifacts capture immutable prefixes. Live page backing is more mutable and granular, but
must reuse the artifact segment chain, representation descriptors, pinned ring, validation vocabulary,
and companion transaction patterns wherever their contracts fit.

### 5.3 Retention and cache-plan machinery

`common/common-retention-sidecar.*`, `tools/server/server-cache-yield.*`, cache accounting, and cache
budgeting already model bounded candidates, marginal resource, stable identities, recency, frequency,
leases, and fail-closed authority. They rank whole live/host/checkpoint artifacts, not tokens inside one
attention cache. Reuse their value/accounting patterns and terminology, but create a distinct page
policy so request-level lineage frequency is never confused with per-page attention evidence.

### 5.4 Hybrid memory precedents

`llama_memory_hybrid`, `llama_memory_hybrid_idx`, and `docs/qwen4-vbr-plan.md` show how one attention
child owns VBR while recurrent and index companions remain exact and fixed. The unmerged
`feature/dsv4-vbr` branch similarly demonstrates grouped cache tiering, but it is model-specific and
diverged from master. Consult it; do not base this work on it or merge its 1,200-line change.

## 6. What current llama.cpp adds

Latest llama.cpp was inspected at `67a17c17`. Buun is synced through upstream `0f3a71be`; after that,
the following are particularly relevant and should be integrated through Buun’s normal upstream-sync
process, not casually cherry-picked into a feature patch:

- `2d8d612e4` / PR #27991 batches state restoration by contiguous destination cell runs. Its run
  construction and tests are useful for sparse page restores even though live paging must not use the
  public state serializer as its transport.
- `8e93a9773` / PR #27970 adds sparse flash-attention plumbing for DSV4/GLM. Its typed sparse metadata,
  graph operator, CUDA dispatch, and backend tests are the closest current upstream kernel precedent.
- `36b101543` / PR #27941 fixes block-position keying, sequence copying, and rollback in Qwen4 sparse
  memory. Those invariants overlap selected logical pages.
- `0eadefebd` adds recurrent rollback coverage and should be compared with Buun’s companion handling.

None is a generic Turbo4 page-offload implementation.

### 6.1 Current issue audit with authenticated GitHub CLI

The issue/PR inventory was rechecked on 2026-09-03 with the authenticated `gh` CLI. No open Buun
issue specifically proposes attention-aware host paging. The following active issues must still be
treated as dependencies or regression inputs:

- Buun #109: dynamic VBR multi-slot context accounting. Initial pager work remains `-np 1`; multi-slot
  authority is not enabled until this issue is resolved or the pager independently proves per-slot
  limits.
- closed Buun #95/#96: VBR auto-fit previously overcommitted context-scaled dequant scratch. The
  present budget design must test this failure class rather than assuming the historical fix covers a
  new compact/paged cache.
- llama.cpp #28115: target RAM placement unintentionally drags MTP KV to RAM. The pager avoids the
  coupled flag for its normal fast mode, but independent draft placement is still a useful generic
  upstream slice.

## 7. Existing community paged-attention work

Do not open a generic duplicate without acknowledging:

- llama.cpp Discussion #21961, “Paged Attention Implementation for llama.cpp”;
- draft PR #22569 and the local reference clone `/srv/repos/matiaslin/llama.cpp`;
- llama.cpp Issue #28115 about keeping speculative/MTP KV on GPU when target KV is in RAM.

The Matias Lin branch provides a useful `llama_block_manager`, CPU/GPU block pools, scheduler, block
table, and CUDA/CPU paged-attention operator. It does not solve this project because it is F16-only,
synchronous, dense in logical block order, limited to single-GPU/full-offload, lacks working state and
sequence operations, and does not support hybrid recurrent memory, MTP, Turbo4, arbitrary selected
native positions, or Buun VBR artifacts. Its default 16-token blocks are also too fine for a 4.125-MiB
cross-layer transfer unit.

Treat it as a kernel-interface and scheduler reference. Prefer contributing generic improvements to
that discussion/PR when they are independently useful; keep Buun-specific VBR/Turbo4 integration in
Buun until maintainers choose a common direction.

## 8. Architecture

### 8.1 Ownership tree

```text
Qwen3.8 request
├── exact recurrent state (GPU, never paged)
├── target attention residency controller (one authority)
│   ├── 1,024 logical token pages, 256 tokens each
│   ├── ~304 GPU physical slots, Turbo4 K+V across all 16 attention layers
│   ├── canonical host backing for every sealed target page
│   ├── all-page routing summaries and structural metadata
│   └── transactional logical↔physical residency table
└── MTP context
    ├── full 262,144-token Turbo4 K+V allocation in GPU VRAM
    ├── separate budget/accounting identity
    └── no target pager/VBR victim eligibility
```

The target page is cross-layer and includes K and V for the same 256 logical positions across all 16
full-attention layers. This yields a 4.125-MiB transfer. A common token-page residency decision avoids
faulting different pages at every layer and keeps recurrent/attention sequence operations coherent.

### 8.2 Page states

Use a closed internal state machine. The current partial write page remains pinned and GPU-authoritative;
once sealed, one batched D2H captures all 32 target K/V tensor segments and makes the host copy canonical:

```text
absent -> filling_gpu
filling_gpu -> sealing_host -> gpu_host_clean
host_clean -> loading_gpu -> gpu_host_clean
gpu_host_clean -> host_clean                 # normal eviction: drop/unmap only
gpu_host_clean -> gpu_dirty                  # logical mutation
gpu_dirty -> resealing_host -> gpu_host_clean
* -> invalid (generation mismatch, clear, failed validation)
```

Only GPU-resident states can appear in the published attention table. Only host-valid states can be
restored without recomputation. A normal sealed-page eviction performs no D2H transfer. Transfer
completion is not publication: validate identity,
generation, representation, position runs, and all layer/side segments before atomically swapping the
table. A dirty/mutated page must reseal successfully before it becomes evictable. A failed transaction
leaves the old table usable.

### 8.3 Logical and physical identity

Every reference must include at least:

```text
request/session generation
sequence id and sequence generation
logical page index
logical position begin/end or explicit position run digest
target representation epoch
page mutation generation
model/execution/topology identity
Turbo4 codec, codebook, rotation, mean-subtraction digests
```

Physical slot IDs are ephemeral and never serialized as logical identity. The table owns forward and
reverse maps, state, last-use fence, last-attention epoch, pin count, dirty flag, and host handle.

### 8.4 CPU backing representation

The canonical payload for every **sealed** target page is exactly the device Turbo4 row encoding,
stored as opaque host bytes. Do not route it through CPU attention and do not silently transcode it to
Q8_0. Seal D2H and restore H2D use representation descriptors and per-segment checksums. The first
implementation copies once when a 256-token page seals, while it is still recent and pinned; it does
not issue 32 small host writes for every generated token. A measured write-through variant may be
tested later. A direct pinned allocation for the complete 4.125-GiB target store is optional; prefer
pageable catalog storage plus a bounded pinned transfer ring unless measurement proves whole-store
pinning safe and faster.

### 8.5 Physical layout

Start with a compact physical cache of `resident_pages * 256` rows for every full-attention K/V tensor.
Selected logical pages are copied into physical slots. The corresponding native logical positions are
stored in a compact position vector, so RoPE/masking uses original positions rather than slot order.

Before policy or a custom kernel, prove the backing design with a fixed/manual 304-page GPU window:
fill and seal a 256K logical cache, promote a chosen window, evict by dropping clean GPU slots, restore
from host, and reconcile byte ledgers/checksums. The first attention milestone may materialize the
compact table before graph launch. The optimized
milestone passes a block/page table and native-position metadata directly to flash attention. Keep both
behind the same internal view contract so the simpler implementation remains a correctness oracle.

Do not allocate 262K target tensor rows merely to reserve virtual addresses; that would hide physical
savings behind VMM semantics and reproduce the old prototype mistake. A sparse VMM mapping approach is
acceptable only if the backend tensor and kernel can address arbitrary logical pages without mapping
all intervening pages and the ledger proves physical occupancy.

### 8.6 Retrieval router

Required candidate union per decode decision:

- current write page;
- configurable recent window;
- first-token/system and conversation-boundary anchors;
- M mandatory pages supplied by application policy;
- pages already pinned by an in-flight graph or speculative checkpoint;
- top-K pages from an all-page query-to-summary scorer;
- a small exploration budget so cold pages can recover from stale summaries.

Initial summaries should be deliberately simple and measurable. Begin with 4–8 representative rotated
K vectors per page for a selected subset of full-attention layers/heads, stored in F16 or a measured
lower-precision format. Score all summaries on GPU, reduce to one page score, and take top-K using the
existing radix top-k machinery where suitable. Do not freeze this format in a public API until recall,
memory, and latency are measured. Clustering or learned summaries are later optimizations.

Retrieval runs ahead of the layer that will consume the page. At minimum, use the prior token’s query
or summary score and prefetch for the next token. If required pages are not ready, use an explicit
policy: wait, use the old hot set, or fall back to a larger resident union. Never consume a partially
published page.

### 8.7 Retention signal

The Turbo4 FA path should optionally reduce attention probability mass into a bounded per-logical-page
accumulator on GPU. Aggregate sum mass and recent peak across heads and chosen layers, update an EMA,
and copy only the bounded page-score vector to the controller: 1,024 `float` scores are about 4 KiB,
not an attention matrix. Combine normalized attention EMA/peak with recency, frequency/hit count,
fault cost, dirty cost, structural pins, and hysteresis. Instrumentation must be allocation-free on the
decode hot path and disabled by default.

Victim ordering is deterministic. A useful first formula is documented inputs rather than magic:

```text
keep_score = normalized_attention_ema
           + normalized_recent_peak
           + normalized_frequency
           + normalized_recency
           + dirty_writeback_penalty
           + structural_pin_infinity
           - age_decay
```

Do not initially hardcode proposed weights such as 0.60/0.20/0.10/0.10: those values are meaningless
until the inputs have compatible scales. Tune weights only through captured traces and replayable policy
tests, then record the normalization and coefficients in configuration/evidence.

### 8.8 Fault and transfer pipeline

One scheduler owns a bounded queue of page intents. It must:

1. snapshot generations and the published table;
2. compute required additions/removals and validate capacity;
3. pin current graph pages;
4. select only unpinned victims;
5. reject dirty victims or reseal them in layer/side batches before eviction;
6. evict clean victims by removing their GPU mapping/slot without D2H;
7. H2D required host pages on a dedicated upload stream using the shared pinned ring;
8. wait on backend events outside global metadata locks while the compute stream continues where safe;
9. revalidate all snapshots;
10. publish the complete new map atomically;
11. release pins and defer/reclaim old mappings only after consumer fences.

Coalesce contiguous physical runs as in llama.cpp PR #27991. Prefer one descriptor batch per device and
direction. Add counters for useful bytes, amplified bytes, queueing, copy time, wait time, faults,
prefetch hits, canceled transfers, stale-generation rejects, and evictions.

### 8.9 Hybrid, speculative, and server atomicity

A Qwen3.8 sequence mutation is one composite operation across target page metadata, recurrent state,
and MTP state. `seq_cp`, `seq_rm`, shifts, prompt reuse, context rewind, speculative rejection, and
checkpoint restore either update every child consistently or leave the prior generation authoritative.

MTP does not share the target page table. Its carry state and cache length must follow the target’s
accepted frontier, using the latest Buun checkpoint companion support. Rejecting draft tokens rolls back
MTP and target write-page metadata without evicting unrelated target pages.

Server slot reuse must mint a new session/sequence generation before any old async completion can
publish. Prompt-cache artifacts may seed host pages, but live mutable pages must not be published as
immutable prefix artifacts until sealed under the existing artifact contract.

### 8.10 Exact page-wave reference

Exact mode reuses the host catalog, page table, native positions, Turbo4 FA tile loader, and bounded
transfer scheduler, but it does not run retrieval or omit pages. For each layer/query/head, every
partition returns an unnormalized online-softmax state `(m, l, o)`: local maximum logit `m`, shifted
exponential sum `l`, and shifted weighted-value vector `o`. Merge two states `a` and `b` as:

```text
m = max(m_a, m_b)
l = l_a * exp(m_a - m) + l_b * exp(m_b - m)
o = o_a * exp(m_a - m) + o_b * exp(m_b - m)
attention_output = o / l              # only after all partitions
```

Process the current hot table and then every cold logical page exactly once in deterministic bounded
waves. Double-buffer H2D staging and compute only after the serial reference passes. The page-coverage
ledger rejects duplicates, gaps, stale generations, bad tails, or mismatched native masks. The first
implementation stays on GPU because Buun CPU kernels currently fall back from Turbo types to Q8_0;
adding a CPU Turbo4 partial-attention kernel is an optional later optimization, not an exactness shortcut.

## 9. Internal interfaces to add

Names are provisional and internal. Reuse existing files where ownership is clear; avoid public
`llama_context_params` fields until maintainers approve the CLI/API.

### 9.1 Page descriptor and table

- `llama_kv_page_id`: logical page plus session/sequence generations;
- `llama_kv_page_record`: position digest, mutation/representation generations, state, host and
  physical handles, pins, scores, fences;
- `llama_kv_residency_snapshot`: immutable table version consumed by one graph;
- `llama_kv_residency_tx`: plan/reserve/transfer/recheck/publish/rollback phases.

### 9.2 Backend residency pool

Add a distinct provisional backend object such as `ggml_backend_kv_residency_pool`. It owns reserved
address ranges or compact physical slots, logical-page mappings, upload events, and resident-byte
accounting. Its minimum internal operations are reserve, map, unmap/drop, asynchronous upload, optional
download/reseal, residency query, and resident-byte query.

Factor/share low-level CUDA VMM allocation, map/unmap, and event helpers with `ggml_vbr_vmm_pool`; do
not add token-page semantics or page-policy state to the representation pool. VBR is allowed to change
an encoding only through the composite coordinator, which invalidates/reseals affected host pages and
advances both representation and residency generations atomically.

### 9.3 Selected capture/adoption seam

Extend VBR capture/adoption below the whole-prefix manifest layer with a bounded list of logical
256-cell page ranges. The adapter must produce/consume existing representation descriptors and segment
chains and must use the pinned ring. It must never build one multi-gigabyte `std::vector`.

Keep full prompt artifacts unchanged. Page backing may use a smaller catalog envelope with the same
identity/checksum/generation primitives. A page restore into an occupied slot uses occupied-replacement
guards and publishes only after every layer/side segment succeeds.

### 9.4 Memory attention view

Add an internal `llama_memory_i`/KV-cache view that supplies:

- compact K/V tensors or a page table;
- native logical positions and masks;
- table epoch as a graph reuse key;
- physical row count bounded by resident capacity;
- page pin/fence lifetime tied to graph completion.

Do not make `get_n_kv()` equal the 262K logical frontier in selected mode; that would make attention and
graph scratch scale with logical context. It must reflect the compact selected row count.

### 9.5 CUDA operator

The final operator needs Turbo4 K and V, GQA, native position/mask semantics, arbitrary selected page
order, tail-page masking, and per-page score reduction. Start from current Turbo4 FA loaders in Buun and
the sparse/block metadata shape in upstream PR #27970 / community PR #22569. It must not dequantize the
whole cache into F16 scratch.

## 10. CLI and configuration contract

Keep initial switches fork-local and experimental. Proposed shape:

```text
--spec-draft-kv-device auto|gpu|cpu
--kv-pager off|observe|selective|exact
--kv-page-size 256
--kv-vram-budget SIZE|auto
--kv-host-budget SIZE
--kv-pin-recent 24576
--kv-hotset-policy attention
--kv-hot-tokens 77824               # diagnostic override; budget remains authority
--kv-router-top-k N
--kv-router-explore N
--kv-prefetch-depth N
--kv-pager-debug
```

The exact spelling remains experimental. During transition, an older fork-local `--kv-page-*` spelling
may be accepted as a deprecated alias, but one normalized configuration object must own the semantics.

Required companion flags remain Turbo4 target and draft types. The current parser accepts
`-ctk t4`, `-ctv t4`, `-ctkd t4`, and `-ctvd t4` (and the `turbo4` spelling).

Fail closed for unsupported backends, non-causal attention, incompatible K/V types, multiple target
VBR authorities, unsupported sequence layouts, missing host budget, and insufficient MTP reservation.
`off` must preserve existing behavior exactly.

## 11. Execution phases and gates

The authoritative task order, status, cluster, model recommendation, and packet path live in
`WORK_STATE.json`. Each packet is self-contained for Luna Medium (or Luna High where the task is
cross-repository, CUDA, lifecycle, or numerically demanding) and names its reads, allowed writes,
tests, stop conditions, and handoff evidence.

All task model recommendations are now Luna-family only: Luna Low for genuinely checklist/documentation
or pure-arithmetic work (`00-01`, `00-02`, `00-04`, `01-01`, `01-03`), Luna Medium for ordinary
implementation/benchmark/policy work, and Luna High for cross-worktree placement, residency
transactions, CUDA/operator integration, concurrent lifecycle, CUDA acceptance, and exact numerical
reference work. The runner converts these labels to `gpt-5.6-luna` with the corresponding reasoning
effort; no task relies on Terra or Sol.

### Phase 00 — governance and reproducible baseline

1. Resolve the human upstream issue/discussion gate and record URLs without AI-authored GitHub text.
2. Pin commits, GGUF/model hash, build flags, GPU/driver, and profile.
3. Reproduce all-GPU 77K, full-context CPU KV, and Turbo4 MTP baselines with a shared prompt corpus.
4. Produce a salvage matrix from the old prototype and community pager.
5. Add independent draft-KV placement first on a connected `vanwho/llama.cpp` branch, port the same
   contract to Buun, verify native/external speculative contexts, and document parser-supported Turbo4
   draft types in Buun.

Gate: repository provenance is immutable, baseline artifacts are machine-readable, no result is claimed
without raw logs, and the draft placement truth table passes without changing legacy `auto` behavior.

### Phase 01 — contracts before data movement

1. Freeze selective-vs-exact semantics and observability.
2. Implement/test pure page identity, position runs, forward/reverse map, and table epochs.
3. Implement/test target/MTP/recurrent/summary/transfer budget arithmetic.
4. Implement/test pure retrieval and retention policy replay.

Gate: CPU-only deterministic tests cover ABA, duplicate map, partial tail, pins, hysteresis, and budget
rounding. No CUDA mutation is allowed before this gate.

### Phase 02 — KV host-page residency substrate

1. Add bounded selected-page capture/adoption descriptors.
2. Add host page catalog/storage with checksums, budget, and stale-entry rejection.
3. Add batched async D2H/H2D transfer plans through existing backend/pinned-ring APIs.
4. Add transactional residency publication and rollback.
5. Fill/seal 256K host backing and exercise a fixed/manual 304-page GPU window without dynamic policy.

Gate: fake-backend fault injection proves that failure before/after every phase leaves the old table
valid and leaks no pin, catalog charge, or physical slot. The fixed-window CUDA proof, when hardware is
available, shows a sealed clean eviction has zero D2H bytes and target GPU payload stays bounded.

### Phase 03 — bounded physical attention path

1. Materialize a compact selected-cache reference view with native positions.
2. Add backend-neutral block/page table operator plumbing.
3. Add Turbo4 CUDA paged/sparse FA decode kernel plus optional score reduction.
4. Integrate prefill/decode graphs and graph-cache epoch invalidation.

Gate: selected-all-pages equals existing Turbo4 FA within tolerance at small context; permuted physical
slots and gapped native positions match a gather-based reference.

### Phase 04 — dynamic attention policy

1. Build and update all-page routing summaries.
2. Collect actual per-page attention telemetry in observe mode.
3. Implement the hot-set controller with pins and hysteresis.
4. Add predictive prefetch, bounded faults, cancellation, and backpressure.

Gate: trace replay retrieves synthetic cold needles, exploration prevents permanent starvation, and a
stable focus workload reaches high prefetch hit rate without oscillation.

### Phase 05 — Qwen3.8/MTP/server integration

1. Make hybrid recurrent/page transactions atomic.
2. Reserve full-context Turbo4 MTP in VRAM and keep it out of pager authority.
3. Integrate server slots, clears, prompt reuse, cancellation, and concurrent requests.
4. Integrate MTP carry/checkpoints/speculative rollback with page generations.

Gate: rejection, slot reuse, and checkpoint restore cannot publish stale pages; MTP remains GPU-resident
under target page pressure.

### Phase 06 — correctness and performance campaign

1. Land CPU/fake-backend tests in existing test targets unless maintainers approve a new test file.
2. Run CUDA correctness and sanitizer/fault matrix.
3. Add a reproducible paging benchmark/trace harness.
4. Run 256K long-context quality, churn, and throughput acceptance.
5. Implement and validate the exact page-wave reference by merging online-softmax states; consider a
   CPU/GPU split only after a correct CPU Turbo4 attention primitive exists.

Required release gates on the reference system:

- logical target context is 262,144 and no target allocation scales secretly to full GPU residency;
- target hot payload is near 304 pages, with exact ledger output;
- MTP Turbo4 K/V is full-length and GPU-resident;
- every sealed target page has canonical Turbo4 host backing and cold pages remain available in RAM;
- focused steady-state throughput is at least 2x the ordinary CPU-KV baseline and within 20% of the
  comparable 77K all-GPU path after warmup; stretch goal is 3x CPU-KV;
- page-fault and churn workloads disclose degraded throughput rather than hiding stalls;
- retrieval/needle and conversation quality thresholds are defined before tuning and met afterward;
- exact page waves cover every valid logical position exactly once and match dense all-pages Turbo4
  within a predeclared numerical tolerance on bounded fixtures;
- `off` mode matches unmodified Buun performance/correctness within benchmark noise.

### Phase 07 — upstreamable slicing

1. Separate generic, Buun-VBR, Turbo4-CUDA, and Qwen3.8/server changes into reviewable slices.
2. Update docs and final evidence; user-authorized fork commits/pushes may be made by the outer agent,
   while a human owns upstream-facing issue/PR prose, review replies, PR creation, and merging.

### 11.1 Connected branch and commit discipline

Keep the initial implementation Turbo4-locked and do not combine it with Buun's dynamic VBR precision
ladder until selective and exact-reference gates pass. Use dependency-ordered connected branches such
as `pager/00-draft-placement`, `pager/10-page-core`, `pager/20-host-backing`,
`pager/30-t4-paged-fa`, `pager/40-telemetry`, `pager/50-policy`, `pager/60-qwen-server`, and
`pager/70-exact-reference`. Names may change, but each branch/commit series must have one responsibility,
tests at its boundary, and no unrelated generated churn.

Before beginning a slice, fetch and fast-forward the appropriate `vanwho/*` fork's default branch from
upstream in a clean worktree, re-read repository instructions, and record the base SHA. Fork-only
feature branches must be based on that recorded default-branch SHA; do not merge upstream into an
in-progress slice. Before handoff, rebase or
range-diff deliberately and rerun the slice's tests. The clustered runner remains `CODEX_GIT_MODE=local`
and never commits automatically; the user-authorized outer agent may review, commit, and push fork-only
branches. Upstream submissions remain a later explicit, human-owned action.

## 12. Test matrix

### 12.1 Pure/unit tests

- page-number and tail arithmetic at 0, 1, 255, 256, 257, 77,824, and 262,144;
- row-byte and payload math from actual tensor types/dimensions;
- logical↔physical bijection and deterministic victim ties;
- sequence generation ABA and stale async completion rejection;
- dirty victim, pinned victim, all-pinned exhaustion, and host-budget exhaustion;
- partial D2H/H2D failures at every transaction phase;
- table epoch/graph reuse rules;
- routing top-K, structural union, exploration, retention EMA, and hysteresis trace replay;
- recurrent/MTP companion rollback.

### 12.2 GPU correctness

- T4/T4 K/V, GQA=4, head width 256;
- one page, tail page, 304 pages, permuted slots, logical gaps, and mixed contiguous runs;
- decode batch 1 and supported multi-sequence batches;
- prefill chunks and transition from prefill to decode;
- score reduction on/off produces the same attention output;
- compare compact gather reference, paged kernel, existing dense T4, and F16 quality reference;
- compute-sanitizer on bounded fixtures;
- CUDA graph capture/reuse across unchanged and changed table epochs.

### 12.3 Lifecycle/fault tests

- server cancellation during each transfer phase;
- slot reuse while old event completes;
- `seq_cp`, `seq_rm`, shift, rewind, clear, and full reset;
- MTP draft accept/reject sequences and checkpoint restore;
- host corruption, short read, wrong codec/topology/model digest;
- VRAM map denial, pinned-ring exhaustion, host-budget pressure, and all pages pinned;
- clean shutdown with transfers in flight.

### 12.4 Performance evidence

Record build SHA/options, model SHA, full CLI, driver/runtime, clocks/power state where available,
prompt hash, warmup, repetitions, median/p10/p90, pp/tg, acceptance rate, VRAM/RAM peak, copy bytes,
faults, prefetch hits, and per-stage time. Compare:

1. current all-GPU 77K Turbo4 target + Turbo4 MTP;
2. current full logical context with ordinary CPU KV placement;
3. pager observe mode;
4. pager selective warm focus;
5. pager selective cold-needle;
6. pager selective adversarial page churn;
7. pager exact page-wave correctness/quality reference (reported separately from speed gates).

The canonical existing service harness is `/srv/ai/benchmarks/run-profile-benchmark.sh`; its contract
is documented in `/srv/ai/benchmarks/README.md` and dated results in
`/srv/ai/benchmarks/benchmarks.md`. Do not replace it with an incomparable microbenchmark. Extend it
carefully (the `/srv/ai` worktree already contains unrelated user edits) or add a repo-local adapter
that invokes it without changing its established `run-config.json`, `records.jsonl`, raw response,
and summary formats.

The harness already performs clean isolated profile startup, restores the previously active profile,
captures model/backend/GPU/profile identity, runs warmups and repeated trials, records MTP acceptance,
and writes partial summaries on interruption. Pager integration must add, without removing existing
fields:

- pager mode, page size, logical/hot token counts, host/VRAM budgets, router K, exploration, recent
  window, and prefetch depth;
- target/MTP physical placement and K/V codec evidence from server startup;
- target page faults, useful prefetches, evictions, canceled/stale transfers, D2H/H2D useful and actual
  bytes, queue/copy/wait time, selected-page count, and attention-table epoch changes;
- peak and steady VRAM/RAM plus transfer-ring bytes;
- prompt/corpus identity and expected needle answers.

Use its current modes as fixed historical anchors:

- `BENCH_SIZE=default`, thinking off, one warmup and three measured 400-token requests per prompt for
  short decode/MTP acceptance;
- `BENCH_SIZE=large`, approximately 44K input tokens, two warmups and five measured trials for scratch
  pressure and long prefill/decode;
- a new pager corpus/variant for focus locality, cold needles at several page distances, focus shifts,
  and adversarial churn. Keep the same request/manifest plumbing so results can be joined.

Existing reference observations that must appear in the Phase 00 baseline report include:

- production Fast: 77,824 target context, target and MTP Turbo4, batch/ubatch 1024/256;
- RTX 4080 reported capacity: 16,376 MiB;
- 44,018-token Fast validation at 77,824 passed at 1,411.64 prompt tok/s and 37.43 decode tok/s;
- 81,920 failed its long prompt on the context-scaled F16 dequant scratch despite passing startup;
- recorded large Fast/off median: 71.05 decode tok/s and 1,326.38 prompt tok/s over five trials;
- short Turbo4-vs-F16 MTP aggregate acceptance: 81.591% vs 80.974%, with about 214 MiB startup VRAM
  recovered in that configuration.

These historical values are orientation, not acceptance results for a new build. Re-run the exact
corpus against the pinned base and pager candidate, and preserve both manifests.

## 13. Upstream contribution rules

The current `CONTRIBUTING.md` in Buun and current llama.cpp `CONTRIBUTING.md`/`AGENTS.md` were read in
full. Every execution packet must re-read applicable instructions because upstream may change.

The synchronized fork defaults at this revision are Buun `cb703be37` and llama.cpp `67a17c17`; both
matched their upstream and `origin` default branches when re-fetched on 2026-09-03. Task 00-02 records
the durable provenance, and every later slice rechecks rather than trusting this sentence.

Mandatory process:

- search existing issues/PRs first and link #21961, #22569, and #28115 where relevant;
- a feature starts with an issue/discussion and maintainer agreement on direction;
- first-time contributors submit one PR at a time;
- keep changes simple, focused, formatted, and tested;
- do not add a new test file without maintainer approval; prefer an existing target/file;
- disclose AI assistance as required, while the human author understands every line;
- the clustered runner does not commit or push;
- the user-authorized outer agent may create commits and push iterative branches only to `vanwho/*`;
- a human writes upstream GitHub posts, issue/PR descriptions, and reviewer replies, creates upstream
  PRs, and performs merges;
- use reviewed, focused commits and preserve a clean range-diff for any upstream-bound work.

The clustered runner must therefore use `CODEX_GIT_MODE=local`. Its historical managed mode is not
permitted for this project’s upstream-bound work.

## 14. Runner and resume procedure

The shared runner now accepts project-specific state clusters and a no-Git mode. From this worktree:

```bash
CODEX_PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
CODEX_GIT_MODE=local \
/srv/codex/run_until_complete_clustered.sh --status

CODEX_PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
CODEX_GIT_MODE=local \
/srv/codex/run_until_complete_clustered.sh --show-clusters
```

For an unattended fork-only run that creates task branches, commits each completed task, pushes the
branches, merges them into the fork's integration branch, and continues until every task is `done` or
`deferred`, invoke the shared runner directly in managed mode. The checked-in wrapper must not be used
for this variant because it deliberately forces `CODEX_GIT_MODE=local`. Keep the plan branch as the
integration branch until the execution package and its history have been deliberately reviewed:

```bash
cd /srv/repos/vanwho/buun-llama-cpp
CODEX_PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
CODEX_PROJECT_REMOTE=origin \
CODEX_PROJECT_BRANCH=plan/attention-aware-kv-paging \
CODEX_GIT_MODE=managed \
CODEX_SESSION_MAX_TURNS=4 \
CODEX_SESSION_MAX_INPUT_TOKENS=90000 \
/srv/codex/run_until_complete_clustered.sh
```

Do not set `MAX_TASKS_PER_RUN`; its default `0` means no per-run task cap. Task packets supply the
Luna Low/Medium/High recommendation, so no model override is needed. Managed mode operates only on the
configured `origin` fork, creates temporary `codex/task-<id>` branches, pushes them, merges each into
`plan/attention-aware-kv-paging`, and removes the temporary remote branch. It still stops on a persisted
blocked task, the runner stop/pause controls, an unavailable required artifact, a failing hard gate, or
an unrecoverable restart/health-check failure. Upstream branches, issues, pull requests, and merges remain
human-owned. To inspect before or after a run, use `CODEX_GIT_MODE=local` with the wrapper's `--status`
or `--show-clusters` commands above.

The 33 tasks are divided into 16 contiguous ownership-oriented clusters of one to three tasks; the
mapping and rationale live in `docs/execution/clusters/README.md`. Each fresh/resumed runner prompt
loads `docs/execution/clusters/<cluster>.md` and dependency handoffs before task-specific source. The
checked-in wrapper defaults to four turns or 90,000 reported input tokens per cluster thread, after
which the runner rotates to a fresh session. This prevents a five-task phase from carrying unrelated
repository, CUDA, benchmark, or numerical context.

For supervised review/commit boundaries, run one task at a time while retaining saved cluster threads:

```bash
MAX_TASKS_PER_RUN=1 tool/codex/run_clustered.sh
```

The outer agent can then review, commit, and push the completed fork-only task before invoking the same
command again. Omitting `MAX_TASKS_PER_RUN` continues through all tasks in local-Git mode and preserves
changes in one working tree; it does not create the connected commits described in section 11.1.

Task `00-01` is complete: the user authorized experimental commits in the `vanwho/*` forks while
upstream submission remains deferred. Normal execution now resumes at `00-02`, the first task not
`done` or `deferred`. Every task maintains `docs/execution/handoffs/<task>.md`; raw benchmark artifacts
belong under ignored build/result storage, with stable summaries linked from the handoff.

The status helper supports:

```bash
python3 tool/codex/task_state.py validate
python3 tool/codex/task_state.py show
python3 tool/codex/task_state.py start 00-02
python3 tool/codex/task_state.py checkpoint 00-02 --summary "..."
python3 tool/codex/task_state.py complete 00-02 --summary "..."
python3 tool/codex/task_state.py block 02-03 --reason "..."
python3 tool/codex/task_state.py unblock 02-03 --summary "blocker resolved ..."
```

Never mark a hardware gate complete without its raw output. Use `deferred` only when the plan explicitly
permits deferral; lack of the reference CUDA system is a blocker for Phase 06 release acceptance.

## 15. Risks and explicit non-goals

### Highest risks

- routing recall, not page transfer mechanics, determines long-context quality;
- a 4.125-MiB cross-layer page can amplify transfers under rapidly shifting focus;
- graph/scratch allocation may reduce the safe hot set below 77K;
- CUDA kernels may lose more to irregular page metadata than they save in bounded sequence width;
- MTP and recurrent rollback can expose stale-generation bugs under server concurrency;
- conflating representation/VBR with residency/paging would create ambiguous authority and make both
  lifecycle reasoning and upstream review substantially harder.

### Mitigations

- observe mode and offline trace replay before selective authority;
- structural anchors, exploration, hysteresis, and prefetch;
- compact gather reference before custom paged kernel;
- separate VBR and pager objects under one composite transaction/accounting coordinator;
- fail-closed generation checks and exhaustive fault injection;
- measured gates at every phase and no performance claims from synthetic metadata tests.

### Non-goals for the first implementation

- generic multi-model/backend support;
- a CPU Turbo4 exact-attention implementation or optimized CPU/GPU split (the bounded GPU page-wave
  exact reference remains required late in Phase 06);
- disk/remote cold storage;
- paging recurrent state or live MTP pages;
- changing Turbo4 codec quality;
- merging the Matias branch or the DSV4 VBR feature branch wholesale;
- exposing a stable public C API before the internal design is accepted.

## 16. Definition of done

This project is complete only when a human-reviewed build on the reference machine demonstrates a
262,144-token logical Qwen3.8 session with full Turbo4 MTP in VRAM, approximately 77K target tokens hot
in VRAM, canonical host Turbo4 backing for every sealed target page, stable attention-aware selection,
correct recurrent/speculative/server lifecycle, disclosed quality behavior, and the Phase 06
performance gates.
An exact page-wave run must also match dense all-pages attention within declared numerical tolerance on
bounded fixtures and provide the full-context storage/selection correctness reference; it is not held to
the selective throughput gate. Documentation, raw evidence, status, and reviewable upstream slices must
agree with the actual code.

## 17. Decision ledger for context-free execution

Later agents must treat this table as the concise record of conclusions, not reopen the design from
scratch. A decision marked **measured** may change only with captured evidence recorded in its task
handoff; **provisional API** may change before upstream review without changing the underlying contract.

| ID | Status | Decision and reason | Implementation consequence | Owning tasks |
| --- | --- | --- | --- | --- |
| D01 | Locked | Scope is Qwen3.8-27B/qwen35 on the reference RTX 4080 first. Genericity before proof would multiply lifecycle and kernel risk. | Fail closed on other geometry/backends; no claim of generic support. | 00-02, 01-03, 03-03, 06-04 |
| D02 | Locked | Logical pages contain 256 tokens across all 16 target attention layers and both K/V sides. This matches Buun's generation granularity and gives a 4.125-MiB reference transfer. | One target residency decision and generation per cross-layer page; 1,024 logical pages at 256K. | 01-02, 02-01, 02-05 |
| D03 | Locked | VBR answers “what representation?”; the pager answers “where resident?”. Combining them creates ambiguous authority. | Separate `llama_kv_*` pager/controller and backend residency pool, sharing low-level VMM/events/descriptors under one composite transaction. | 01-01, 01-02, 02-03, 02-04 |
| D04 | Locked | Draft KV placement must be independent before pager work because target CPU placement currently leaks into draft context construction. | Add one tri-state draft policy in both fork worktrees; `auto` preserves legacy behavior. | 00-05, 05-02 |
| D05 | Locked initially | Target and MTP K/V remain Turbo4. Dynamic VBR precision changes are not combined with initial paging. | Direct Turbo4 storage/transfers/FA; representation epoch still guards future cooperation. | 02-01–03-04, 05-02 |
| D06 | Locked | Every sealed target page has canonical host Turbo4 backing; the partial write page remains pinned and GPU-authoritative. Literal per-token write-through is likely excessive. | One batched D2H at seal/reseal; normal clean eviction only drops/unmaps GPU residency and performs zero D2H. | 02-02–02-05 |
| D07 | Measured admission | 77,824/304 pages is a target after reserving weights, graphs, scratch, MTP, recurrent state, routing, staging, and headroom. | Budget is authoritative; diagnostic token override may only reduce admitted capacity. | 01-03, 02-05, 06-04 |
| D08 | Initial policy | Exact 304-page split is 96 recent, 32 structural, 140 attention-selected historical, 36 transient. The originally suggested 24K+8K+36K+9K page-rounded split totaled 78,848. | Deduplicate pools, refuse mandatory overflow, and tune only from traces. | 01-04, 04-03, 06-04 |
| D09 | Locked | Resident attention width must scale with physical rows, not the 262K logical frontier; no hidden full-size GPU target allocation is acceptable. | Start with compact slots/reference view; sparse VMM is allowed only when mappings and ledgers prove bounded physical occupancy. | 02-05, 03-01, 03-02 |
| D10 | Locked | Paged FA consumes Turbo4 K/V directly and dequantizes tiles in registers/shared memory. | Never build a whole-cache F16 gather buffer. | 03-03, 03-04 |
| D11 | Locked | Retrieval of cold candidates and retention of observed resident pages are different problems. Resident attention alone cannot rediscover a cold page. | All-page routing summaries plus anchors/exploration feed retrieval; FA attention EMA/peak feeds retention. | 01-04, 04-01–04-03 |
| D12 | Measured | GPU reduces page attention statistics; CPU/controller receives bounded summaries only. Fixed score weights are unjustified before normalization. | About 4 KiB for 1,024 float scores; calibrate normalization/weights with captured trace replay. | 04-02, 04-03 |
| D13 | Locked | Promotions are asynchronous and predictive; partially transferred pages are never published. | Dedicated upload stream, bounded/double-buffered staging, events, revalidation, atomic table swap, explicit not-ready behavior. | 02-03, 02-04, 04-04 |
| D14 | Locked | MTP owns a separate full-length 262,144-token Turbo4 GPU allocation and is never a target pager victim. | Reserve/account its expected 264-MiB payload before target hot pages and keep rollback/frontier atomic. | 01-03, 05-01–05-04 |
| D15 | Locked | Selective mode is approximate. Exact mode is a late correctness/quality oracle, initially using bounded GPU page waves because CPU Turbo4 attention is absent. | Merge per-partition online-softmax `(m,l,o)` states; CPU/GPU split is optional later. | 01-01, 06-05 |
| D16 | Locked | Off mode and existing prompt artifacts must remain compatible; state/server/MTP/recurrent changes publish as one generation-safe operation. | Fail closed on stale identity, rollback every partial operation, and test cancellation/slot reuse/checkpoints. | 02-04, 05-01–06-02 |
| D17 | Locked | Upstreamability requires small connected branches and repository conventions. | Generic draft placement first; page core, backing, FA, telemetry, policy, integration, and exact work remain separate; no giant initial PR. | 00-05, 07-01, 07-02 |
| D18 | Locked | Performance claims use the existing `/srv/ai/benchmarks` result contract and same model/corpus controls. | Preserve raw manifests; compare all-GPU 77K, ordinary CPU KV, observe, selective focus/needle/churn, exact separately. | 00-03, 06-03–06-05 |
| D19 | Locked | Task execution uses Luna Medium by default, Luna Low only for genuinely simple checklist work, and Luna High only where cross-repo, CUDA, lifecycle, or numerical risk warrants it. | `WORK_STATE.json` recommendations are all `Luna Low`, `Luna Medium`, or `Luna High`; runner/session clusters remain model-homogeneous enough for efficient reuse. | 00-01–07-02 |

### 17.1 Deliberately unresolved choices

Do not guess these prematurely:

- public CLI spellings are provisional until maintainer discussion, while semantics are fixed;
- routing-summary shape starts with 4–8 representative rotated K vectors but is selected by measured
  recall, bytes, and latency;
- attention/peak/frequency/recency weights require normalized trace evidence;
- whole-store pinned host allocation competes with pageable backing plus a bounded pinned ring and must
  be benchmarked before selection;
- compact physical slots are the correctness-first layout; sparse VMM mappings may replace them only if
  arbitrary page addressing and physical savings are demonstrated;
- multi-slot/multi-sequence pager authority is unsupported initially and must not be inferred from
  whole-artifact multi-GPU support;
- optimized CPU/GPU exact attention waits for a real CPU Turbo4 primitive.

## 18. Source and evidence pointer index

| Subject | Primary pointer | Why it matters |
| --- | --- | --- |
| Canonical execution | `docs/execution/WORK_STATE.json`, `docs/execution/tasks/`, `docs/execution/clusters/` | Sole task order, task acceptance, and per-session context |
| Resume tooling | `tool/codex/task_state.py`, `tool/codex/run_clustered.sh`, `/srv/codex/run_until_complete_clustered.sh` | Validates state, resumes at the first unfinished task, and reuses only one cluster's session |
| Buun VBR/VMM | `src/llama-vbr-*`, `src/llama-kv-cache.*` | Representation epochs, VMM precedents, transactions, capture/adopt, generations, pinned ring |
| CUDA VMM backend | `ggml/src/ggml-cuda/vbr-vmm.cu`, `vbr-vmm-policy.h` | Low-level reserve/map/unmap/physical-accounting primitives to factor beneath a separate pager pool |
| Turbo4 FA CUDA | `ggml/src/ggml-cuda/fattn-mma-turbo.cuh`, `fattn-common.cuh`, `fattn.cu`, `template-instances/` | Existing direct Turbo K/V tile loaders, dispatch, and generated specializations |
| GPU top-k | `ggml/src/ggml-cuda/top-k.cu` | Candidate implementation for bounded all-page summary selection after measurement |
| Host artifacts | `src/llama-vbr-artifact-{capture,adopt,stage,validate}.*`, `src/llama-vbr-explicit-capture.*` | Bounded segment chains, identity/checksum validation, occupied replacement |
| Budget/accounting | `common/fit.cpp`, `src/llama-cache-accounting.*`, `src/llama-cache-budget.*` | Reserve MTP/scratch/headroom before deriving target page capacity |
| Hybrid/recurrent ownership | `src/llama-memory-hybrid.*`, `src/llama-memory-recurrent.*`, `docs/qwen4-vbr-plan.md` | Composite operations and exact non-attention companions |
| MTP construction | `common/speculative.*`, `common/common.cpp`, `tools/server/server-context.cpp` | Current target/draft placement coupling and native/external context creation |
| Generic draft-placement worktree | `/srv/repos/vanwho/llama.cpp-kv-pager` on `pager/00-draft-placement` | Prepared clean branch based on synchronized llama.cpp `67a17c17`; task 00-05 may edit it |
| Server retention | `common/common-retention-sidecar.*`, `tools/server/server-cache-yield.*` | Reusable bounded ranking/accounting vocabulary, not page-attention authority |
| Relevant Buun history | commits `34941d33b`, `2714303b5`, `2174ad63`, `283ba19ed`, `65eb44ebc`, `87b37eac9`, `764f3a044`, `cb703be37` | Prompt lifecycle, restore, multi-GPU, entry tiers, batched maps, MTP fit/carry safety |
| Relevant llama.cpp history | commits `2d8d612e4`, `8e93a9773`, `36b101543`, `0eadefebd` | Non-contiguous restore batching, sparse FA plumbing, Qwen sparse positions/rollback |
| Community pager | `/srv/repos/matiaslin/llama.cpp`, Discussion #21961, draft PR #22569 | Block manager/table and operator reference; not directly mergeable |
| Old prototype | `/srv/ai/paged-kv/repos/buun-llama-cpp` | Reference-only identity/table/policy ideas; dirty and never modified by this project |
| Archived original plan | `/srv/ai/paged-kv/qwen38_256k_attention_aware_kv_paging_implementation_plan.md` | Historical rationale only; its banner points here |
| Model/profile | `/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf`, `/srv/ai/config/profiles/qwen38-fast.env` | Actual geometry, Turbo4 types, MTP2, 77,824 baseline profile |
| Benchmarks | `/srv/ai/benchmarks/README.md`, `benchmarks.md`, `run-profile-benchmark.sh` | Required comparable harness and output schema |
| Raw historical evidence | `/srv/ai/paged-kv/results/` | Prior performance/MTP orientation; never substitute it for new-build acceptance |

Task packets narrow these pointers further. A low-reasoning agent reads repository instructions, state,
its cluster context, its task packet, the prior dependency handoffs, and only then the explicitly named
source files. It does not need this entire plan in every task session.

## 19. Agentic-run preflight and genuine blockers

The clustered runner can execute the implementation sequence without conversational clarification when
these preconditions hold. This section distinguishes a safe automatic retry/defer from a condition that
must stop the run; agents must not “solve” a blocker by ad-hoc killing an unrelated service, weakening a
test, inventing hardware evidence, or publishing upstream text. The user has authorized controlled
stop/restart of the active Qwen service through the established benchmark/profile workflow.

| Condition | Runner behavior | Can the agent resolve it? |
| --- | --- | --- |
| A task is incomplete, Codex hits a transient network/rate-limit, or a cluster exceeds its context guardrail | Retry with the same task/session, then rotate the cluster session; preserve state/handoff. | Yes, automatically, subject to service availability. |
| The current task is blocked in `WORK_STATE.json` | Exit with code 3 and preserve the concrete blocker. | No; an external state change or user decision is required. |
| Required model/sidecar is absent or metadata differs from the pinned Qwen3.8 geometry | Stop at provenance; do not guess or expand scope. | No, unless the artifact becomes available or the user changes scope. |
| Relevant upstream default branch moves after provenance | Stop and request a deliberate re-sync/rebase decision; never merge over local work. | Not safely without an explicit policy choice. |
| A live service would be disrupted by `/srv/ai/benchmarks` | Enter a controlled maintenance window: capture active profile/PID/command/health, let the established profile harness stop/restart the service with non-interactive sudo, and verify restoration plus health afterward. Never kill an unrelated service or leave a failed restart hidden. | Yes, now that the user has explicitly authorized the plan to bring down/restart the active server and passwordless sudo is available. A restart failure remains a blocker. |
| Reference RTX 4080/CUDA is unavailable | Complete pure/fake-backend work, record hardware checks as deferred where packets permit, and stop release acceptance that requires CUDA evidence. | No; the requested speed/capacity goal cannot be honestly certified without the reference system. |
| Selective quality, exact coverage, or throughput gate fails | Preserve raw evidence and mark the task blocked; do not retune thresholds after seeing results. | No; the implementation must be repaired or the user must accept a changed goal. |
| Upstream issue/PR direction, prose, review, or merge is needed | Prepare local slice maps/evidence only; leave GitHub-facing action for the human. | No, but this does not block fork-local implementation completion. |
| Local-Git mode has uncommitted changes | Continue agentically if desired; status/handoffs are persisted. Use `MAX_TASKS_PER_RUN=1` for an outer agent to review/commit between tasks. | Yes for code execution; fork commit/push is an outer-agent action. |
| Another clustered runner holds `.codex-runner/runner.lock`, or stop/pause is requested | Wait for pause/lock or exit safely; never run two writers concurrently. | Yes after the other run exits or the operator clears the control condition. |

Current preflight facts were checked on 2026-09-03: `nvidia-smi` reports an RTX 4080 with 16,376 MiB,
the Qwen3.8 GGUF is readable, both fork defaults match their upstream defaults, Codex CLI is installed,
passwordless sudo is available for the established activation workflow, and a Qwen `llama-server` is
active on port 8080. The user has authorized controlled service restart, so this is a planned preflight
action rather than a standing blocker. The benchmark harness's profile restore and post-run health check
must succeed before the task is complete.
