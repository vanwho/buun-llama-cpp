# Qwen3.8 256K attention-aware Turbo4 KV paging plan

Status: corrective execution phase ready, revised 2026-09-04
Canonical worktree: `/srv/repos/vanwho/buun-llama-cpp`
Pinned Buun base: `cb703be37e3628dadb71912f3b3b25b82090555b`
Compared llama.cpp base: `67a17c17caa95742186f8b1ecadd1b5abd6d5ebb`
Execution state: `docs/execution/WORK_STATE.json`

## 1. Outcome

Build an opt-in Qwen3.8 path, validated through the model's 262,144-token
maximum context, in which:

- all target and MTP K/V are TurboQuant Turbo4;
- the number of target-attention tokens physically resident in VRAM is derived at runtime from the
  requested context, actual model/MTP allocations, graph and kernel scratch, transfer staging, and
  configured safety headroom;
- every sealed target-attention page has a canonical opaque Turbo4 copy in CPU RAM; hot pages also have
  a GPU mapping, while transfers use either measured pinned backing or a bounded pinned staging ring;
- the MTP draft K/V capacity exactly follows the resolved target context for the current run, remains
  resident in VRAM, and is never selected as a paging victim;
- recurrent/linear-attention state remains exact and GPU-resident;
- a small all-page routing index can recall relevant cold pages;
- selected target pages are compacted into a bounded physical working set before Turbo4 flash attention;
- transfers are page-batched and asynchronous, so a focused workload is much faster than ordinary
  `--no-kv-offload` execution.

The previous Fast profile is a historical all-GPU control, not a pager capacity target and not
a permitted production default. Admission must derive the largest safe multiple of the runtime page
size after model weights, graphs, Turbo4 dequant scratch, the context-sized MTP cache, recurrent state,
transfer rings, routing summaries, allocator granularity, and configurable VRAM headroom have been
reserved. Every run must report the resulting hot-page count and explain every reserved byte. No
production constant, default, policy split, test expectation, or operator example may encode the old
estimate. Historical evidence files may retain the value only when naming the configuration that was
actually measured.

### 1.1 Portability and repository-boundary invariant

The implementation is Buun/llama-compatible code, not a deployment script for this particular server.
All committed source, headers, tests, CLI options, backend abstractions, and user-facing documentation
must be portable to another checkout, model path, host OS, GPU, and supported backend. In particular,
implementation files must not contain `/srv/ai`, `/srv/repos`, `/home/ninja`, this server's systemd unit
names, process IDs, fixed ports, absolute model/profile paths, or assumptions that an RTX 4080 is present.
Use existing repository configuration/CLI seams, runtime capability and GGUF metadata checks, injected
allocators/streams, and caller-provided paths instead.

Qwen3.8-27B is the first acceptance fixture and may have a dedicated model-capability adapter, but its
geometry, Turbo4 choice, page size, maximum context, and discovered hot set are runtime metadata or
test inputs—not compile-time global constants hidden in generic pager/VMM/attention code. Unsupported
geometry must fail closed with a clear diagnostic. `/srv/ai` paths, profiles, service lifecycle commands,
machine facts, benchmark result directories, and raw manifests belong only in fork-local execution docs,
handoffs, ignored build/evidence storage, or the external `/srv/ai` benchmark repository; they must never
be required by the Buun library or binary at runtime.

Every implementation task must inspect its changed-file diff for this invariant. The final documentation
task must run a changed-file portability scan and record any intentional test fixture exceptions. A code
commit that embeds server-specific paths or operational state fails review even if the local benchmark
passes.

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
- canonical existing server profile: `/srv/ai/config/profiles/qwen38-fast.env`;
- native context: 262,144 tokens;
- 65 total blocks: 64 target trunk blocks plus one native MTP block;
- 16 full-attention target layers (`full_attention_interval = 4`);
- four K/V heads;
- K width 256 and V width 256 per full-attention layer;
- 4.125 effective bits/value for Turbo4 rows, including block scales.

The plan must re-read the deployed GGUF metadata and fail closed if this geometry differs. No generic
API should silently assume it.

The existing `qwen38-fast.env` is the canonical Qwen3.8-27B UD-IQ4_XS/Turbo4/MTP baseline and
the starting point for every live comparison. The existing `qwen38-big.env` is the same model's
ordinary non-MTP/control profile and remains the paired CPU-KV comparison and rollback target;
the harness must capture whichever profile is active and preserve the tested profile after a successful
run for the next task; restoration is explicit for control/revert, teardown, failed-start recovery, or
final cleanup. The historical
`qwen38-fast.env` `CTX=77824` value is retained only as a reproducibility control; it is **not** a
pager or MTP default. New runs must clone the appropriate existing profile and set context from
the benchmark ladder, with native MTP capacity resolved to that same requested context on every
run. No production code may restore that historical number.

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
| Target VRAM working set | `H * P` / `H` | `H * target_page_bytes` |
| Target host cold set | `C - H * P` / `ceil(C/P) - H` | derived from valid rows per cold page |
| Full target logical context | `C` / `ceil(C/P)` | `C * target_bytes_per_token` |
| Full MTP VRAM cache | `C` | `C * mtp_bytes_per_token` |
| Target + MTP Turbo4 payload | context-dependent | target host payload plus MTP GPU payload |

These are encoded payloads only. The budget must separately report allocator/VMM granularity,
alignment, page tables, positions, masks, routing summaries, pinned staging, CUDA graph reserves,
recurrent state, and dequant scratch.

### 3.2 Dynamic residency partition

Let `C` be the resolved target context in tokens, `P` the validated page size, `L = ceil(C/P)` logical
pages, and `H` the number of GPU page slots admitted by the measured memory ledger. `H` is never a
compile-time or default page count. The controller partitions `H` after deduplicating mandatory pages:

| Pool | Capacity rule | Purpose |
| --- | --- | --- |
| Mandatory | exact union, no quota | write/tail, in-flight, application pins, structural anchors, and required attention sinks |
| Recent | configured tokens rounded to pages, or calibrated `auto` share of remaining `H` | preserve local continuity |
| Retrieved history | remaining calibrated share | router-selected and attention-retained historical pages |
| Transient/prefetch | bounded calibrated share with a minimum only when `H` permits | overlap promotions and absorb focus changes |

The default `auto` policy is expressed as ratios/minima and clamps against the discovered `H`; it never
contains a server-specific target count. Mandatory overflow is a typed refusal. At very small contexts,
`H >= L` selects all pages and must reduce to dense-equivalent attention. Structural pages come from
explicit server/token metadata, never from guessed natural-language semantics. Final ratios and minima
are locked only after observe traces and quality/performance Pareto measurements.

### 3.3 Why MTP must be Turbo4 and pinned

At the model's maximum context, F16 MTP would consume 1 GiB while Turbo4 consumes 264 MiB, saving
760 MiB. At every other context both values scale with the resolved context; neither allocation may use
the trained maximum implicitly. The current profile and benchmark already prove that Buun accepts Turbo4 for
draft K/V and that acceptance/throughput were indistinguishable within the small sample:

- F16 result: `/srv/ai/paged-kv/results/profile-benchmark-default-fast-20260902-233315`;
- Turbo4 result: `/srv/ai/paged-kv/results/profile-benchmark-default-fast-20260902-233429`.

Therefore “add Turbo4 draft support” is complete before this plan starts. The remaining work is to
make the MTP allocation independently budgeted, sized to the resolved target context, GPU-pinned, and
excluded from the target pager/VBR controller. For native MTP, an explicit draft-context override that
does not equal the resolved target context must fail clearly rather than silently shorten the draft KV.

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

Before policy or a custom kernel, prove the backing design with a fixed/manual GPU window whose capacity
is an explicit fixture input or the output of admission:
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
--kv-host-budget SIZE|auto
--kv-pin-recent TOKENS|auto
--kv-hotset-policy attention
--kv-hot-pages N|auto                # optional upper bound; budget remains authority
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
effort. Retry 1 is only the original task model/reasoning with an artifact-aware direct prefix. Before
retry 2, the task's own model family runs a High-reasoning assessment; before retry 3, the assessor
advances through `luna -> terra -> sol` (Sol is the ceiling). Retry 2 and retry 3 task attempts retain
the original model/reasoning and receive no retry prefix; explicit model overrides may override assessor
models.

### Phase 00 — governance and reproducible baseline

1. Resolve the human upstream issue/discussion gate and record URLs without AI-authored GitHub text.
2. Pin commits, GGUF/model hash, build flags, GPU/driver, and profile.
3. Reproduce the historical all-GPU Fast profile, full-context CPU KV, and Turbo4 MTP baselines with a
   shared prompt corpus; completion phases replace it with a freshly discovered safe all-GPU ladder.
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
5. Fill/seal host backing through the configured context and exercise a fixed/manual GPU window whose
   size is returned by the admission ledger, without dynamic policy.

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

- logical target context reaches the model maximum and no target allocation scales secretly to full GPU residency;
- target hot payload is the largest safe page-rounded result of the measured ledger, with exact output;
- MTP Turbo4 K/V matches the resolved target context and is GPU-resident;
- every sealed target page has canonical Turbo4 host backing and cold pages remain available in RAM;
- focused steady-state throughput is at least 3x the same-context ordinary CPU-KV baseline, targets 5x,
  and reaches at least 70% of the automatically discovered safe all-GPU control at its comparable
  resident attention width after warmup;
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
range-diff deliberately and rerun the slice's tests. The clustered runner remains `GIT_MODE=manual`
and never commits automatically; the user-authorized outer agent may review, commit, and push fork-only
branches. If auto mode is explicitly selected for a fully unattended fork run, its implementation
commit excludes `docs/execution/**`; the subsequent `chore(<task>): record task completion` commit contains
the state, work log, handoff, and other execution metadata. Thus code commits remain suitable for
upstream range-diffs while the controller history stays fork-local. Upstream submissions remain a later
explicit, human-owned action.

## 12. Test matrix

### 12.1 Pure/unit tests

- page-number and tail arithmetic at zero, page boundaries, non-multiple tails, several runtime contexts,
  and the context derived from model metadata;
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
- one page, tail page, several explicit capacities, the admission-derived maximum, permuted slots,
  logical gaps, and mixed contiguous runs;
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

1. historical all-GPU Fast Turbo4 target + Turbo4 MTP, followed by a fresh safe-context ladder;
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

The harness performs clean isolated profile startup, captures the previously active profile, captures
model/backend/GPU/profile identity, runs warmups and repeated trials, records MTP acceptance, and writes
partial summaries on interruption. By default, a successful run leaves its tested profile/server loaded
and health-checked so the next task or retry can inspect and reuse the exact state. Restoration is an
explicit operation for a control/revert benchmark, teardown, failed-start recovery, or final cleanup;
the lifecycle record must state which policy was used. Pager integration must add, without removing existing
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

- historical production Fast target context, target and MTP Turbo4, batch/ubatch 1024/256;
- RTX 4080 reported capacity: 16,376 MiB;
- 44,018-token Fast validation at its recorded historical context passed at 1,411.64 prompt tok/s and
  37.43 decode tok/s;
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

The checked-in wrapper defaults to `GIT_MODE=auto`; use `GIT_MODE=manual` for supervised
upstream-bound work. Auto mode is permitted only for the user-authorized fork-local completion run
documented in section 14; its
implementation and execution-metadata commits remain separate, and no upstream GitHub action is
automated.

## 14. Runner and resume procedure

The shared runner now accepts project-specific state clusters and a no-Git mode. From this worktree:

```bash
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
GIT_MODE=manual \
/srv/codex/run_until_complete_clustered.sh --status

PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
GIT_MODE=manual \
/srv/codex/run_until_complete_clustered.sh --show-clusters
```

For an unattended fork-only run that creates task branches, commits each completed task, pushes the
branches, merges them into the fork's integration branch, and continues until every task is `done` or
`deferred`, invoke the shared runner directly in auto mode with the plan branch explicitly selected as
the integration branch. Keep the plan branch as the
integration branch until the execution package and its history have been deliberately reviewed:

```bash
cd /srv/repos/vanwho/buun-llama-cpp
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
PROJECT_REMOTE=origin \
PROJECT_BRANCH=plan/attention-aware-kv-paging \
GIT_MODE=auto \
CODEX_SESSION_MAX_TURNS=0 \
CODEX_SESSION_MAX_INPUT_TOKENS=0 \
/srv/codex/run_until_complete_clustered.sh
```

Do not set `MAX_TASKS_PER_RUN`; its default `0` means no per-run task cap. Task packets supply the
Luna Low/Medium/High recommendation, so no model override is needed. Auto mode operates only on the
configured `origin` fork, creates temporary `codex/task-<id>` branches, pushes them, merges each into
`plan/attention-aware-kv-paging`, and removes the temporary remote branch. It still stops on a persisted
blocked task, the runner stop/pause controls, an unavailable required artifact, a failing hard gate, or
an unrecoverable restart/health-check failure. Upstream branches, issues, pull requests, and merges remain
human-owned. To inspect before or after a run, use `GIT_MODE=manual` with the wrapper's `--status`
or `--show-clusters` commands above.

The 33 tasks are divided into 16 contiguous ownership-oriented clusters of one to three tasks; the
mapping and rationale live in `docs/execution/clusters/README.md`. Each fresh/resumed runner prompt
loads `docs/execution/clusters/<cluster>.md` and dependency handoffs before task-specific source. The
The shared runner and checked-in wrapper default both proactive session-rotation guardrails to `0`
(disabled). Set a positive `CODEX_SESSION_MAX_TURNS` or `CODEX_SESSION_MAX_INPUT_TOKENS` only when you
deliberately want conversation rotation for context hygiene; these settings do not truncate prompts. The
Codex service's own hard context/usage limits still apply, and the runner can rotate after a resulting
failure.

For supervised review/commit boundaries, run one task at a time while retaining saved cluster threads:

```bash
MAX_TASKS_PER_RUN=1 tool/codex/run_clustered.sh
```

The outer agent can then review, commit, and push the completed fork-only task before invoking the same
command again. Omitting `MAX_TASKS_PER_RUN` continues through all tasks in manual mode and preserves
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
- graph/scratch allocation can materially reduce the safe hot set unless it is included in admission;
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
262,144-token logical Qwen3.8 session with context-sized Turbo4 MTP in VRAM, a target hot set derived
from the runtime memory ledger, canonical host Turbo4 backing for every sealed target page, stable attention-aware selection,
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
| D07 | Superseded by D21 | The old fixed-page admission target was only a historical estimate. | Completion phases must remove it from production defaults and derive capacity from the live ledger. | 08-01, 08-03, 09-02, 14-02 |
| D08 | Superseded by D22 | The old absolute policy partition was a test fixture, not a production default. | Use capacity-relative auto shares plus exact mandatory pins and calibrate them from frozen traces. | 08-03, 11-04, 13-05 |
| D09 | Locked | Resident attention width must scale with physical rows, not the 262K logical frontier; no hidden full-size GPU target allocation is acceptable. | Start with compact slots/reference view; sparse VMM is allowed only when mappings and ledgers prove bounded physical occupancy. | 02-05, 03-01, 03-02 |
| D10 | Locked | Paged FA consumes Turbo4 K/V directly and dequantizes tiles in registers/shared memory. | Never build a whole-cache F16 gather buffer. | 03-03, 03-04 |
| D11 | Locked | Retrieval of cold candidates and retention of observed resident pages are different problems. Resident attention alone cannot rediscover a cold page. | All-page routing summaries plus anchors/exploration feed retrieval; FA attention EMA/peak feeds retention. | 01-04, 04-01–04-03 |
| D12 | Measured | GPU reduces page attention statistics; CPU/controller receives bounded summaries only. Fixed score weights are unjustified before normalization. | About 4 KiB for 1,024 float scores; calibrate normalization/weights with captured trace replay. | 04-02, 04-03 |
| D13 | Locked | Promotions are asynchronous and predictive; partially transferred pages are never published. | Dedicated upload stream, bounded/double-buffered staging, events, revalidation, atomic table swap, explicit not-ready behavior. | 02-03, 02-04, 04-04 |
| D14 | Superseded by D23 | MTP remains separate, Turbo4, GPU-only, and never a target victim, but its capacity must not implicitly expand to the trained maximum. | Completion work derives MTP rows from the resolved target context and reserves the measured bytes before admitting target pages. | 08-02, 09-02, 14-02 |
| D15 | Locked | Selective mode is approximate. Exact mode is a late correctness/quality oracle, initially using bounded GPU page waves because CPU Turbo4 attention is absent. | Merge per-partition online-softmax `(m,l,o)` states; CPU/GPU split is optional later. | 01-01, 06-05 |
| D16 | Locked | Off mode and existing prompt artifacts must remain compatible; state/server/MTP/recurrent changes publish as one generation-safe operation. | Fail closed on stale identity, rollback every partial operation, and test cancellation/slot reuse/checkpoints. | 02-04, 05-01–06-02 |
| D17 | Locked | Upstreamability requires small connected branches and repository conventions. | Generic draft placement first; page core, backing, FA, telemetry, policy, integration, and exact work remain separate; no giant initial PR. | 00-05, 07-01, 07-02 |
| D18 | Locked | Performance claims use the existing `/srv/ai/benchmarks` result contract and same model/corpus controls. | Preserve historical raw manifests; new comparisons use same-context CPU KV, an automatically discovered safe all-GPU control, observe, selective focus/needle/churn, and exact separately. | 08-04–08-05, 12-02, 14-03–14-05 |
| D19 | Locked | Task execution uses Luna Medium by default, Luna Low only for genuinely simple checklist work, and Luna High only where cross-repo, CUDA, lifecycle, or numerical risk warrants it. Three bounded substantive retries are allowed: retry 1 is a direct artifact-aware prefix only; retry 2 receives a same-family High assessment; retry 3 receives the next family from `luna -> terra -> sol` in High mode. | `WORK_STATE.json` retains task recommendations; the runner records recovery count under ignored run state, promotes only the separate assessment model, and keeps every task retry on its original model and reasoning with no prefix after the assessment. | 00-01–16-03 |
| D20 | Locked | Acceptance is Qwen3.8-on-reference-machine-specific, but committed implementation must remain portable Buun/llama code. | Keep server paths, profiles, service/PID/port facts, model filenames, GPU assumptions, and mutable benchmark data outside production source; express geometry through runtime metadata/capability and injected configuration, and scan changed files before upstream slicing. | 00-02, 01-01–06-05, 07-01–07-02 |
| D21 | Locked | Hot target capacity is a runtime result, never a named token/page target. | Compute `H=floor(usable_after_all_reservations/page_charge)`, clamp to logical pages, expose the complete ledger, and permit only user-supplied upper bounds. | 08-03, 09-01–09-02, 12-02, 14-02 |
| D22 | Locked | Auto policy allocation scales with admitted `H`; absolute page counts are valid only as explicit test inputs. | Mandatory pages consume exact capacity first; recent/history/transient shares use calibrated ratios/minima and fail on mandatory overflow. | 08-03, 11-04, 13-05 |
| D23 | Locked | Native MTP KV capacity always equals the resolved target per-sequence context for the run. | Never use `n_ctx_train` as an MTP allocation floor; reject conflicting native-MTP `-cd`, reserve actual Turbo4 bytes first, and verify every MTP KV buffer is GPU-backed. | 08-02, 09-02, 14-02 |
| D24 | Locked | The first 33 tasks delivered tested components, not a live pager product. | Phases 08–16 must wire real model storage, CUDA transfers, attention, routing, telemetry, policy, lifecycle, CLI, benchmarks, corrective acceptance, and handoff; required live gates cannot be marked deferred. | 08-01–16-03 |
| D25 | Locked | Speed is the primary optimization objective after correctness and frozen quality floors. | Require at least 3x warm-focused decode over same-context ordinary CPU KV, target 5x, retain at least 70% of the comparable safe all-GPU control, and report fault/churn tails without averaging them away. | 08-04, 13-01–14-05 |

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
| Model/profile | `/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf`, `/srv/ai/config/profiles/qwen38-fast.env` | Actual geometry, Turbo4 types, MTP2, historical Fast baseline profile |
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
| The current task reports a blocker | Reopen it for three automatic recovery attempts (fresh session each time); retry 1 gets only the direct prefix, retry 2 gets a same-family/high assessment, and retry 3 gets the next-tier/high assessment from `luna -> terra -> sol`. Preserve the block and exit code 3 only if the fourth total task approach still fails. | Yes, through distinct diagnostics/configuration/fallback paths; a persisted block remains when all four approaches fail. |
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
action rather than a standing blocker. The benchmark harness's post-run health check must succeed before
the task is complete; profile restoration is required only for an explicitly declared control/revert,
teardown, failed-start recovery, or final-cleanup operation under section 27.

## 20. Production-completion series: why phases 08–16 exist

Phases 00–07 completed 33 tasks and left reviewable, tested components. They did not connect those
components into the production Qwen graph. Their final acceptance record correctly deferred the live
product. Phases 08–14 are the original production wiring, phase 15 is the corrective repair/re-acceptance
series, and phase 16 is the final handoff. They end only when the requested runtime works and is measured;
a task may not count a callback fake, arithmetic fixture, documentation update,
or `not_configured` metric as a substitute for its required model-backed acceptance.

The known gaps at the start of phase 08 are concrete:

- native MTP sizing currently uses the larger of the active target context and trained model context;
- `llama-cache-budget.h`, policy defaults, telemetry bounds, fixed-window tests, and older documentation
  still contain fixed maximum/hot-page assumptions;
- the host catalog, transfer scheduler, residency transaction, controller, exact executor, and selected
  attention objects are internal seams rather than an owner reachable from `llama_context` execution;
- the CUDA paged Turbo4 fixture launches a bounded decode callback, but the production Qwen graph does
  not dispatch it layer-by-layer from a live page table;
- there is no production CUDA transfer adapter joining real Turbo4 K/V tensor segments to the host
  catalog and compact physical slots;
- routing summaries and attention scores are not produced and consumed at live graph boundaries;
- the server has no functional pager mode, public experimental configuration, or pager metrics;
- exact mode is a callback executor, not a live all-page graph route;
- the benchmark adapter reports pager telemetry as `not_configured`, and the frozen corpus lacks the
  expected-answer stream required for quality acceptance.

Task 08-01 must verify this list against the then-current tree and turn it into a symbol-level wiring
map. Discovering that a gap was already closed is acceptable only with a direct source pointer and an
executed model-backed test; otherwise the owning task remains required.

## 21. Corrected dynamic capacity contract

### 21.1 Context-sized MTP

For native MTP, define `C` as the resolved target per-sequence context reported by the constructed
target context after CLI parsing and auto-fit. MTP must use exactly `C` logical rows. It must not use the
GGUF trained context as a lower bound. `-c 65536` therefore creates a 65,536-row MTP cache; `-c 262144`
creates a 262,144-row MTP cache. `-c 0` follows the target's normal resolved value. Auto-fit must evaluate
the target and its MTP sidecar as one candidate so the value cannot change between projection and
allocation.

The native-MTP use of `-cd` is intentionally strict: omission means “match target”; an explicitly
different value is rejected with a diagnostic. External draft models retain their existing independently
configurable context semantics. For native MTP, Turbo4 K and V and `--spec-draft-kv-device gpu` are
required by selective/exact pager modes. Allocation succeeds only if the realized memory breakdown shows
all MTP K/V buffers on a GPU. MTP model-weight placement remains governed by existing model/device rules;
this contract concerns its K/V cache.

### 21.2 Target hot capacity

For each device participating in target attention, admission records:

```text
usable_device_bytes = min(user_budget_or_device_free, backend_safe_limit)
fixed_bytes = weights + immutable_model_state + recurrent_state
reserved_bytes = context_sized_mtp + graph_peak + turbo4_kernel_scratch
               + transfer_staging + routing_and_telemetry + allocator_guard
page_charge_bytes = round_up(actual_cross_layer_turbo4_page_bytes,
                             backend_mapping_granularity)
H = floor((usable_device_bytes - fixed_bytes - reserved_bytes - safety_headroom)
          / page_charge_bytes)
H = min(H, logical_page_count)
```

All terms come from realized allocations, backend queries, a measured dry allocation, or an explicit
user budget. Estimates must be tagged and reconciled after context creation. Negative or mandatory-page
insufficient capacity fails before graph execution. Multi-GPU support is not inferred: the first live
implementation may require one target-attention device, but it must diagnose that requirement.

There is no default hot-token count. An optional user page cap can reduce `H`; it cannot increase it or
override safety. Page-table, graph, mask, dequantization, and temporary buffers must scale with `H * P`
or current wave size, never with `C`, except canonical host storage and intentionally bounded all-page
routing metadata.

### 21.3 Dynamic policy split

Policy receives `H` and first inserts mandatory pages. It then computes recent, historical, and
transient quotas from ratios/minima calibrated in phase 13. The sum is reconciled exactly to `H` after
deduplication. Tests use many explicit capacities, including capacities below every nominal minimum,
non-multiples of transfer batches, `H >= L`, and one-slot pressure. Absolute page counts may appear only
as test inputs or measured output, never as production defaults.

## 22. Required live dataflow

The completed runtime must follow this ownership sequence:

1. The target context resolves `C`, validates Qwen/Turbo4 capability, constructs context-sized GPU MTP,
   and reserves all fixed/scratch/staging costs before computing `H`.
2. The pager owns `H` compact GPU slots and `L=ceil(C/P)` logical identities. Native positions remain
   logical; physical slot order never becomes position.
3. New target K/V is written to the current GPU page. When a page seals, every attention-layer K/V
   segment is copied as opaque Turbo4 bytes into canonical host RAM, authenticated, and entered into the
   catalog. The partial write page remains pinned and GPU-authoritative.
4. Retrieval unions mandatory/recent/structural pages with candidates from an all-page routing index.
   Retention uses completed attention telemetry only for pages that actually participated.
5. The controller produces the next complete target table. Promotions copy canonical host bytes to free
   GPU slots on a dedicated stream. Events and generations are checked before one immutable table is
   published. Clean eviction drops a mapping/slot without D2H.
6. Prefill and decode attention consume Turbo4 directly through logical-page metadata and native masks.
   There is no full-cache F16 gather. Decode uses the optimized paged kernel; prefill has its own bounded
   selected-page path.
7. MTP and recurrent state advance or roll back atomically with the target logical frontier but are not
   target pager victims.
8. Exact mode visits every valid logical page exactly once in bounded GPU waves and merges online-softmax
   state. It is the storage/quality oracle, not the speed path.
9. Server telemetry exposes the reconciled resource ledger, page state, transfers, stalls, attention
   sampling, policy decisions, and MTP placement without copying attention matrices to the CPU.

## 23. Quality architecture and frozen evaluation

Selective attention is approximate, so quality protection is part of the architecture rather than a
post-hoc benchmark concession:

- mandatory system/tool/application anchors and the current partial page always participate;
- a configurable recent region protects local coherence;
- cold retrieval uses summaries generated from every sealed page, not resident-only attention history;
- bounded exploration prevents a permanently invisible cold-page trap;
- unavailable attention observations are unknown, never zero importance;
- focus changes use hysteresis plus transient capacity, while required-page misses wait or use the prior
  safe table instead of consuming partial transfers;
- small contexts where every logical page fits must match the dense Turbo4 route;
- exact page waves provide a full-context oracle for output/logit and attention-mass comparisons.

`pager-corpus-v2` is frozen in task 08-04 before selective tuning. It contains at least: warm repeated
focus, early and middle cold needles, multiple competing needles, system/tool anchors, coding-repository
facts, long conversations with referents, abrupt focus shifts, adversarial churn, and a recent-only
control. Each scored item has an immutable expected answer or deterministic checker, page-distance
metadata, and a stable content hash. Phase 13 may tune implementation parameters against a designated
calibration split only. Phase 14 evaluates a held-out split and may not change thresholds or answers.

Required quality reporting includes exact match/task score, failure examples, routing recall@selected
pages, fraction of dense/exact attention mass captured, top-page recall, selective-versus-exact output
agreement, perplexity or token-distribution divergence on the designated sample, MTP acceptance, and
the effect of each mandatory/recent/router/exploration component. Any fundamental retrieval failure
returns work to its owning phase; quality thresholds are never weakened after results are visible.

## 24. Performance method and gates

Every timing run uses a Release CUDA build, fixed clocks/power state if the operator can establish them,
clean isolated server state, recorded model/binary SHA, temperature 0, seed 42, identical prompt/output
length, warmups, and at least five measured trials for checkpoint work and ten for final acceptance.
Capture median plus p10/p90 or p5/p95, never only the fastest sample.

Required comparisons are discovered, not hardcoded:

1. same-context ordinary Turbo4 target K/V in CPU RAM with context-sized Turbo4 MTP forced to GPU;
2. same-build `off` behavior;
3. `observe` overhead against `off` where dense execution fits;
4. selective warm focus at the model maximum context;
5. selective cold needle, focus shift, and churn at the model maximum context;
6. the largest safe all-GPU Turbo4 target+MTP context found by a fresh context ladder for that build;
7. exact mode separately, because it is not subject to the selective speed gate.

Record prompt throughput, decode throughput, TTFT, inter-token p50/p95/p99, MTP acceptance, CPU usage,
host RSS/pinned bytes, total/free/peak VRAM, `H`, `L`, attention rows, graph/scratch bytes, faults,
prefetch hits, late waits, evictions, D2H/H2D useful and aligned bytes, transfer bandwidth, overlap time,
table rebuilds, and errors. Preserve raw per-request data.

The final warm-focus selective gate is all of:

- at least 3x median decode throughput over same-context ordinary CPU K/V;
- explicit report of whether the 5x target is achieved;
- at least 70% of the comparable safe all-GPU control's median decode throughput after normalizing the
  comparison to the closest available selected attention width;
- p95 inter-token latency no worse than 1.5x its median in the no-fault steady segment;
- observe-mode overhead no more than 5% on its applicable dense control;
- no unexplained allocation whose size follows the full logical target context on GPU.

Cold-fault and churn cases have no hidden pass average: report steady and faulting segments separately.
If the 3x floor fails, phase 13 continues profiling and optimization. The task blocks only after at least
three evidence-driven optimization hypotheses have been tested and preserved with results.

## 25. Completion phases, clusters, and gates

The exact task order and model recommendations live in `WORK_STATE.json`; the following phase gates
explain why each cluster exists.

### Phase 08 — correct contracts and freeze controls

- `08a-dynamic-contract`: verify the live gap map, make native MTP context-sized, and remove fixed hot-set
  capacity/policy assumptions from production code.
- `08b-benchmark-contract`: freeze `pager-corpus-v2`, thresholds, result schemas, and new same-build
  controls across multiple context sizes.

Gate: tests prove MTP rows follow resolved contexts, admission has no fixed hot count, policy scales over
many capacities, expected answers and thresholds are hashed before tuning, and model-backed baseline
artifacts include explicit MTP GPU placement.

### Phase 09 — live compact storage and CUDA residency

- `09a-live-memory`: normalize experimental configuration, instantiate one real pager owner, allocate
  compact physical target storage, and bind target writes/positions/mutations.
- `09b-live-transfers`: bind canonical host Turbo4 sealing plus actual CUDA promotion/eviction and run a
  fixed-selection model-backed storage smoke.

Gate: a long prompt seals and authenticates host pages, target VRAM is bounded by the ledger, promotions
round-trip byte-exactly, clean evictions perform zero D2H, stale events cannot publish, MTP remains GPU,
and teardown returns allocations to baseline.

### Phase 10 — live Turbo4 attention and exact oracle

- `10a-live-decode`: route selected-all-pages reference attention through the real Qwen graph, then route
  qualified decode to the direct paged Turbo4 kernel and validate graph epochs/performance.
- `10b-live-prefill-exact`: implement bounded paged prefill and connect exact page waves to production
  catalog/transfer/kernel callbacks.

Gate: dense, selected-all-pages, direct paged, and exact-wave routes agree within frozen tolerance on
supported contexts; output/scratch scale with physical rows or wave size; no full F16 gather exists;
prefill-to-decode, tails, table changes, and graph reuse pass on the real model.

### Phase 11 — live retrieval, telemetry, policy, and prefetch

- `11a-live-routing`: create all-page summaries, run query-to-page retrieval, and publish GPU-reduced
  attention telemetry.
- `11b-live-policy`: publish dynamic hot tables and overlap transfers with decoding while preserving
  server/MTP/recurrent/checkpoint atomicity.

Gate: a previously cold needle can be recalled, telemetry overhead is bounded, policy never exceeds `H`,
required pages are never partially consumed, prefetch produces measured overlap, and lifecycle faults
leave one valid generation.

### Phase 12 — usable experimental server surface

- `12a-operator-surface`: finalize CLI/config, metrics/ledger output, benchmark telemetry ingestion, and
  off/observe/selective/exact lifecycle behavior.

Gate: generated help/docs match parser behavior; server startup fails closed on unsupported combinations;
`off` remains unchanged; every benchmark manifest reads real telemetry rather than environment stand-ins.

### Phase 13 — profile and optimize for speed

- `13a-kernel-transfer-opt`: establish reproducible microprofiles, then optimize direct Turbo4 attention
  and transfers/overlap.
- `13b-system-opt`: optimize prefill/graph/scratch behavior and calibrate the routing/policy/prefetch
  Pareto frontier without touching held-out answers.

Gate: every optimization has before/after raw data, output/quality equivalence at its boundary, no
server-specific production constant, and the integrated checkpoint reaches the 3x floor or records three
failed evidence-driven hypotheses for further repair.

### Phase 14 — full model-backed acceptance

- `14a-correctness-quality`: run CPU/CUDA/sanitizer/fault coverage, the dynamic context/capacity/MTP
  ladder, and held-out exact/dense/selective quality.
- `14b-performance-soak`: run final speed comparisons plus endurance, churn, concurrency, checkpoint,
  restore, and teardown campaigns.

Gate: every definition-of-done item has raw machine-readable evidence. Phase 14 tasks cannot be deferred;
they either pass or remain blocked with reproducible evidence.

### Phase 15 — corrective live-product repair and re-acceptance

Phase 15 exists because the V2 audit correctly preserved blocked quality, performance, and soak gates.
It is an implementation-and-evidence repair phase, not a documentation retry. Tasks are intentionally
small so a lower-reasoning model can resume from a handoff without loading the whole plan:

- `15a-corrective-integration`: rebase the implementation onto the synced fork and preserve V2 evidence;
- `15b-corpus-contract`: construct and validate real multi-page fact-bearing prompts;
- `15c-harness-lifecycle`: make the benchmark launcher explicit and keep successful server state loaded;
- `15d-mtp-fit`: solve dynamic context-sized Turbo4 MTP GPU admission and target hot capacity;
- `15e-runtime-lifecycle`: repair speculative rollback and page-generation atomicity;
- `15f-exact-telemetry`: bind live exact page waves and publish real residency metrics;
- `15g-quality`: rerun dense/reference/exact/selective model-backed quality;
- `15h-performance`: run paired dynamic-capacity Turbo4 pager+MTP speed trials;
- `15i-soak`: run lifecycle endurance and leave the accepted profile loaded for handoff.

Gate: V3 quality, performance, and soak manifests contain live model-backed evidence with one consistent
source/model/profile/corpus provenance. A failed gate returns to its implementation owner; it cannot be
closed by changing thresholds, shrinking context, or relabeling `not_configured` as a result.

### Phase 16 — clean handoff and reproducibility

The former phase-15 tasks are shifted to phase 16 and run only after phase 15's corrective gates pass:

- `16a-upstream-handoff-v2`: recheck upstream, rebase/slice locally, update portable operator/evidence
  documentation, and reproduce the final build/run from a clean worktree.

Gate: the fork integration branch is clean and pushed, implementation commits contain no execution data,
the evidence commits contain no upstream code dependency, all source is portable, and the status is
complete only if phase 15 and all required live gates passed. Agents prepare but never post AI-authored
GitHub issue/PR text.

## 26. Autonomous execution rules for phases 08–16

Every new task packet contains exact reads, likely write ownership, implementation steps, commands,
benchmark conditions, acceptance, recovery paths, and handoff requirements. An executing agent must:

1. start only the task reported by `tool/codex/task_state.py next`;
2. read the cluster file once, the packet, and dependency handoffs; avoid loading unrelated old packets;
3. treat existing callbacks as untrusted until a live caller and test are demonstrated;
4. keep implementation generic; machine/service/profile facts live in execution metadata or `/srv/ai`;
5. use Release builds for timing and separate test builds for sanitizer/debug evidence;
6. preserve active-profile identity, stop/start only the named benchmark service through the established
   scripts, keep `ai-long-memory.service` healthy for the restore gate, and never touch the unrelated
   service on port 8092. A successful benchmark leaves its tested server/profile loaded by default for
   the next task or retry; restoration is required only when the packet explicitly requests a control or
   revert benchmark, teardown, failed-start recovery, or final cleanup, and the lifecycle state records it;
7. use passwordless `sudo` non-interactively when the packet calls for controlled service work;
8. retain raw results under `/srv/ai/paged-kv/results/` and put only summaries/hashes/pointers in Git;
9. checkpoint after each major sub-gate so a restarted Luna session can resume from evidence;
10. try distinct diagnosis and repair paths before blocking; never satisfy a gate by disabling it,
    shrinking the requested maximum-context test, or relabeling missing telemetry as zero;
11. mark a task complete only when every required item in its packet passes; required live phase 15
    corrective work may not be deferred;
12. leave GitHub issues, PR creation, descriptions, replies, and merges to a human under `CONTRIBUTING.md`.

The outer runner owns task branches, implementation-versus-execution commits, pushes to the user's fork,
and merges into the plan branch in `auto` mode. New implementation commits exclude `docs/execution/**`;
task state, logs, handoffs, and raw-evidence pointers follow in their own metadata commit.

## 27. Persistent benchmark lifecycle contract

Benchmark tasks are a dependent diagnostic sequence, not independent disposable runs. Each live task
records the active profile, server command, PID, health result, loaded model, resolved context, MTP
placement, pager mode, and lifecycle action in its handoff. If startup and its acceptance sub-gate pass,
the tested server/profile remains loaded so the next task or retry can inspect the same allocation and
cache state. This is the default `keep_loaded` policy and is especially important for diagnosing transfer,
rollback, and telemetry failures.

An explicit revert is required only when a task declares a paired control benchmark, requests a clean
teardown/restart, recovers from a failed start, or performs final cleanup. Control and revert tasks must
capture before/after health and profile identity. A failed run must leave either a usable tested process
with a recorded failure or a known stopped state; agents must never silently restore a different profile
and thereby erase the state needed by the next recovery attempt. The unrelated service on port 8092 is
always out of scope.
