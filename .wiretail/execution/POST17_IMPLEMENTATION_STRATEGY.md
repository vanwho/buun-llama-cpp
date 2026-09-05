# Post-17 strategy: make the live data path real, then measure it

Revision: 2026-09-05. Applies to work **after the already-running 17-15**.
This amendment and `BENCHMARK_PROTOCOL_V5.md` supersede conflicting historical
execution directions, speed gates, and phase-18 review numbering in the large
plan. `WORK_STATE.json` remains the sole task-order authority. Do not reread
old phase-14/15/16 acceptance ledgers to execute this sequence.

For execution detail, each packet's concrete recipe selects sections of
[EXECUTION_COOKBOOK.md](EXECUTION_COOKBOOK.md) (commands, fixtures, receipts,
resume) and [IMPLEMENTATION_CONTRACTS.md](IMPLEMENTATION_CONTRACTS.md)
(allocation/write/graph/kernel/query interfaces). Use the recipe's checkpoint
order. Complex changes still need numerical/live proof and a recovery assessment
when a checkpoint fails; the plan does not make unimplemented interfaces exist.

## 1. Non-negotiable outcome

The available target is Qwen3.8-27B UD-IQ4_XS, whose GGUF architecture maps to
the Qwen35 implementation. Use the existing model, not a substitute tiny model
for live proof. Tiny fixtures remain useful for deterministic unit tests.

- Target K **and** V and native-MTP K **and** V remain Turbo4 throughout.
- Logical target capacity must reach 262,144 tokens. Distinguish capacity from
  occupied prompt/history tokens and reserve room for generation.
- Full-attention target pages have canonical, recoverable CPU-RAM backing and
  a bounded hot GPU cache. Recurrent/linear state is separate, exact, and GPU
  resident. This is not moving model weights or all model compute to CPU.
- Native MTP capacity follows the resolved target per-sequence context on
  every launch; all its KV remains GPU resident, including at 262,144. Never
  silently shrink the draft, disable it, or change its codec to make a result fit.
- Hot target capacity is the **remaining safe runtime byte budget**, not a
  historical token estimate. No fixed production hot-token or page count.
- Attention-aware retrieval can find a previously cold page. Residency and
  precision are separate abstractions. VBR precision changes stay disabled.
- Speed is the main optimization objective. The old 3x floor, 5x target and
  70% all-GPU ratio are comparison annotations, **not pass/fail gates**. Record
  honest speed curves, including regressions and unavailable denominators.
- Selective attention is approximate. Dense/selected-all parity, correct
  positions/masks, lossless residency, and a working exact reference are hard
  correctness requirements; optional quality/performance tradeoffs must be
  explicitly labeled, measured and reversible.
- YaRN beyond 256K is a conditional stretch experiment, not a reason to delay
  fixing the base pager or to claim that addressable positions imply quality.

Development tests choose whatever context exposes their specific behavior
efficiently. The six 20K/40K/60K/100K/175K/256K speed coordinates belong only to
final results collection after functionality works, not every task or gate.
Use tiny forced-hot budgets to test page faults cheaply and short identical
prefixes for parity; reserve populated 256K runs for explicit capacity/final
proof tasks. Small diagnostic context is valid, just not overall-goal proof.

## 2. What the evidence actually says

Snapshot source tree: `23411197aa446e4766ce921160ebe2e33cf53992`, plus the
active 17-15 work. These observations are a starting diagnosis, not a claim
that the active run has finished. Task 17-16 will capture its final results.

| Observation | Interpretation / next owner |
| --- | --- |
| 17-09: observe/native median decode 111.006690 tok/s, selective/native 74.225274; MTP acceptance 96.6667% versus 72.2222% | About 33.1% slower in this short diagnostic. Very short outputs, different acceptance, and no cold-page traffic prevent a pager-speed claim. 18-05/06 isolate parity before 20-04 profiles overhead. |
| Same 17-09 selective snapshot: one resident page, zero host pages, zero H2D/D2H/faults/evictions | Not a CPU-RAM-offload measurement. An all-fit short prompt should not lose quality merely because it is called selective. |
| 17-10: 16K configured but about 3,245 occupied tokens, host bookkeeping present, zero transfer/fault/eviction counters | Lifecycle diagnostics only, not physical pressure. 19-03 and 21-04 must prove actual transfers and reuse. |
| 17-13: 22,016 native-MTP startup segfault after listening; old exact path refuses missing callbacks | A runtime identity/backtrace problem and an unimplemented exact path, not evidence that 22K or 256K is intrinsically impossible. 18-01 and 19-06 own repairs. |
| Active 17-15 dense summary: 24 completed, 22 correct, 2 incorrect; selected-all: 20 completed, 0 correct, 4 timeouts | The selected-all label is not proof of full coverage. Check request/provenance, causal rows, compressed bytes, layer results and teacher-forced logits before tuning selection. Dense itself is not a perfect 24/24 oracle. |
| Active 17-15 exact summary: 24 failures | Inspect response classes before calling these quality failures. Repeating a known unsupported capability over 24 cases adds no evidence. |
| Active 17-15 soak raw requests report 53,797 prompt tokens against 22,016 capacity | Word-count padding is broken. Current `prompt_context_words()` and soak `prompt_words` guess two tokens per word. Fix actual rendered-token accounting in 18-02. |
| Historical ~42 prompt tok/s | From a fully CPU-loaded 27B diagnostic, not a measurement of GPU model + CPU KV or this pager. |

Raw roots (never overwrite):

- `/srv/ai/paged-kv/results/17-09-performance-20260905T020000Z`
- `/srv/ai/paged-kv/results/17-10-soak-20260905T021200Z`
- `/srv/ai/paged-kv/results/17-15-quality-20260905T052900Z`

The active 17-15 quality summaries refer to executable SHA256
`404622c75944d27c054c0dcf48919d5af0793c35c95758347c011637f023b790`,
model SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`,
and corpus semantic hash
`37111506c8ddc7f9d04086121797cc9c2b7a2842cc1a56ef4b33f506c2252007`.
These are historical snapshot identities, not defaults for subsequent builds.

## 3. Source-level gaps: inspect, reproduce, repair

Use symbols rather than fragile line numbers. Recheck each observation against
the task's current tree; if already repaired, prove that with a live regression.

1. `src/llama-context.cpp::init_kv_pager()` runs against existing memory and
   charges context allocations. `src/llama-kv-cache.cpp` still constructs
   per-layer K/V tensors with `kv_size` rows; the pager allocates a separate
   residency store. Audit physical buffers and aliases, not tensor shapes
   alone. A full-size GPU target allocation **plus** a pager slab cannot deliver
   bounded target VRAM. See 19-01/02; preserve logical metadata without dense
   physical GPU backing.
2. `llama_kv_pager::apply_live_policy()` exists in `src/llama-kv-pager.cpp`, but
   the inspected `src` tree contains only its declaration and definition, no
   production invocation. Pure policy/lifecycle implementations and unit tests
   are not a live query -> retrieval -> transfer -> publication pipeline.
   Trace alternate callers as well; implement the missing production boundary
   and show CUDA-copy evidence in 19-03, then real query routing in 20-01.
3. `prepare_kv_attention_graph()` builds selected pages from all current
   residents. Its direct capability requires single-query decode; MTP verify
   blocks and prefill use the reference gather. Make direct compressed attention
   work for verification blocks (19-04) and bounded prefill (19-05).
4. `prepare_kv_attention_graph()` explicitly refuses exact cold-page waves:
   `exact CUDA page-wave callbacks are not configured`. HTTP 501 from 17-14
   documents that missing implementation; it does not implement the reference.
   Wire the layer-wise GPU wave execution and online-softmax merge in 19-06.
5. `get_kv_pager_metrics()` first reads actual native-MTP buffers, then assigns
   `snapshot.mtp_rows`/`snapshot.admission.mtp_bytes`. Separate measured state,
   estimates and requested settings (18-04); do not overwrite observed residency
   with an admission promise.
6. `set_kv_pager`, `pager_host_prepare`, `finish_pager_batch`,
   `selected_attention_rows`, and graph write indices must agree on logical
   position versus physical slot and exact compressed representation. The
   capture helper's `physical_slot * page_tokens` is safe only when actual graph
   writes use the same mapping. Check tails, non-contiguous pages, M-RoPE layout,
   rotations/codebooks/meansub, and every full-attention layer (18-05, 19-02).
   The technical specification T1a identifies a particularly concrete candidate:
   GET_ROWS returns still-rotated Turbo4 keys as floating-point tensors, so the
   ordinary FA dispatch can omit the query rotation previously triggered by a
   Turbo4 K type. Reproduce/fix this before calling the selected-all loss a
   permissible sparse-quality tradeoff. T1 also checks hybrid layer indexing;
   T2 checks graph-write versus direct-read physical storage.
7. Current routing summaries start at layer/head zero. That cannot be assumed
   sufficient for all layers. Feed real per-layer/group queries to an all-page
   index; keep conservative unions initially, then measure per-layer masks and
   budget allocation (20-01/02). Never invent a query from token IDs or hidden
   state without the correct projection/rotation.

## 4. Chosen implementation architecture

### 4.1 Storage, ownership and capacity

Keep the existing compact physical-slot pool first. CUDA VMM can be reused
internally where helpful, but stable sparse virtual mappings are not a
prerequisite and must not become another infrastructure rewrite. Use existing
backend buffer/event abstractions; never dereference an unmapped logical row.

Logical identity includes sequence, logical page, layer/representation identity
and generation. Preserve runtime-derived layer offsets. Start with 256-token
cross-layer transfer groups because existing capture code uses that granularity;
selection masks and score ownership must nevertheless be per layer or explicit
layer group. Do not require physical per-head paging for the first working path.
If smaller layer-granular transfers help, 20-02 measures that change with the
same correctness fixtures. Do not hardcode the reference's 16 layers in generic
allocators or assume all transformer layers use full KV attention.

The host catalog holds opaque Turbo4 bytes, not a lossy re-encoding. GPU writes
are unavoidable: seal newly committed pages D2H asynchronously once, then keep
host copies authoritative and evict clean GPU copies without D2H. The writable
tail/speculative pages remain pinned until their accepted bytes have reached
host storage. Flush partial tails before checkpoint/release; never publish
unaccepted speculative KV as committed host truth. Prefer pageable backing
plus bounded pinned rings initially; compare whole pinned backing only with
measured host limits. There is a **writeback cost**, not free CPU canonicality.

Budget before physical allocation, reserve actual graph worst cases, then
verify after warmup:

`usable = device_total - external/driver - weights - recurrent_state - full_MTP_KV
          - target_and_MTP_compute - routing - page_tables - pinned/device_rings
          - checkpoint/rollback_reserves - configurable_safety_margin`

`hot_pages = min(logical_pages, floor(usable / aligned_physical_page_charge))`

Avoid double charging buffers shared between contexts, and do not hide native
MTP compute behind the much smaller KV estimate. Use checked arithmetic, actual
CUDA allocation granularity, query-block/batch maxima and peak measurements.
If a configuration narrowly OOMs, keep requested context and MTP rows constant
and reduce target hot pages, then batch/ubatch/workspace or optional checkpoint
retention. Only reduce logical context as an explicitly labeled diagnostic.
If fixed allocations alone cannot fit, report the ledger and repair/optimize
them; do not silently turn a 256K task into another 16K acceptance run.

### 4.2 Attention correctness and execution

All-resident/all-selected must first agree with ordinary dense Turbo4 on the
**same token prefix**, with MTP off, and then with native MTP on. Compare layer
outputs and final logits using quantified tolerances anchored in dense numeric
variation; do not force byte-identical text at near-tie logits or loosen
tolerances to hide large errors. Check all query positions, masks, GQA heads,
tails, page permutations, RoPE coordinates, and Turbo4 transform conventions.

Direct kernels load compressed K/V tiles from physical slots; reference gather
is a diagnostic oracle, not the performance path. Support the actual multiquery
MTP verify shape and bounded prefill tiles. Preserve an efficient all-fit path:
avoid routing/copy/graph churn when all relevant KV is already accessible, but
keep a forced-paged diagnostic path so a dense bypass cannot hide pager bugs.

Exact reference means all logical pages participate. The first correct cold
implementation stages compressed waves on GPU and merges, per query/head/layer,
`m=max(m_i); l=sum(exp(m_i-m)*l_i); o=sum(exp(m_i-m)*o_i); output=o/l`.
Handle masked/empty partitions without NaNs, causal tails and sequence identity.
CPU cold attention is optional later, not a new prerequisite. Exact prefill is
the conservative way to build canonical KV without accumulating sparse-prefill
errors; bounded selective prefill is separately labeled, quality-tested and
opt-in until its effect is understood. Host backing preserves computed KV, not
the hypothetical dense KV if the prefix was already computed approximately.

### 4.3 Attention-aware retrieval, retention and overlap

Keep GPU-resident summaries for **all** host pages. Query-aware scores should
estimate candidate relevance in the same key/query coordinate system consumed
by attention. Compare representative keys and min/max bounds; do not assume
a technique validated for another model or codec transfers without tests.
Mandatory sinks, active tail and caller-declared anchor spans override scores.
Budget recent/history/prefetch shares relative to available pages, not fixed
token windows. Always reserve a bounded promotion/write workspace.

Reduce page attention mass on GPU from the final softmax normalization (including
all tiles/partitions); update retention EMA/peaks without confusing unobserved
cold pages with observed zero mass. Query retrieval recovers cold candidates;
retention reduces churn. Per-layer score scales need normalization and explicit
aggregate/union semantics. Bounded exploration and fallback expansion protect
against a self-reinforcing cold-page blind spot; measure cold multi-hop tasks.

First wire synchronous transfers correctly. Then overlap coalesced promotions
with compute using events and a bounded ring. Current-query candidates cannot
be prefetched before that query exists: use measured previous-token/layer
predictions for lookahead, and explicitly wait/expand/fall back on an essential
late page. Never read an in-flight slot or use a stale speculative generation.
Stable descriptor addresses and shape-based graph reuse must avoid recapture
on content changes alone. Profile actual MTP verification as well as decode.

### 4.4 References and why they matter

- [Quest](https://arxiv.org/abs/2406.10774): page min/max keys and the current
  query motivate a cheap all-page candidate index. This is a design reference,
  not proof of Qwen/Turbo4 quality or end-to-end speed here.
- [InfiniGen](https://arxiv.org/abs/2406.19707): predictive selection for later
  attention motivates measured lookahead; adopting an additional rehearsal
  path is optional only if its compute cost beats saved copy time.
- [YaRN](https://arxiv.org/abs/2309.00071): RoPE extension is a separate
  model-quality experiment, not merely allocating more pages.
- Local upstream Buun commits `6690273bfde56fa94b276361d1cf0bcd076a0a15`
  (self-contained MTP tensors) and `c9c52d7183bd75d7ae4a71d02f1aba6d34546fe5`
  (MTP recovery after target-only restore) are relevant foundations. Inspect
  ancestry/current upstream at 18-06; reuse equivalent fixes without duplicate
  patches. Do not merge or switch the active 17-15 branch during this revision.

## 5. Execution sequence and context economy

| Phase | Purpose | Exit |
| --- | --- | --- |
| 17-16 | Compact bridge from the running campaign; replace repeated failed matrices | Small factual gap/measurement index |
| 18 | Reliable experiment identity, exact sizing, resumable clients, telemetry, all-fit/MTP parity | Trustworthy short live tests |
| 19 | Bounded physical storage, canonical bytes, wired transfers, direct kernels, exact waves, full-context proof | Real host-backed 256K capacity with GPU MTP |
| 20 | Real per-layer query retrieval, retention, async movement, graph/fast-path optimization, quality tradeoffs | Profiled attention-aware candidate |
| 21 | Frozen quality, original-three-prompt context curve, paired controls, physical-pressure soak, optional YaRN, compact summary | Measured findings, including shortcomings |
| 22-01 | Sol High benchmark-only assessment | Verdict; if needed, new measured remediation and review chain |

Old **unstarted** 17-16/17-17/17-18/17-19/18-01 packet contents are archived in
`archive/pre-post17/`; no completed task, active packet or historical result is
deleted. 17-16 is now the bridge; the old reviewer is superseded by 22-01.
The runner reads the edited state after 17-15 completes and proceeds normally.

All task recommendations remain Luna High per the user's project policy;
22-01 remains Sol High. This does not change Wiretail's tool-wide defaults.
Packets spell out decisions so smaller reasoning settings remain practical.
Clusters contain at most three adjacent tasks with the same source ownership.
Read this strategy once per new cluster, that cluster, the current packet,
`BENCHMARK_PROTOCOL_V5.md` for live work, and direct dependency handoffs only.
Do not load whole raw roots, million-token transcripts or all old acceptances.

## 6. Completion, repair and handoff rules

An implementation task must demonstrate its named live behavior, not merely
introduce an API/test, improve an error message, or document an absent callback.
On failure, minimize the case, locate the source owner and repair it. Assessments
may insert a concrete prerequisite before remaining work, but cannot use task
renumbering to reset an unchanged failure forever. Keep a failure fingerprint,
attempted hypotheses, changed variables and raw pointers. After distinct paths
are exhausted, report a real blocker without weakening correctness or fabricating
timings. A benchmark task can finish with valid negative findings and a precise
repair owner; 22-01 decides overall progress from those findings.

Each task writes `handoffs/<id>.md` (aim <=120 lines) and a small machine-readable
receipt in `evidence/`. Include source/bundle/model/corpus IDs, commands, actual
runtime observation, next experiment, and raw paths, not pasted logs. Update
task state via the shared helper; preserve other tasks and token accounting.
Raw runs live under `/srv/ai/paged-kv/results/`, never `/tmp` for durable evidence.

Do not edit the currently running task or switch its branch. The outer auto
runner owns commits/pushes/merges on the fork and checkpoints plan metadata
separately from generic code. Read CONTRIBUTING/AGENTS in every repo touched.
Re-read the actual contribution policy rather than assuming a blanket ban on
new tests: extend existing tests first; add a focused regression where necessary
and justified. AI-generated upstream posts/PR prose are prohibited by the
current Buun policy; a human owns those and must disclose/review AI code.
No execution metadata, machine paths, credentials, raw data or their commits
belong in upstream slices. Do not inspect unrelated issue #116 for this work.
