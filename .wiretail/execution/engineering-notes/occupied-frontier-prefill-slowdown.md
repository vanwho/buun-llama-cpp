# Occupied-frontier prefill slowdown: code diagnosis and optimization direction

Recorded 2026-10-02 from the 102-04 128K occupancy run. This is a diagnostic
note, not an accepted speed result or a conclusion that the selected-attention
architecture is fundamentally too slow. Keep it distinct from later 102-05
measurements, which use a different logical/hot geometry.

Updated 2026-10-04 by source inspection only, at repository HEAD
`9e1e469617b20537542aa7ccae3e6d743b38b25f`. No tests, benchmarks, server
changes, or runtime-code edits were performed for this update. Line numbers
below identify this checkout; use the named symbols after later edits.

## Main finding and recommended decision

The strongest concrete defect is **per-token, usually no-op, routing-summary
invalidation that performs whole-history quadratic accounting on the CPU**.
This is on the normal selected-cache write path, before GPU graph execution.
It is unnecessary work, not an intrinsic cost of storing cold Turbo4 pages in
RAM, and it directly defeats the intended bounded-hot-window architecture.
Repair this first, without changing attention routes, cache precision, batch
sizes, or MTP semantics. Full-history GPU MTP attention is a separate genuine
context-dependent cost; query replay explains intermittent double processing.

The source proves the expensive operations exist and are reachable during
ordinary single-sequence prefill. It does not prove their exact share of the
recorded seconds: that requires the separate later testing task. Do not claim
a measured speedup or that this accounts for every second of the slowdown.

## Run identity and observed curve

The measurements below came from L=131,072, H=16,384 tokens (64 pages at 256
tokens/page), B=1024/U=256, GPU Turbo4 target KV and full-L GPU Turbo4 MTP. The
task reduced H from its initial 59,904-token request because measured target
allocation and scratch reserve could not admit that hot pool; it kept all other
server settings fixed. The raw request journal is
`/srv/ai/paged-kv/results/forward/102-04/attempt-01/occupancy-resume/request-journal.jsonl`.

| occupied after request | fresh tokens | timed prompt tokens | prefill time | reported prefill rate |
| ---: | ---: | ---: | ---: | ---: |
| 20,303 | 15,615 | 15,615 | 13.73 s | 1,136.9 tok/s |
| 36,319 | 15,617 | 15,617 | 21.42 s | 729.2 tok/s |
| 52,334 | 15,616 | 15,616 | 29.91 s | 522.1 tok/s |
| 68,342 | 15,609 | 15,609 | 40.18 s | 388.5 tok/s |
| 84,223 | 15,482 | 15,885 | 52.58 s | 302.1 tok/s |
| 100,228 | 15,606 | 31,201 | 133.34 s | 234.0 tok/s |
| 116,238 | 15,611 | 15,611 | 79.55 s | 196.2 tok/s |

These are occupancy-campaign request timings, not the prescribed final three-
prompt speed benchmark. The later 145-second request at C=100,228 is especially
important: the campaign counted 15,606 newly committed tokens, but the server
timed 31,201 prompt tokens. Its API usage reported 84,223 cached plus 15,606
new prompt tokens. This suggests extra replay/re-evaluation work in that
request; investigate target versus draft/replay accounting before treating its
234 tok/s denominator as fresh-input throughput. The row also reports roughly
twice the usual summary-build and host-seal counts. A later row returned to
15,611 timed prompt tokens, so the double-work event is intermittent rather
than a simple smooth context-scaling effect.

## What the existing counters establish, with corrected semantics

- The candidate used selected direct/packed prefill routes; reference-route
  counts were zero. Thus this slowdown is not explained by an observed
  `selected_reference` fallback.
- Ordinary 15.6K chunks report about 59–60 direct plus 4–5 packed prefill route
  submissions, near the roughly 61 subbatches expected at U=256. Total graph
  completion counts also include output generation and MTP verification, and
  must not be mistaken for prompt graph count.
- The old `graph_construction_us` counter is misleadingly named:
  `llama_context::prepare_kv_attention()` (lines 2624–2653) times the execution
  planner and proof-page-list bookkeeping, **not** `model.build_graph()` or
  scheduler allocation. Actual graph build/allocation is in
  `process_ubatch()` (lines 6825–6869); `graph_build_us` measures it only when
  hotpath profiling is enabled. Therefore the old sub-millisecond number
  cannot rule out graph-build/allocation costs. There is still no evidence of
  a graph rebuilt for every input token: route submissions are microbatches.
- Scheduler `wait_us` in `synchronize()` (lines 2454–2484) includes outstanding
  GPU computation as well as any backend waiting. It is not a measurement of
  GPU-idle time. Likewise `total_token_us` spans more than a kernel. These
  overlapping totals must not be added together or subtracted as independent
  stages, and are not enough to diagnose a synchronization-only bottleneck.
- Normal chunks showed roughly 4,000 summary builds / 134–138 MB and about
  6,000 host-seal D2H calls / 270–279 MB. The host copy is part of maintaining
  canonical RAM backing; counters give volume/count, not elapsed-time
  attribution. Determine whether these are efficiently batched and overlapped
  before changing the required backing behavior.
- H2D promotion and eviction deltas were zero in these request rows. They
  establish continued occupancy and selected-route execution, not that these
  particular turns naturally promoted a cold historical page.

## 1. Highest-priority defect: no-op invalidation for almost every appended token

Exact call chain:

1. `llama_kv_cache::apply_ubatch()` in `src/llama-kv-cache.cpp`, around
   7210–7264, calls `pager_->begin_write_batch()` for one sequence.
2. `llama_kv_pager::begin_write_batch()` in `src/llama-kv-pager.cpp`,
   3442–3696, performs useful batch admission, **but still calls
   `begin_write()`/`begin_write_planned()` separately for every position**.
3. `begin_write_planned()`, 3215–3230, unconditionally calls
   `invalidate_routing_summaries({previous_id})` whenever the physical page
   already exists, then advances its content version and sets
   `summary_content_version=0`. For a newly appended 256-token page, roughly
   255 subsequent rows repeat this operation on an already-unsummarized tail.
4. `invalidate_routing_summaries()`, 2344–2361, calls invalidation on the
   compatibility store plus every valid layer/head store. The configuration
   list covers every attention layer × KV head (2318–2342).
5. `llama_kv_routing_summary_store::invalidate_pages()` in
   `src/llama-kv-routing-summary.cpp`, 732–762, copies the store descriptor
   vector, scans it for removal, and calls `rebuild_accounting()` **even when
   no summary matched and nothing was removed**.
6. `rebuild_accounting()`, 832–922, contains a nested pointer-deduplication
   loop: each payload is compared with every preceding payload. This is
   O(P²) for P historical pages, even though ordinary different-page payloads
   have distinct allocations. It additionally recounts and hashes all page
   descriptors. These operations are unconditional; the profiling flag only
   controls timing/log output.

With T fresh tokens and S layer/head stores, the common append path therefore
contains roughly O(T × S × P²) CPU work that should not exist. At fixed H,
P grows approximately as occupied C / 256 because cold summaries intentionally
survive eviction. Around 20K versus 100K occupancy, P rises roughly fivefold;
the nested-accounting work rises roughly 25-fold. For illustration, T≈15,600,
S≈64 plus the compatibility store, and P≈390 imply on the order of **tens of
billions of pointer comparisons** if those stores contain the corresponding
history. This is a static operation-count estimate, not a runtime measurement.
It plausibly explains a much larger slowdown than ordinary one-layer draft
attention, and leaves the GPU waiting while the CPU prepares the next graph.

Important distinction: summary payloads already use
`std::shared_ptr<const payload>` (`llama-kv-routing-summary.h`, private `page`);
this is **not** a newly discovered deep copy of all old float vectors. The
remaining descriptor copies, repeated scans and especially quadratic
accounting are sufficient to be a problem. Do not reimplement the earlier
payload-sharing repair or claim all historical KV bytes are being recopied.

### Specific repair direction

- In `begin_write_planned()`, invalidate an existing sealed summary only on
  the transition from summarized/immutable to dirty. A page already marked
  `summary_content_version==0` should not repeatedly invalidate every store
  during ordinary append. Keep content-version advancement, ticket identity,
  host invalidation, pinning, and rollback semantics intact. Audit restore,
  cancellation, codec mutation and partial rewrite paths before making that
  flag the authoritative fast-path guard; stale summaries must never remain
  selector-eligible after a real rewrite.
- Add a real no-change path in summary invalidation: if the requested exact
  identities are absent, do not recount payloads or rehash descriptors. Any
  required snapshot-epoch rebinding is metadata work, not permission to do
  whole-store accounting. A by-value `return *this` still copies descriptors;
  eliminate the common call at the pager first rather than only optimizing
  its return statement.
- Replace the nested payload-pointer deduplication in `rebuild_accounting()`
  with a standard-library pointer set or sorted/unique pointer inventory.
  Preserve unique-allocation charging and saturating/overflow behavior. Aim
  for O(P) expected or O(P log P), never O(P²). Do not assume payloads are
  unique without establishing that invariant for all APIs.
- For a genuine update/invalidation wave, deduplicate changed identities and
  publish once per layer/head store. The existing `update_pages()` and
  staging wave in `seal_ready_pages()` are the starting point, not a reason
  to redesign the pager. Preserve atomic all-store success/failure handling
  and old payload lifetime for in-flight readers.
- Do not add per-token counters, per-token logs, GPU readbacks or a new fence
  to demonstrate this repair. The follow-up measurement can time the existing
  apply/seal/policy boundaries over a bounded batch.

This fix is equally valuable for decode, but prefill magnifies it by writing
thousands of rows. It does not change selected attention quality or require
less hot capacity.

## 2. Other avoidable whole-history metadata costs

These are secondary to the per-token defect, but should be corrected in the
same architectural direction if they remain material after the first repair.

- `exact_page_records()` (pager 1090–1126) and `routing_inventory()`
  (2262–2287) build resident+cold unions by checking each catalogue entry
  against an output vector that grows with every cold page. Both can become
  O(P²), followed by sorting. Use an indexed union or sorted merge keyed by
  the full stable identity, retaining resident precedence. Cache a coherent
  immutable inventory per catalogue/residency version, not per token.
- `remember_logical_page()` (2289–2306) linearly scans the entire cold
  catalogue on every publication. `publish_page()` (2966–3009) is called by
  per-row writes/completions, so this introduces O(T × P) work even after
  no-op invalidation is removed. Maintain an internal page-to-record index
  keyed by session/sequence/generation/logical-page/layer; deletion and slot
  reuse must update it. Start with that local index rather than weakening
  residency transactions or their commit/rollback boundaries.
- `routing_summary_content_version()` (2238–2248) falls back to a linear
  store lookup for each cold record; `range_min()`, `range_max()`, `mean_k()`
  and `content_version()` in the summary store (772–830) all linearly search
  sorted descriptors. A catalogue walk that repeatedly calls them becomes
  quadratic again. Use binary search with the existing identity ordering or
  a validated index, not logical page alone across multiple generations.
- At the final-user selector graph, `set_kv_page_select_inputs()`
  (`llama-kv-cache.cpp`, 18578 onward) walks the inventory and issues small
  tensor uploads for changed bounds, metadata and membership. Preserve its
  existing content-version delta logic; batch adjacent dirty ranges and
  retain stable device inputs. Do not upload the entire cold catalogue every
  microbatch. `capture_kv_routing_query()` (2582–2668) already defers the full
  selector refresh to the graph containing the last user row; intermediate
  graphs accumulate Q. Do not regress that cadence.

`update_pages()` already shares old payloads and only builds supplied inputs,
but its descriptor copy, sort, accounting and input/inventory searches still
scale with P. Address these after eliminating the much more frequent no-op
calls; do not start with a wholesale replacement of the summary subsystem.

## 3. Full-L GPU MTP adds genuine C-dependent attention work

The previous note's inference that *all* per-token attention work should be
bounded by H was incomplete. The target is bounded, but the draft is not.

- `common_speculative_impl_draft_mtp::process()` (`common/speculative.cpp`,
  2877–3194) mirrors each target prompt batch into `ctx_dft` with
  `llama_decode(ctx_dft, batch)` around 3111. This is an actual model forward
  pass, not a CPU-RAM storage operation. The ordinary process path catches up
  the incoming batch, not the entire previous prefix. There is a separate
  target-only recovery path when the carry is invalid; do not mistake it for
  evidence that full-prefix replay is the usual behavior.
- `src/models/qwen35.cpp`'s MTP builder (583 onward) creates one MTP block and
  uses normal cached attention at 698–701. Its GPU Turbo4 cache spans L.
- For nonpaged draft cache, `llama_kv_cache::get_n_kv()` (15890–15903) derives
  the read width from the padded occupied cell watermark. Thus the MTP
  attention width grows with C, up to L, rather than stopping at target H.
- `set_input_kq_mask_impl()` (16291–16439) also writes a CPU mask of roughly
  U × occupied-draft-width elements. It optimizes later rows by copying the
  first row and updating nearby indices, but the row copies still scale with
  C. At U=256/C=100K an F16 mask is about 51 MB per microbatch. This is
  auxiliary metadata, not CPU model attention. It can contribute CPU and
  upload cost even when all draft weights/KV are GPU-resident.

After removing the pathological pager work, some slope can remain from this
one full-history MTP attention layer. Do not promise flat 1500 tok/s up to
250K with the present exact full-history draft semantics. A fivefold context
increase does not by itself prove that one layer explains a fivefold total
slowdown across the whole model; its contribution is not yet timed separately.

Optimization order without sacrificing intended MTP:

1. Keep full-L GPU Turbo4 draft storage and correct streaming hidden carry.
   The device-hidden handoff already exists around process() 3026–3052 and
   `process_ubatch()` 6890–7004. Verify its path later rather than rewrite it
   or materialize every target hidden row on the host again.
2. For the common one-sequence contiguous causal draft, avoid constructing
   and uploading a U×C host mask. Extend the supported fused CUDA attention
   contract with causal position/valid-prefix metadata, generated or consumed
   on GPU. Keep the existing general mask path for holes, multiple sequences,
   M-RoPE cases needing distinct equal-position rules and unsupported shapes.
   This requires an explicit kernel/graph contract; simply passing null mask
   to today's generic FA would change causality and is not a fix.
3. Preserve the existing matched Turbo4 fused CUDA path:
   `ggml_cuda_flash_attn_ext()` in `ggml/src/ggml-cuda/fattn.cu`, around
   4309–4425, already accepts supported multi-token matched Turbo4 shapes.
   An old comment saying all Turbo prefill dequantizes to F16 is not proof
   that it does so for this shape. Profile dispatch later before proposing
   another precision/route change. Optimize GPU tiles/partitioning if this
   full-width layer is the remaining measured owner.
4. Only if full-history MTP remains dominant, consider a separate explicitly
   approximate **draft-attention** policy while retaining full-L stored
   Turbo4 KV. Bounded draft attention can reduce attention arithmetic, but
   changes proposal behavior and may lower acceptance. It is not the first
   repair and cannot be silently enabled merely to improve prompt timing.

## 4. Intermittent double processing has a concrete query-replay path

`llama_kv_cache::commit_kv_pager_query()` (2930–2976) compares the historical
selection at turn entry with the final committed selection. A change can be
a resident rerank; it does **not** require an H2D promotion. At the server
boundary, `server_query_replay_transition()` and its caller
(`tools/server/server-context.cpp`, 21945–22027) restore the query checkpoint,
truncate the logical prompt to `query_begin`, and process the complete final
user span again once. `server_query_checkpoint` (1940 onward) preserves hybrid
recurrent state and speculative carry, not a stale full attention map.

That mechanism fits the C=100,228 observation: roughly a 15.6K user turn is
processed twice, with about 31.2K timed target prompt tokens and doubled seal/
summary work. It is a code-backed explanation, not a confirmed match to that
row's replay log. The old speculation that draft-prefill alone causes the
target's doubled prompt counter is less direct than this server replay path.
Fresh-input throughput for that row is **15,606 / 133.34 ≈117 tok/s**; the
reported 234 tok/s is throughput of processed/replayed prompt rows. These are
different findings and should both be labeled correctly.

Do not remove replay indiscriminately: Qwen's recurrent state and MTP carry
must reflect the final selected history before generation. Better direction:

- Keep exact one-replay behavior for a genuine recall/query whose selected
  historical set changed, and zero replay when the effective selection did
  not change. Compare authenticated content/visibility, not incidental epoch
  movement or a harmless page-list permutation.
- Avoid treating 15K of newly ingested file data plus a short question as an
  unavoidable 15K replay query. In the later workload/task, use ordinary
  ingestion turns followed by a short semantic question, or implement an
  explicitly declared ingest/query seam with the matching recurrent/carry
  checkpoint. This reduces the replay span; it must be reported as a workload
  or API-semantic change, not disguised as faster processing of identical
  requests. Keep a paired same-request comparison for implementation speed.
- A two-stage selector/probe followed by one authoritative query forward pass
  is a larger architectural option. Do not implement a Q-only shortcut that
  assumes deep-layer Q is independent of earlier attention/recurrent state.

## 5. GPU route, RAM backing and fences: what not to change speculatively

The selected target graph is explicitly bounded by resident H in
`prepare_kv_attention_graph()` (`llama-context.cpp`, 3106 onward): it checks
snapshot pages against admitted physical capacity and builds a selected view.
Packed allocation uses `pager_snapshot.physical_rows`, not logical L
(`llama-graph.cpp`, 4555–4565). This is evidence against a simple accidental
full-C target-KV tensor being the default explanation, although the exact
per-layer submitted shapes still need the later verification.

Host sealing is queued for changed completed pages; `seal_ready_pages()`
already uses a maintenance queue and asynchronous full-page host capture
(`llama-kv-pager.cpp`, 2569–2760). Required host traffic per fresh chunk is
approximately constant, not proportional to total historical C. CPU summary
dequantization in `pager_routing_summary_build()` (`llama-kv-cache.cpp`,
4932–5110) processes newly changed page data and can be moved to a GPU page
summary producer later, but the first fix is unnecessary **old-store
accounting**, not removing canonical RAM backing or re-reading all old KV.

Real capacity pressure still requires a completion boundary before reusing a
slot with in-flight writes/copies. Existing fresh/cached-wave coalescing in
`llama_context::decode()` (around 7600–7650) is dependency-conditional; repeated
fences should be improved only with accurate page ownership/events. Do not
replace valid asynchronous transfers with synchronous host attention, weaken
pinning, or remove the fence protecting a source slot.

Keep B=1024/U=256 and Turbo4 for target and draft. Do not route decode/MTP to
direct attention based on an arbitrary packed-owner byte cutoff: that does
not fix this prefill CPU pathology and can select an unsupported split/page-
mass shape. Likewise, requested H is an upper bound; use actual admitted
physical rows for memory/workload planning. Packed storage is compressed
Turbo4; a two-owner workspace reservation is not proof of two live F16 copies.

## Recommended implementation order and later measurement boundary

1. Eliminate repeated invalidation of unsummarized mutable pages and the
   quadratic payload-accounting loop. Preserve all page identity, transaction,
   rollback and summary-readiness invariants. This is the best-supported
   first optimization, with no model approximation or route changes needed.
2. Remove quadratic inventory unions and linear per-row catalogue publication
   lookups; use versioned indexes and batched real summary changes.
3. Attribute residual time to target graph, native MTP catch-up, actual backend
   execution, replay, and page maintenance separately. Correct the misleading
   graph-construction and wait interpretations in future reports.
4. Only then optimize the draft's full-width causal mask/attention and any
   measured GPU kernel owner; bound actual replay query work without weakening
   recurrent/MTP consistency or falsely changing the benchmark denominator.

The separate testing task should use a short fixed fresh chunk at a few saved
frontiers, one immutable candidate and actual admitted H, not another full
250K fill merely to establish this defect. Measure **fresh input divided by
wall time**, processed target rows, draft rows, replay rows, CPU apply/seal/
policy time, and GPU kernel time separately. Operation counts for no-op
summary updates should not scale with every token × historical pages, and
accounting must no longer scale quadratically. Keep test design/proof gates
out of this research update; no live or deterministic tests were run here.

Raw evidence root remains
`/srv/ai/paged-kv/results/forward/102-04/attempt-01/`. This document supplies
code-directed optimization decisions for later tasks, not a passed speed gate
or permission to silently alter the workload, target quality, or MTP policy.

## Source implementation — 2026-10-04

Implemented on `codex/task-102-05` in source commit `785764d5d`, following the
separate instruction to implement the code fixes. **No compilation, regression suite, benchmark,
generation request, server reload, or service change was performed.** The
earlier source-only investigation remains distinct from this implementation;
the currently loaded known-good binary has not been replaced. These edits are
not runtime acceptance evidence and do not close an occupancy or speed task.

Implemented changes:

- `llama_kv_pager::begin_write_planned()` skips summary invalidation for an
  already-unsummarized append tail. The guard also checks the exact retained
  compatibility-store identity: promoted cold pages can retain a summary
  despite having a zero resident `summary_content_version`. Content versions,
  host invalidation, tickets, pins and normal write completion are preserved.
- `invalidate_routing_summaries()` avoids copying unaffected stores, stages
  every affected compatibility/layer/head store, then replaces existing stores
  without index growth. A failed staged invalidation refuses the corresponding
  write/cancellation before that operation changes page state. Explicit host
  invalidation clears the resident summary marker; cancellation invalidates
  any summary published since its write ticket was issued.
- Summary-store `invalidate_page()`, `invalidate_pages()` and `reconcile()`
  reuse charges/digests when no descriptor was removed, while retaining epoch
  rebinding and zero newly hashed payload floats. Exact invalidation identities
  are sorted/deduplicated once per call rather than scanned for each descriptor.
- `rebuild_accounting()` deduplicates immutable payload allocations with a
  standard-library pointer set in expected O(P), replacing O(P²) comparisons.
  Unique-allocation charging, source accounting and digest ordering remain
  intact. Allocation/accounting failure propagates as `overflow` rather than
  accepting a zero charge or throwing through this `noexcept` boundary.
- The pager's `logical_catalogue_` is a stable-owner-keyed map. Row publication
  and deletion are O(log P), not whole-history scans. The key preserves the old
  session/sequence/sequence-generation/logical-page/layer equality; the value
  still contains every authenticated identity and content field.
- `exact_page_records()` and `routing_inventory()` check a bounded resident
  key set when combining resident and cold records. Resident precedence and
  their existing output sort order remain, without scanning the growing union.
- Exact summary identity lookups use binary search over sorted descriptors.
  Ordering includes `attention_layer`, matching complete identity equality.
  Legacy logical-page-only accessors retain their existing cross-sequence
  semantics; runtime exact-identity lookups no longer use those linear scans.
- Summary inventory validation, full construction, reconciliation, seal-wave
  updates and CPU score validation use owner-keyed indexes instead of nested
  whole-history searches. Each changed seal wave merges sorted descriptors
  linearly, sorting only its new inputs and sharing unchanged payloads.

The common dirty-tail append now performs an exact retained-summary lookup and
catalogue publication in O(log P), rather than copying/recounting S historical
stores per token. Genuine summary changes still cost a store-level wave;
history remains discoverable rather than being discarded to improve timings.
This is an algorithmic improvement, **not a measured throughput result**.

Deliberately unchanged: GPU attention dispatch/precision, full-L GPU Turbo4
MTP, selector cadence/device inputs, replay semantics, asynchronous canonical
RAM backing, transfer fences, page size, batch/microbatch and hot capacity.
The possible GPU causal-mask contract and replay-span changes described above
are separate architectural options, not proven defects to implement blindly.
No versioned inventory cache was added: indexed construction already removes
the quadratic union without introducing another cache-invalidation authority.

Follow-up testing should build this source into a separate immutable candidate,
check append/rewrite/promotion/cancellation and exact-identity summary behavior,
then compare the same short fresh chunk at saved occupied frontiers. Preserve
the previous binary and fixed B/U/Turbo4/MTP settings. Report fresh and replayed
rows separately; do not rerun a full 250K fill just to assess this repair.
