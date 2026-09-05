# Technical change specification for post-17 implementation

Read only the sections linked by your task. These are concrete implementation
directions for the inspected tree, not claims that the changes already exist.
Use the existing types and source files; names for new internal fields below
are proposals, not invented public APIs. Preserve off-mode behavior and prove
each transition with the tests named in its packet.

The packet's concrete recipe and `IMPLEMENTATION_CONTRACTS.md` narrow the
interfaces and checkpoint order. `EXECUTION_COOKBOOK.md` supplies actual CLI
entrypoints, reusable minimal fixtures and the common receipt schema. In
particular, a client mode label is not proof of server dispatch, and a CPU
exact-state helper is not a GPU wave executor.

## T1 — One explicit model-layer to attention-layer mapping (18-05, 19-02)

There are two different indices. `llama_kv_cache::pager_geometry()` appends
offsets while iterating cache `layers`. Those vectors are indexed by **cache
attention-layer ordinal**. `llm_graph_context::build_attn()` receives model
layer `il` and currently accesses `direct_layer_k_offsets[size_t(il)]` and V
the same way. A hybrid model need not have contiguous full-attention layer IDs.
This is a concrete indexing hazard to test, not yet a proven cause of the
observed bad answers.

Add/derive a checked mapping from `llama_kv_cache::get_layer_ids()` (or existing
equivalent): model layer -> attention ordinal. Store model-layer IDs with
geometry and use the same mapping in direct views, host units, summaries and
telemetry. Never use model `il` to index a compact attention-only vector unless
the mapping proves identity. Test non-contiguous IDs, first/last layer, absent
recurrent-only layer, and every live Qwen full-attention layer. Do not hardcode
the apparent reference spacing or count as a generic fix.

### T1a — Selected-reference K/Q coordinate mismatch (18-05, highest-priority reproducer)

`llm_graph_context::build_attn()` gathers Turbo4 K/V with `ggml_get_rows()`.
CUDA `getrows.cu` calls `dequantize_turbo4_0()` in `turbo-quant-cuda.cuh`, which
returns centroid * norm values **without inverse FWHT**. Thus the gathered
floating-point K is still in Turbo's rotated domain. The subsequent ordinary
FA sees floating-point K instead of a Turbo4 type. In `fattn.cu`, the automatic
query FWHT is conditional on the K type being Turbo; it can therefore disappear
on this reference route. This is a specific likely correctness bug, not proof
that every observed mismatch has that cause.

Reproducer: quantize a known K row using the actual Turbo4 encoder, gather it,
and compare (a) dense Turbo4 QK logits, (b) selected-reference QK logits and
(c) explicit same-domain dot products. Include two 128-wide blocks, multiple
GQA heads and current channel-scale settings. In the live model capture the
first all-selected layer's logits/output with MTP off before changing code.

Preferred fix: carry **representation domain separately from tensor dtype**
through selected-reference metadata. Apply exactly the same Q transform used
by dense Turbo4 FA before dotting gathered rotated K, including signs,
normalization and InnerQ inverse channel scale where active. Reuse a proven
backend transform or add the smallest internal graph operator needed; a plain
FWHT alone is not sufficient if the dense contract includes channel scaling.
Keep gathered V in the documented rotated domain and preserve exactly one
`build_attn_v_unrotate()`/mean restoration after attention. Do not fix this by
changing canonical Turbo4 bytes or pretending gathered F32 values are Turbo4.
An alternative explicit K inverse-transform reference is valid only if all
associated scaling is handled and matches dense logits; record the choice.

After this minimized regression passes, check model-layer indexing and physical
authority separately. One fixed transform does not prove cold paging is wired.

## T2 — A single physical target authority (19-01/02/03)

Current graph stores use `mctx_cur->cpy_k/cpy_v` into cache tensors while the
direct attention branch reads `inp->direct_storage` from the residency pool.
Prove aliasing or explicit synchronization; assuming these buffers contain the
same bytes is invalid. This is a priority candidate for stale/zero KV reads.

Chosen destination: the existing compact residency slab is the only hot target
physical authority in selective/exact pager mode. Rebind cache views/writes to
it, or implement the smallest bounded physical-row write operation if current
`cpy_k/cpy_v` cannot express the strided layout. Do not keep a second full-size
GPU cache and copy between the two on every request. Dense/off stays separate.

For page P, layer ordinal A, row R and KV head H, validate addressing against
the real row layout:

```
slot = logical_to_physical(sequence, P, generation)
row_base_K = slab + slot * bytes_per_slot + layer_k_offset[A]
               + R * k_row_bytes
head_K = row_base_K + H * ggml_row_size(TURBO4, head_dim_k)
```

Apply the analogous V formula. Use `ggml_row_size` and real strides, not raw
4-bit arithmetic that forgets Turbo4 block metadata. Byte views used by GGML
must describe allocation bounds and dependencies even when the kernel uses
explicit strides. Graph pruning must not move the attention read before the
KV write to its aliased physical storage. Include an explicit dependency if
an alias alone is not visible to the scheduler.

Keep **logical cell metadata** sized to requested context in CPU memory;
physical backing/compute scratch scale with hot slots. A `begin_write` ticket
must bind graph destination, complete/cancel callback and host-capture source
to the same slot/generation. Test with deliberately poisoned old slots, fresh
zeroed slabs and a permuted mapping so accidental dense-cache reads fail.
Compare source/destination Turbo4 bytes before comparing model text.

## T3 — Admission before allocation, with measured MTP compute (19-01)

Split geometry/planning from materialization. Geometry can be computed from
GGUF hparams/type/row sizes without allocating context-sized GPU tensors.
Account for shared model buffers once, external CUDA occupancy and separate
target/MTP graph workspaces. Reserve worst-case selected query tile, verify
block and recurrent/checkpoint workspace before allocating target slots.

Add post-warmup reconciliation: projected category bytes versus actual category
bytes and peak free margin. A small allocation failure during MTP startup must
trigger a page-aligned reduction of target hot capacity and clean retry of
context construction, not repeated restart of the same allocation layout.
Cap retries by a deterministic descending page/workspace search and preserve
each attempted ledger. Never hide the change by reducing requested `n_ctx`.

## T4 — Host seal and real transaction ordering (19-02/03/07)

Use the existing `llama_kv_live_policy_boundary`,
`llama_kv_residency_transaction_request`, transport and hooks. Do not add a
parallel pager state machine. The actual caller supplies complete host+resident
inventory, not just current resident pages. Trace call coverage from live
request to `llama_kv_pager::apply_live_policy()` and transfer completion.

```
reserve writable slots -> enqueue all layer K/V writes -> GPU completion
 -> commit accepted target/recurrent/MTP frontier
 -> seal accepted bytes to canonical host (tail remains pinned until complete)
 -> choose desired physical set -> reserve/pin -> enqueue H2D
 -> event complete -> recheck full identity -> publish page table -> retire
```

Keep host copy authority distinct from GPU residency. Copy-completion counters
advance only when the event completes; failed/stale transfers do not count as
useful hits. A failed replacement after a clean drop must restore the old
graph-usable set from host or leave the old mapping intact until replacement
is safe; test this with occupied slots, not only empty-pool tests.

## T5 — Multiquery direct attention and the serial-kernel bottleneck

Owners: 19-04/05/09, 20-04. Actual implementation is in
`ggml/src/ggml-cuda/fattn.cu`, not just the declaration in
`fattn-paged-turbo4.cuh`. The present
`ggml_cuda_fattn_turbo4_paged_decode_kernel` launches one 256-thread CTA per
query head and loops over **every selected row**, with block synchronizations
per row and online softmax. It is a correctness-first serial-in-KV kernel,
not Buun's tiled MMA flash attention. Increasing n_query_tokens alone will not
make it fast. Keep this kernel as a small diagnostic oracle.

19-04: extend the raw parameter and graph interfaces with explicit query/head
strides, query count, query-position stride, output/partial/page-mass query
strides. Update `ggml.h`, `ggml.c`, CUDA shape validator/dispatcher and graph
reshape assumptions together. Do not reinterpret an old layout with new fields
without a version/layout check. Query positions cannot always use `[0]`.
Preserve the dense contract: Q Turbo pre-rotation happens once, output remains
in the Turbo V domain until `build_attn_v_unrotate()`, also exactly once.

19-05: adapt the existing Turbo4 tile loader/MMA path where possible: replace
contiguous tile row addressing by page descriptor -> physical row, split tiles
at page/tail boundaries and preserve per-query native causal masking. Prefill
tiles and multiquery verify reuse compressed K/V across queries; do not launch
an independent entire-cache serial scan for every query as the final path.

19-09: for small-query decode, partition selected pages across multiple CTAs
per head/query group (split-KV). Each writes an unnormalized `(m,l,o)` partial
to a bounded scratch buffer; a second GPU reduction merges these states and
normalizes. Choose partition count from active rows/device occupancy and a
scratch cap; do not bake in reference GPU SM counts. Compare this approach to
existing tiled Turbo MMA for multiquery verify/prefill, retaining the faster
qualified dispatch by shape. Profile register/shared memory and occupancy.

Page attention statistics must use the **global** normalization after partial
merges. Local per-partition probabilities cannot just be summed. Reserve this
scratch in T3, and include telemetry-off/on timings so score collection cannot
silently dominate the claimed fast path.

## T6 — The exact-wave operation belongs inside each attention layer (19-06)

Build the model's Q/K/V, stage a bounded wave of that layer's canonical KV,
launch the direct partial-state kernel, merge on GPU and then continue that
layer's output projection/recurrent graph. Do not call a whole-model forward
once per KV page; do not normalize each wave independently and add outputs.
The graph operation must own an immutable per-layer wave plan and runtime
completion fences, with explicit scratch and aliases. Start with a noncaptured
dynamic cold path; enable capture only for stable, fully qualified layouts.

Reuse `llama-kv-attention-exact.*` numerical/coverage helpers and the existing
backend partial-state API. Add the graph execution door that is currently
missing. If a host catalog/pinned ring is needed by CUDA execution, pass a
backend-neutral lifetime-owned runtime descriptor via the established internal
adapter, not a dangling stack pointer embedded in op params. Avoid recursive
scheduler execution from inside a node callback. Test exact prefill, not just
decode, so conservative full-history prefix construction is available.

## T7 — Current queries do not exist before the graph runs (20-01/03/04)

This timing boundary is essential. `prepare_kv_attention_graph()` cannot score
the current layer query on CPU before the layer's projection has executed.
Do not pretend a pre-graph query index implements current-query retrieval.

Implement two explicitly distinguished paths using existing graph/backend seams:

1. **Fresh selection correctness path:** add a query/score boundary after Q
   projection and positional/codec transformations. GPU scores all-page
   summaries and produces small candidate IDs/scores. At an explicit graph
   split, transfer those small results, run the host residency transaction,
   wait only for necessary uploads, then execute that layer's attention. Carry
   a lifetime-owned routing context and generation. A scheduler eval callback
   may prototype the boundary, chaining existing callbacks, but must be labeled
   diagnostic: `src/llama-context.cpp` already documents its synchronization
   overhead near `eval_callback_dormant()`. Do not replace a user callback or
   install a permanently active full-sync callback as the final fast path.
2. **Stable warm/predictive path:** retain a prepared hot set and graph-stable
   descriptors. GPU current-query selection among resident pages stays on GPU;
   current-query all-page candidates can prefetch for the next step. Label
   query-generation age and prediction mode. Use fresh selection on new focus/
   request boundaries and confidence/recall failure; essential missing pages
   require the fresh wait/fallback, not an unreported miss. Predictions from
   previous-token/layer real Q are hints, never described as current-query proof.

Initial correctness can be synchronous. 20-03/04 must measure and remove its
warm-path host waits, not just add another stream. A cold/fresh path may require
breaking capture; warm fully resident replay should not. Keep selection policy
honest about query freshness and show cold first-answer recall in quality tests.
If strict current-query scoring remains faster than predictive complexity at
measured sizes, retain it and report the profile; complexity is not a goal.

## T8 — Layer-aware index and retention, with an explicit starting design

Owners: 20-01/02/05. The current summary store holds one configured layer/head.
Introduce stores indexed by runtime attention ordinal and KV-head/group, budgeted
in the ledger. Reuse existing representative-key format first to validate the
whole pipeline. Then add a component-wise min/max candidate form:

`bound(q,page) = sum_i (q_i >= 0 ? q_i * max_key_i : q_i * min_key_i)`

Keys and queries must be in the same **scored** coordinate system, including
Turbo4 transforms/scales and RoPE; only call this a bound if extrema cover all
valid decoded keys, not four sampled keys. A bound is a ranking hint, not a
calibrated attention probability. Measure representative recall versus bounds
and budget their bytes/build cost. Build/update summaries from all committed
rows on GPU at seal, avoiding full key D2H just to make summaries.

For GQA, score the actual query heads and reduce explicitly within each KV
group (start with max for candidate recall); do not invent an averaged query
without measuring cancellation errors. Per-layer selected masks can share a
cross-layer physical transfer group. Initially union groups conservatively,
then measure traffic/compute waste before introducing layer-granular eviction.
No unbounded union may exceed admission; mandatory pins reserve space first.

Retention stores observation count, EMA mass, peak and last-used per layer/page.
Normalize score families on calibration traces. Update only observed mass;
cold unknown state does not decay as an observed zero. Stable ties, bounded
exploration and pin priorities must be deterministic and tested at tiny H.

## T9 — Graph keys and metadata allocation (20-04)

The inspected graph reuse checks size input tensors from selected row/page
counts. Allocate admitted-capacity descriptor arrays once and carry active
counts/valid bits as data where backend semantics permit. Graph identity should
include shape/capacity, device, codec, layer layout and query tile, but not
every content generation. Data epochs still protect stale consumers through
fences. A table update is not automatically a graph recapture.

Avoid clearing/copying a full logical-page score array per query head every
step unless measurement justifies it. Use cadence or touched-index updates with
versioning, and explicitly reset on generation changes. Never optimize by
reusing stale scores/slots unnoticed. The all-fit fast path must still have a
forced-paged parity probe, and observe must not acquire pager mutation costs.

## T10 — Specific benchmark algorithm changes (18-01/02/03/04)

These harness repairs are verified with small purpose-built test contexts.
They do not require the final six-point speed curve. Use synthetic token
boundary tests, one small live preflight, a few-page forced-pressure request
and an interrupted two/three-case client to validate the mechanisms. Explicit
full-context capacity tests and the final results campaign have their own
packets and are not repeated after every harness/kernel change.

- Token fitter input is complete messages + generation reserve + target model/
  template identity. Render once, tokenize, binary-search only padding until
  within requested occupied-token bound, then persist the exact final request
  hash and tokenizer count. Validate fact positions and suffix unchanged.
- Durable case state is planned -> started -> completed, or interrupted with
  a new attempt ID. Only complete provenance-identical cases feed statistics.
  Flush response/metrics first, atomically mark case complete second.
- Timeout categories: connect, server startup, prefill no-progress, decode
  no-progress, total campaign. Progress restarts idle timers, not hard total
  deadlines. Outer wall expiry checkpoints rather than fabricates a bad answer.
- Capability probe precedes a mode. Store one 501 refusal and remaining cases
  unevaluated; a 400 overflow goes back to sizing. A quality sentinel mismatch
  creates a minimized regression, not a 24-case repeated failure matrix.
- Lifecycle captures MainPID/exe/loaded-DSOs/effective argv and verifies those
  after restore; one bounded owner controls sudo, locks and child process IDs.
- Metrics table has explicit bytes/tokens/pages/microseconds plus provenance.
  The measured native-MTP snapshot cannot be overwritten by an admission
  estimate. A zero transfer counter is distinct from missing measurement.
