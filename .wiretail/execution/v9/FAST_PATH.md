# V9 fast-path implementation contract

Authoritative for phases 34/36. Resolve symbols with `rg -n`; line numbers
below refer to pre-merge SHA 4af26bd68 and are navigation hints only.

## F1. Physical addressing fails closed before CUDA

`src/llama-kv-pager.cpp::physical_row` maps a **model layer ID**, not its
ordinal among attention layers. Preserve the corrected geometry lookup.
In `src/llama-kv-cache.cpp::set_input_k_idxs` / `set_input_v_idxs` (13478 /
13508), remove the paged-target `compact ? pager_row : logical_idx` fallback.
For paged target writes, require a valid physical mapping, sequence/generation,
row within resident slot, valid native position and ready write reservation.
Propagate the existing recoverable memory failure before any index upload or
graph launch; do not write logical index L into a slab of capacity H. Non-pager
contexts retain their ordinary addressing. Do not assert/crash to test refusal.

Test non-contiguous model IDs, missing ID, stale page/sequence generation,
capacity overflow, tail boundary and unpublished destination; verify zero
backend writes on refusal. Check K and V independently. Use existing pager
CPU fixtures, then one short native-MTP CUDA append across a page boundary.

## F2. Persistent compressed selected workspace

Owners: `src/llama-kv-attention-execution.{h,cpp}`,
`llama_kv_attention_packed_cache::{find_or_create,release_completed}`, current
entry fields and `llama_context::prepare_kv_attention_graph` (~2174).

Replace content-dependent ownership with a structural key:

`(model-layer-id, sequence-slot, backend/device, K/V type and head geometry,
  A_capacity, source-allocation-lifetime, owner-generation)`.

Do not key on graph pointer, vector of selected pages, live row count, newest
token, or a content/selection epoch. Allocate K/V at fixed padded A once per
key. Track separately a fixed slot table: logical page ID, source physical
slot/generation, native row positions, valid row count and copied content
version. Keep existing selected pages in the same packed slots. Assign newly
selected pages to free/replaced slots deterministically; do not shift/repack
all slots when one page is promoted. Duplicate page IDs are invalid.

Context owns storage across graph rebuilds; graph inputs hold a use lease.
`release_completed` must not free a persistent owner merely because it was not
used in the most recent graph build. Release only on context/sequence teardown
or structural replacement after its last backend completion event. Bound to
one active owner plus at most one draining replacement per layer/slot, charge
both while draining, and refuse runaway replacement. Distinguish allocation
lifetime from mutable content versions. Test repeated reserve/build/replay,
rollback, slot reuse, changed A, owner retirement and constant allocation count.

Initial packed layout: per-layer Turbo4 K/V, A rows, page-aligned fixed slots,
unused rows masked. Non-chronological slot order is legal; use native positions
for causal masking. Page size initially uses the implementation's validated
256-token contract; other requested sizes must be supported and tested or
explicitly refused, not silently rounded into incompatible layouts.

## F3. Mature FA for every production query shape

Switch `llama_kv_attention_execution::planned_route` (~351): legal contiguous
typed view -> selected packed -> explicit unsupported/refusal. Do not choose
custom direct before packed. Keep `off` ordinary, `observe` ordinary + sampled
diagnostics, `exact` explicitly separate. Existing explicit direct override
can remain diagnostic; do not enable it automatically because `direct_shape`
succeeds. The custom route's split scratch/MMA/head/Q bugs are not this plan's
production implementation target.

Graph sites: `src/llama-graph.cpp::build_attn_inp_kv_impl` (~3725), selected
attention graph construction (~4333–4700), ordinary `ggml_flash_attn_ext`
(~3517), `llm_graph_input_attn_kv::set_input` packed refresh (~864).

Important ordering:

1. Immutable newly selected historical pages may be copied compressed D2D on
   the owning backend stream before the graph, ordered before attention.
   Use `ggml_backend_tensor_copy_async` only when its ordering/lifetime contract
   is met. Do not loop synchronous `ggml_backend_tensor_copy` over all A rows
   for every token. Data/version updates become visible only after the copy.
2. **Current microbatch K/V is written inside the graph.** Copy its new/dirty
   ranges into the packed slots using graph CPY/view nodes whose source is
   the actual K/V write node output, then make attention consume those nodes.
   Inspect the existing cache `cpy_k`/`cpy_v` graph methods after merge. Do not
   make a pre-graph host copy of data that has not been written yet; merely
   putting independent aliasing nodes in a list is not a dependency. If GGML
   views do not preserve the write-to-read edge, add the smallest explicit
   dependency to the existing copy construction and test scheduling order.
   The current source already appends `ggml_cpy` nodes for pages receiving
   current queries just after `mctx_cur->cpy_k/cpy_v` (~4395–4425). Preserve
   and tighten that path; do not add a second full-page copy. Its existence
   is not proof of a bug or of correct alias/lifetime handling. The regression
   determines whether an explicit dependency is missing after refactoring.
3. Copy only new row intervals/touched partial pages, handling a microbatch
   spanning multiple pages. Immutable historical pages with unchanged
   `(logical ID, generation, content version)` produce zero D2D work.
4. Mask uses native token/sequence positions. All queries see valid earlier
   selected rows and their own causal prefix within the current microbatch;
   rejected/future/padding rows get `-inf`. Require A to fit the mandatory
   recent/current set. Never truncate a multi-query MTP verification to its
   first row. Never use packed slot index as the RoPE position.
5. Use mature Turbo4 Q rotation/dequantization and the existing V output
   unrotation exactly once. The stored representation stays Turbo4. Do not
   wrap it in a permanent F16 K/V cache or allocate an F16 L/H mirror. If the
   mature backend legitimately requires bounded dequant scratch, charge A
   rows explicitly; no hidden full-context materialization.

Dense-view fast path is valid only if the **selected** rows form a compatible
contiguous physical span with correct mask and lifetime. Do not equate all
resident H rows with selected A or silently widen selection to obtain a view.

Parity fixture: use four KV heads and the real GQA ratio from the model plus a
smaller synthetic ratio. Exercise query counts 1, 2, 3, 64 and 128; multiple
KV heads must have distinct data. Compare identical quantized selected K/V
and masks against mature contiguous FA using the same dequantized reference.
Include shuffled physical slots, logical gaps, partial last page, all-masked
query, invalid slot, current microbatch and MTP rollback. Compare values, not
just finiteness. Reject NaN; use the existing backend reference tolerance and
record error distribution. Do not loosen tolerances to hide wrong heads or
FWHT. CPU scalar reference is a fixture, never a selective runtime fallback.

## F4. Crash diagnosis on the chosen path only

Prior observed illegal access surfaced at synchronization; that does not name
the first bad kernel. The old synchronous setter completed its own upload,
so a different-stream theory alone was not proof of cause.

Use a tiny fresh candidate sequence: prompt -> append crossing a page ->
native-MTP multi-row verification -> q0 repeat/slot clear -> cold recall.
First run without full profiling. If failure: enable CUDA line info and run
the smallest failing request under compute-sanitizer memcheck. Toggle graphs
off as a diagnostic, then MTP off only if needed to localize. Do not run a
Cartesian matrix. Record the first failing operation, range, allocation
owner/generation, expected address and lifetime; add a regression reproducer.
Fix caller indexing, dirty-copy order, upload ownership or consumer retirement
in the responsible owner. No device-wide synchronization after every token as
the final fix. Restore graphs/native MTP and replay the same sequence three
times. A successful MTP-off/graphs-off probe alone does not complete repair.

## F5. Runtime-derived memory ledger

Pointers: `src/llama-kv-pager-config.{h,cpp}`, existing pager plan/geometry and
memory admission code, `llama_context::prepare_kv_attention_graph`,
`llama_kv_attention_scratch_request`, `common/speculative.cpp`,
`src/llama-context.cpp`, `common/arg.cpp` and `tools/server/bench/cache_plan_common.py`.
Locate existing admission functions instead of inventing a parallel estimator.

Use checked 64-bit arithmetic and actual `ggml_row_size`, KV head counts,
attention model-layer list, padding/alignment and backend allocation sizes.
Solve for H **after** accounting for weights, recurrent state, full-L GPU
Turbo4 draft K/V, draft graph/sampler, target graph at B/U, selected A duplicate,
any bounded dequant scratch, summary catalogue, copy ring, pending replacement
owners, and a safety reserve. Reuse upstream budget helpers where applicable;
representation precision and residency remain separate policies.

Catalogue reserve before phase35 exists: for each full-attention layer use
`2 * D_k * n_head_kv * ceil(L/page_tokens) * sizeof(fp16)` plus aligned
validity/generation/position metadata. Sum actual layer geometries. This is
the R1 whole-page min/max layout; no need to load routing implementation
history to calculate the reserve in phase34.

H = floor(remaining target bytes / complete physical page-bundle bytes), after
all fixed charges; a transfer destination must be included within H, not an
unbudgeted extra allocation. Recent/current/in-flight pins must fit useful H
with at least one replaceable destination. Apply per-layer A limits separately.
Do not sum every layer's scratch if the allocator shares it; do not omit
simultaneously live graphs or double-count aliases. Record predicted and actual
peak by category and residual error. Use the allocation plan's real maxima,
then a short workload to validate headroom, not free VRAM at idle alone.

Fixture arithmetic only: this model has 16 attention layers, 4 KV heads,
D256, Turbo4 blocks 66 bytes/128 values; 16,896 bytes/token target payload;
256-token whole-layer bundle 4,325,376 bytes. 128K target host payload 2.0625
GiB, draft payload roughly 132 MiB; 256K doubles both. Derive all of this in
code. Names and numbers are not general architecture constants.

Keep configured **L** unchanged during H/U fitting. Explicit unsafe user sizes
return a clear ledger-based error. Auto mode lowers H first after accounting
for measured scratch; try U128 -> 64 -> 32 if scratch dominates, recomputing H
each time. B >= U and draft B sized for verification, not L. Lower U changes
microbatching/throughput, not the user's maximum submitted prompt length.
Runtime L smaller than the requested test is diagnostic only, never relabeled
success at the original L. Native draft context follows resolved per-sequence
L, not H/A, trained maximum, or a hardcoded 70K/77K.

## F6. Later speed work, after the path works

- Keep table/buffer addresses and tensor shapes stable; content epochs update
  inputs, not graph signatures. Graph keys contain structural capacity/shape,
  not token position or selected page vector. Copy slots and masks through
  upstream owned pinned input uploads; query events without blocking.
- Do not touch unchanged page metadata or scan L on every token. Maintain
  dirty/ready queues and a bounded resident/page-ID lookup.
- Summarize sealed pages once per generation. Router and transfer cadence are
  separate from attention mass telemetry; default mass output off.
- Preserve upstream GDN/RMSNorm fusions. Verify Qwen Turbo4 store predicates;
  do not force unsupported fusions. Compare B/U at identical L/H/A/workload.
- MTP n-max begins at 2. Report proposed/accepted **and committed** tokens,
  hidden-carry continuity and KV rollback. At stable correctness, test n-max
  1 vs 2; retain whichever improves committed decode, not proposal throughput.
  An MTP-off control is diagnostic, not the final profile.
- Try increasing A or recent pin fraction only when MTP acceptance/one-fact
  retrieval benefit outweighs extra work. Defaults for small tuning: B128,
  U64, A2048, recent 512, page 256; comparisons A4096 and U128/256 are bounded
  experiments, not another 48-case validation campaign. Final H is budgeted.
