# V9 cold-page selection and asynchronous publication

Authoritative for phase 35. Use the existing `llama_kv_pager`, backend residency
transactions, and two-slot candidate mailbox. This is **not** the old scalar
direct-attention router with extra counters attached.

## R1. GPU all-history catalogue, not just resident summaries

Implement a context-owned catalogue per full-attention model layer/sequence:
`min[D, kv_heads, logical_pages]`, `max[...]` in FP16, and compact validity,
logical position and generation metadata. Entries cover every committed,
sealed logical K page, including cold RAM pages. Initially one min/max pair
per whole page: finer subblocks are an optional later quality experiment.

Build from actual Turbo4 K coefficients **after the same representation
transform used by the mature FA dot product**, before inverse FWHT. Reuse the
existing Turbo4 codebook/dequant helpers; do not summarize GGUF model weights
or mix original-space Q with rotated-space K. q_i >= 0 selects max_i, else
min_i; sum(q_i * bound_i) is the page score upper-bound heuristic. It is not
actual attention probability or a guarantee of good recall. Reduce over
query heads belonging to each KV group, then max over groups for a layer.
Page ID is the deterministic tie-breaker.

Hook existing `llama_kv_pager::seal_ready_pages` dirty/committed queues, plus
the backend implementation responsible for host-ready copies. On seal, enqueue
one CUDA summary reduction from the still-valid device K page, then publish
its catalogue entry only when the producing event completes. A page cannot
lose its only summary source before publication. Canonical host KV and summary
readiness are separate event states; both are required before clean eviction.
The current partial page is pinned and included in recent attention; no full
page-summary recomputation for every new row. Clear/reuse/rollback invalidates
only affected generations. Reconstructed restored pages may summarize from
host Turbo4 once during explicit restore, not per-step CPU attention.

Catalogue allocation is O(L/page_size), not O(L*heads*queries). Fixture size
at 128K is ~32 MiB for this model (16*4*256*512*2 bounds*2 bytes), plus metadata.
Charge all layers and alignment in the ledger. No full F16 K history copy.

Tests compare GPU summaries with CPU decoding of the same Turbo4 bytes, then
page scores with a CPU reference for mixed-sign Q, nonzero KV groups, invalid
generation and tails. FP16 rounding can shrink a bound; round min outward
down and max outward up, or document/use a conservative representable margin.
Skip invalid/unsealed/future entries. No NaN-to-topK promotion.

## R2. A real current-Q producer, independent of FA

Current defects to replace:

- `src/llama-kv-cache.cpp::capture_kv_routing_query` (~2248) discards Q/layer.
- `apply_pager_live_policy` (~2298) polls a mailbox with no production producer.
- `ggml/src/ggml-cuda/fattn.cu::ggml_cuda_fattn_turbo4_route_kernel` (~2024)
  uses a one-thread scan and only the supplied attention page domain.
- The paged FA operator's production setup does not bind the routing buffers
  which fixtures populate. Do not fix that by forcing every FA through it.

Add **one internal GGML graph op** `GGML_OP_KV_PAGE_SELECT` (follow existing
op registration patterns in `ggml/include/ggml.h`, `ggml/src/ggml.c`, CPU
forward dispatch, `ggml/src/ggml-cuda/ggml-cuda.cu`, op names/count and backend
support tables). Inputs: live target Q, catalogue min/max, page validity/
native-position metadata, resident bitmap. Output: I32 logical IDs of bounded
ranked resident and cold candidates in fixed regions; `-1` padding. Return
IDs sorted by score, so CPU can use rank without transferring float scores.
Internal params contain capacities, head geometry, query row selection and
fixed layout only; validate params and overflow. Do not use float-encoded
page IDs or introduce a public stable API/second pager controller.

Concrete initial op layout (one sequence slice per node):

| Input | GGML type / dimensions | Contract |
| --- | --- | --- |
| src0 | F32 `[D, n_head_q, n_queries, 1]` | Target q_cur after RoPE and any existing self_k_rot, before build_attn_mha's head/query permutation. Honor nb strides. |
| src1 | F16 `[D, 2, n_head_kv, logical_pages]` | Stored-space min/max catalogue from R1; bound index0=min,1=max. |
| src2 | I64 `[4, logical_pages]` | Native position_begin, valid_length (0 means unavailable), sequence_generation, page_generation. |
| src3 | I32 `[logical_pages]` | Ready resident membership for this sequence; not inflight membership. |
| src4 | I64 `[4]` | Mutable query_position, sequence_generation, snapshot_generation, refresh_enabled. |
| output | I32 `[K_resident + K_cold]` | First region resident ranked IDs; second region cold ranked IDs; -1 padding. |

Use shape-derived head geometry, params containing only fixed capacities,
page size and query_row=0. **Do not store changing token positions/epochs in
graph structural keys or constant op_params.** Upload src4 through owned graph
input storage; it permits a skipped refresh without capture/reallocation.
Keep the host snapshot descriptor paired with the submitted src4 generation.
For multi-sequence execution slice the real per-sequence Q/metadata correctly
or refuse selective support clearly; never silently execute this op on CPU
with a full Q readback. CPU implementation is the numerical test reference.

Provide a simple CPU reference implementation and cases in
`tests/test-backend-ops.cpp`; run CPU and CUDA implementations against the
same reference, as CONTRIBUTING requires for added GGML ops. CUDA path:
one cooperative CTA per page (128/256 threads, warp reductions over D and
head groups), then bounded deterministic top-K reduction over page scores.
Use existing backend top-K primitives if they fit fixed workspaces; otherwise
a two-stage tile top-K and small candidate merge, not a single thread scanning
all pages. Resident top-K size is A/page minus pinned slots, cold K initially
2 per layer. Scratch is O(logical pages + tile candidates), preallocated.
Keep valid past cold pages in the domain regardless of physical residency.

Insert producer after **target** post-RoPE Q exists. Transform Q with the same
forward FWHT helper if the chosen tensor is not yet in Turbo4 dot-product
space. Do not rotate twice. Explicitly expand the routing output into the
executed graph; an unused node or a capture callback with a null tensor is not
production integration. It must run on the owning CUDA backend with Q's real
dependencies, no synchronous host Q retrieval. Draft Q is not a substitute.

Selection is for a **subsequent** graph boundary. For prefill use only the
first query row of the microbatch and summaries valid before that row: using
the last future query to select history for earlier queries can leak future
information even if the attention mask is causal. For multi-token native-MTP
verification use its first already-authorized query boundary. Discard the
snapshot if the associated sequence/position is rolled back; never publish
rejected speculative history. Decode uses the current single target query.

Run selection on the first request boundary, committed page crossings and
every 8 accepted target tokens initially. Skip intermediate steps using
bounded graph input params/early-return, not graph allocation or mass output.
If a request starts with no usable snapshot, retain recent/pinned selection
while the first real query produces one; do not synchronously stall attention
waiting for historical top-K.

## R3. Two-slot mailbox and per-layer selection

Extend existing mailbox snapshots, not another queue. Payload carries sequence
generation, query position, table/catalogue epoch, actual model-layer IDs,
bounded resident IDs and ranked cold IDs. Own fixed pinned buffers until
completion. After backend graph submission, copy compact output IDs on an
ordered stream, record an event, then `publish_pending`; polling publishes
ready only after completion. Never mark ready immediately after enqueue.
If both slots are in flight, coalesce/drop the older requested refresh and
keep serving the last valid selection. No `cudaDeviceSynchronize` or blocking
wait just to consume telemetry. Reuse upstream owned upload ring for input
metadata; a CPU vector reused on the next graph must not be the async source.

Consumer `capture_kv_routing_query` / `apply_pager_live_policy` must distinguish
produced, pending, consumed and stale snapshots. Wire these from the actual
context/backend graph completion boundary. Move CPU policy application to
completed-snapshot/boundary handling, not a scan every token. Remove the
oldest-cold-once fallback from production; retain only an explicit diagnostic
forcing hook in tests, labeled `forced`, never a natural-proof receipt.

Maintain a bounded selected set for each **model layer ID** in context/attention
metadata; replace the global `selected_attention_pages` union as production A.
Initial physical residency remains whole-layer bundles for simplicity. Merge
cold suggestions by best **rank** across layers and deterministic page ID,
not by comparing unnormalized layer logits. At most two bundle promotions per
refresh globally. Layer selections use only resident ready pages plus mandatory
recent/current pins, never widen every layer to the union or H. After a
promotion, only layers ranking that page need update their packed slots.

This intentionally allows lag and approximates relevance. A small exploration
slot can be a later quality option; it cannot become a repeated forced page
copy used to manufacture attention-directed counters.

## R4. Transfers without token-path stalls

Reuse `llama_kv_residency_pool` and `llama_kv_residency_pool_backend`,
`src/llama-kv-residency-transaction.cpp::llama_kv_residency_execute_transaction`,
`src/llama-kv-residency-transfer.cpp::llama_kv_residency_execute_transfer`,
page ownership, event fences and `seal_ready_pages`. The mailbox lives in
`src/llama-kv-prefetch.{h,cpp}`. Extend the summary layout/helpers in
`src/llama-kv-routing-summary.{h,cpp}` rather than creating another catalogue
API. Resolve renamed symbols after merge with targeted `rg`.

Promotion: reserve a destination from **budgeted** H slots -> retain canonical
host generation -> enqueue compressed H2D -> event complete -> atomically
publish layer mappings/epoch -> allow next attention selection. Cancellation
before publication invalidates its generation; drain/recycle event/buffer
only after completion. No consumer accesses an in-flight destination.

Victim: not recent/current/pinned, not held by an executing attention graph,
not awaiting host canonical commit or summary completion. Retire mapping only
after last use event. A clean host-ready page requires no eviction D2H.
Use bounded pinned staging rings instead of pinning the entire RAM history
by default. K and V remain opaque Turbo4 bytes in RAM/transfers/VRAM.

Host coherence is committed-token based. Seal immutable full pages once; a
mutable tail may use dirty interval copies. Rejection/rollback removes dirty
speculative intervals and invalidates stale snapshots. Full-L native-MTP
allocation is never a victim, and its recurrent/hidden carry is not target KV.

Small proof must retain a compact sampled timeline: target query boundary,
catalogue generation, selected formerly cold page, copy event/generation,
published slot, the actual layer attention descriptor/packed slot consuming
it, and successful committed generation. Sample these records only in the
proof/profile mode. An answer alone can come from recurrent state or MTP;
an H2D byte counter alone can come from forced fallback. Neither suffices.

## R5. Bounds and failure decisions

Missing/late snapshot: keep recent and last valid selected pages; schedule
another refresh. OOM: report ledger and reduce auto H/U, not L or draft
residency. Stale map: discard safely before launch. No usable cold candidate:
inspect catalogue validity, Q transform and top-K using the deterministic
injected-Q fixture, then one natural recall. Do not run long contexts hoping
the producer will spontaneously become connected. If natural recall selects
poorly despite correct wiring, try the prescribed rank/recency/A tuning in
36-02/03; report approximation limits, not a fake transport failure.
