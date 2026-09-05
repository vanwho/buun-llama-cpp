# Narrow interfaces and checkpoint contracts

Read only the I-sections listed in your packet. They settle implementation
choices left open by the strategy; they are proposed internal interfaces,
not claims that code is present. Reuse equivalent existing members after
checking the current tree. Don't expand the public API merely to match names.

## I1. Planner -> storage construction (19-01/02)

Use `llama_kv_pager_geometry` and `llama_cache_budget_admission_input` as the
shared output of a pure geometry/reservation pass. Extend geometry with the
model-layer IDs / checked ordinal map. Compute row sizes from model metadata
and GGML type helpers before making context-sized tensors. Return a plan with
logical cells, admitted physical slots, layer offsets, byte charge per slot,
MTP rows/row bytes, scratch categories and safety reserve. Missing mandatory
geometry/reservations is an explicit error, not zero cost.

Pass that plan into cache/storage constructors. Off/observe keep dense backing;
selective/exact use bounded backing but preserve logical host-side cell metadata.
19-01's compile/allocation boundary may use a temporary bounded compatibility
view so existing short generation works. This view must alias the one physical
allocation, not allocate a second dense GPU KV. 19-02 changes write/capture
indices to the final slab layout. This division prevents a dependency cycle
where 19-01 tries to prove all pressured mutations before 19-02 implements them.

OOM fallback is a deterministic monotonic search: hold logical/MTP rows fixed;
reduce target slots by the estimated byte deficit rounded to a page plus the
configured guard; unwind failed allocations; retry. If no legal slots remain,
reduce supported workspace/ubatch constraints and recompute the same plan.
Record tried tuples to prohibit repeating an identical failed allocation.
Reconcile estimates after successful warmup. Never reset device state globally.

## I2. Graph write -> commit -> host capture -> attention read (19-02/03)

`begin_write` returns a ticket that must include or resolve one immutable
logical identity, physical slot, row range and expected generation. Derive
`cpy_k/cpy_v` physical indices from those tickets, not `ubatch.pos` directly.
Host capture reads exactly the same slab offsets after a completion event.

If the existing row-store op cannot express the page-interleaved stride, add
a bounded internal paged row-store variant with explicit slot/row indices,
layer offsets and slab stride. Reuse Turbo4 encoding, don't add a codec.
Add an explicit graph edge from both K/V writes to paged attention if byte-view
aliasing alone isn't sufficient for scheduling. Keep the owner/pool alive until
all graph readers and upload/download events release it.

Host metadata has separate committed-valid and GPU-ready states. Tail flush
records valid row count; padding bytes don't become valid tokens. A successful
CUDA submission does not mean data finished. Host publication and clean-eviction
eligibility wait for completion. After partial graph failure, cancel tickets in
reverse order, preserving older accepted rows. A clean page can be restored
from host after a failed replacement, before returning to the old visible table.

The live transaction adapter implements existing hooks in
`llama_kv_residency_transaction_hooks` and `llama_kv_residency_transfer_transport`.
Bind host_read/write to canonical catalog identity and bounded rings, not a
callback returning success without bytes. `continue_transfer` checks cancellation;
`recheck` checks sequence/session/page/representation generations and table epoch.
Use `llama_kv_residency_ggml_adapter::pool_backend()`/slot_tensor for real copies.
Start with force_synchronous=true; flip only in 20-03 after real movement passes.

## I3. CUDA query/partition layout (19-04/05/09)

Extend raw params and graph op metadata together. Recommended explicit strides
in bytes: q_head/q_query, output_head/output_query, partial_head/partial_query/
partial_partition, and page_mass_head/page_mass_query. Existing page/row/head K/V
strides stay authoritative. Internally index a query as
`q_base + query*q_query_stride + head*q_head_stride`; use its own native position.
Keep batch one initially, GQA grouping from runtime head counts and the existing
supported Turbo4 head geometry. Reject unsupported shapes before launch.

Partition state logically has `[partition][query][head][2 + head_dim_v]` floats
for max, sum and unnormalized weighted V. Storage strides may differ; record
them explicitly. Merge empty partitions by l==0 without evaluating
exp(-infinity - -infinity). Merge nonempty states with max-rescaling; normalize
once after all partitions. V is still in the rotated domain until graph unrotate.

Checkpoint order: shape/stride validator -> one-query numerical regression ->
multiquery causal kernel -> graph interface/reshape -> live MTP -> tiled prefill
-> split-KV optimization. A two-query kernel must not silently store both query
outputs in one head buffer. Add guard/canary output memory tests. Page-mass
reduction applies partition normalization, not a sum of locally normalized mass.

Modify all relevant seams: `ggml/include/ggml.h`, `ggml/src/ggml.c`,
`ggml/src/ggml-cuda/fattn-paged-turbo4.cuh`, `fattn.cu`,
`ggml-cuda.cu` support/dispatch as needed, and `src/llama-graph.cpp` inputs/reshape.
Snapshot a descriptor layout/version and validate host/device agreement. Do not
stash an unowned pointer in fixed-size GGML op_params or overflow that storage.

## I4. Exact runtime: CPU numerical reference is not the CUDA implementation

Owner 19-06. Current `llama_kv_attention_exact_backend::compute_wave` returns
`llama_kv_attention_online_state`, which contains a **CPU std::vector**. Keep
that interface for CPU/fake numerical tests; don't shuttle all per-query/head
partial V through the host to pretend it is the desired GPU executor.

Add a CUDA/backend-owned wave runtime through the existing internal execution
adapter: immutable logical coverage plan; layer ordinal; query tensor/positions;
host catalog identity and upload ring; GPU staging slots/events; device partial
and accumulation tensors; output destination. The graph/execution owner holds
its lifetime until completion. Keep project-specific host objects out of the
generic GGML public interface; pass an existing internal backend handle/adapter
with a checked ownership contract. Document its exact final symbol in the handoff.

For the initial implementation execute a noncaptured attention subgraph/node
between Q/K/V creation and output projection: ensure required KV writes complete,
upload one cold wave, event-wait, partial kernel, device merge, next wave,
normalize, then exactly one V inverse transform and the ordinary downstream
graph. Do not recursively invoke the entire scheduler from a node callback.
If existing op encoding lacks scratch/plan lifetime, extend the internal
execution door rather than repurposing the public page-mass output as storage.

First wire one layer/query and two waves; then multiple heads/queries; then all
model layers; then prefill/native MTP. Every stage compares against the existing
CPU numerical state helper or dense same-prefix driver. Remove the known 501
only when a supported real cold-wave path is dispatched, not unconditionally.
All-hot direct exact can remain a fast specialization. CPU cold attention is
not an additional prerequisite.

## I5. Inventory and query boundary (20-01/02/03/04)

The query/retention controller needs two different inputs: complete **logical
inventory** of committed pages from host catalog plus resident write tail, and
current **physical residency snapshot**. Do not validate cold membership solely
by looking in a GPU-only snapshot. Authenticate host-only records using full
identity and generation; cold physical_slot is UINT32_MAX until upload completes.
Extend routing validation to accept that inventory without weakening identity
checks for resident records. Test cold-only candidate selection explicitly.

Use per-attention-ordinal/KV-group summary sets. Start with existing representative
vectors and correct layer queries; min/max is a later candidate within the same
task only after pipeline proof. Query tensor transform domain is explicit, as
in T1a. Bind summaries to model/codec/position/page content generation, not only
residency epoch: eviction alone does not invalidate immutable host-page content.

Per-layer fresh path ordering is Q projection/position transform -> GPU score
node -> small candidate IDs/scores readback at a deliberate graph split ->
transaction -> completed slot descriptors -> attention. An eval callback is a
correctness prototype and must chain existing callbacks; its synchronization
cost is recorded. 20-04 implements the steady-hot fast path without leaving
this callback installed unconditionally. Table inputs are per layer/group;
changing one layer's table must not change descriptors already consumed by
another in-flight layer. Use graph fences/double-buffered metadata as needed.

Cross-layer transfer groups may be shared while selection masks differ. Preserve
write pins and budget the largest simultaneous live layer selection plus staging.
If a union doesn't fit, reuse slots only after the prior layer's completion or
use a recorded capacity-relative selection/fallback; don't overallocate.
Prediction is explicit prior-query evidence, never current-query proof. First
answer after focus change tests the fresh path, not merely next-token promotion.

## I6. Test drivers, mode switching and sequential contexts (18-03/05, 21)

18-03 adds explicit case/repetition selection and resume behavior to the
existing Python runners, with help/tests before use. Store candidate mode and
observed execution mode separately. An arbitrary `--mode` label in the quality
client cannot stand in for selecting all pages in engine execution.

18-05's optional model driver supplies a forced page selection/test route through
internal APIs and reports actual dispatched route. Tests needing all-selected
must select the full valid history, including the tail, per layer. A forced
direct test cannot be satisfied by a dense bypass. Do not add a public server
flag merely to run the private driver if an internal seam suffices.

The driver runs one context at a time to avoid two simultaneous target/MTP
allocations causing spurious OOM. Serialize small captured reference outputs
to the test result root, destroy/drain context, create the comparison context
at identical parameters and teacher-force the exact token IDs. Use actual model
geometry, with targeted tensor captures bounded by selected positions/layers.
This harness is reusable by 18-06, 19-04/06/09 and 20-04; don't rebuild a separate
one-off server/client loop for each parity task.

## I7. Task boundaries that prevent circular prerequisites

18-05 owns production selected-reference parity and direct operator/domain/
indexing correctness for the currently supported single-query shape. It does
not own 19-01/02's full bounded-storage conversion or 19-04's multiquery direct
kernel. If direct graph reads a separate stale slab, first fix a valid alias
where the layouts permit it. Otherwise its diagnostic driver may populate a
bounded slab explicitly from captured live Turbo4 bytes, then compare the
direct operator against the same captured Q/K/V reference. This is a controlled
operator proof on real target tensors, not production write-path proof. Record
`input_source=captured_live_kv` versus `production_slab` and the unresolved
write binding; don't introduce an always-on copying workaround as a fast path.

18-05 cannot finish with failed/missing reference parity or an untested direct
operator. Its receipt may explicitly reserve production direct write binding
for 19-02. 19-02 must then pass the same driver with `production_slab` and no
diagnostic preload, including a slot mutation; its own acceptance cannot use
the earlier controlled copy. Production end-to-end direct verification is
19-04; early multiquery parity uses the repaired selected-reference path.

18-06 tests MTP state/rollback with that repaired reference path and validates
GPU draft allocation independently. The fully functioning ordinary CPU-Turbo4-KV
baseline is 20-06, not a hidden prerequisite on CPU attention implementation
here. The bounded host/transfer replacement in 19-02/03 repeats speculative
commit invariants; 19-07 proves pressure plus rollback/restart on the final
storage. Earlier all-resident tests cannot stand in for that later pressure.

19-03 wires deterministic selection and real movement. 20-01 wires actual
query-driven selection. Neither a manually chosen page nor previous-query
prediction counts as fresh current-query cold recall. Each receipt names which
boundary it actually proved, so a later task cannot mistake a component proof
for the final capability.
