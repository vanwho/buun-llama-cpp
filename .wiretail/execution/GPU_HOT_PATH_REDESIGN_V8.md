# V8: efficient GPU attention with a host-backed Turbo4 cache

Authority: this design and BENCHMARK_PROTOCOL_V8.md supersede V7 and older
instructions for unfinished tasks from 27-02. Source audit: 2026-09-12,
9173600e2b50fd0257b57186bbd6af3eef75023d. Line numbers are navigation hints;
use the named symbols after preceding tasks change them. State defines order.

## 1. What went wrong, and what is established

The interrupted maximum campaign used L=131072, H=319*256=81664, B=128,
U=64, native MTP with full-L Turbo4 GPU cache. Its cached-prefix accounting
worked: raw-23 processed 1997 NEW tokens after 47073 cached tokens, taking
161.067 seconds (12.399 pp/s). Its 104-token output reported 5.940 tg/s.
At the later audit, all 899 reported prefill routes were selected-direct;
dense and packed were zero. Selected pages=226, below H=319, with zero
faults/evictions. This is a bad GPU execution path BEFORE cold pressure,
not evidence that PCIe offload necessarily costs this much. Raw location and
limitations: evidence/GPU_HOT_PATH_AUDIT_20260912.md.

Confirmed source problems:
- ggml-cuda/fattn.cu:ggml_cuda_fattn_turbo4_paged_query_tile_kernel (about1301)
  visits rows serially within a partition, has two block barriers per KV row,
  eight five-shuffle reductions per query-row instead of accumulating lane
  products before one reduction, and one CTA per query head rather than GQA
  group. Q=1 uses only one warp for most arithmetic. This is not the normal MMA
  prefill engine. Its local arrays and repeated scalar exponentials add cost.
- The same kernel reserves 32 floats for reductions, but its dispatcher
  ggml_cuda_flash_attn_ext_paged_turbo4 (about1884) counts only8.
  The shared allocation is short by24 floats/96 bytes. Fix this actual
  bounds defect before trusting numerical or timing results.
- llama-kv-attention-execution.cpp:planned_route chooses direct for
  non-contiguous Q<=128, even Q=64 interactive prefill. This unmeasured threshold
  sends the chosen small-U profile through the slow kernel indefinitely.
- llama-context.cpp:prepare_kv_attention_graph (2586+) uses every resident page
  when the routed list is absent/stale. llama-kv-cache.cpp:apply policy path
  clears that list and unions layer selections. Thus nominal attention budget A
  can expand to H; the live A rose with occupancy. H is not a compute budget.
- llama-graph.cpp:llm_graph_input_attn_kv::can_reuse compares exact row counts,
  page counts, and telemetry-node presence. These are avoidable shape changes.
  Separately, published frontend graph counts are NOT actual CUDA captures.
- collect_pager_routing_queries copies Q to CPU after execution, and host
  routing scans inventory/summaries. This is a control-path cost to remove,
  not a demonstrated explanation for all the measured slowdown.
- Existing sealed-page/packed caches are useful but their lifetime is tied too
  closely to rebuilt graphs. Do not keep repacking historical bytes after an
  unrelated tail or graph change.

Not established: precise wall-time share of each kernel, true CUDA recapture
cost, why the live contiguous-prefix fast-path was rejected, or a matched
1500 pp/s control at these exact B/U/C settings. 27-02/03 must measure these.
Do NOT repeat the earlier claim that queue/wait counters or graph_replay=0
prove a CPU or CUDA-graph bottleneck. Their intervals overlap and queue time
can include GPU execution.

## 2. Execution architecture — three activities, not a new framework

Keep the existing ggml scheduler, cache, host backing, Turbo4 codecs and MTP.
Do not import a second serving engine, Python runtime, or scheduler.

1. GPU compute: normal weights, GDN, native MTP, and optimized attention over
   a bounded per-attention-layer selected set. Store K/V as Turbo4. Dense
   contiguous views use existing FA; non-contiguous views change the K/V tile
   address calculation, not the attention algorithm.
2. GPU selection: small key summaries, current Q, bounded selected-page IDs.
   Reuse a selection for several decode steps; refresh at a new user/query
   block, page boundary, invalidation, or a measured cadence.
3. Background movement: seal newly committed Turbo4 bytes to canonical RAM;
   prefetch ranked cold layer-pages into free device slots. Publish only after
   copy completion. Evict clean GPU mappings without downloading again.

Diagnostics are consumers of these activities, never prerequisites for them.
Production does not export softmax matrices, full Q tensors, or host copies
of all selected IDs per layer per token. Page-mass diagnostics remain optional
and sparse; no attention kernel is selected merely because it emits a counter.

## 3. Four independent sizes and a memory contract

L = logical capacity; C = occupied committed history; H_l = GPU capacity of
attention layer l; A_l = rows actually attended at layer l. U = microbatch;
B = batch scheduling limit. MTP capacity is always resolved L, Turbo4/GPU.
Never alias any of these quantities or silently shorten history.

For this model, source-derived payload estimates (verify runtime metadata):
16 full-attention layers, 4 KV heads, head width256; Turbo4 block128=66bytes.
Per layer K+V =1056bytes/token, all target layers=16896bytes/token.
A 256-token all-layer bundle=4,325,376bytes; a single layer-page=270,336bytes.
Full128K MTP payload is132MiB before padding/other state. These are payload,
not complete VRAM estimates and not portable production constants.

Charge once per actual owner: weights, GDN, full-L MTP K/V, target H_l,
summary index, graph scratch, selected staging, split states, upload/seal
rings, allocator granularity and runtime headroom. Full H is NOT charged as
selected scratch when only A_l is materialized. Existing materialize-to-F16
FA temporarily costs4096bytes per attended row for K+V here, shared across
sequential target layers per backend owner. A=4096 costs16MiB, not H or L
times16. Include separate native-MTP owner and growth/rounding accurately.

Keep H high if helpful, but require append/promotion workspace and measured
headroom. Do not fill VRAM until 196MiB remains and assume later scratch fits.
Lower U does not limit how much text a user can submit; it chunks that input.
Choose U by measured throughput/headroom, not by preserving U=64 at all costs.

## 4. Restore the fast arithmetic first (27-02/03, 28-03/04)

Immediate recovery: route contiguous selections to existing Turbo4 FA.
Canonicalize physical/native page order when this describes the SAME selected
set, and construct masks from native positions. Never relax the contiguity
check to reinterpret gaps as adjacent native tokens. Record its actual
rejection reason once per shape, not on every token.

For non-contiguous prefill, existing compact Turbo4 packing + mature FA is an
acceptable bridge IF its copy cache persists across graphs and only changes
new/rewritten/replaced rows. It must be bounded by A, measured against the
custom direct kernel at Q=16/64/128, and may not become a new default based on
intuition. The selected F16 materialization scratch is allowed as a temporary
measured bridge. The earlier absolute prohibition prevented reusing efficient
kernels; full-L F16 materialization remains prohibited for selective target KV.

Final kernel direction:
- Decode/small native-MTP verification: reuse fattn-vec.cuh / existing fused
  Turbo4 MMA scheduling and GQA sharing. A page-aware loader resolves physical
  slot, row-in-page, layer/head stride. Process a KV tile cooperatively, reduce
  once per tile, reuse K/V across query heads sharing a KV head.
- Prefill: reuse fattn-mma-turbo.cuh tile pipeline, online softmax and MMA.
  Substitute paged tile loads; unpack Turbo4 in registers/shared memory.
  Reuse the decoded tile across query columns/GQA where the existing algorithm
  permits. No row-by-row scalar attention body in the production prefill path.
- Use fixed, bounded workspace with device counts; choose splits from active
  query/GQA parallelism, SM count and A, not one split per page regardless of
  work. Avoid needless split/merge for short rows; no Hopper-only requirement.
- Keep the repaired old scalar kernel as a small numerical test oracle, not
  an automatic performance fallback. Unsupported optimized shapes refuse
  explicitly or use the verified bounded mature-FA bridge, never CPU attention.
- Respect current Turbo4 K inverse-WHT vs Q-rotation conventions. Inspect
  ggml_cuda_turbo_prefill_attend, fattn-vec.cuh, quantize.cu and
  build_attn_v_unrotate. No double rotation/unrotation; do not assume the
  experimental TURBO_FUSED_PREFILL=1 is faster: upstream comments report losses.

The optimized paged prefill kernel is an implementation task, not “deferred
until profiling”. Keep the bridge only where it actually wins a matched test.

## 5. Query-aware selection without per-token host scoring (28-01)

Reuse routing summary/index structures but add GPU storage and scoring. At
cache write, update per-layer, per-KV-head summary ranges in the SAME coordinate
domain used by the query dot product. Do not run inverse/forward WHT twice.
Capture post-RoPE Q, with Turbo4 channel transform applied consistently.

Estimator for summary block b:
score_b(q) = sum_d max(q_d * kmin_b,d, q_d * kmax_b,d).
Score physical page p as max over its summary blocks and grouped query heads.
This is an approximate ranking heuristic, not a normalized attention mass.
Build summaries from decoded represented K, or conservatively account for
quantization/rounding if calling the result an upper bound. Arbitrary FP16
min/max rounding must not be advertised as a strict mathematical bound.

Start with page256 and summary subblock64; compare subblock16 only if recall
needs it. At128K,16 layers,4heads,D256, two FP16 bounds:
subblock64 ~128MiB index; subblock16 ~512MiB. At256K double these. Charge it!
Do not unconditionally lose half a GB of hot cache for “tiny” metadata.
Coarse page-only bounds and finer summaries are budgeted quality/speed options.

Selection is per full-attention layer, initially shared across its KV-head
groups with max score aggregation; support finer head-group selection in the
descriptor design without requiring it immediately. Do not union all layers'
selected IDs and feed that union to every layer.

A_l is explicitly bounded: mandatory current query block + recent/sink rows +
top-ranked history <= A_l <= H_l. Reserve append/prefetch space separately.
When mandatory rows exceed A, split the query block or reject the configuration;
never drop the write frontier or quietly increase A to H/L.
At missing/stale summary, use deterministic bounded recent/sink + prior VALID
resident selection; stale IDs cannot be attended. Do not fall back to all H.
No cold page can become important through attention-mass feedback alone:
the summary index covers ALL valid host pages, even when not resident.

Current-Q selection for resident pages is a GPU dependency immediately before
attention. No CPU readback is required for that decision. Refresh decode at
an initial experimental cadence8, compare1/8/16 on the SMALL fixture once;
force refresh for a new input block/topic and invalidation. Prefill refresh is
per query block, not every row. Keep current-block causal semantics.

## 6. Stable graph inputs and bounded host work (28-02)

Extend existing input/descriptors, not an independent graph cache:
fixed-capacity per-layer page-index arrays, active counts, tail lengths,
query positions and selection generations in persistent device buffers.
Mutable values must be DEVICE INPUTS actually read at replay, not stale
captured scalar launch arguments or host pointers in op extras.
Storage pointers, workspace offsets, launch shapes stable within a small
query-shape class (decode1, verification smallQ, prefill admitted tile).
Validate generation at publication, keep producer/copy/consumer events and
slot reader lifetimes. Completion gates reuse; graph caching must not pin
every hot page forever. No host cudaMalloc/free, full inventory rehash, row-sized
native-position vector construction, full-table scans or blocking tensor_get
on the ordinary no-change decode boundary.

A page boundary may update counts/IDs without changing graph topology. A model,
representation, tensor stride/capacity or query-shape change may rebuild.
Use ggml-cuda.cu actual_capture_count, actual_update_count,
actual_instantiate_count and actual_launch_count with a short CUDA trace.
GGML_CUDA_GRAPH_DIAGNOSTICS exposes those existing counters in a diagnostic
run. Repeated launches without capture/update establish reuse; there is no
backend field named actual_replay_count. Frontend counts do not certify it.
Do not insist on CUDA graphs where eager is measured faster. Telemetry cadence
must not toggle nodes in the main graph; allocate an optional side diagnostic
outside the production fast graph.

## 7. Layer-granular residency and useful asynchronous promotion (28-05/06)

Existing canonical host page bundles can stay on disk/RAM as they are. Split
GPU mapping identity into (sequence, logical_page, full-attention-layer,
representation/content generation). The payload offsets already distinguish
layer K and V in llama-kv-residency-transfer.cpp. Fetch the requested layer's
K+V (270336bytes for this fixture), not all16 layers (4.125MiB) because one
layer's query liked a page. Geometry remains runtime-derived.

Reuse residency pool/host store/upload ring. Extend transaction records and
layer views; do not instantiate16 complete copies of the whole existing
global pager including duplicating host canonical data. A single host bundle
owner exposes layer slices and per-layer readiness. Track valid lengths and
ownership so graph cpy_k/cpy_v and attention use the same slot mapping.
All16 layer slices eventually form the canonical committed page; no host_valid
whole-page claim before required layer copies complete.

D2H sealing: only newly committed or rewritten Turbo4 ranges, async after the
producer event, coalesced by source layout. Ring memory pinned and bounded.
A full page needs one versioned seal, not whole-page downloads each token.
Mutable tail/rejection writes invalidate only affected versions. An incomplete
seal means “not yet evictable”; do not fake a clean backing copy. Avoid making
CPU checksums/dequantization a production dependency.

H2D promotion: GPU ranking emits a small bounded cold-request list into a
double-buffered mailbox; CPU polls completed mailbox/events at boundaries
without waiting for every token. Copy stream uploads into spare layer slots;
compute continues on current valid selection. Publish on completion; stale
version or rejected speculation cancels publication. Clean eviction drops
mapping without D2H. Host offload is not CPU attention.

Use bytes/time budgets per refresh and hysteresis; refresh frequency and
prefetch count must not grow with C. Start with one cold layer-page per refresh,
two spare layer slots if budget permits; these are test settings, not fixed
production constants. Permit larger coalesced batches only when measured.
A single candidate may be needed by several layers; prioritize by normalized
per-layer score/usefulness, not raw incomparable dot products.

No causality lie: current-layer Q is available only after its projections.
You cannot hide a current-query cache miss behind computation already finished.
The speed mode uses resident approximation now and staged promotion for the
next query/step. For immediate cold recall, allow one explicitly bounded
demand wait or a refresh boundary before the measured decode segment, charge
its latency as cold TTFT, and label the mode. Do not silently wait for every
layer's cold top-k each token. Cross-layer prediction is a later measured
option, not assumed possible from the previous layer in this hybrid model.

## 8. MTP, quality and things not to build now

Native MTP retains every logical row in Turbo4 VRAM. It is not the paged
target and is not admitted with H/A in place of L. Resolve its cache capacity
from the actual context on every launch. Separate target verification from
draft generation in timing; verify phase classification against actual callers,
not just a flag that may never be set. Do not label accepted drafts as proof
that draft attention was paged (it should not be).

Inspect common_speculative_mtp_context_params_apply, native ctx_mtp construction,
draft hidden-state handoff and synchronization in common/speculative.cpp.
Optimize needless host hidden-state round trips only if trace confirms them;
reuse existing device handoff. Do not confuse DFlash's batch policy with MTP.
Compare n-max1/2 only after the small pager path works, at same L/C/H/A/B/U.
Keep MTP enabled in final controls. Avoid target offload flags moving weights
or MTP to CPU. GDN/recurrent states remain exact/on GPU.

Keep finite outputs, correct native masks/RoPE, Turbo4 parity to the SAME
selected set, speculation rollback, event/slot lifetime and real cold use.
Do not add24-case quality campaigns, exact CPU/GPU attention, multi-hour soak,
VBR ladders, new metrics dashboards, or upstream PR work to the critical path.
Sparse prefill can alter old representations even though all produced KV is
preserved. Say so. Optional larger A/early-layer budgets, finer summaries,
immediate-miss waits and periodic dense reference comparisons are explicit
quality tradeoffs, not hidden production work.

## 9. Research basis and limits

[Quest](https://arxiv.org/html/2406.10774v2) motivates current-query min/max
page ranking including history whose importance changes; historical attention
mass alone misses newly relevant pages. Its reported quality/models do not
validate this Qwen/Turbo4 implementation. Our represented-domain summary,
GPU budget and hybrid-layer integration require their own tests.

[LServe](https://arxiv.org/html/2502.14866v1) separates small summary blocks
from larger physical pages and reuses selections across nearby decode steps.
We adopt those two ideas with explicitly priced metadata, not its offline
head classifications, engine or claimed speedups.

[FlashInfer](https://arxiv.org/html/2501.01005v1) describes tiled attention,
GQA-aware work sizing, split-work scheduling and fixed-address graph
workspaces. Its [paged decode API](https://docs.flashinfer.ai/api/attention.html)
also warns that graph constraints can cost efficiency. These guide changes
inside ggml; FlashInfer is not a drop-in Turbo4 dependency.

[InfiniGen](https://www.usenix.org/conference/osdi24/presentation/lee)
illustrates selective prefetch rather than full-cache transfers. Its
cross-layer predictor uses additional model-specific transformations; this
plan does not assume those are already available or copy them blindly.

[NVIDIA CUDA guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
is the authority for events, memory lifetimes and graph execution. Preserve
same-stream producer dependencies and explicit cross-stream fences.

## 10. Execution and completion

27-02 safety/profile harness;27-03 mature FA recovery.
28-01 GPU bounded per-layer selection;28-02 stable graph inputs;
28-03 fast paged decode;28-04 tiled paged prefill;
28-05 layer-granular movement;28-06 asynchronous cold promotion;
28-07 integrate MTP/maintenance/budget and remove obsolete production paths.
29-01 small real-offload speed;29-02 controlled larger-context findings;
29-03 compact summary;30-01 review only that summary.

All implementation tasks require the specified executable small fixture.
A missing auth preflight is not a hardware deferral on this server:
sudo -n true / exact authorized command, not sudo -n -v. Available hardware
must be used. Evidence tasks may honestly finish with negative measurements;
an implementation task must not call its missing required live proof complete.
Record one compact handoff plus raw pointers, not a repeated audit diary.
No task runs the maximum campaign until the production kernel and actual
small cold promotion/use are demonstrated. Rates remain findings, not fixed
3x/5x acceptance thresholds. 256K and YaRN remain future extensions, not an
excuse to extend this repair campaign before128K works.
