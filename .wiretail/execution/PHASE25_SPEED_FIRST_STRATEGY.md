# Speed-first implementation amendment — phases 25–27

> Historical phase25 reference. For unfinished tasks26-01 onward, use
> PHASE26_INTERACTIVE_KV_STRATEGY.md and BENCHMARK_PROTOCOL_V7.md. They replace
> this document's256K-first/six-coordinate campaign and ambiguous B notation.
> Current campaign maximum is131072; primary speed fixture8192/4096.


Revision: 20 26-09-11. This is the execution authority for tasks 25-02 onward.
Read the current packet, its cluster, and BENCHMARK_PROTOCOL_V6.md. Do not
load the previous 24 phases, rerun their gates, or resume the old 25-02 campaign.
Historical scope is preserved in archive/phase25-before-speed-first-20260911/.
WORK_STATE.json controls order. All implementation/measurement tasks are Luna
High; the independent reviewer 27-01 is Sol High. Wiretail defaults are unchanged.

## 1. Deliverable and deliberate scope reduction

Build the fastest measured experimental Qwen3.8-27B UD-IQ4_XS configuration
with target AND native-MTP K/V Turbo4, target canonical CPU-RAM backing,
GPU-resident attention-selected target pages, and full resolved-context MTP
K/V on GPU. Demonstrate near-full occupation of a 262144-token context and
generation, not merely allocation. Both prefill and decoding matter.

Weights, 48 recurrent/Gated-DeltaNet layers' working state, the 16 full-attention
layers' compute, and the native MTP layer stay on GPU on this target if admitted.
Verify geometry from the GGUF/runtime; internal QWEN35 names are not evidence
the available Qwen3.8 model is unsupported. Do not switch models or weight
quantization to obtain a favorable result.

“Low attention” means low-importance historical pages at an attention layer,
not low-numbered network layers. Moving transformer weights or attention
compute to CPU defeats this speed goal. CPU holds compressed historical KV;
GPU computes attention over the selected resident subset.

Broad quality matrices, exact CPU/GPU hybrid attention, quality tuning,
checkpoint/restart soak, multi-user serving, YaRN, VBR precision changes and
upstream PR preparation leave the critical path. Keep them as explicit later
options, not todo gates blocking this prototype. Retain minimal numerical,
causal, byte-identity, placement and memory-lifetime checks: a fast wrong-page
kernel is not an optimization. Review answers as diagnostics, not 24/48-case
acceptance. Do not claim production quality or upstream readiness.

## 2. What the evidence actually establishes

Source inspection was at e154151d62e583563343a15128392b69030dcf48 plus the
existing uncommitted 25-02 implementation. Preserve that work; inspect its
diff before modifying the same functions. No new throughput benchmark was run
during this plan revision.

The old PHASE25_FULL256K receipt proves allocation at 262144 with GPU Turbo4
MTP and a derived 251-page target budget, but only partial prompt population.
Its final receipt uses an 8192-row snapshot; the separate v6 raw progress
series reached 11264. These are different observations, not a completed
262K benchmark. Do not combine their counters into a fictional single run.

Raw source:
`/srv/ai/paged-kv/results/25-02-full256k-20260906T161500Z/telemetry-progress-v6.jsonl`.
It advances 2048 -> 4096 rows in 30 seconds (68.3 rows/s), then 10240 ->
11264 in about 120 seconds (8.5 rows/s). These are sampled population slopes,
not validated decode or isolated GPU-attention timings. The historical
roughly 11 tok/s claim is not a fundamental PCIe/CPU-KV throughput limit.
The user's approximately 1400 tok/s dense prefill is a target comparison to
remeasure on matched occupancy, ubatch, MTP and source, not an accepted ratio.

Source-confirmed bottlenecks, with remaining measurement uncertainty:

| Mechanism | Concrete owner | Consequence / planned repair |
| --- | --- | --- |
| Immutable pages reread on every seal | src/llama-kv-pager.cpp, seal_ready_pages | A host-clean page still rebuilds summaries whenever a provider exists. Use content-versioned dirty queues. |
| Many tiny synchronous D2H reads | src/llama-kv-cache.cpp, pager_routing_summary_build | Four representative tensor_get calls per layer/page; 16 tables, head zero. Reuse canonical bytes initially, then batch GPU summary work. |
| Repeated whole-store copies/sorts | src/llama-kv-routing-summary.cpp, update_page/reconcile; pager seal loop | Copies current store, then result=*this, inventory snapshots and sorts inside page/layer loop. Update a bounded delta once per batch. |
| Kernel tile caps the whole model | llama-context.cpp, prefill_ubatch_size; llama-kv-attention-execution.cpp, prefill_chunk_size | The 64-query primitive forces tiny matmul/GDN graphs even when -ub requests hundreds. Decouple CTA tile from model batch. |
| Correctness-first serial attention | ggml-cuda/fattn.cu, paged_query_tile_kernel | Row-wise traversal, warp reductions and barriers; 64-query shared workspace even for tiny queries. Compare mature Turbo FA reuse and tiled Tensor-Core paged path. |
| Unnecessary host/controller work | synchronize, prepare_kv_attention_graph, policy/telemetry publication | Frequent fences, per-row descriptors, allocations and CPU scoring can dominate. Profile and batch/sparsify maintenance. |
| Potential native-MTP handoff cost | common/speculative.cpp, draft-MTP process | Target embeddings copied through host and replayed into draft batches. Measure; preserve pending-row semantics while moving contiguous handoff to device. |

For 44 immutable pages, one full head-zero summary sweep can issue
44 * 16 * 4 = 2816 small tensor_get calls, before host publication itself.
Repeated sweep/store work can grow superlinearly even below hot capacity.
This is a concrete bad algorithm; the percentage of elapsed time it consumes
still requires stage timings in 25-02.

The graph_capture_count/replay counters in llama-kv-attention-execution
describe backend-neutral graph admission/rebuild decisions. They are NOT
counts of cudaStreamBeginCapture or cudaGraphLaunch. kernel_us=0 means
uninstrumented/unavailable, not free CUDA kernels; total_token_us is not an
attention-kernel duration. Do not repeat the earlier “5.4s kernel” inference.

Zero promotion/eviction transfer counters do not imply zero PCIe work:
summary tensor_get and canonical-host sealing have separate/unreported copy
paths. Host valid rows prove some host publication, not production H2D recall.
A short fully resident speed result is not a hot-cache + cold-RAM speed result.

## 3. Architecture: four budgets and two distinct paths

Keep KV representation (Turbo4), logical history, residency, and attention
selection separate. Track:

- L: logical capacity and occupied history; final capacity 262144.
- H: physically allocated target hot-page capacity, admitted after weights,
  full-context MTP, GDN state, scratch, rings and safety margin.
- A[layer, phase]: selected attention rows, at most the resident rows; need
  not equal H. Keeping more pages cached need not mean scanning all of them.
- B: model prefill microbatch; CUDA query tile Q is an independent subdivision.

The resident set includes selected old pages, recent rows, append workspace
and transfer slack. The next write wave's pages must be reserved before graph
submission. Do not give the selector all H slots and then discover there is
nowhere to write or promote.

Use runtime ggml_row_size/strides. On the inspected geometry, a 128-value
Turbo4 block is 66 bytes; 16 layers * K,V * 4 KV heads * 256 dimensions yields
16896 bytes/token (16.5 KiB), or 4.125 MiB per 256-token all-layer page and
4.125 GiB at 262144 tokens before alignment/metadata. This is illustrative,
not an allocation constant. MTP bytes must come from its own tensors.

Prefill:

1. Embed/project/GDN on useful batched GPU graphs; process all input tokens.
2. Attend on GPU over current causal query block plus bounded resident
   history. Dense optimized Turbo FA for an eligible contiguous prefix;
   compressed packing or native paged tile loads for selected history.
3. Quantize/write new Turbo4 K/V once. Commit each changed range to host once
   asynchronously. Build compact routing summaries from those writes.
4. Refresh sparse selection on block/page-wave boundaries, not a CPU scan and
   synchronous readback after every token or 64-query tile.
5. Native MTP consumes every committed target hidden-row batch once, with its
   original full-context GPU attention/cache contract.

Decode:

1. Reuse a stable resident/selected set across multiple decode/verify steps.
2. Run the winning grouped-query/split-KV or packed dense Turbo kernel.
3. On a sampled cadence, update attention mass and query-to-cold-summary
   relevance, promote only a bounded number of useful pages, and publish
   completed mappings. Reuse hot pages without copying them again.
4. Draft and verify on GPU. Account for accepted tokens, not proposal count.

A truncated/sparse prefill changes downstream KV and GDN states. Keeping all
resulting KV in RAM does NOT reconstruct the counterfactual dense-prefill
states later. This approximation is allowed for this speed-first prototype
and must be labeled. Host recall recovers stored pages, not exact dense
semantics. Optional dense/chunk-exact prefill can be assessed later; it must
not covertly become the default CPU-attention path.

## 4. Reuse fast kernels before building more custom machinery

Compare two real GPU implementations in 25-04/25-05:

- Existing fast Turbo4 FA with no pack for eligible contiguous all-fit data;
  for selected history, a bounded compressed-Turbo4 GPU pack assembled once
  per selection change, not each decode. This is NOT F16/F32 KV expansion.
  Track duplication against H; prefer owning compact slots/in-place safe
  relocation where practical. Pack only A, not every host or resident page.
- Native paged Turbo4 loads inside tiled Flash Attention. Retain per-page
  logical position and per-head strides, dequantize a tile in registers/shared
  memory, use the existing mature MMA/softmax machinery where possible.
  Cover decode and multi-query MTP separately from large-query prefill.

For sorted historical pages entirely before a contiguous query block, compact
row order plus the appended query block can use a correctly aligned causal
mask: missing old positions remain already in the past. Otherwise pass native
positions/mask to tile loads. RoPE coordinates must NEVER be renumbered to
compact indices. Test this eligibility rather than assuming it.

Do not merely increase MAX_QUERY_TOKENS from 64 to 1024: that inflates per-CTA
shared memory and does not recover Tensor-Core utilization. Use grid query
tiles and shape-sized scratch. The existing serial kernel stays as a tiny
numerical oracle/fallback for supported shapes, not a required large-run route.
Never create a full-history F16/F32 KV gather or Q-by-L score matrix.

Avoid importing FlashInfer as a runtime dependency or its unsupported Turbo4
codec. Use its documented prefill/decode separation and GQA scheduling as
design references; implement against Buun's existing backend and license
rules. Measure whole-model benefit, not just a kernel benchmark.

## 5. Canonical host memory without serialization

GPU-generated KV must cross D2H at least once to become canonical host data.
Make this cheap by coalescing changed page ranges into a bounded pinned
staging ring and copying into pageable canonical slabs, or measured pinned
backing if capacity/OS cost permits. Do not pin all free system RAM.

Separate committed_content_version, host_committed_version, inflight_version,
summary_version and slot_generation. Stable old content is never invalidated
solely by a mapping/table epoch. New tail rows and speculative overwrite do
invalidate the relevant content. Publish host-clean only after its copy fence;
eviction of host-clean pages drops a GPU mapping; no additional eviction D2H.
Dirty/inflight pages cannot be reused until the relevant event completes.

Start with one compute stream and a bounded transfer stream; measure whether
separate upload/download streams help. Wait only on dependencies, not the
entire device. Retain source page pins until read completion, destination pins
until promotion completion, and host/staging lifetime until reuse is safe.
Generation/fence correctness stays mandatory despite reduced resilience scope.
Normal append should not hash/dequantize/archive every historical page.
Keep sampled checksums for the pressure fixture; metadata is not the payload.

GPU summary/index update should scale with changed/new pages. Cold lookup can
scan compact O(number of logical pages) summaries on GPU at a sampled cadence,
but must not reread full KV or clone all CPU stores each step. Use query/K
vectors in the same RoPE and Turbo transform domain. Attention EMA from only
resident pages cannot recall a cold page by itself; retain cold summaries and
a small bounded exploration allowance.

Initial residency may group the 16 layers to use the existing ownership path.
Selection and attention statistics must be layer-specific where available.
Do not claim head-zero routing is full-head attention scoring. First optimize
per-layer masks/lists within shared residency; implement layer/page-specific
promotion only if measured transferred-unused-layer bytes make grouping a
material bottleneck. The API must distinguish layer and logical-page identity
so that this does not require a future rewrite.

## 6. Material speed optimization coverage and owners

“Fastest” is an empirical choice on this hardware, not a promise that every
possible optimization is known. No category below may disappear into a generic
future-work sentence. Its owner must measure applicability, implement a
positive candidate or record why it is not material. Negative experiments keep
raw timings; do not ship slower code just to check a box.

| Category / concrete change | Owner | Deciding measurement |
| --- | --- | --- |
| Stage/CPU/CUDA timing, actual graph events, cheap persistent driver | 25-02 | matched 2K/4K/8K prefill slopes, 128-token decode; no phantom timings |
| Dirty queues, summary from existing host bytes, incremental index | 25-03 | reads/bytes/allocations per new page independent of old-page count |
| Dense Turbo FA eligibility and amortized compressed packing | 25-04 | kernel + pack + total request time, packed bytes per selection epoch |
| Tensor-Core paged prefill, query tiles, GQA tile reuse, tail masks | 25-05 | useful B=256/512, attention milliseconds, no Q*L scratch |
| Macro-batch append admission, fewer graph waves/fences, GDN batching | 25-06 | requested/effective B, matmul/GDN share and wall pp/s |
| Coalesced async D2H, host slabs/ring, no repeated tail/full-page copies | 25-07 | new-KV bytes, D2H calls, overlap, copy wait, CPU allocation cost |
| Real RAM->VRAM recall under deliberately small H | 25-08 | host-only page checksum/slot/event -> GPU consumption; pp/tg under pressure |
| Decode specialized query count, split-KV occupancy, GQA reuse | 25-09 | q=1/2/3/5 kernels, shared memory/occupancy, accepted tok/s |
| Stable graph descriptors, shape buckets, actual CUDA graph reuse | 25-10 | real captures/updates/launches, CPU launch gaps, descriptor-copy bytes |
| GPU cold routing, sparse per-layer A, telemetry sampling, prefetch/churn | 25-11 | selector/attention/transfer time vs A, useful bytes, held-hot reuse |
| Device target-hidden->MTP handoff, batched draft catch-up, GPU sampling | 25-12 | target/draft/pending-copy/verify costs and committed tok/s |
| Weight/GDN kernels, WHT/fusions, CPU thread overhead, hardware transport | 25-13 | whole-model attribution, matched dense baseline and top remaining costs |
| Budget/A/H/B joint tuning, transient/scratch reserve, reusable allocation | 25-14 | measured Pareto frontier for pp/tg, safe 262144 MTP allocation |
| Whole-pipeline integration, limited engineering ablations | 25-15 | one all-fit and one real-pressure result, live export for final campaign |
| Long-run prompt fitting/progress without compulsory timeouts/matrices | 26-01 | near-full actual occupancy and continuation with real movement |
| Ordinary CPU-KV + GPU MTP and safe GPU controls | 26-02 | matched occupancy/prompt/native MTP, confidence/censoring |
| Original questions at six final context points, once initially | 26-03 | cold-prefill curve and accepted-decode curve, not pass/fail speed gates |
| Compact evidence and independent performance review | 26-04 / 27-01 | traceable findings, no historical gate audit |

25-13 specifically checks: unintended CPU tensor placement/fallback,
actual -b/-ub and n_threads_batch, MMQ/cuBLAS/quantized matmul selection and
Tensor-Core utilization, GDN/fused-op support, redundant Turbo WHT/dequant/
mean-restoration kernels, full-vocabulary logits for unrequested rows,
GPU synchronization for sampling, hot-loop allocations/copies/logging,
checkpoint/cache-idle overhead (disable in the performance profile), PCIe
link/NUMA locality/pinned bandwidth, GPU clocks/thermal throttling, other GPU
processes, and CPU profiling overhead. Use existing portable switches first.
Server affinity and service/power configuration stay in /srv/ai; no overclock
or unrelated service shutdown. Memory bandwidth can be the irreducible dense
27B decode floor; reduced target attention cannot remove reading its weights.

VMM is optional implementation machinery, not the pager design. Current
stable slabs plus slot indirection may already suffice. Consider VMM only
when measured allocation/address instability or compressed-pack duplication
justifies it; account for mapping granularity and synchronization costs.
Do not revive VBR precision degradation: Turbo4 remains fixed.

## 7. Iteration, findings and termination

Use BENCHMARK_PROTOCOL_V6.md. Start with one short pair and one source
hypothesis; profile a few waves, not the full 256K prompt. No 48-case matrix,
mandatory multi-hour soak, 10 warm repeats, or full curve during development.
A paused HTTP client is not a durable model checkpoint. Raw progress is
durable evidence; resumable compute requires verified target + GDN + MTP +
pending-activation state or a still-owned live request.

Implementation tasks complete when the feature's stated minimal live behavior
and focused regression work. Performance experiments may choose an existing
faster route after rejecting an alternative. “Not implemented” for the only
paging path is not a completed implementation. If a new concrete defect is
outside a packet, insert a bounded repair before its dependent, update state
and cluster references, and retain the reproducer. Never repeat an unchanged
full-context campaign or rename the same exhausted hypothesis.

Before 26-01, the integration receipt must demonstrate nonzero real movement
and useful prefill/decode results. If not, repair the minimized pressure path;
do not try to make a 262K campaign pass by more waiting. Numerical speed
ratios are findings, not arbitrary acceptance thresholds. A measured slow
result directs source work; it is neither zero throughput nor a quota failure.

The 27-01 reviewer considers only the new summary and selected raw records.
It may add a new, evidence-driven remediation phase and identical subsequent
summary-review chain if a material goal remains. It may not resurrect exact,
quality/soak or upstream gates removed by this user-directed scope revision.
YaRN and production hardening remain unscheduled options after the base result.

## 8. References and verified local entrypoints

- [FlashInfer attention design](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html):
  separate paged prefill and decode kernels, GQA reuse and Tensor-Core scheduling.
- [Paged prefill API](https://docs.flashinfer.ai/api/attention.html):
  query indptr and KV page metadata are distinct; not a reason to restrict
  the full model's query batch to one CUDA tile.
- [CUDA graph updates](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html):
  dynamic data/eligible parameter updates need not imply graph reinstantiation.
  These are design references, not measured speedups for this card/codec.

Verified source owners above are search anchors, not fragile line-number-only
instructions. Existing test executables include test-kv-pager,
test-kv-attention-execution, test-kv-attention-telemetry,
test-cuda-fattn-paged-turbo4, test-server-prompt-cache. Extend existing tests.
The baseline build tree is build-cuda; nsys and ncu are installed. Profiling
permission/support must be checked, not assumed. Do not require a full CUDA
template rebuild for a small host edit; wait for active incremental compilation
instead of restarting at an arbitrary 240-second limit.

Existing tools: tools/server/bench/run-pager-profile-benchmark.py,
run-final-curve.py, prompt_sizing.py, pager_benchmark_contract.py and their
unit tests; /srv/ai/benchmarks/run-profile-benchmark.sh owns site profiles.
Do not confuse run-final-curve.py with a nonexistent run-context-curve.py.
New flags described by V6 are proposed task-25-02 work until --help/tests
confirm them. Preserve a working invocation in every handoff.
