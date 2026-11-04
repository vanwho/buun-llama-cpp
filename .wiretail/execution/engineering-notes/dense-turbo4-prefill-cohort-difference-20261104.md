# Dense Turbo4 prefill: upstream and fork cohort difference

## Matched fresh-content observation

R6 completed the upstream and fork GPU/MTP-on ingest-only series at L=65536,
actual C=62465 (requested 62464), with five mapped repository appends each.
B=1024/U=256, t=16/tb=16, GPU Turbo4 target/draft, native MTP n-max=2, model,
source map and template are pinned in `benchmarks/COMPARISON_COMPLETION_20261104.json`.
The fork has pager off. Neither series ran canonical decode questions.

| Build | Backend/net-new tok/s | HTTP-wall net-new tok/s | HTTP-wall seconds |
|---|---:|---:|---:|
| pristine master d2407c33f959 | 1346.87 | 1341.45 | 46.57 |
| fork 8a79008b5473 | 910.57 | 908.05 | 68.79 |

These are weighted aggregate fill rates, not the final interval rate. Both
processed 62465 backend and net-new tokens; repeated historical input or HTTP
queueing does not account for this difference. This is a systematic build-cohort
difference, not evidence that the repo chunk mapping malfunctioned.

Raw summaries remain under
`/srv/ai/paged-kv/results/comparison-completion-20261104-r6/{master-gpu-mtp-on,fork-gpu-mtp-on}/attempt-001-L65536/`.
Do not mix this fresh mapped ingest with canonical question-suffix processing.

## Verified source difference; runtime attribution remains a hypothesis

In the fork `ggml/src/ggml-cuda/fattn.cu`,
`ggml_cuda_turbo4_fused_shape_supported()` (around 4409) admits matched Turbo4
K/V, query head width 128/256, and any positive query length when its tensor,
mask and geometry checks pass. `ggml_cuda_flash_attn_ext()` (around 4513/4542)
can dispatch that fused raw-Turbo4 MMA path before the established
`ggml_cuda_turbo_prefill_attend()` path (around 4648).

In pristine master the analogous fused dispatch (around 2624) requires
`Q->ne[1] <= 4`; larger prefill batches reach Turbo prefill attention
(around 2722). U=256 makes that predicate difference relevant. Build flags
checked by the read-only audit matched for CUDA, FA, graphs, sm89-real and
forced-MMQ/CUBLAS choices. The fork pager-off path builds dense attention and
skips routing-query capture; there was no evidence of unconditional selector
or page-ranking work accounting for the 32% gap.

The fork's `llama_context::reserve_mtp_attention_graph_scratch()` (around 6958)
also consults the fused-shape predicate and may avoid materialization scratch.
This suggests a possible throughput/scratch tradeoff. It does not prove which
kernel each measured graph dispatched. The R6 records lack per-kernel dispatch
traces, and this observation alone does not justify changing selection,
checkpointing, page storage, B/U, or quants.

## Later investigation, outside the frozen comparison

Keep both measured binaries/configurations unchanged during this matrix.
If optimization is pursued later, confirm real dispatch first with the existing
`ggml_cuda_fattn_turbo4_fused_last_dispatch_was_mma()` indicator or a separately
labeled bounded CUDA timeline. Compare identical tensor shapes/content and
record scratch/VRAM alongside throughput. Do not silently disable fused MMA or
replace primary control values with a modified diagnostic run. Any adaptive
choice must retain bounded scratch at the larger ACO context allocations.

## Later GPU-only fill failures: real pressure, conservative reservation

In the same R6 matrix with MTP off and L131072, fork completed actual C100352
then failed the C120832 append. Master completed C120832 then failed the
C130048 append. Both sampled a minimum of 9 MiB device-free memory, with
server process GPU usage reaching 15886 MiB; host MemAvailable remained about
50.7 GiB. These extrema are not an instantaneous allocator measurement. The
runs establish resource exhaustion under these exact allocations, not an
intrinsic context limit for every configuration.

The fork log reserved 234.5 MiB each for F16 K and V scratch, then failed a
4 MiB physical increase even after flushing deferred unmaps. This is a real
allocation failure: `vbr_scratch_reserve()` calls the backend reserve;
`kv_dequant_scratch_try()` attempts VMM allocation/map and cudaMalloc fallback.
It is not a ledger-only rejection of a hypothetical buffer.

However, this scratch is requested by a conservative **pre-graph target**
watermark, not proof that a materializing attention kernel executed:
`llama_kv_cache::apply_batch()` (around 8228–8263) takes the pager-off path when
`graph_scratch_admission` is false. It sizes K/V scratch from occupied/padded
rows and static tensor codecs. `ggml_vbr_kv_dequant_sides()` treats both Turbo4
sides as needing F16 materialization. Native MTP enables graph-time admission
in `llama-model.cpp` (around 3274), but ordinary target caches retain the
watermark policy. Thus the main target can reserve scratch before the actual
fused-attention shape is available, even if a later fused dispatch would not
consume that scratch. Confirm real dispatch before changing this behavior.

The predefined L122880 GPU-only profiles will collect separate lower-ceiling
mapped curves after the primary queue. B/U, quants and binaries stay fixed.
A separate Nsight trace of the frozen GPU/MTP-on profiles will identify actual
attention kernels; profiler timings will not replace the primary measurements.
