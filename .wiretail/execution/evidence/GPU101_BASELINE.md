# GPU101 matched prefill and GPU attribution

Task `101-01`; source commit `389b09016907c3e907c3c499738936d6be44257e`; candidate `8d66334886f13f727c6d81f00a0cc44abe43c1806075ebe3997c24358a8be860`.

## Matched fresh rows

| Mode | Prompt | Cache hit | Prefill tok/s | Decode tok/s | MTP accepted/drafted |
|---|---:|---:|---:|---:|---:|
| Pager off | 4326 | 0 | 1626.53 | 66.16 | 36/56 (64.29%) |
| Selective | 4326 | 0 | 202.14 | 38.71 | 39/55 (70.91%) |

Both rows used frozen prefix SHA `ad528d87807f615a95b0694c0b0dd097e3facd24be14b71914a5ecf9694f4067`, fresh/reset slots, reasoning off, 40-token warmup, 80 measured output tokens, context8192, B1024/U256, target/draft Turbo4, native GPU MTP nmax2. The measured wire bodies have SHA `26cb0006b16ca506fef34792439edebced036aec737eade625befc2778413079` and match byte-for-byte. Selective is about 8.05x slower than pager off and misses the 500 tok/s floor; the regression is selective-specific.

## Physical page geometry

Selective allocator telemetry reports 16 resident physical GPU pages × 256 = H4096, 32 logical pages for L8192, and 16 host pages. Resolved/admitted target capacity is 4096 tokens. Automatic generation tail plus one admitted transfer destination reserve two physical pages (512 tokens), leaving 14 pages/3584 tokens for retrieval. MTP reserved 8192 Turbo4 rows / 8,781,824 bytes on GPU. Checkpoint capture frontiers were 1130, 4299, and 4322, with state sizes recorded in the JSON.

Effective U was 256. With a 4,326-token prompt this implies a 17-item schedule (16×256 plus230); the service did not emit the per-call `context_ubatch` records even with `LLAMA_HOTPATH_PROFILE=1` and verbose logging, so that vector is marked inferred.

## CUDA attribution

Nsight Systems 2025.5.2 profiled the exact candidate executable on CUDA device0. In the bounded 19-second request/error interval, the report contains 32,359 kernel records, 35,788 copy records, 43 graph spans and 1.098s union GPU activity (5.78% of the interval). CUDA kernel families include quantized `mul_mat_q`, `gated_delta_net_cuda`, flash attention and Turbo4 dequant. Inclusive CUDA API time includes 1.735s of stream synchronization and 0.159s of asynchronous memcpy API time; these overlap and are not additive.

The profiled replay completed its 4,326-token prefill but returned `KV pager batch write reservation failed: no_victim` during decode. It is a valid prefill-attribution trace, not a successful profiled end-to-end row. The separate matched benchmark rows above both completed.

## Opt-in gate

`LLAMA_HOTPATH_PROFILE` now enables the context detailed timers only for exact value `1`; null, `0`, and `true` disable them. `test-kv-attention-execution` passed. The successful ordinary-mode selective journal has no `context_ubatch` or `context_fence` records.

Raw inputs, full report, statistics and build/fixture logs are listed with hashes in `GPU101_BASELINE.json`; task raw root: `/srv/ai/paged-kv/results/forward/101-01/attempt-01`.
