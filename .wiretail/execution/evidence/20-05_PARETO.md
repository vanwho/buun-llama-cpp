# 20-05 quality/speed calibration receipt

Result: incomplete, with the scoped production fix and local verification complete. The frozen `pager-corpus-v4` fixture is valid (24 records; calibration and held-out copies of 12 IDs; corpus hash `37111506c8ddc7f9d04086121797cc9c2b7a2842cc1a56ef4b33f506c2252007`). The named Qwen3.8-27B UD-IQ4_XS model was used unchanged (`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`).

## Implementation and live result

`src/llama-kv-cache.cpp` now creates compressed cache views against the byte-anchor slab using `ggml_row_size(type, width)` as the view width, then restores the typed cache shape. This corrected a live startup assertion where GGML interpreted the compressed row width as I8 elements and let later layers exceed the slab. The repaired immutable candidate loaded on managed port 8080 with full 22016 context, selective paging, target/MTP Turbo4, native GPU MTP, and a 371,982,000-byte physical pool.

The short managed probe reported 24.84–26.08 prompt tok/s, 3.94–4.14 decode tok/s, and 66.7–100% MTP acceptance. Telemetry reported 35 hot tokens, 1 prefill reference route, 2 decode reference routes, 58 MTP verification reference routes, zero faults, and zero evictions. These are actual server counters; no dense-equivalence claim is made.

The first frozen quality probe was `cold-early` at diagnostic context 5120. Its rendered request entered prefill and reached 3072/5120 tokens at 398.96 seconds, but the 600-second bounded campaign ended without a response, so answer score and logit/PPL delta are null. The durable result is classified `incomplete_timeout`, not as a quality pass or a guessed failure.

The requested exact-prefill/selective-decode control is not exposed as a mixed CLI mode. A whole-runtime `--kv-pager exact` probe was attempted and repeatedly aborted during `ggml_backend_sched_split_graph` (`ggml-backend.cpp:1359`, no backend assignment); the harness restored the selective candidate and verified health. This is routed as implementation work to the exact graph binding, not a threshold adjustment. The conservative default remains `selective` because it is the only full-context Turbo4/GPU-MTP mode that loaded and produced runtime telemetry; no held-out tuning or dense-equivalence claim is made.

## Verification and raw evidence

- `validate-pager-benchmark.py` — valid corpus.
- Python bench unit tests — 46 passed.
- CUDA build of `llama-server` and seven focused pager/attention targets — passed.
- Focused CUDA ctest regex — 7/7 passed.
- Final managed service — `active/running`, `NRestarts=0`, repaired selective candidate on 8080; unrelated 8092 was not touched.
- Full receipt: `.wiretail/execution/evidence/20-05_PARETO.json`.
- Durable results: `/srv/ai/paged-kv/results/20-05-runtime-selective-20260905T200000Z`, `/srv/ai/paged-kv/results/20-05-runtime-exact-20260905T200000Z`, and `/srv/ai/paged-kv/results/20-05-quality-selective-cold-early-20260905T200000Z`.

## Deferred verification

After exact page-wave backend repair, rerun the frozen calibration and held-out sentinels at increasing occupied lengths, then execute the one-variable equal-hot-byte policy sweep (recent-only, query, retention, exploration, layer group, confidence fallback) and freeze the selected configuration before held-out scoring. The 262144-token ultimate target and ordinary CPU-KV control remain later scope. No live answer, dense parity, or PPL result is inferred from the timeout.
