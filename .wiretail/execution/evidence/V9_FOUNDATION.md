# V9 foundation receipt

Task 34-06 measured the current CUDA foundation before dynamic cold selection.
The fixture passed with CUDA-event timings for fixed selected A, small and
large prefill shapes, decode Q=1/2/3/5, and a four-KV-head incremental tail.
Packed Turbo4 FA was 0.022/0.025/0.030 ms at U=16/64/128 for the 530-row case,
versus 0.020/0.023/0.028 ms contiguous. Across A=2048, 4096, and 8192 with
U=16/64/128, packed was 0.001–0.002 ms slower. This is a small 1–3% event-time
overhead, so no opt-in profile was warranted.

The incremental active-tail H2D event was 0.002048, 0.005888, and 0.001728 ms
for 1203, 1267, and 1331 rows with four KV heads. The complete append+attention
events were 22.969280, 1.246208, and 1.299456 ms; the first event is the
dominant unresolved cost and includes first-use/warmup behavior.

The planner and telemetry fixtures passed. Automatic selected typed views use
packed routing; direct is only exercised by an explicit diagnostic fixture
route. No automatic full-A copy, full-H selection, or blocking upload remains.
The current service receipt has graph=268435000, packed workspace=69206000,
packed dequant=8388610 bytes, capture=1, replay=0, rebuild=1 before a request.

The matched current all-GPU q0 completed with 512 prompt and 16 output tokens:
384.392 ms prompt and 34.831 tok/s decode, compared with 463.142 ms and 35.315
tok/s in the 33-03 receipt. Both had zero accepted speculative tokens, so this
is recorded as a real short control, not inflated throughput.

Raw: `/srv/ai/paged-kv/results/v9/34-06/20260913T184319Z-foundation/`,
`20260913T184443Z-b2-quick/`, and `20260913T184502Z-allgpu-q0/`.

The canonical B2 adapter could not run because `BENCH_ENDPOINT` and
`CANONICAL_BENCHMARK_RUNNER` are not configured; its non-diagnostic and
diagnostic attempts are retained. The tested selective native-MTP candidate is
restored and loaded by `llama-server.service` from `build-cuda/bin`.
