# Fresh ACO prefill to a 256K allocation — 2026-11-03

Completed **261,121 occupied tokens** in one fresh server session, leaving **1,023 tokens** of reserve in L=262,144. H=65,536 (256 pages), B/U=1024/256, Turbo4 target/draft K/V, full-L GPU native MTP2, CUDA0, FA on, one slot, threads16, no context shift. No decode benchmark was run; fill requests used one-token ACKs.

Candidate: `d2d9c7291e3b1e057d90dea42c9bfe7c145a0ff6`; binary SHA-256 `da58c02c2c9a9cbc7b1879e07fa96fa86b02f4ca155feab09a0a2c1b12587dc2`. This candidate stayed unchanged across attempts. Full suite remains paused.

Each append adds the next distinct repository bytes. The successful attempt used 32K content groups, with smaller checkpoint tails. Its 912,974-byte source stream is identical to the failed 64K-group attempt; request boundaries differ. Old history was reused from cache.

| Occupied tokens | New tokens in interval | Backend tok/s | Net-new prefill tok/s | Net-new request-wall tok/s |
|---:|---:|---:|---:|---:|
| 6,144 | 6,144 | 1,450.72 | 1,450.72 | 1,445.19 |
| 12,288 | 6,144 | 1,266.10 | 633.67 | 632.22 |
| 28,673 | 16,385 | 983.34 | 491.85 | 491.41 |
| 62,465 | 33,792 | 628.38 | 314.30 | 313.94 |
| 78,849 | 16,384 | 817.02 | 408.66 | 407.91 |
| 88,065 | 9,216 | 819.65 | 410.09 | 408.58 |
| 100,352 | 12,287 | 831.70 | 416.05 | 414.79 |
| 130,049 | 29,697 | 870.24 | 435.21 | 434.46 |
| 196,608 | 66,559 | 814.61 | 407.42 | 406.18 |
| 256,001 | 59,393 | 756.38 | 378.27 | 377.19 |
| 261,121 | 5,120 | 591.07 | 295.88 | 292.25 |

Whole fill: **515,930 backend prompt tokens / 261,121 net-new occupied tokens**; weighted rates **790.36 backend**, **400.01 net-new prefill**, **398.97 net-new request-wall tok/s**. Summed prompt time 652.78s; summed request-wall time 654.50s. These exclude harness preparation and service restart time.

Backend rate counts actual prompt evaluations. Net-new prefill divides occupancy growth by that same prompt time; request-wall rate also includes HTTP/request overhead. Rates are summed, not averaged across chunks. Different append sizes and short tails contribute to variation.

**Replay remains:** initial request replayed zero times; 14 later requests replayed their new user span once. No old-context reload was recorded. This run does not establish that the new automatic replay bypass is effective, and it measures no retrieval accuracy, decode speed, or MTP acceptance.

Sampled resources: GPU used peak 15,589 MiB; reported free minimum 357 MiB; process VmRSS peak 8,696,184 KiB; VmHWM peak 14,322,216 KiB; process VmSwap maximum 0 KiB. Samples were taken every five seconds, so these are observed sample extrema. The GPU-free value is the reported free counter, not total-minus-used arithmetic.

Attempt 01 failed due to an API-key variable being overwritten by continuation metadata. Benchmark-only fix `7038a241a` and its existing continuation regression passed. Attempt 02 reached C130,049, then a roughly 64K new-user request failed exact-history publication (`transaction=0 generation=0 winners=0`). Neither failed attempt contributes to the successful curve. No OOM was recorded.

The likely code path for attempt 02 is a Q span of about 257 pages exceeding the retrieval budget, producing no admitted winners while exact candidates remain nonempty. See `llama_kv_query_history_budget` in `src/llama-kv-live-policy.cpp`, the capacity calculation around `src/llama-kv-cache.cpp:5032`, and the publication check around `src/llama-kv-cache.cpp:3859`. The precise rejection predicate was not instrumented. Completing with 32K chunks does not fix this limitation.

Runtime identity: [candidate-identity.json](/srv/ai/paged-kv/results/forward/automatic-attention-reuse-20261103/candidate-identity.json). Exact argv, per-append sizes, counters, artifact hashes and the separate failed-attempt data are in the companion JSON. Raw primary results: [attempt-03](/srv/ai/paged-kv/results/forward/automatic-reuse-prefill-20261103/attempt-03). Restoration: [restoration.txt](/srv/ai/paged-kv/results/forward/automatic-reuse-prefill-20261103/attempt-03/restoration.txt).

Managed server and gateway restored; 8080/8091 returned HTTP 200. Jellyfin stayed running at the same PID and returned HTTP 200 on 8096. Port 8092 was untouched.

Exact tested server command (device auto resolved to CUDA0 for the full-L Turbo4 MTP reservation):

```bash
/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/automatic-reuse-prefill-20261103/attempt-03/cell/slot-state --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --kv-pager selective --kv-page-size 256 --kv-hot-pages 256 --kv-router probe-rerank --kv-pin-recent 0 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4
```

Configured hot pool was 256 pages (65,536 tokens); actual admitted physical-page telemetry was not sampled. This does not invalidate the measured prefill timings.

## Historical ACO comparison

Old cohort: runtime `1d74e9a31`, same L/H/B/U/MTP, variable per-file chunks up to 4,080 tokens. New cohort: runtime `d2d9c7291`, mapped 32K groups and checkpoint tails. This is an observed whole-approach comparison, not an isolated code A/B; interval boundaries differ by at most one token.

| Target C | Old backend | Old net-new | New backend | New net-new | Net-new change |
|---:|---:|---:|---:|---:|---:|
| 6,144 | 1,287.72 | 797.28 | 1,450.72 | 1,450.72 | +82.0% |
| 12,288 | 1,009.81 | 953.48 | 1,266.10 | 633.67 | -33.5% |
| 28,672 | 909.54 | 520.61 | 983.34 | 491.85 | -5.5% |
| 62,464 | 595.93 | 317.36 | 628.38 | 314.30 | -1.0% |
| 78,848 | 569.29 | 297.56 | 817.02 | 408.66 | +37.3% |
| 88,064 | 607.64 | 304.47 | 819.65 | 410.09 | +34.7% |
| 100,352 | 592.65 | 296.94 | 831.70 | 416.05 | +40.1% |
| 130,048 | 560.71 | 281.00 | 870.24 | 435.21 | +54.9% |
| 196,608 | 513.95 | 257.61 | 814.61 | 407.42 | +58.2% |
| 256,000 | 443.04 | 222.20 | 756.38 | 378.27 | +70.2% |

Historical source: [ACO_FIXED_RESULTS_20261103.md](ACO_FIXED_RESULTS_20261103.md). No new full-GPU control was run in this prefill-only request.

For the same overall occupied span **0→256,001**, weighted rates were **541.93 backend / 281.27 net-new** previously and **795.79 backend / 402.85 net-new** now: **+43.2% net-new**. Request grouping differs as noted above.

## Historical full-GPU MTP-on references

These are recorded fresh-source-fill controls, not a new matched 32K-map A/B. Builds, total allocations, request grouping and possibly fill-template/reasoning differ; raw config and schedule references are in JSON. Each comparison below aggregates from empty to the same final occupied count, rather than comparing one final chunk to a whole run.

| Span | Historical full-GPU control | Control backend / net-new | Current ACO backend / net-new | ACO net-new change |
|---:|---|---:|---:|---:|
| 0→12,288 | master full GPU + MTP2 (L16,384) | 1,488.11 / 1,488.11 | 1,322.23 / 882.06 | -40.7% |
| 0→78,849 | fork full GPU, ACO off + MTP2 (L80,896) | 703.95 / 703.95 | 777.49 / 404.66 | -42.5% |

The larger master GPU run stopped short of its target, so it is not presented as a completed control. No matched full-GPU MTP-on reference at 256K is available; full GPU cannot be extrapolated to that occupancy on this card.
