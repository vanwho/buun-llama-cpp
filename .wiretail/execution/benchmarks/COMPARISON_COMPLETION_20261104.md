# Comparison completion plan R6 — active

**Active and frozen.** R5 was held after its completed C6144/C12288/C28672 CPU-MTP-off stages. Their valid triplets are retained. R5 JSON/MD are archived for provenance only.

Output root: `/srv/ai/paged-kv/results/comparison-completion-20261104-r6`. Workload identity/build/model/template/map metadata is copied from R5; model/build/DSO identities were not rehashed.

## Protocol

All occupied controls use normal server prefix caching of the unchanged filled repository base. Only mapped repository ingestion is append-only; previous benchmark questions and answers do not accumulate in the base. No benchmark-owned slot save/restore/erase operations are part of the protocol. `checkpoint_free_control=false` on every profile; normal 8 / 8192 / 12288 tiered cache flags are retained. Actual C is recorded separately from its nominal target. The server reuses the filled repo prefix for each independent question.

Canonical sampled decode remains 40-token warmup + 3 measured requests, max 400, T=0, seed42. Ingest-only targets have empty selections and no decode requests.

## Ordered series

| # | Family / series | Purpose | L profiles | Target C sequence(s) | Selected decode | Requests incl warmup |
|---:|---|---|---|---|---|---:|
| 1 | master-gpu-mtp-on | Retain every completed historical decode cell. Collect only the uniform mapped-chunk ingest curve at a safe 64K GPU allocation, so all three prefill metrics have a same-method comparison. | L65536 (65536) | 6144,12288,28672,62464 | ingest only / no decode | 0 |
| 2 | fork-gpu-mtp-on | Retain every completed historical decode cell. Collect only the uniform mapped-chunk ingest curve at a safe 64K GPU allocation; no canonical prompt/mode generations are repeated. | L65536 (65536) | 6144,12288,28672,62464 | ingest only / no decode | 0 |
| 3 | fork-gpu-mtp-off | Only the previously requested conditional 127K occupied endpoint: lower points already have all four modes. If 128K allocation cannot fit, record the verified lower ceiling and collect prefill only. | L131072 (131072); L122880 (122880) | 6144,12288,28672,62464,78848,88064,100352,120832,130048; 6144,12288,28672,62464,78848,88064,100352,120832 | C130048/off/P1, C130048/off/P2, C130048/off/P3, C130048/low/P1, C130048/low/P2, C130048/low/P3, C130048/medium/P1, C130048/medium/P2, C130048/medium/P3, C130048/xhigh/P1, C130048/xhigh/P2, C130048/xhigh/P3 | 48 |
| 4 | master-gpu-mtp-off | Capture a uniform mapped-chunk ingest curve without repeating the already reproduced resident-prefix decode failure. Requested decode gaps remain visible through requested_decode_requirements and reused_decode_outcomes; an ingest-only completion is not a successful decode measurement. | L131072 (131072); L122880 (122880) | 6144,12288,28672,62464,78848,88064,100352,120832,130048; 6144,12288,28672,62464,78848,88064,100352,120832 | ingest only / no decode | 0 |
| 5 | fork-cpu-mtp-on | Capture a uniform mapped-chunk ingest curve without repeating the already reproduced resident-prefix decode failure. Requested decode gaps remain visible through requested_decode_requirements and reused_decode_outcomes; an ingest-only completion is not a successful decode measurement. | L262144 (262144); L200704 (200704); L131072 (131072) | 6144,12288,28672,61440,126976,196608,256000; 6144,12288,28672,61440,126976,196608; 6144,12288,28672,61440,126976 | ingest only / no decode | 0 |
| 6 | master-cpu-mtp-on | Capture a uniform mapped-chunk ingest curve without repeating the already reproduced resident-prefix decode failure. Requested decode gaps remain visible through requested_decode_requirements and reused_decode_outcomes; an ingest-only completion is not a successful decode measurement. | L200704 (200704); L131072 (131072) | 6144,12288,28672,61440,126976,196608; 6144,12288,28672,61440,126976 | ingest only / no decode | 0 |
| 7 | master-cpu-mtp-off | The historical off-mode curve is already complete through the 256K allocation. Run only the never-completed low/medium/xhigh curves, not off again. | L262144 (262144) | 6144,12288,28672,61440,126976,192512,258048 | C61440/low/P1, C61440/low/P2, C61440/low/P3, C61440/medium/P1, C61440/medium/P2, C61440/medium/P3, C61440/xhigh/P1, C61440/xhigh/P2, C61440/xhigh/P3, C126976/low/P1, C126976/low/P2, C126976/low/P3, C126976/medium/P1, C126976/medium/P2, C126976/medium/P3, C126976/xhigh/P1, C126976/xhigh/P2, C126976/xhigh/P3, C192512/low/P1, C192512/low/P2, C192512/low/P3, C192512/medium/P1, C192512/medium/P2, C192512/medium/P3, C192512/xhigh/P1, C192512/xhigh/P2, C192512/xhigh/P3, C258048/low/P1, C258048/low/P2, C258048/low/P3, C258048/medium/P1, C258048/medium/P2, C258048/medium/P3, C258048/xhigh/P1, C258048/xhigh/P2, C258048/xhigh/P3 | 144 |
| 8 | master-cpu-mtp-on-outlier-a | Independent clean rechecks of the one irregular C28672/off/P3 cell; one mapped fill per fresh series, do not cherry-pick faster. | L32768-series-a (32768) | 6144,12288,28672 | C28672/off/P3 | 4 |
| 9 | master-cpu-mtp-on-outlier-b | Independent clean rechecks of the one irregular C28672/off/P3 cell; one mapped fill per fresh series, do not cherry-pick faster. | L32768-series-b (32768) | 6144,12288,28672 | C28672/off/P3 | 4 |
| 10 | master-cpu-mtp-on-outlier-c | Independent clean rechecks of the one irregular C28672/off/P3 cell; one mapped fill per fresh series, do not cherry-pick faster. | L32768-series-c (32768) | 6144,12288,28672 | C28672/off/P3 | 4 |

Expected selected decode requests including warmups: **204**; primary fill-target runs including the three outlier series: **55**; all listed startup-fallback profiles total **87** fill-target runs. Prefill append request counts depend on the frozen source map and are kept separately from decode requests.

## Retained outcomes and open findings

R4 fork CPU MTP-off remains whitelisted. R5 fork CPU MTP-on, master CPU MTP-on and master CPU MTP-off attempt directories are individually whitelisted as partial series; only valid complete triplets may be imported. R4 CPU MTP-on target-only restore series remain quarantined and excluded.

Master GPU MTP-off C6144 retains the prior normal-prefix fault as an explicit missing outcome; no identical fourth decode attempt is planned. The exact source receipt must be pinned before this draft is frozen. The known same-master CPU MTP-on 256K startup CUDA OOM remains a capacity result, not a missing required decode cell. The master GPU runtime fault remains classified separately from allocation OOM.

## Independent outlier series 8–10

Each uses the master CPU MTP-on configuration at L32768, with one fresh normal resident history filling 6144→12288→28672 and only C28672/off/P3 decoded (one 40-token warmup + 3 measured requests capped at 400). Each series has four decode requests; total 12. A valid complete triplet can replace the old cell as a whole; keep all three independent series for robustness, never cherry-pick.

## Commands and provenance

Each profile records argv regenerated from current `matrix.argv_for_row`, including server template, first-attempt argv, server command, driver argv and orchestrator command. Lightweight Python driver SHA-256 values are refreshed; R5 model/build/DSO/template/map identities are copied unchanged. R5 remains the only active plan.

## Coverage remains explicit

`requested_decode_requirements` retains the earlier requested decode cells, including those that cannot currently be measured because a native control fails. An ingest-only step does not turn missing decode cells into passes. The 204 primary requests are gap/outlier requests, not a repeat of the full historical matrix. No inference binary or quant/batch/sampling setting changes.
