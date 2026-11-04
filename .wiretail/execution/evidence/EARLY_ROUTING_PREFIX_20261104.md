# Automatic early routing prefix: implementation and validation

Run: `/srv/ai/paged-kv/results/candidate-full256k-20261104T005437547968773Z`  
Status: **complete**, 11/11 requested checkpoints, 15 append requests; prefill-only, no decode campaign.

## Identity and configuration

- Source equivalent commit: `6068d4a003c6ad0e117a4808097f34f0056e0e85`; Build-06 differs from the Build-05 small-cell source by cleanup-only allocation-guard placement.
- Server SHA256: `f0f44fdefb920245811908f9a75e96ec5fce782f962836e024bc59f10d837cd0`; loaded DSO hashes: `runtime-dependencies.json`.
- Qwen3.8-27B UD-IQ4_XS, L=262144, Turbo4 target K/V on GPU, native MTP draft max 2, B1024/U256, H=65536 = 256 pages ×256; one server.
- Fixed map: 32K token chunks, SHA256 `5ca53650661d610ff2632eae262cb794bceb5b72efaf47364b8bcd43247dcf4f`; actual final occupancy 261,121.

## Aggregated checkpoint curve

Each milestone aggregates every fill request since the prior checkpoint. Rates use net-new tokens / server prompt time, backend tokens / prompt time, and net-new tokens / HTTP wall time. Server prompt time is not a CUDA-kernel-only measurement. The leading-layer routing prepass is not counted as another full-model token evaluation; its work is included in measured request wall time. New-token counts include the new file content and required chat framing.

| Requested → actual | New / backend tokens | Prompt / wall seconds | Net tok/s | Backend tok/s | Wall tok/s |
|---:|---:|---:|---:|---:|---:|
| 6,144 → 6,144 | 6,144 / 6,144 | 4.24 / 4.25 | 1450.8 | 1450.8 | 1444.8 |
| 12,288 → 12,288 | 6,144 / 6,144 | 5.11 / 5.13 | 1202.2 | 1202.2 | 1196.6 |
| 28,672 → 28,673 | 16,385 / 16,385 | 17.30 / 17.33 | 947.4 | 947.4 | 945.5 |
| 62,464 → 62,465 | 33,792 / 33,792 | 55.50 / 55.63 | 608.9 | 608.9 | 607.5 |
| 78,848 → 78,849 | 16,384 / 16,384 | 18.46 / 18.54 | 887.3 | 887.3 | 883.7 |
| 88,064 → 88,065 | 9,216 / 9,216 | 11.33 / 11.41 | 813.8 | 813.8 | 808.0 |
| 100,352 → 100,352 | 12,287 / 12,287 | 14.89 / 14.98 | 825.4 | 825.4 | 820.3 |
| 130,048 → 130,049 | 29,697 / 29,697 | 34.98 / 35.10 | 849.0 | 849.0 | 846.1 |
| 196,608 → 196,608 | 66,559 / 67,540 | 88.00 / 88.49 | 756.4 | 767.5 | 752.1 |
| 256,000 → 256,001 | 59,393 / 59,393 | 81.57 / 81.98 | 728.1 | 728.1 | 724.5 |
| 261,120 → 261,121 | 5,120 / 5,120 | 8.62 / 8.84 | 594.0 | 594.0 | 579.4 |

## Append-level fill curve

Rates are `net / backend seconds`, `backend prompt_n / backend seconds`, and `net / HTTP wall seconds`, respectively. Backend tokens may exceed newly added tokens when a short query suffix is replayed.

| Actual occupied | New / backend tokens | Prompt / wall seconds | Net tok/s | Backend tok/s | Wall tok/s | Route prefix tokens / state bytes | Query replay |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6,144 | 6,144 / 6,144 | 4.24 / 4.25 | 1450.8 | 1450.8 | 1444.8 | — | 0 |
| 12,288 | 6,144 / 6,144 | 5.11 / 5.13 | 1202.2 | 1202.2 | 1196.6 | — | 0 |
| 28,673 | 16,385 / 16,385 | 17.30 / 17.33 | 947.4 | 947.4 | 945.5 | — | 0 |
| 61,456 | 32,783 / 32,783 | 53.11 / 53.17 | 617.3 | 617.3 | 616.5 | — | 0 |
| 62,465 | 1,009 / 1,009 | 2.39 / 2.46 | 421.3 | 421.3 | 410.8 | — | 0 |
| 78,849 | 16,384 / 16,384 | 18.46 / 18.54 | 887.3 | 887.3 | 883.7 | 16,369 / 9,805,824 | 0 |
| 88,065 | 9,216 / 9,216 | 11.33 / 11.41 | 813.8 | 813.8 | 808.0 | 9,201 / 9,805,824 | 0 |
| 100,352 | 12,287 / 12,287 | 14.89 / 14.98 | 825.4 | 825.4 | 820.3 | 12,272 / 9,805,824 | 0 |
| 130,049 | 29,697 / 29,697 | 34.98 / 35.10 | 849.0 | 849.0 | 846.1 | 29,682 / 9,805,824 | 0 |
| 162,832 | 32,783 / 32,783 | 40.28 / 40.43 | 813.9 | 813.9 | 810.9 | 32,768 / 9,805,824 | 0 |
| 195,615 | 32,783 / 32,783 | 42.11 / 42.28 | 778.5 | 778.5 | 775.4 | 32,768 / 9,805,824 | 0 |
| 196,608 | 993 / 1,974 | 5.61 / 5.79 | 177.1 | 352.0 | 171.6 | — | 1 (981 toks) |
| 229,391 | 32,783 / 32,783 | 44.12 / 44.31 | 743.1 | 743.1 | 739.9 | 32,768 / 9,805,824 | 0 |
| 256,001 | 26,610 / 26,610 | 37.45 / 37.67 | 710.6 | 710.6 | 706.4 | 26,595 / 9,805,824 | 0 |
| 261,121 | 5,120 / 5,120 | 8.62 / 8.84 | 594.0 | 594.0 | 579.4 | 5,105 / 9,805,824 | 0 |

Total: 261,121 net tokens / 262,102 backend tokens; 339.98s prompt time, 341.68s wall; 768.04 net-new/prompt-time tok/s, 770.93 backend tok/s, 764.22 net/wall tok/s.
Earlier full baseline: 261,121 net / 459,657 backend tokens; 580.34s prompt time / 582.07s wall; 449.94 net-new/prompt-time, 792.05 backend, 448.61 net/wall tok/s.

Early prefix state was 9,805,824 bytes on all 9 routed appends (9.35 MiB; one target-query pass each). Those nine appends had no replay. The lone replay was a 981-token suffix during the 993-token tail to C196608; no full populated-prefix replay was recorded.

## Small retrieval/decode control

At L16384/H8192, frozen baseline prompt 1 produced 400 tokens at C12288 in 4.731s / 101.47 decode tok/s; MTP accepted 261/276 (94.57%). Baseline recall answered ~41.6% from 0.00163 vs 0.00279 in 293 tokens.
Adaptive Build-05 prompt 1 produced 400 tokens in 4.693s / 101.27 decode tok/s; MTP 259/278 (93.17%). The short fact prompt returned `39.4% (0.00163 vs 0.00279)` in 24 tokens; MTP 14/20 (70%). The initial unrestricted prefix-only proxy missed this fact; the <=B all-layer selector recovered the source figures and approximately 40% statement. Its calculated percentage is not numerically identical to the baseline; this is a bounded retrieval check, not a general accuracy certification.

## Resources and restoration

Runtime capture at the final checkpoint confirms admitted `hot_pages=256` (`H=65,536`, 256-token pages). GPU-wide max used/free minimum: 15,567/379 MiB; candidate process max: 15516.0 MiB. Process RSS/HWM peaks were 8,550,228/14,321,444 KiB (5s harness samples), process VmSwap remained 0 KiB. System SwapFree minimum was 6,130,644 KiB, MemAvailable minimum 44,384,520 KiB (1s external samples).
CUDA compute buffer: 168.3167 MiB; CUDA_Host: 118.3286 MiB; MTP KV allocation: 276,955,136 bytes for 262,144 rows. No request errors or truncation. Server VmSwap was zero in process samples; system swap figures include unrelated host processes.
Services restored with exit 0; gateway and managed llama-server active; Jellyfin PID 1000914 unchanged; :8091 remained PID 76881; :8092 remained untouched/unbound.

## Raw evidence

- Candidate full run curve: `/srv/ai/paged-kv/results/candidate-full256k-20261104T005437547968773Z/cell/prefill-curve.json`
- Startup log: `/srv/ai/paged-kv/results/candidate-full256k-20261104T005437547968773Z/cell/server-startup-20261104T005438Z.log`
- 1s GPU/system sampler: `/srv/ai/paged-kv/results/candidate-full256k-20261104T005437547968773Z/resource-loop-1s.jsonl`
- 5s process sampler: `/srv/ai/paged-kv/results/candidate-full256k-20261104T005437547968773Z/cell/resource-samples.jsonl`
- Runtime DSO identity: `/srv/ai/paged-kv/results/candidate-full256k-20261104T005437547968773Z/runtime-dependencies.json`
- Restoration: `/srv/ai/paged-kv/results/candidate-full256k-20261104T005437547968773Z/restoration.txt`
- Earlier full baseline: `/srv/ai/paged-kv/results/forward/allhot-reuse-repair-20261104/full256k/summary.md`

## Comparison and scope

HTTP request wall time fell from 582.07 to 341.68 seconds (41.30% less); wall throughput rose from 448.61 to 764.22 tok/s (1.70x). GPU-wide peak used was 22 MiB lower in this sample, not a claim of exact allocation savings. The existing CUDA compute allocator was 168.3167 MiB in both runs, with an unchanged 118.3286 MiB host compute allocator. Routing state stays at 9.35 MiB; no prompt-length hidden tape is retained.

Four focused checks passed: attention-view coverage, prefetch, assertion-enabled policy (including early-mass provenance), and router job CPU fixtures. All builds used `--parallel 16`. The full decode/reasoning matrix remains paused; the long run measured prefill only. The 400-token Prompt-1 small run checks MTP behavior, while short recall deliberately retains the more accurate all-layer route.
