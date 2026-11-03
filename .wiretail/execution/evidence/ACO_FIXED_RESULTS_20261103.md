# Fixed-candidate ACO results — 2026-11-03

Status: partial. Candidate SHA `189473719da99f8c391593c61a4c928769c80f72df8ceb6e527e707f95f35794`; source/build `1d74e9a315acaf0379eb2af1245202b820cff1e1`.
Progress: 120/240 prompt-mode pairs; 360/720 measured requests and 120/240 warmups (480/960 total).
Only the v2 fixed-candidate output root is included. Earlier ca549/v1 ACO rows are excluded from current comparisons.
Decode and acceptance values are medians across three passing repetitions, reported per prompt. Context-ingest rates use repository-fill schedule chunks; benchmark-question suffix prefill is excluded.

## C=6,144 (actual 6,145)

**Decode tok/s median**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 101.65 | 80.94 | 96.82 |
| low | 99.25 | 75.53 | 87.95 |
| medium | 99.08 | 77.01 | 89.54 |
| xhigh | 99.38 | 74.77 | 84.18 |

**MTP acceptance median (%)**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 92.31 | 63.33 | 86.64 |
| low | 89.90 | 56.64 | 74.38 |
| medium | 90.14 | 58.85 | 76.75 |
| xhigh | 90.87 | 56.19 | 68.96 |

## C=12,288 (actual 12,289)

**Decode tok/s median**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 99.50 | 82.67 | 86.03 |
| low | 96.24 | 74.43 | 91.19 |
| medium | 93.44 | 76.69 | 89.35 |
| xhigh | 95.67 | 77.77 | 85.41 |

**MTP acceptance median (%)**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 93.85 | 69.35 | 74.69 |
| low | 89.47 | 58.29 | 82.67 |
| medium | 85.19 | 61.51 | 79.55 |
| xhigh | 88.58 | 63.24 | 74.38 |

## C=28,672 (actual 28,673)

**Decode tok/s median**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 91.84 | 75.14 | 82.06 |
| low | 89.69 | 70.44 | 86.63 |
| medium | 90.34 | 69.90 | 82.65 |
| xhigh | 93.45 | 65.44 | 75.27 |

**MTP acceptance median (%)**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 91.67 | 66.67 | 77.88 |
| low | 89.62 | 60.28 | 84.80 |
| medium | 90.49 | 59.34 | 78.71 |
| xhigh | 96.79 | 52.92 | 67.75 |

## C=62,464 (actual 62,465)

**Decode tok/s median**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 79.55 | 59.35 | 74.42 |
| low | 76.25 | 54.11 | 74.97 |
| medium | 74.20 | 60.99 | 71.01 |
| xhigh | 72.51 | 54.49 | 61.09 |

**MTP acceptance median (%)**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 93.51 | 57.33 | 84.80 |
| low | 88.07 | 48.69 | 86.30 |
| medium | 85.08 | 60.83 | 78.96 |
| xhigh | 81.70 | 49.40 | 61.17 |

## C=78,848 (actual 78,849)

**Decode tok/s median**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 70.77 | 56.65 | 68.08 |
| low | 68.53 | 49.59 | 65.98 |
| medium | 70.26 | 52.10 | 60.05 |
| xhigh | 65.45 | 51.90 | 53.52 |

**MTP acceptance median (%)**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 90.31 | 61.54 | 85.37 |
| low | 86.28 | 48.70 | 80.66 |
| medium | 89.82 | 53.65 | 69.16 |
| xhigh | 80.21 | 53.60 | 56.84 |

## C=88,064 (actual 88,065)

**Decode tok/s median**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 69.44 | 50.46 | 65.13 |
| low | 66.06 | 45.98 | 64.82 |
| medium | 66.02 | 51.52 | 61.01 |
| xhigh | 65.06 | 48.97 | 54.68 |

**MTP acceptance median (%)**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 89.88 | 51.01 | 80.39 |
| low | 82.77 | 43.07 | 80.66 |
| medium | 82.39 | 54.19 | 72.62 |
| xhigh | 81.12 | 49.00 | 60.39 |

## C=100,352 (actual 100,353)

**Decode tok/s median**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 67.71 | 49.03 | 63.25 |
| low | 67.04 | 46.66 | 61.71 |
| medium | 68.62 | 51.93 | 61.50 |
| xhigh | 64.55 | 47.35 | 54.11 |

**MTP acceptance median (%)**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 87.65 | 50.00 | 78.71 |
| low | 86.28 | 46.07 | 75.71 |
| medium | 90.14 | 56.42 | 75.95 |
| xhigh | 81.39 | 47.48 | 60.22 |

## C=130,048 (actual 130,049)

**Decode tok/s median**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 65.01 | 49.22 | 62.78 |
| low | 63.00 | 45.59 | 60.30 |
| medium | 66.42 | 49.69 | 57.62 |
| xhigh | 65.22 | 44.83 | 48.12 |

**MTP acceptance median (%)**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 86.75 | 53.41 | 82.67 |
| low | 82.74 | 47.20 | 77.56 |
| medium | 90.14 | 55.70 | 72.09 |
| xhigh | 87.19 | 45.21 | 52.58 |

## C=196,608 (actual 196,609)

**Decode tok/s median**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 61.45 | 45.95 | 56.48 |
| low | 58.73 | 41.74 | 55.29 |
| medium | 58.20 | 44.31 | 48.90 |
| xhigh | 52.59 | 45.34 | 48.95 |

**MTP acceptance median (%)**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 88.36 | 54.00 | 78.39 |
| low | 82.92 | 45.39 | 75.95 |
| medium | 82.12 | 51.40 | 63.07 |
| xhigh | 70.18 | 53.65 | 62.89 |

## C=256,000 (actual 256,001)

**Decode tok/s median**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 55.78 | 41.48 | 53.59 |
| low | 54.87 | 39.98 | 51.04 |
| medium | 55.70 | 40.09 | 50.54 |
| xhigh | 54.90 | 41.23 | 45.95 |

**MTP acceptance median (%)**

| Reasoning | P1 | P2 | P3 |
|---|---:|---:|---:|
| off | 85.64 | 49.47 | 80.66 |
| low | 81.82 | 47.94 | 74.61 |
| medium | 84.46 | 47.91 | 73.15 |
| xhigh | 84.68 | 51.27 | 61.45 |

## Context-ingest curve — MTP-on

| C target | actual interval start–end | net-new tokens | chunks | backend prompt tok/s | net-new/prompt s | net-new/wall s | wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6,144 | 0–6,145 | 6,145 | 4 | 1287.7 | 797.3 | 791.4 | 7.8 |
| 12,288 | 6,145–12,289 | 6,144 | 4 | 1009.8 | 953.5 | 941.7 | 6.5 |
| 28,672 | 12,289–28,673 | 16,384 | 6 | 909.5 | 520.6 | 517.6 | 31.7 |
| 62,464 | 28,673–62,465 | 33,792 | 12 | 595.9 | 317.4 | 315.5 | 107.1 |
| 78,848 | 62,465–78,849 | 16,384 | 8 | 569.3 | 297.6 | 294.3 | 55.7 |
| 88,064 | 78,849–88,065 | 9,216 | 3 | 607.6 | 304.5 | 301.8 | 30.5 |
| 100,352 | 88,065–100,353 | 12,288 | 4 | 592.6 | 296.9 | 294.3 | 41.7 |
| 130,048 | 100,353–130,049 | 29,696 | 11 | 560.7 | 281.0 | 277.6 | 107.0 |
| 196,608 | 130,049–196,609 | 66,560 | 27 | 513.9 | 257.6 | 253.4 | 262.7 |
| 256,000 | 196,609–256,001 | 59,392 | 30 | 443.0 | 222.2 | 216.8 | 273.9 |

### Resources and artifacts

- MTP-on peak VRAM: 15723 MiB; sampled process VmHWM: 13986 MiB; minimum system available RAM: 35409 MiB.
- Full MTP-on aggregate: `/srv/ai/paged-kv/results/forward/final-after-packed-repair-20261103-v2/FULL_MTP_ON_REPORT.json` (SHA-256 `5590b1a0e0943dc4ee2b808b7cca56f4f5af6060cb7528a17ed78270019350ef`, 223,927 bytes). Compact JSON links the same aggregate.
- MTP-off: pending (0/120 prompt-mode pairs); no prior-generation rows are substituted.

Prefill chunk-size/replay variance is explained in [the investigation](../engineering-notes/prefill-chunk-variance-20261103.md). The 256K decode row is complete; MTP-off remains paused pending the prefill review.
