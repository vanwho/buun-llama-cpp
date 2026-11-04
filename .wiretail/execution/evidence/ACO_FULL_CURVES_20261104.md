# ACO full curves — 2026-11-04

Current run observations from the frozen campaign build. Warmups are excluded from decode metrics. A cell is complete with three passing measured repeats; incomplete cells remain visible in JSON. Full raw SSE/request references and hashes are in JSON; payloads are not embedded.

- Compiled build label: `8a79008b5 maskrepair`; per-row binary and loaded-DSO SHA-256 values are in JSON.
- Planned: 22 checkpoint rows; 264 mode/prompt cells; 792 measured requests + 264 warmups = 1056 requests.
- Discovered: 9/22 rows (9 complete); 108/108 discovered cells complete; 324/324 observed passing measured repeats complete/observed, out of 792 planned.
- Variant roots discovered: 1 (`/srv/ai/paged-kv/results/candidate-full256k-aco-mtp-on-20261104T105341378940038Z`); checkpoints are rows within a variant root.
- Baseline index: `/srv/ai/paged-kv/results/aco-fullcurves-20261104-baseline-index.json`. Historical comparison requires same model/geometry/MTP/nominal checkpoint/mode/prompt; actual C may differ by up to 8 framing tokens, which is labeled. A complete cell replaces history when median or max decode tok/s is higher; both deltas and the full selected distribution are retained.
- These are observations; differences are not causal or significance claims.

## MTP ON

Rows discovered: 9; complete: 9. Frozen build/model identity and exact refs are in JSON.
Variant-root sampled peaks: GPU 15713.00 MiB; server RAM HWM 13.66 GiB; server swap 0.52 GiB; system swap used 3.40 GiB (sampled wrapper-wide; JSON has fields and source).

### Decode median (tok/s)

| Context / actual C | Mode | P1 | P2 | P3 |
|---:|---|---:|---:|---:|
| 6,144 / 6,144 | off | 102.42 | 81.97 | 92.69 |
| 6,144 / 6,144 | low | 100.11 | 75.56 | 92.64 |
| 6,144 / 6,144 | medium | 99.85 | 80.45 | 90.73 |
| 6,144 / 6,144 | xhigh | 99.90 | 76.09 | 80.00 |
| 12,288 / 12,288 | off | 101.63 | 79.77 | 90.69 |
| 12,288 / 12,288 | low | 95.20 | 80.11 | 88.83 |
| 12,288 / 12,288 | medium | 92.32 | 77.53 | 89.79 |
| 12,288 / 12,288 | xhigh | 93.50 | 74.29 | 77.66 |
| 28,672 / 28,673 | off | 92.17 | 74.00 | 81.16 |
| 28,672 / 28,673 | low | 92.01 | 68.64 | 85.44 |
| 28,672 / 28,673 | medium | 88.49 | 73.45 | 82.25 |
| 28,672 / 28,673 | xhigh | 87.97 | 68.52 | 79.64 |
| 62,464 / 62,465 | off | 79.26 | 60.64 | 70.19 |
| 62,464 / 62,465 | low | 75.86 | 58.43 | 73.95 |
| 62,464 / 62,465 | medium | 78.29 | 66.26 | 70.71 |
| 62,464 / 62,465 | xhigh | 75.31 | 55.37 | 64.34 |
| 78,848 / 78,849 | off | 73.93 | 55.98 | 66.45 |
| 78,848 / 78,849 | low | 70.39 | 53.05 | 68.47 |
| 78,848 / 78,849 | medium | 69.49 | 56.10 | 61.54 |
| 78,848 / 78,849 | xhigh | 69.66 | 52.95 | 58.82 |
| 88,064 / 88,065 | off | 73.12 | 57.40 | 65.67 |
| 88,064 / 88,065 | low | 71.02 | 56.39 | 67.34 |
| 88,064 / 88,065 | medium | 66.42 | 56.62 | 63.35 |
| 88,064 / 88,065 | xhigh | 68.34 | 54.25 | 53.07 |
| 100,352 / 100,352 | off | 71.83 | 56.72 | 63.60 |
| 100,352 / 100,352 | low | 70.95 | 55.82 | 62.33 |
| 100,352 / 100,352 | medium | 67.02 | 56.98 | 62.71 |
| 100,352 / 100,352 | xhigh | 67.35 | 52.59 | 54.61 |
| 130,048 / 130,049 | off | 69.13 | 53.71 | 63.16 |
| 130,048 / 130,049 | low | 67.16 | 53.09 | 62.43 |
| 130,048 / 130,049 | medium | 66.15 | 53.43 | 61.11 |
| 130,048 / 130,049 | xhigh | 65.77 | 50.45 | 51.65 |
| 196,608 / 196,608 | off | 65.39 | 48.83 | 60.13 |
| 196,608 / 196,608 | low | 62.40 | 48.79 | 56.46 |
| 196,608 / 196,608 | medium | 62.72 | 51.53 | 58.55 |
| 196,608 / 196,608 | xhigh | 61.99 | 45.09 | 48.37 |

### Decode maximum (tok/s)

| Context / actual C | Mode | P1 | P2 | P3 |
|---:|---|---:|---:|---:|
| 6,144 / 6,144 | off | 102.48 | 82.01 | 92.93 |
| 6,144 / 6,144 | low | 100.63 | 75.74 | 92.87 |
| 6,144 / 6,144 | medium | 100.07 | 80.47 | 90.88 |
| 6,144 / 6,144 | xhigh | 100.23 | 76.36 | 80.10 |
| 12,288 / 12,288 | off | 101.72 | 79.85 | 91.34 |
| 12,288 / 12,288 | low | 95.36 | 80.19 | 89.20 |
| 12,288 / 12,288 | medium | 92.85 | 77.84 | 90.16 |
| 12,288 / 12,288 | xhigh | 93.66 | 74.32 | 77.99 |
| 28,672 / 28,673 | off | 92.32 | 74.02 | 81.19 |
| 28,672 / 28,673 | low | 92.43 | 68.82 | 85.60 |
| 28,672 / 28,673 | medium | 88.84 | 73.48 | 83.04 |
| 28,672 / 28,673 | xhigh | 88.19 | 68.60 | 79.76 |
| 62,464 / 62,465 | off | 79.69 | 60.90 | 70.33 |
| 62,464 / 62,465 | low | 75.94 | 58.44 | 74.18 |
| 62,464 / 62,465 | medium | 78.53 | 66.33 | 70.71 |
| 62,464 / 62,465 | xhigh | 75.71 | 57.69 | 64.79 |
| 78,848 / 78,849 | off | 74.20 | 56.14 | 66.69 |
| 78,848 / 78,849 | low | 70.52 | 58.17 | 68.83 |
| 78,848 / 78,849 | medium | 69.72 | 57.07 | 64.40 |
| 78,848 / 78,849 | xhigh | 70.09 | 54.97 | 65.66 |
| 88,064 / 88,065 | off | 73.31 | 57.51 | 66.44 |
| 88,064 / 88,065 | low | 71.19 | 59.37 | 67.89 |
| 88,064 / 88,065 | medium | 67.25 | 57.18 | 65.77 |
| 88,064 / 88,065 | xhigh | 69.94 | 55.18 | 53.84 |
| 100,352 / 100,352 | off | 72.01 | 57.32 | 64.90 |
| 100,352 / 100,352 | low | 71.29 | 56.90 | 62.47 |
| 100,352 / 100,352 | medium | 68.66 | 57.25 | 63.15 |
| 100,352 / 100,352 | xhigh | 67.77 | 53.07 | 54.85 |
| 130,048 / 130,049 | off | 70.61 | 55.31 | 64.14 |
| 130,048 / 130,049 | low | 67.53 | 53.96 | 62.80 |
| 130,048 / 130,049 | medium | 66.27 | 53.77 | 61.43 |
| 130,048 / 130,049 | xhigh | 66.35 | 51.36 | 53.14 |
| 196,608 / 196,608 | off | 66.52 | 50.56 | 60.35 |
| 196,608 / 196,608 | low | 63.05 | 51.82 | 57.00 |
| 196,608 / 196,608 | medium | 62.81 | 52.23 | 58.64 |
| 196,608 / 196,608 | xhigh | 62.35 | 46.68 | 52.42 |

### MTP acceptance median (%)

| Context / actual C | Mode | P1 | P2 | P3 |
|---:|---|---:|---:|---:|
| 6,144 / 6,144 | off | 94.20 | 64.66 | 80.66 |
| 6,144 / 6,144 | low | 91.20 | 56.84 | 80.39 |
| 6,144 / 6,144 | medium | 90.81 | 63.53 | 77.88 |
| 6,144 / 6,144 | xhigh | 91.07 | 57.88 | 62.71 |
| 12,288 / 12,288 | off | 96.69 | 65.75 | 82.39 |
| 12,288 / 12,288 | low | 88.10 | 66.57 | 79.22 |
| 12,288 / 12,288 | medium | 84.15 | 63.25 | 80.39 |
| 12,288 / 12,288 | xhigh | 85.38 | 58.56 | 63.07 |
| 28,672 / 28,673 | off | 93.31 | 64.62 | 76.27 |
| 28,672 / 28,673 | low | 93.20 | 57.31 | 83.00 |
| 28,672 / 28,673 | medium | 87.59 | 64.84 | 78.39 |
| 28,672 / 28,673 | xhigh | 87.36 | 56.85 | 73.91 |
| 62,464 / 62,465 | off | 92.31 | 60.27 | 77.32 |
| 62,464 / 62,465 | low | 87.50 | 55.88 | 83.89 |
| 62,464 / 62,465 | medium | 92.14 | 70.18 | 78.14 |
| 62,464 / 62,465 | xhigh | 87.06 | 51.36 | 67.06 |
| 78,848 / 78,849 | off | 95.62 | 60.56 | 82.67 |
| 78,848 / 78,849 | low | 89.81 | 55.51 | 86.30 |
| 78,848 / 78,849 | medium | 87.87 | 61.80 | 72.84 |
| 78,848 / 78,849 | xhigh | 89.17 | 54.88 | 67.06 |
| 88,064 / 88,065 | off | 95.62 | 64.29 | 82.06 |
| 88,064 / 88,065 | low | 92.92 | 63.71 | 85.37 |
| 88,064 / 88,065 | medium | 83.50 | 64.18 | 77.32 |
| 88,064 / 88,065 | xhigh | 87.19 | 60.00 | 57.57 |
| 100,352 / 100,352 | off | 95.62 | 63.82 | 78.71 |
| 100,352 / 100,352 | low | 92.78 | 62.71 | 75.47 |
| 100,352 / 100,352 | medium | 85.83 | 65.51 | 76.51 |
| 100,352 / 100,352 | xhigh | 86.05 | 56.60 | 60.56 |
| 130,048 / 130,049 | off | 92.50 | 61.27 | 81.52 |
| 130,048 / 130,049 | low | 89.84 | 61.00 | 80.66 |
| 130,048 / 130,049 | medium | 87.89 | 61.62 | 78.14 |
| 130,048 / 130,049 | xhigh | 87.71 | 55.84 | 58.15 |
| 196,608 / 196,608 | off | 94.20 | 58.33 | 84.46 |
| 196,608 / 196,608 | low | 88.94 | 60.11 | 76.51 |
| 196,608 / 196,608 | medium | 89.82 | 65.32 | 80.92 |
| 196,608 / 196,608 | xhigh | 88.19 | 51.60 | 58.58 |

### Incremental prefill append curve

Only repository-context append requests from each campaign `schedule.jsonl` are included. Every append group is listed; canonical question prompt-time requests are excluded. Rates: backend prompt tok/s, net-new prefill tok/s, and HTTP wall tok/s.

| Checkpoint target / actual C | Append # | Chunk | Net-new tokens | Backend tok/s | Net-new tok/s | HTTP wall tok/s | Raw append ref |
|---:|---:|---|---:|---:|---:|---:|---|
| 6,144 / 6144 | 0 | checkpoint_tail | 6144 | 1475.37 | 1475.37 | 1469.77 | `append-0000.sse` |
| 12,288 / 12288 | 1 | checkpoint_tail | 6144 | 1267.83 | 1268.45 | 1263.00 | `append-0001.sse` |
| 28,672 / 28673 | 2 | checkpoint_tail | 16385 | 1006.17 | 1006.35 | 1004.21 | `append-0002.sse` |
| 62,464 / 61456 | 3 | full_group | 32783 | 671.38 | 671.44 | 670.62 | `append-0003.sse` |
| 62,464 / 62465 | 4 | checkpoint_tail | 1009 | 455.50 | 455.50 | 445.07 | `append-0004.sse` |
| 78,848 / 78849 | 5 | checkpoint_tail | 16384 | 888.88 | 889.04 | 885.39 | `append-0005.sse` |
| 88,064 / 88065 | 6 | checkpoint_tail | 9216 | 826.17 | 826.44 | 820.43 | `append-0006.sse` |
| 100,352 / 100352 | 7 | checkpoint_tail | 12287 | 833.87 | 834.07 | 828.88 | `append-0007.sse` |
| 130,048 / 130049 | 8 | checkpoint_tail | 29697 | 849.45 | 849.53 | 846.51 | `append-0008.sse` |
| 196,608 / 162832 | 9 | full_group | 32783 | 815.90 | 815.98 | 812.63 | `append-0009.sse` |
| 196,608 / 195615 | 10 | full_group | 32783 | 775.82 | 775.82 | 772.41 | `append-0010.sse` |
| 196,608 / 196608 | 11 | checkpoint_tail | 993 | 346.02 | 174.06 | 169.03 | `append-0011.sse` |

Append lineage and full raw/request paths plus hashes are stored with each checkpoint in JSON.

## Historical selection

See the compact [best historical report](.wiretail/execution/evidence/ACO_BEST_RESULTS_20261104.md). Current comparable cells promoted: 82 (75 improved both, 4 median-only, 3 max-only). Non-promoted comparable cells retained prior best: 26. Each comparison records both deltas, old/new actual C, source refs, and match note in JSON.

