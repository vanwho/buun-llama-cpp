# ACO full curves — 2026-11-04

Current run observations from the frozen campaign build. Warmups are excluded from decode metrics. A cell is complete with three passing measured repeats; incomplete cells remain visible in JSON. Full raw SSE/request references and hashes are in JSON; payloads are not embedded.

- Compiled build label: `8a79008b5 maskrepair`; per-row binary and loaded-DSO SHA-256 values are in JSON.
- Planned: 22 checkpoint rows; 264 mode/prompt cells; 792 measured requests + 264 warmups = 1056 requests.
- Discovered: 5/22 rows (4 complete); 55/60 discovered cells complete; 165/165 observed passing measured repeats complete/observed, out of 792 planned.
- Variant roots discovered: 1 (`/srv/ai/paged-kv/results/candidate-full256k-aco-mtp-on-20261104T105341378940038Z`); checkpoints are rows within a variant root.
- Baseline index: `/srv/ai/paged-kv/results/aco-fullcurves-20261104-baseline-index.json`. Historical comparison requires same model/geometry/MTP/nominal checkpoint/mode/prompt; actual C may differ by up to 8 framing tokens, which is labeled. A complete cell replaces history when median or max decode tok/s is higher; both deltas and the full selected distribution are retained.
- These are observations; differences are not causal or significance claims.

## MTP ON

Rows discovered: 5; complete: 4. Frozen build/model identity and exact refs are in JSON.
Variant-root sampled peaks: GPU 15657.00 MiB; server RAM HWM 13.66 GiB; server swap 0.00 GiB; system swap used 2.15 GiB (sampled wrapper-wide; JSON has fields and source).

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
| 78,848 / 78,849 | medium | 69.49 | — (0/3) | — (0/3) |
| 78,848 / 78,849 | xhigh | — (0/3) | — (0/3) | — (0/3) |

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
| 78,848 / 78,849 | medium | 69.72 | — (0/3) | — (0/3) |
| 78,848 / 78,849 | xhigh | — (0/3) | — (0/3) | — (0/3) |

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
| 78,848 / 78,849 | medium | 87.87 | — (0/3) | — (0/3) |
| 78,848 / 78,849 | xhigh | — (0/3) | — (0/3) | — (0/3) |

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

Append lineage and full raw/request paths plus hashes are stored with each checkpoint in JSON.

## Historical selection

See the compact [best historical report](.wiretail/execution/evidence/ACO_BEST_RESULTS_20261104.md). Current comparable cells promoted: 33 (30 improved both, 1 median-only, 2 max-only). Non-promoted comparable cells retained prior best: 27. Each comparison records both deltas, old/new actual C, source refs, and match note in JSON.

