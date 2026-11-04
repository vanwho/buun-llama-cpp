# ACO full curves — 2026-11-04

Current run observations from the frozen campaign build. Warmups are excluded from decode metrics. Each measured request has a 400-token output cap; completion-token counts below are the actual usage counts, with the terminal `finish_reason` shown separately (`length` means the cap was reached; `stop` means generation ended on a stop/EOS condition). A cell is complete with three passing measured repeats; incomplete cells remain visible in JSON. Full raw SSE/request references and hashes are in JSON; payloads are not embedded.

- Compiled build label: `8a79008b5 maskrepair`; per-row binary and loaded-DSO SHA-256 values are in JSON.
- Planned: 22 checkpoint rows; 264 mode/prompt cells; 792 measured requests + 264 warmups = 1056 requests.
- Discovered: 12/22 rows (11 complete); 140/144 discovered cells complete; 420/421 observed passing measured repeats complete/observed, out of 792 planned.
- Variant roots discovered: 2 (`/srv/ai/paged-kv/results/candidate-full256k-aco-mtp-off-20261104T115523300080953Z`, `/srv/ai/paged-kv/results/candidate-full256k-aco-mtp-on-20261104T105341378940038Z`); checkpoints are rows within a variant root.
- Baseline index: `/srv/ai/paged-kv/results/aco-fullcurves-20261104-baseline-index.json`. Historical comparison requires same model/geometry/MTP/nominal checkpoint/mode/prompt; actual C may differ by up to 8 framing tokens, which is labeled. A complete cell replaces history when median or max decode tok/s is higher; both deltas and the full selected distribution are retained.
- These are observations; differences are not causal or significance claims.

## MTP ON

Rows discovered: 11; complete: 11. Frozen build/model identity and exact refs are in JSON.
Variant-root sampled peaks: GPU 15713.00 MiB; server RAM HWM 13.66 GiB; server swap 0.52 GiB; system swap used 3.43 GiB (sampled wrapper-wide; JSON has fields and source).

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
| 256,000 / 256,001 | off | 60.81 | 48.41 | 55.30 |
| 256,000 / 256,001 | low | 59.09 | 41.94 | 53.16 |
| 256,000 / 256,001 | medium | 58.53 | 47.86 | 53.45 |
| 256,000 / 256,001 | xhigh | 57.86 | 42.98 | 45.41 |
| 261,120 / 261,121 | off | 60.83 | 46.04 | 54.26 |
| 261,120 / 261,121 | low | 58.22 | 45.78 | 55.30 |
| 261,120 / 261,121 | medium | 57.60 | 45.27 | 50.54 |
| 261,120 / 261,121 | xhigh | 57.75 | 44.03 | 44.83 |

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
| 256,000 / 256,001 | off | 61.13 | 48.83 | 55.35 |
| 256,000 / 256,001 | low | 60.61 | 43.08 | 53.25 |
| 256,000 / 256,001 | medium | 59.30 | 48.28 | 55.07 |
| 256,000 / 256,001 | xhigh | 59.06 | 43.26 | 47.08 |
| 261,120 / 261,121 | off | 61.23 | 46.46 | 54.29 |
| 261,120 / 261,121 | low | 58.79 | 45.82 | 56.73 |
| 261,120 / 261,121 | medium | 58.03 | 45.65 | 51.39 |
| 261,120 / 261,121 | xhigh | 58.03 | 44.65 | 49.30 |

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
| 256,000 / 256,001 | off | 93.86 | 65.32 | 82.39 |
| 256,000 / 256,001 | low | 91.23 | 51.97 | 77.07 |
| 256,000 / 256,001 | medium | 89.51 | 65.70 | 78.39 |
| 256,000 / 256,001 | xhigh | 87.90 | 53.57 | 60.11 |
| 261,120 / 261,121 | off | 93.85 | 60.32 | 80.39 |
| 261,120 / 261,121 | low | 90.00 | 61.34 | 82.39 |
| 261,120 / 261,121 | medium | 88.24 | 60.83 | 72.09 |
| 261,120 / 261,121 | xhigh | 88.66 | 56.50 | 59.62 |

### Actual completion tokens and terminal reason (400-token cap)

Each cell shows the three measured usage completion-token counts in repetition order, followed by the corresponding SSE finish reasons. `stop` is a stop/EOS termination; `length` indicates the output cap ended generation. Counts are actual, not the configured cap.

| Context / actual C | Mode | P1 actual tokens (repeats; reason) | P2 actual tokens (repeats; reason) | P3 actual tokens (repeats; reason) |
|---:|---|---|---|---|
| 6,144 / 6,144 | off | 400/400/400 (length/length/length) | 134/134/134 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 6,144 / 6,144 | low | 306/306/306 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 6,144 / 6,144 | medium | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 6,144 / 6,144 | xhigh | 238/238/238 (stop/stop/stop) | 315/315/315 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 12,288 / 12,288 | off | 400/400/400 (length/length/length) | 169/169/169 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 12,288 / 12,288 | low | 291/291/291 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 12,288 / 12,288 | medium | 331/331/331 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 12,288 / 12,288 | xhigh | 353/353/353 (stop/stop/stop) | 317/317/317 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 28,672 / 28,673 | off | 365/365/365 (stop/stop/stop) | 150/150/150 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 28,672 / 28,673 | low | 296/296/296 (stop/stop/stop) | 280/280/280 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 28,672 / 28,673 | medium | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 28,672 / 28,673 | xhigh | 240/240/240 (stop/stop/stop) | 266/266/266 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 62,464 / 62,465 | off | 186/186/186 (stop/stop/stop) | 161/161/161 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 62,464 / 62,465 | low | 298/298/298 (stop/stop/stop) | 289/289/289 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 62,464 / 62,465 | medium | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 62,464 / 62,465 | xhigh | 233/233/233 (stop/stop/stop) | 223/234/223 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 78,848 / 78,849 | off | 400/400/400 (length/length/length) | 186/169/158 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 78,848 / 78,849 | low | 306/303/302 (stop/stop/stop) | 250/263/315 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 78,848 / 78,849 | medium | 400/400/376 (length/length/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 78,848 / 78,849 | xhigh | 236/330/334 (stop/stop/stop) | 173/169/266 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 88,064 / 88,065 | off | 400/400/400 (length/length/length) | 159/169/161 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 88,064 / 88,065 | low | 317/324/324 (stop/stop/stop) | 400/356/400 (length/stop/length) | 400/400/400 (length/length/length) |
| 88,064 / 88,065 | medium | 400/276/304 (length/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 88,064 / 88,065 | xhigh | 332/332/242 (stop/stop/stop) | 263/265/264 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 100,352 / 100,352 | off | 400/400/400 (length/length/length) | 167/162/174 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 100,352 / 100,352 | low | 277/277/277 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 100,352 / 100,352 | medium | 328/327/400 (stop/stop/length) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 100,352 / 100,352 | xhigh | 235/242/242 (stop/stop/stop) | 227/227/229 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 130,048 / 130,049 | off | 400/400/400 (length/length/length) | 144/159/166 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 130,048 / 130,049 | low | 345/250/312 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 130,048 / 130,049 | medium | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 130,048 / 130,049 | xhigh | 326/325/243 (stop/stop/stop) | 330/291/255 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 196,608 / 196,608 | off | 400/400/400 (length/length/length) | 130/131/132 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 196,608 / 196,608 | low | 289/289/297 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 196,608 / 196,608 | medium | 400/390/400 (length/stop/length) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 196,608 / 196,608 | xhigh | 351/355/330 (stop/stop/stop) | 255/274/245 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 256,000 / 256,001 | off | 400/400/186 (length/length/stop) | 166/170/144 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 256,000 / 256,001 | low | 380/323/365 (stop/stop/stop) | 259/303/266 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 256,000 / 256,001 | medium | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 256,000 / 256,001 | xhigh | 338/342/400 (stop/stop/length) | 291/255/253 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 261,120 / 261,121 | off | 186/188/188 (stop/stop/stop) | 130/118/140 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 261,120 / 261,121 | low | 295/295/295 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 261,120 / 261,121 | medium | 400/377/383 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 261,120 / 261,121 | xhigh | 347/330/330 (stop/stop/stop) | 263/261/264 (stop/stop/stop) | 400/400/400 (length/length/length) |

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
| 256,000 / 229391 | 12 | full_group | 32783 | 742.79 | 742.86 | 739.49 | `append-0012.sse` |
| 256,000 / 256001 | 13 | checkpoint_tail | 26610 | 706.83 | 706.83 | 702.91 | `append-0013.sse` |
| 261,120 / 261121 | 14 | checkpoint_tail | 5120 | 605.26 | 605.62 | 590.53 | `append-0014.sse` |

Token weighted across these 15 append groups: backend 784.72 tok/s; net-new 781.88 tok/s (backend prompt time); HTTP wall 777.90 tok/s. Totals: 261121 net-new / 262072 backend tokens.
Append lineage and full raw/request paths plus hashes are stored with each checkpoint in JSON.

## MTP OFF

Rows discovered: 1; complete: 0. Frozen build/model identity and exact refs are in JSON.
Variant-root sampled peaks: GPU 14589.00 MiB; server RAM HWM 13.66 GiB; server swap 0.00 GiB; system swap used 2.89 GiB (sampled wrapper-wide; JSON has fields and source).

### Decode median (tok/s)

| Context / actual C | Mode | P1 | P2 | P3 |
|---:|---|---:|---:|---:|
| 6,144 / 6,144 | off | 49.04 | 49.16 | 49.01 |
| 6,144 / 6,144 | low | 48.95 | 49.00 | 49.00 |
| 6,144 / 6,144 | medium | 49.00 | 49.00 | — (1/3) |
| 6,144 / 6,144 | xhigh | — (0/3) | — (0/3) | — (0/3) |

### Decode maximum (tok/s)

| Context / actual C | Mode | P1 | P2 | P3 |
|---:|---|---:|---:|---:|
| 6,144 / 6,144 | off | 49.05 | 49.17 | 49.06 |
| 6,144 / 6,144 | low | 48.98 | 49.02 | 49.05 |
| 6,144 / 6,144 | medium | 49.04 | 49.03 | — (1/3) |
| 6,144 / 6,144 | xhigh | — (0/3) | — (0/3) | — (0/3) |

### MTP acceptance median (%) — N/A

| Context / actual C | Mode | P1 | P2 | P3 |
|---:|---|---:|---:|---:|
| 6,144 / 6,144 | off | N/A | N/A | N/A |
| 6,144 / 6,144 | low | N/A | N/A | N/A |
| 6,144 / 6,144 | medium | N/A | N/A | N/A |
| 6,144 / 6,144 | xhigh | N/A | N/A | N/A |

### Actual completion tokens and terminal reason (400-token cap)

Each cell shows the three measured usage completion-token counts in repetition order, followed by the corresponding SSE finish reasons. `stop` is a stop/EOS termination; `length` indicates the output cap ended generation. Counts are actual, not the configured cap.

| Context / actual C | Mode | P1 actual tokens (repeats; reason) | P2 actual tokens (repeats; reason) | P3 actual tokens (repeats; reason) |
|---:|---|---|---|---|
| 6,144 / 6,144 | off | 400/400/400 (length/length/length) | 168/168/168 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 6,144 / 6,144 | low | 313/313/313 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 6,144 / 6,144 | medium | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) | 400 (length) |
| 6,144 / 6,144 | xhigh | — | — | — |

### Incremental prefill append curve

Only repository-context append requests from each campaign `schedule.jsonl` are included. Every append group is listed; canonical question prompt-time requests are excluded. Rates: backend prompt tok/s, net-new prefill tok/s, and HTTP wall tok/s.

| Checkpoint target / actual C | Append # | Chunk | Net-new tokens | Backend tok/s | Net-new tok/s | HTTP wall tok/s | Raw append ref |
|---:|---:|---|---:|---:|---:|---:|---|
| 6,144 / 6144 | 0 | checkpoint_tail | 6144 | 1674.96 | 1674.96 | 1666.11 | `append-0000.sse` |

Token weighted across these 1 append groups: backend 1674.96 tok/s; net-new 1674.96 tok/s (backend prompt time); HTTP wall 1666.11 tok/s. Totals: 6144 net-new / 6144 backend tokens.
Append lineage and full raw/request paths plus hashes are stored with each checkpoint in JSON.

## Historical selection

See the compact [best historical report](.wiretail/execution/evidence/ACO_BEST_RESULTS_20261104.md). Current comparable cells promoted: 95 (86 improved both, 5 median-only, 4 max-only). Non-promoted comparable cells retained prior best: 28. Each comparison records both deltas, old/new actual C, source refs, and match note in JSON.

