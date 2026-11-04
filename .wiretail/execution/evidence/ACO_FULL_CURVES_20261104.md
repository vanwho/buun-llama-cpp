# ACO full curves — 2026-11-04

Current run observations from the frozen campaign build. Warmups are excluded from decode metrics. Each measured request has a 400-token output cap; completion-token counts below are the actual usage counts, with terminal `finish_reason` shown separately (`length` means the cap was reached; `stop` means generation ended on a stop/EOS condition). Decode and MTP acceptance min/median/max use the same three measured repeats per cell. A cell is complete with three passing measured repeats; incomplete cells remain visible in JSON. Full raw SSE/request references and hashes are in JSON; payloads are not embedded.

**Status: COMPLETE** — 22/22 rows, 264/264 cells, 792/792 measured requests, and 264/264 warmups.

- Compiled build label: `8a79008b5 maskrepair`; per-row binary and loaded-DSO SHA-256 values are in JSON.
- Planned: 22 checkpoint rows; 264 mode/prompt cells; 792 measured requests + 264 warmups = 1056 benchmark requests (48 expected cases/row derived from campaign files).
- File-derived requests: 1056 benchmark records (792 measured + 264 warmups), 30 repository append requests, 1086 HTTP requests total.
- Discovered: 22/22 rows (22 complete); 264/264 discovered cells complete; 792/792 observed passing measured repeats complete/observed, out of 792 planned.
- Variant roots discovered: 2 (`/srv/ai/paged-kv/results/candidate-full256k-aco-mtp-off-20261104T115523300080953Z`, `/srv/ai/paged-kv/results/candidate-full256k-aco-mtp-on-20261104T105341378940038Z`); checkpoints are rows within a variant root.
- Baseline index: `/srv/ai/paged-kv/results/aco-fullcurves-20261104-baseline-index.json`. Historical comparison requires same model/geometry/MTP/nominal checkpoint/mode/prompt; actual C may differ by up to 8 framing tokens, which is labeled. A complete cell replaces history when median or max decode tok/s is higher; both deltas and the full selected distribution are retained.
- These are observations; differences are not causal or significance claims.

## MTP ON

Rows discovered: 11; complete: 11. Frozen build/model identity and exact refs are in JSON.
Variant-root sampled peaks: GPU 15713.00 MiB; server RAM HWM 13.66 GiB; server swap 0.52 GiB; system swap used 3.43 GiB (sampled wrapper-wide; JSON has fields and source).

### Decode tok/s (min / median / max)

| Context / actual C | Mode | P1 | P2 | P3 |
|---:|---|---:|---:|---:|
| 6,144 / 6,144 | off | 102.23/102.42/102.48 | 81.87/81.97/82.01 | 92.54/92.69/92.93 |
| 6,144 / 6,144 | low | 99.93/100.11/100.63 | 75.45/75.56/75.74 | 92.55/92.64/92.87 |
| 6,144 / 6,144 | medium | 99.69/99.85/100.07 | 80.42/80.45/80.47 | 90.54/90.73/90.88 |
| 6,144 / 6,144 | xhigh | 99.82/99.90/100.23 | 76.03/76.09/76.36 | 79.98/80.00/80.10 |
| 12,288 / 12,288 | off | 101.53/101.63/101.72 | 79.36/79.77/79.85 | 90.48/90.69/91.34 |
| 12,288 / 12,288 | low | 95.05/95.20/95.36 | 79.84/80.11/80.19 | 88.81/88.83/89.20 |
| 12,288 / 12,288 | medium | 92.18/92.32/92.85 | 77.53/77.53/77.84 | 89.73/89.79/90.16 |
| 12,288 / 12,288 | xhigh | 93.17/93.50/93.66 | 74.18/74.29/74.32 | 77.64/77.66/77.99 |
| 28,672 / 28,673 | off | 92.07/92.17/92.32 | 74.00/74.00/74.02 | 81.05/81.16/81.19 |
| 28,672 / 28,673 | low | 91.89/92.01/92.43 | 68.50/68.64/68.82 | 85.33/85.44/85.60 |
| 28,672 / 28,673 | medium | 88.36/88.49/88.84 | 73.27/73.45/73.48 | 82.07/82.25/83.04 |
| 28,672 / 28,673 | xhigh | 87.62/87.97/88.19 | 68.29/68.52/68.60 | 79.51/79.64/79.76 |
| 62,464 / 62,465 | off | 79.22/79.26/79.69 | 60.60/60.64/60.90 | 69.90/70.19/70.33 |
| 62,464 / 62,465 | low | 75.83/75.86/75.94 | 58.31/58.43/58.44 | 73.79/73.95/74.18 |
| 62,464 / 62,465 | medium | 78.24/78.29/78.53 | 66.12/66.26/66.33 | 70.38/70.71/70.71 |
| 62,464 / 62,465 | xhigh | 75.06/75.31/75.71 | 55.05/55.37/57.69 | 64.23/64.34/64.79 |
| 78,848 / 78,849 | off | 73.85/73.93/74.20 | 54.54/55.98/56.14 | 64.27/66.45/66.69 |
| 78,848 / 78,849 | low | 70.35/70.39/70.52 | 52.53/53.05/58.17 | 68.05/68.47/68.83 |
| 78,848 / 78,849 | medium | 68.96/69.49/69.72 | 55.77/56.10/57.07 | 61.38/61.54/64.40 |
| 78,848 / 78,849 | xhigh | 69.43/69.66/70.09 | 52.44/52.95/54.97 | 53.21/58.82/65.66 |
| 88,064 / 88,065 | off | 73.04/73.12/73.31 | 55.43/57.40/57.51 | 63.53/65.67/66.44 |
| 88,064 / 88,065 | low | 68.64/71.02/71.19 | 53.57/56.39/59.37 | 66.61/67.34/67.89 |
| 88,064 / 88,065 | medium | 66.10/66.42/67.25 | 56.11/56.62/57.18 | 62.98/63.35/65.77 |
| 88,064 / 88,065 | xhigh | 68.12/68.34/69.94 | 51.52/54.25/55.18 | 51.65/53.07/53.84 |
| 100,352 / 100,352 | off | 71.04/71.83/72.01 | 55.88/56.72/57.32 | 63.09/63.60/64.90 |
| 100,352 / 100,352 | low | 70.71/70.95/71.29 | 54.80/55.82/56.90 | 61.54/62.33/62.47 |
| 100,352 / 100,352 | medium | 66.34/67.02/68.66 | 56.75/56.98/57.25 | 62.31/62.71/63.15 |
| 100,352 / 100,352 | xhigh | 67.01/67.35/67.77 | 50.21/52.59/53.07 | 50.27/54.61/54.85 |
| 130,048 / 130,049 | off | 69.08/69.13/70.61 | 50.76/53.71/55.31 | 62.57/63.16/64.14 |
| 130,048 / 130,049 | low | 65.47/67.16/67.53 | 52.24/53.09/53.96 | 61.46/62.43/62.80 |
| 130,048 / 130,049 | medium | 65.97/66.15/66.27 | 52.28/53.43/53.77 | 60.32/61.11/61.43 |
| 130,048 / 130,049 | xhigh | 65.63/65.77/66.35 | 49.61/50.45/51.36 | 50.76/51.65/53.14 |
| 196,608 / 196,608 | off | 65.12/65.39/66.52 | 48.46/48.83/50.56 | 59.53/60.13/60.35 |
| 196,608 / 196,608 | low | 62.23/62.40/63.05 | 48.70/48.79/51.82 | 56.04/56.46/57.00 |
| 196,608 / 196,608 | medium | 59.35/62.72/62.81 | 49.34/51.53/52.23 | 57.52/58.55/58.64 |
| 196,608 / 196,608 | xhigh | 61.03/61.99/62.35 | 44.16/45.09/46.68 | 44.97/48.37/52.42 |
| 256,000 / 256,001 | off | 60.62/60.81/61.13 | 48.01/48.41/48.83 | 51.57/55.30/55.35 |
| 256,000 / 256,001 | low | 58.13/59.09/60.61 | 41.60/41.94/43.08 | 52.17/53.16/53.25 |
| 256,000 / 256,001 | medium | 58.19/58.53/59.30 | 46.00/47.86/48.28 | 51.91/53.45/55.07 |
| 256,000 / 256,001 | xhigh | 55.31/57.86/59.06 | 42.61/42.98/43.26 | 43.05/45.41/47.08 |
| 261,120 / 261,121 | off | 60.35/60.83/61.23 | 44.37/46.04/46.46 | 53.00/54.26/54.29 |
| 261,120 / 261,121 | low | 57.86/58.22/58.79 | 45.52/45.78/45.82 | 54.71/55.30/56.73 |
| 261,120 / 261,121 | medium | 57.49/57.60/58.03 | 43.64/45.27/45.65 | 50.00/50.54/51.39 |
| 261,120 / 261,121 | xhigh | 57.30/57.75/58.03 | 43.36/44.03/44.65 | 42.27/44.83/49.30 |

### MTP acceptance % (min / median / max)

| Context / actual C | Mode | P1 | P2 | P3 |
|---:|---|---:|---:|---:|
| 6,144 / 6,144 | off | 94.20/94.20/94.20 | 64.66/64.66/64.66 | 80.66/80.66/80.66 |
| 6,144 / 6,144 | low | 91.20/91.20/91.20 | 56.84/56.84/56.84 | 80.39/80.39/80.39 |
| 6,144 / 6,144 | medium | 90.81/90.81/90.81 | 63.53/63.53/63.53 | 77.88/77.88/77.88 |
| 6,144 / 6,144 | xhigh | 91.07/91.07/91.07 | 57.88/57.88/57.88 | 62.71/62.71/62.71 |
| 12,288 / 12,288 | off | 96.69/96.69/96.69 | 65.75/65.75/65.75 | 82.39/82.39/82.39 |
| 12,288 / 12,288 | low | 88.10/88.10/88.10 | 66.57/66.57/66.57 | 79.22/79.22/79.22 |
| 12,288 / 12,288 | medium | 84.15/84.15/84.15 | 63.25/63.25/63.25 | 80.39/80.39/80.39 |
| 12,288 / 12,288 | xhigh | 85.38/85.38/85.38 | 58.56/58.56/58.56 | 63.07/63.07/63.07 |
| 28,672 / 28,673 | off | 93.31/93.31/93.31 | 64.62/64.62/64.62 | 76.27/76.27/76.27 |
| 28,672 / 28,673 | low | 93.20/93.20/93.20 | 57.31/57.31/57.31 | 83.00/83.00/83.00 |
| 28,672 / 28,673 | medium | 87.59/87.59/87.59 | 64.84/64.84/64.84 | 78.39/78.39/79.22 |
| 28,672 / 28,673 | xhigh | 87.36/87.36/87.36 | 56.85/56.85/56.85 | 73.91/73.91/73.91 |
| 62,464 / 62,465 | off | 92.31/92.31/92.31 | 60.27/60.27/60.27 | 77.32/77.32/77.32 |
| 62,464 / 62,465 | low | 87.50/87.50/87.50 | 55.88/55.88/55.88 | 83.89/83.89/83.89 |
| 62,464 / 62,465 | medium | 92.14/92.14/92.14 | 70.18/70.18/70.18 | 78.14/78.14/78.14 |
| 62,464 / 62,465 | xhigh | 87.06/87.06/87.06 | 51.36/51.36/54.95 | 67.06/67.06/67.75 |
| 78,848 / 78,849 | off | 95.62/95.62/95.62 | 57.69/60.56/60.71 | 77.32/82.67/82.67 |
| 78,848 / 78,849 | low | 89.35/89.81/89.91 | 54.67/55.51/66.37 | 85.71/86.30/86.64 |
| 78,848 / 78,849 | medium | 87.24/87.87/88.54 | 61.00/61.80/63.71 | 72.84/72.84/78.96 |
| 78,848 / 78,849 | xhigh | 88.24/89.17/89.83 | 54.32/54.88/59.50 | 56.12/67.06/80.39 |
| 88,064 / 88,065 | off | 95.62/95.62/95.62 | 61.18/64.29/65.22 | 78.14/82.06/83.28 |
| 88,064 / 88,065 | low | 87.39/92.92/92.92 | 57.88/63.71/69.16 | 83.84/85.37/86.30 |
| 88,064 / 88,065 | medium | 82.89/83.50/84.80 | 63.35/64.18/65.32 | 77.32/77.32/82.94 |
| 88,064 / 88,065 | xhigh | 86.93/87.19/90.68 | 53.94/60.00/61.02 | 54.19/57.57/58.58 |
| 100,352 / 100,352 | off | 95.62/95.62/95.62 | 63.70/63.82/65.00 | 77.07/78.71/81.52 |
| 100,352 / 100,352 | low | 92.78/92.78/94.27 | 60.83/62.71/64.66 | 74.38/75.47/76.51 |
| 100,352 / 100,352 | medium | 84.02/85.83/89.16 | 65.03/65.51/65.99 | 76.27/76.51/77.32 |
| 100,352 / 100,352 | xhigh | 85.39/86.05/86.93 | 51.79/56.60/57.62 | 51.79/60.56/61.00 |
| 130,048 / 130,049 | off | 92.50/92.50/95.62 | 55.15/61.27/65.28 | 80.66/81.52/83.84 |
| 130,048 / 130,049 | low | 86.81/89.84/90.09 | 59.34/61.00/62.89 | 78.39/80.66/80.66 |
| 130,048 / 130,049 | medium | 87.59/87.89/87.89 | 59.62/61.62/62.71 | 75.47/78.14/78.14 |
| 130,048 / 130,049 | xhigh | 87.50/87.71/88.14 | 54.10/55.84/57.84 | 56.99/58.15/61.34 |
| 196,608 / 196,608 | off | 93.86/94.20/96.69 | 57.38/58.33/61.21 | 82.94/84.46/85.08 |
| 196,608 / 196,608 | low | 88.32/88.94/89.81 | 59.18/60.11/65.99 | 74.69/76.51/77.32 |
| 196,608 / 196,608 | medium | 82.12/89.82/89.93 | 60.72/65.32/67.06 | 78.64/80.92/80.92 |
| 196,608 / 196,608 | xhigh | 86.54/88.19/89.83 | 49.19/51.60/55.38 | 52.44/58.58/68.96 |
| 256,000 / 256,001 | off | 92.31/93.86/94.20 | 64.86/65.32/66.90 | 73.37/82.39/83.28 |
| 256,000 / 256,001 | low | 88.93/91.23/93.56 | 51.15/51.97/54.48 | 76.75/77.07/78.64 |
| 256,000 / 256,001 | medium | 88.54/89.51/90.49 | 61.45/65.70/66.67 | 74.14/78.39/83.28 |
| 256,000 / 256,001 | xhigh | 82.67/87.90/90.83 | 53.25/53.57/54.13 | 54.74/60.11/64.18 |
| 261,120 / 261,121 | off | 92.31/93.85/96.88 | 57.27/60.32/61.21 | 79.22/80.39/82.67 |
| 261,120 / 261,121 | low | 90.00/90.00/90.00 | 61.17/61.34/61.97 | 81.79/82.39/86.30 |
| 261,120 / 261,121 | medium | 87.89/88.24/89.42 | 56.27/60.83/61.00 | 71.65/72.09/74.92 |
| 261,120 / 261,121 | xhigh | 87.30/88.66/89.83 | 55.20/56.50/58.33 | 53.51/59.62/69.16 |

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

### Descriptive append-rate ranges by checkpoint

Min/median/max summarize that checkpoint's 1–3 append groups; group sizes and intervals differ, so these are descriptive samples, not independent replicate trials. With n=1 the three values are equal. Repository question-prompt timings are excluded.

| Target C | Append groups n | Backend tok/s min/median/max | Net-new tok/s min/median/max | HTTP wall tok/s min/median/max |
|---:|---:|---:|---:|---:|
| 6,144 | 1 | 1475.37/1475.37/1475.37 | 1475.37/1475.37/1475.37 | 1469.77/1469.77/1469.77 |
| 12,288 | 1 | 1267.83/1267.83/1267.83 | 1268.45/1268.45/1268.45 | 1263.00/1263.00/1263.00 |
| 28,672 | 1 | 1006.17/1006.17/1006.17 | 1006.35/1006.35/1006.35 | 1004.21/1004.21/1004.21 |
| 62,464 | 2 | 455.50/563.44/671.38 | 455.50/563.47/671.44 | 445.07/557.85/670.62 |
| 78,848 | 1 | 888.88/888.88/888.88 | 889.04/889.04/889.04 | 885.39/885.39/885.39 |
| 88,064 | 1 | 826.17/826.17/826.17 | 826.44/826.44/826.44 | 820.43/820.43/820.43 |
| 100,352 | 1 | 833.87/833.87/833.87 | 834.07/834.07/834.07 | 828.88/828.88/828.88 |
| 130,048 | 1 | 849.45/849.45/849.45 | 849.53/849.53/849.53 | 846.51/846.51/846.51 |
| 196,608 | 3 | 346.02/775.82/815.90 | 174.06/775.82/815.98 | 169.03/772.41/812.63 |
| 256,000 | 2 | 706.83/724.81/742.79 | 706.83/724.84/742.86 | 702.91/721.20/739.49 |
| 261,120 | 1 | 605.26/605.26/605.26 | 605.62/605.62/605.62 | 590.53/590.53/590.53 |

Descriptive append-sample min/median/max across all 15 heterogeneous groups: backend 346.02/815.90/1475.37; net-new 174.06/815.98/1475.37; HTTP wall 169.03/812.63/1469.77 tok/s (n is append groups, not independent replicates).
Aggregate throughput (total tokens / total corresponding request seconds) across these append groups: backend 784.72 tok/s; net-new 781.88 tok/s (backend prompt time); HTTP wall 777.90 tok/s. These are aggregate rates, not arithmetic means of per-append rates. Totals: 261121 net-new / 262072 backend tokens.
Append lineage and full raw/request paths plus hashes are stored with each checkpoint in JSON.

## MTP OFF

Rows discovered: 11; complete: 11. Frozen build/model identity and exact refs are in JSON.
Variant-root sampled peaks: GPU 14611.00 MiB; server RAM HWM 13.66 GiB; server swap 0.00 GiB; system swap used 2.89 GiB (sampled wrapper-wide; JSON has fields and source).

### Decode tok/s (min / median / max)

| Context / actual C | Mode | P1 | P2 | P3 |
|---:|---|---:|---:|---:|
| 6,144 / 6,144 | off | 48.97/49.04/49.05 | 49.16/49.16/49.17 | 49.00/49.01/49.06 |
| 6,144 / 6,144 | low | 48.95/48.95/48.98 | 48.99/49.00/49.02 | 48.98/49.00/49.05 |
| 6,144 / 6,144 | medium | 48.99/49.00/49.04 | 48.97/49.00/49.03 | 48.95/48.98/49.01 |
| 6,144 / 6,144 | xhigh | 48.96/48.97/49.00 | 48.73/48.90/48.90 | 48.96/48.96/48.98 |
| 12,288 / 12,288 | off | 48.03/48.04/48.10 | 48.12/48.18/48.20 | 48.05/48.06/48.06 |
| 12,288 / 12,288 | low | 47.98/47.98/48.01 | 48.00/48.00/48.04 | 48.01/48.03/48.06 |
| 12,288 / 12,288 | medium | 48.00/48.01/48.04 | 47.97/48.04/48.07 | 48.02/48.03/48.08 |
| 12,288 / 12,288 | xhigh | 47.94/47.96/47.97 | 47.96/47.98/47.99 | 48.01/48.03/48.05 |
| 28,672 / 28,673 | off | 46.01/46.09/46.11 | 46.21/46.23/46.25 | 46.09/46.10/46.14 |
| 28,672 / 28,673 | low | 45.95/45.96/46.02 | 45.92/45.98/46.03 | 45.99/46.04/46.09 |
| 28,672 / 28,673 | medium | 46.08/46.08/46.10 | 46.08/46.08/46.11 | 46.06/46.09/46.11 |
| 28,672 / 28,673 | xhigh | 45.94/45.95/46.00 | 45.90/46.00/46.05 | 46.02/46.03/46.07 |
| 62,464 / 62,465 | off | 42.29/42.33/42.33 | 42.31/42.32/42.34 | 42.09/42.17/42.21 |
| 62,464 / 62,465 | low | 42.03/42.05/42.08 | 41.79/41.89/41.98 | 42.11/42.12/42.12 |
| 62,464 / 62,465 | medium | 42.16/42.18/42.18 | 42.15/42.18/42.19 | 42.16/42.18/42.20 |
| 62,464 / 62,465 | xhigh | 41.98/41.99/42.00 | 42.04/42.06/42.08 | 42.12/42.14/42.15 |
| 78,848 / 78,849 | off | 37.49/37.51/37.54 | 37.51/37.55/37.56 | 37.50/37.50/37.55 |
| 78,848 / 78,849 | low | 37.43/37.45/37.48 | 37.40/37.46/37.50 | 37.47/37.50/37.52 |
| 78,848 / 78,849 | medium | 37.49/37.50/37.52 | 37.49/37.49/37.50 | 37.47/37.51/37.54 |
| 78,848 / 78,849 | xhigh | 37.38/37.40/37.49 | 37.42/37.47/37.49 | 37.50/37.50/37.52 |
| 88,064 / 88,065 | off | 37.49/37.49/37.53 | 37.56/37.58/37.58 | 37.46/37.47/37.51 |
| 88,064 / 88,065 | low | 37.46/37.46/37.50 | 37.44/37.49/37.52 | 37.46/37.51/37.52 |
| 88,064 / 88,065 | medium | 37.41/37.47/37.54 | 37.47/37.50/37.52 | 37.48/37.51/37.54 |
| 88,064 / 88,065 | xhigh | 37.36/37.39/37.50 | 37.44/37.49/37.58 | 37.49/37.51/37.53 |
| 100,352 / 100,352 | off | 37.46/37.50/37.53 | 37.53/37.56/37.57 | 37.47/37.49/37.54 |
| 100,352 / 100,352 | low | 37.41/37.41/37.48 | 37.47/37.50/37.52 | 37.42/37.49/37.50 |
| 100,352 / 100,352 | medium | 37.47/37.48/37.53 | 37.49/37.49/37.50 | 37.50/37.50/37.51 |
| 100,352 / 100,352 | xhigh | 37.37/37.47/37.48 | 37.36/37.39/37.46 | 37.48/37.50/37.51 |
| 130,048 / 130,049 | off | 37.48/37.50/37.50 | 37.57/37.57/37.57 | 37.47/37.49/37.52 |
| 130,048 / 130,049 | low | 37.42/37.43/37.47 | 37.46/37.50/37.53 | 37.47/37.49/37.50 |
| 130,048 / 130,049 | medium | 37.45/37.47/37.49 | 37.44/37.47/37.50 | 37.46/37.48/37.52 |
| 130,048 / 130,049 | xhigh | 37.34/37.45/37.47 | 37.40/37.41/37.44 | 37.47/37.48/37.50 |
| 196,608 / 196,608 | off | 37.48/37.49/37.52 | 37.53/37.56/37.58 | 37.47/37.48/37.51 |
| 196,608 / 196,608 | low | 37.36/37.39/37.42 | 37.46/37.48/37.48 | 37.46/37.49/37.50 |
| 196,608 / 196,608 | medium | 37.46/37.48/37.50 | 37.43/37.49/37.50 | 37.46/37.47/37.50 |
| 196,608 / 196,608 | xhigh | 37.42/37.44/37.48 | 37.38/37.38/37.45 | 37.46/37.47/37.47 |
| 256,000 / 256,001 | off | 37.42/37.42/37.51 | 37.57/37.57/37.57 | 37.44/37.44/37.45 |
| 256,000 / 256,001 | low | 37.40/37.42/37.45 | 37.25/37.32/37.36 | 37.42/37.46/37.47 |
| 256,000 / 256,001 | medium | 37.41/37.45/37.49 | 37.44/37.45/37.46 | 37.44/37.44/37.45 |
| 256,000 / 256,001 | xhigh | 37.37/37.41/37.45 | 37.28/37.37/37.39 | 37.43/37.46/37.46 |
| 261,120 / 261,121 | off | 37.48/37.55/37.59 | 37.52/37.57/37.59 | 37.44/37.44/37.47 |
| 261,120 / 261,121 | low | 37.37/37.38/37.43 | 37.43/37.43/37.45 | 37.42/37.44/37.45 |
| 261,120 / 261,121 | medium | 37.41/37.41/37.45 | 37.44/37.46/37.46 | 37.42/37.42/37.42 |
| 261,120 / 261,121 | xhigh | 37.38/37.38/37.43 | 37.33/37.36/37.38 | 37.42/37.45/37.48 |

### MTP acceptance % — N/A (MTP disabled)

| Context / actual C | Mode | P1 | P2 | P3 |
|---:|---|---:|---:|---:|
| 6,144 / 6,144 | off | N/A | N/A | N/A |
| 6,144 / 6,144 | low | N/A | N/A | N/A |
| 6,144 / 6,144 | medium | N/A | N/A | N/A |
| 6,144 / 6,144 | xhigh | N/A | N/A | N/A |
| 12,288 / 12,288 | off | N/A | N/A | N/A |
| 12,288 / 12,288 | low | N/A | N/A | N/A |
| 12,288 / 12,288 | medium | N/A | N/A | N/A |
| 12,288 / 12,288 | xhigh | N/A | N/A | N/A |
| 28,672 / 28,673 | off | N/A | N/A | N/A |
| 28,672 / 28,673 | low | N/A | N/A | N/A |
| 28,672 / 28,673 | medium | N/A | N/A | N/A |
| 28,672 / 28,673 | xhigh | N/A | N/A | N/A |
| 62,464 / 62,465 | off | N/A | N/A | N/A |
| 62,464 / 62,465 | low | N/A | N/A | N/A |
| 62,464 / 62,465 | medium | N/A | N/A | N/A |
| 62,464 / 62,465 | xhigh | N/A | N/A | N/A |
| 78,848 / 78,849 | off | N/A | N/A | N/A |
| 78,848 / 78,849 | low | N/A | N/A | N/A |
| 78,848 / 78,849 | medium | N/A | N/A | N/A |
| 78,848 / 78,849 | xhigh | N/A | N/A | N/A |
| 88,064 / 88,065 | off | N/A | N/A | N/A |
| 88,064 / 88,065 | low | N/A | N/A | N/A |
| 88,064 / 88,065 | medium | N/A | N/A | N/A |
| 88,064 / 88,065 | xhigh | N/A | N/A | N/A |
| 100,352 / 100,352 | off | N/A | N/A | N/A |
| 100,352 / 100,352 | low | N/A | N/A | N/A |
| 100,352 / 100,352 | medium | N/A | N/A | N/A |
| 100,352 / 100,352 | xhigh | N/A | N/A | N/A |
| 130,048 / 130,049 | off | N/A | N/A | N/A |
| 130,048 / 130,049 | low | N/A | N/A | N/A |
| 130,048 / 130,049 | medium | N/A | N/A | N/A |
| 130,048 / 130,049 | xhigh | N/A | N/A | N/A |
| 196,608 / 196,608 | off | N/A | N/A | N/A |
| 196,608 / 196,608 | low | N/A | N/A | N/A |
| 196,608 / 196,608 | medium | N/A | N/A | N/A |
| 196,608 / 196,608 | xhigh | N/A | N/A | N/A |
| 256,000 / 256,001 | off | N/A | N/A | N/A |
| 256,000 / 256,001 | low | N/A | N/A | N/A |
| 256,000 / 256,001 | medium | N/A | N/A | N/A |
| 256,000 / 256,001 | xhigh | N/A | N/A | N/A |
| 261,120 / 261,121 | off | N/A | N/A | N/A |
| 261,120 / 261,121 | low | N/A | N/A | N/A |
| 261,120 / 261,121 | medium | N/A | N/A | N/A |
| 261,120 / 261,121 | xhigh | N/A | N/A | N/A |

### Actual completion tokens and terminal reason (400-token cap)

Each cell shows the three measured usage completion-token counts in repetition order, followed by the corresponding SSE finish reasons. `stop` is a stop/EOS termination; `length` indicates the output cap ended generation. Counts are actual, not the configured cap.

| Context / actual C | Mode | P1 actual tokens (repeats; reason) | P2 actual tokens (repeats; reason) | P3 actual tokens (repeats; reason) |
|---:|---|---|---|---|
| 6,144 / 6,144 | off | 400/400/400 (length/length/length) | 168/168/168 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 6,144 / 6,144 | low | 313/313/313 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 6,144 / 6,144 | medium | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 6,144 / 6,144 | xhigh | 345/345/345 (stop/stop/stop) | 279/279/279 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 12,288 / 12,288 | off | 400/400/400 (length/length/length) | 119/119/119 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 12,288 / 12,288 | low | 295/295/295 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 12,288 / 12,288 | medium | 331/331/331 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 12,288 / 12,288 | xhigh | 286/286/286 (stop/stop/stop) | 307/307/307 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 28,672 / 28,673 | off | 365/365/365 (stop/stop/stop) | 163/163/163 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 28,672 / 28,673 | low | 296/296/296 (stop/stop/stop) | 294/294/294 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 28,672 / 28,673 | medium | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 28,672 / 28,673 | xhigh | 240/240/240 (stop/stop/stop) | 293/293/293 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 62,464 / 62,465 | off | 186/186/186 (stop/stop/stop) | 164/164/164 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 62,464 / 62,465 | low | 302/302/302 (stop/stop/stop) | 224/224/224 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 62,464 / 62,465 | medium | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 62,464 / 62,465 | xhigh | 235/235/235 (stop/stop/stop) | 293/293/293 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 78,848 / 78,849 | off | 400/400/400 (length/length/length) | 167/171/171 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 78,848 / 78,849 | low | 306/291/326 (stop/stop/stop) | 317/263/322 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 78,848 / 78,849 | medium | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 78,848 / 78,849 | xhigh | 355/235/236 (stop/stop/stop) | 237/169/261 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 88,064 / 88,065 | off | 400/400/400 (length/length/length) | 157/141/180 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 88,064 / 88,065 | low | 317/318/327 (stop/stop/stop) | 400/291/400 (length/stop/length) | 400/400/400 (length/length/length) |
| 88,064 / 88,065 | medium | 400/277/350 (length/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 88,064 / 88,065 | xhigh | 334/235/242 (stop/stop/stop) | 165/322/277 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 100,352 / 100,352 | off | 400/400/400 (length/length/length) | 159/172/156 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 100,352 / 100,352 | low | 277/277/277 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 100,352 / 100,352 | medium | 400/365/400 (length/stop/length) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 100,352 / 100,352 | xhigh | 330/242/329 (stop/stop/stop) | 274/226/236 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 130,048 / 130,049 | off | 400/400/400 (length/length/length) | 145/161/165 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 130,048 / 130,049 | low | 328/292/292 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 130,048 / 130,049 | medium | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 130,048 / 130,049 | xhigh | 325/243/325 (stop/stop/stop) | 270/279/275 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 196,608 / 196,608 | off | 400/400/400 (length/length/length) | 132/127/149 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 196,608 / 196,608 | low | 301/301/301 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 196,608 / 196,608 | medium | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 196,608 / 196,608 | xhigh | 353/352/321 (stop/stop/stop) | 274/261/254 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 256,000 / 256,001 | off | 400/400/186 (length/length/stop) | 177/146/147 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 256,000 / 256,001 | low | 372/353/353 (stop/stop/stop) | 252/266/239 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 256,000 / 256,001 | medium | 400/400/383 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 256,000 / 256,001 | xhigh | 340/342/333 (stop/stop/stop) | 261/253/297 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 261,120 / 261,121 | off | 188/188/400 (stop/stop/length) | 140/115/115 (stop/stop/stop) | 400/400/400 (length/length/length) |
| 261,120 / 261,121 | low | 295/295/295 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 261,120 / 261,121 | medium | 377/383/383 (stop/stop/stop) | 400/400/400 (length/length/length) | 400/400/400 (length/length/length) |
| 261,120 / 261,121 | xhigh | 330/330/325 (stop/stop/stop) | 299/296/264 (stop/stop/stop) | 400/400/400 (length/length/length) |

### Incremental prefill append curve

Only repository-context append requests from each campaign `schedule.jsonl` are included. Every append group is listed; canonical question prompt-time requests are excluded. Rates: backend prompt tok/s, net-new prefill tok/s, and HTTP wall tok/s.

| Checkpoint target / actual C | Append # | Chunk | Net-new tokens | Backend tok/s | Net-new tok/s | HTTP wall tok/s | Raw append ref |
|---:|---:|---|---:|---:|---:|---:|---|
| 6,144 / 6144 | 0 | checkpoint_tail | 6144 | 1674.96 | 1674.96 | 1666.11 | `append-0000.sse` |
| 12,288 / 12288 | 1 | checkpoint_tail | 6144 | 1405.16 | 1405.85 | 1399.17 | `append-0001.sse` |
| 28,672 / 28673 | 2 | checkpoint_tail | 16385 | 1123.04 | 1123.25 | 1120.64 | `append-0002.sse` |
| 62,464 / 61456 | 3 | full_group | 32783 | 748.98 | 749.05 | 747.99 | `append-0003.sse` |
| 62,464 / 62465 | 4 | checkpoint_tail | 1009 | 507.71 | 507.71 | 492.23 | `append-0004.sse` |
| 78,848 / 78849 | 5 | checkpoint_tail | 16384 | 976.95 | 977.13 | 972.75 | `append-0005.sse` |
| 88,064 / 88065 | 6 | checkpoint_tail | 9216 | 922.42 | 922.72 | 914.92 | `append-0006.sse` |
| 100,352 / 100352 | 7 | checkpoint_tail | 12287 | 945.61 | 945.85 | 938.90 | `append-0007.sse` |
| 130,048 / 130049 | 8 | checkpoint_tail | 29697 | 995.27 | 995.37 | 991.59 | `append-0008.sse` |
| 196,608 / 162832 | 9 | full_group | 32783 | 988.28 | 988.37 | 984.00 | `append-0009.sse` |
| 196,608 / 195615 | 10 | full_group | 32783 | 970.67 | 970.67 | 965.77 | `append-0010.sse` |
| 196,608 / 196608 | 11 | checkpoint_tail | 993 | 422.28 | 212.42 | 204.72 | `append-0011.sse` |
| 256,000 / 229391 | 12 | full_group | 32783 | 965.31 | 965.40 | 959.63 | `append-0012.sse` |
| 256,000 / 256001 | 13 | checkpoint_tail | 26610 | 940.31 | 940.31 | 933.11 | `append-0013.sse` |
| 261,120 / 261121 | 14 | checkpoint_tail | 5120 | 797.38 | 797.85 | 767.40 | `append-0014.sse` |

### Descriptive append-rate ranges by checkpoint

Min/median/max summarize that checkpoint's 1–3 append groups; group sizes and intervals differ, so these are descriptive samples, not independent replicate trials. With n=1 the three values are equal. Repository question-prompt timings are excluded.

| Target C | Append groups n | Backend tok/s min/median/max | Net-new tok/s min/median/max | HTTP wall tok/s min/median/max |
|---:|---:|---:|---:|---:|
| 6,144 | 1 | 1674.96/1674.96/1674.96 | 1674.96/1674.96/1674.96 | 1666.11/1666.11/1666.11 |
| 12,288 | 1 | 1405.16/1405.16/1405.16 | 1405.85/1405.85/1405.85 | 1399.17/1399.17/1399.17 |
| 28,672 | 1 | 1123.04/1123.04/1123.04 | 1123.25/1123.25/1123.25 | 1120.64/1120.64/1120.64 |
| 62,464 | 2 | 507.71/628.35/748.98 | 507.71/628.38/749.05 | 492.23/620.11/747.99 |
| 78,848 | 1 | 976.95/976.95/976.95 | 977.13/977.13/977.13 | 972.75/972.75/972.75 |
| 88,064 | 1 | 922.42/922.42/922.42 | 922.72/922.72/922.72 | 914.92/914.92/914.92 |
| 100,352 | 1 | 945.61/945.61/945.61 | 945.85/945.85/945.85 | 938.90/938.90/938.90 |
| 130,048 | 1 | 995.27/995.27/995.27 | 995.37/995.37/995.37 | 991.59/991.59/991.59 |
| 196,608 | 3 | 422.28/970.67/988.28 | 212.42/970.67/988.37 | 204.72/965.77/984.00 |
| 256,000 | 2 | 940.31/952.81/965.31 | 940.31/952.85/965.40 | 933.11/946.37/959.63 |
| 261,120 | 1 | 797.38/797.38/797.38 | 797.85/797.85/797.85 | 767.40/767.40/767.40 |

Descriptive append-sample min/median/max across all 15 heterogeneous groups: backend 422.28/965.31/1674.96; net-new 212.42/965.40/1674.96; HTTP wall 204.72/959.63/1666.11 tok/s (n is append groups, not independent replicates).
Aggregate throughput (total tokens / total corresponding request seconds) across these append groups: backend 941.85 tok/s; net-new 938.44 tok/s (backend prompt time); HTTP wall 932.60 tok/s. These are aggregate rates, not arithmetic means of per-append rates. Totals: 261121 net-new / 262072 backend tokens.
Append lineage and full raw/request paths plus hashes are stored with each checkpoint in JSON.

## Historical selection

See the compact [best historical report](ACO_BEST_RESULTS_20261104.md). Current comparable cells promoted: 95 (86 improved both, 5 median-only, 4 max-only); 28 comparable prior cells retained. Among promoted cells, min deltas ranged -1.66 to +12.53, median deltas -0.55 to +10.42, and max deltas -3.06 to +11.02 tok/s. Promotion remains based on median or max only. These are cell-selection observations, not significance or causal claims. Each comparison records all three deltas, old/new actual C, source refs, and match note in JSON.


## Memory observation limitation

The MTP-on candidate process recorded a peak `VmSwap` of **530.4 MiB** at the 100,352-token checkpoint (98K), including changes during some measured requests. The MTP-off candidate recorded zero across all 958 samples. Five-second process samples do not identify which allocations were swapped, prove that K/V pages were swapped, or measure swap-in I/O. These rates remain the observed run results, including any associated effects; the MTP-on run is **not certified RAM-only**. This caveat also applies to current-run cells selected as historical bests.

The hashed offline attribution and per-checkpoint sample windows are referenced in the JSON report; raw measurements remain external.

Attribution source SHA-256: `4542c513a1a89dbf8643192c599fbe77e62d943f872c85ebeee3e66a9520cd86`.
