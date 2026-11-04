# ACO best historical cells — 2026-11-04

Cell-wise selection across history. Each cell preserves its complete repeat distribution and raw/request references in the JSON report. A current complete cell replaces history when median or max decode rate is higher. Framing differences up to 8 actual tokens are labeled in lineage; current and historical observations do not imply causal comparison.

## MTP ON

| Context / actual C | Mode | P1 median/max [source] | P2 median/max [source] | P3 median/max [source] |
|---:|---|---:|---:|---:|
| 6,144 / 6145 | off | 102.42/102.48 [new] | 81.97/82.01 [new] | 96.82/96.93 [prior] |
| 6,144 / 6144 | low | 100.11/100.63 [new] | 75.56/75.74 [new] | 92.64/92.87 [new] |
| 6,144 / 6144 | medium | 99.85/100.07 [new] | 80.45/80.47 [new] | 90.73/90.88 [new] |
| 6,144 / 6145 | xhigh | 99.90/100.23 [new] | 76.09/76.36 [new] | 84.18/84.33 [prior] |
| 12,288 / 12289 | off | 101.63/101.72 [new] | 82.67/82.81 [prior] | 90.69/91.34 [new] |
| 12,288 / 12289 | low | 96.24/96.46 [prior] | 80.11/80.19 [new] | 91.19/91.20 [prior] |
| 12,288 / 12289 | medium | 93.44/93.94 [prior] | 77.53/77.84 [new] | 89.79/90.16 [new] |
| 12,288 / 12289 | xhigh | 95.67/95.90 [prior] | 77.77/77.92 [prior] | 85.41/85.58 [prior] |
| 28,672 / 28673 | off | 92.17/92.32 [new] | 75.14/75.52 [prior] | 82.06/82.16 [prior] |
| 28,672 / 28673 | low | 92.01/92.43 [new] | 70.44/70.56 [prior] | 86.63/86.77 [prior] |
| 28,672 / 28673 | medium | 90.34/90.45 [prior] | 73.45/73.48 [new] | 82.25/83.04 [new] |
| 28,672 / 28673 | xhigh | 93.45/94.00 [prior] | 68.52/68.60 [new] | 79.64/79.76 [new] |
| 62,464 / 62465 | off | 79.26/79.69 [new] | 60.64/60.90 [new] | 74.42/74.63 [prior] |
| 62,464 / 62465 | low | 76.25/76.29 [prior] | 58.43/58.44 [new] | 74.97/75.03 [prior] |
| 62,464 / 62465 | medium | 78.29/78.53 [new] | 66.26/66.33 [new] | 71.01/71.06 [prior] |
| 62,464 / 62465 | xhigh | 75.31/75.71 [new] | 55.37/57.69 [new] | 64.34/64.79 [new] |
| 78,848 / 78849 | off | 73.93/74.20 [new] | 56.65/59.53 [prior] | 68.08/68.53 [prior] |
| 78,848 / 78849 | low | 70.39/70.52 [new] | 53.05/58.17 [new] | 68.47/68.83 [new] |
| 78,848 / 78849 | medium | 70.26/70.30 [prior] | 52.10/55.20 [prior] | 60.05/60.63 [prior] |
| 78,848 / 78849 | xhigh | 65.45/66.24 [prior] | 51.90/52.09 [prior] | 53.52/60.18 [prior] |
| 88,064 / 88065 | off | 69.44/69.50 [prior] | 50.46/51.17 [prior] | 65.13/66.81 [prior] |
| 88,064 / 88065 | low | 66.06/66.53 [prior] | 45.98/48.35 [prior] | 64.82/66.31 [prior] |
| 88,064 / 88065 | medium | 66.02/66.84 [prior] | 51.52/52.50 [prior] | 61.01/61.35 [prior] |
| 88,064 / 88065 | xhigh | 65.06/65.13 [prior] | 48.97/49.16 [prior] | 54.68/59.38 [prior] |
| 100,352 / 100353 | off | 67.71/70.34 [prior] | 49.03/51.34 [prior] | 63.25/64.00 [prior] |
| 100,352 / 100353 | low | 67.04/67.50 [prior] | 46.66/49.01 [prior] | 61.71/63.66 [prior] |
| 100,352 / 100353 | medium | 68.62/69.14 [prior] | 51.93/52.35 [prior] | 61.50/61.68 [prior] |
| 100,352 / 100353 | xhigh | 64.55/65.80 [prior] | 47.35/47.61 [prior] | 54.11/54.19 [prior] |
| 130,048 / 130049 | off | 65.01/66.65 [prior] | 49.22/51.22 [prior] | 62.78/62.93 [prior] |
| 130,048 / 130049 | low | 63.00/64.15 [prior] | 45.59/48.52 [prior] | 60.30/60.36 [prior] |
| 130,048 / 130049 | medium | 66.42/66.45 [prior] | 49.69/49.93 [prior] | 57.62/60.72 [prior] |
| 130,048 / 130049 | xhigh | 65.22/65.31 [prior] | 44.83/52.35 [prior] | 48.12/50.28 [prior] |
| 196,608 / 196609 | off | 61.45/62.09 [prior] | 45.95/47.26 [prior] | 56.48/56.76 [prior] |
| 196,608 / 196609 | low | 58.73/58.76 [prior] | 41.74/41.86 [prior] | 55.29/55.54 [prior] |
| 196,608 / 196609 | medium | 58.20/58.51 [prior] | 44.31/45.45 [prior] | 48.90/54.46 [prior] |
| 196,608 / 196609 | xhigh | 52.59/53.99 [prior] | 45.34/45.84 [prior] | 48.95/55.15 [prior] |
| 256,000 / 256001 | off | 55.78/58.28 [prior] | 41.48/42.27 [prior] | 53.59/53.76 [prior] |
| 256,000 / 256001 | low | 54.87/55.17 [prior] | 39.98/40.29 [prior] | 51.04/51.11 [prior] |
| 256,000 / 256001 | medium | 55.70/56.17 [prior] | 40.09/43.12 [prior] | 50.54/50.86 [prior] |
| 256,000 / 256001 | xhigh | 54.90/55.63 [prior] | 41.23/42.78 [prior] | 45.95/46.09 [prior] |
| 261,120 / 261121 | low | 59.06/60.84 [prior] | 44.45/47.67 [prior] | 55.49/57.47 [prior] |

Source tags are compact; lineage, raw file paths, request hashes, and all repeat values are in the JSON. Promoted cells retain both old and new comparison values in `comparison_to_prior`.
