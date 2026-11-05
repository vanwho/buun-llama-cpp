# Comparison tables — audited snapshot, 2026-11-05

This appendix preserves the full min/median/max summary for every quality-eligible cell, plus exact matched ACO/control comparisons, mapped-prefill checkpoints, weighted independent cohorts, and retained resource observations. Nominal C identifies the planned checkpoint; actual C is the observed occupied context. Per-cell source paths and SHA-256 values are in `decode-summary.csv`; raw answer text is not included.

## Decode and MTP acceptance

Rates are tokens/s; acceptance is percent. MTP-off acceptance is N/A. Each row represents one whole three-repeat triplet.

| Family | Nominal C | Actual C range | MTP | Mode | Prompt | Decode min | median | max | Accept min | median | max |
|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| master-gpu-mtp-off | 12288 | 12288 | off | medium | P1 | 48.40 | 48.41 | 48.45 | N/A | N/A | N/A |
| master-gpu-mtp-off | 12288 | 12288 | off | xhigh | P1 | 48.40 | 48.42 | 48.43 | N/A | N/A | N/A |
| master-gpu-mtp-off | 28672 | 28672 | off | off | P1 | 46.63 | 46.65 | 46.66 | N/A | N/A | N/A |
| master-gpu-mtp-off | 28672 | 28672 | off | off | P2 | 46.60 | 46.62 | 46.65 | N/A | N/A | N/A |
| master-gpu-mtp-off | 62464 | 62465 | off | off | P1 | 42.92 | 42.97 | 42.99 | N/A | N/A | N/A |
| master-gpu-mtp-off | 62464 | 62465 | off | off | P2 | 42.95 | 42.95 | 42.97 | N/A | N/A | N/A |
| master-gpu-mtp-off | 62464 | 62465 | off | off | P3 | 42.87 | 42.90 | 42.92 | N/A | N/A | N/A |
| master-gpu-mtp-off | 62464 | 62464 | off | low | P1 | 42.86 | 42.89 | 42.89 | N/A | N/A | N/A |
| master-gpu-mtp-off | 62464 | 62464 | off | low | P2 | 42.86 | 42.88 | 42.92 | N/A | N/A | N/A |
| master-gpu-mtp-off | 62464 | 62465 | off | medium | P1 | 42.87 | 42.87 | 42.93 | N/A | N/A | N/A |
| master-gpu-mtp-off | 62464 | 62465 | off | medium | P2 | 42.89 | 42.90 | 42.92 | N/A | N/A | N/A |
| master-gpu-mtp-off | 62464 | 62465 | off | medium | P3 | 42.87 | 42.88 | 42.93 | N/A | N/A | N/A |
| master-gpu-mtp-off | 62464 | 62464 | off | xhigh | P1 | 42.89 | 42.91 | 42.91 | N/A | N/A | N/A |
| master-gpu-mtp-off | 62464 | 62464 | off | xhigh | P2 | 42.85 | 42.87 | 42.92 | N/A | N/A | N/A |
| master-gpu-mtp-off | 62464 | 62464 | off | xhigh | P3 | 42.88 | 42.89 | 42.91 | N/A | N/A | N/A |
| master-gpu-mtp-off | 78848 | 78849 | off | off | P1 | 41.13 | 41.15 | 41.23 | N/A | N/A | N/A |
| master-gpu-mtp-off | 78848 | 78849 | off | off | P2 | 41.28 | 41.28 | 41.30 | N/A | N/A | N/A |
| master-gpu-mtp-off | 78848 | 78849 | off | off | P3 | 41.17 | 41.20 | 41.23 | N/A | N/A | N/A |
| master-gpu-mtp-off | 78848 | 78849 | off | low | P1 | 41.18 | 41.19 | 41.26 | N/A | N/A | N/A |
| master-gpu-mtp-off | 78848 | 78849 | off | low | P2 | 41.17 | 41.18 | 41.25 | N/A | N/A | N/A |
| master-gpu-mtp-off | 78848 | 78849 | off | low | P3 | 41.21 | 41.21 | 41.23 | N/A | N/A | N/A |
| master-gpu-mtp-off | 78848 | 78849 | off | medium | P1 | 41.17 | 41.17 | 41.26 | N/A | N/A | N/A |
| master-gpu-mtp-off | 78848 | 78849 | off | medium | P2 | 41.15 | 41.19 | 41.23 | N/A | N/A | N/A |
| master-gpu-mtp-off | 78848 | 78849 | off | medium | P3 | 41.16 | 41.17 | 41.20 | N/A | N/A | N/A |
| master-gpu-mtp-off | 78848 | 78849 | off | xhigh | P1 | 41.17 | 41.20 | 41.26 | N/A | N/A | N/A |
| master-gpu-mtp-off | 78848 | 78849 | off | xhigh | P2 | 41.17 | 41.21 | 41.24 | N/A | N/A | N/A |
| master-gpu-mtp-off | 78848 | 78849 | off | xhigh | P3 | 41.16 | 41.17 | 41.24 | N/A | N/A | N/A |
| master-gpu-mtp-off | 88064 | 88065 | off | medium | P1 | 40.34 | 40.35 | 40.36 | N/A | N/A | N/A |
| master-gpu-mtp-off | 88064 | 88065 | off | medium | P2 | 40.31 | 40.31 | 40.36 | N/A | N/A | N/A |
| master-gpu-mtp-off | 88064 | 88065 | off | medium | P3 | 40.29 | 40.32 | 40.34 | N/A | N/A | N/A |
| master-gpu-mtp-off | 88064 | 88065 | off | xhigh | P1 | 40.30 | 40.32 | 40.36 | N/A | N/A | N/A |
| master-gpu-mtp-off | 88064 | 88065 | off | xhigh | P2 | 40.31 | 40.33 | 40.35 | N/A | N/A | N/A |
| master-gpu-mtp-off | 88064 | 88065 | off | xhigh | P3 | 40.31 | 40.35 | 40.36 | N/A | N/A | N/A |
| master-gpu-mtp-off | 100352 | 100353 | off | off | P1 | 39.28 | 39.29 | 39.37 | N/A | N/A | N/A |
| master-gpu-mtp-off | 100352 | 100353 | off | off | P2 | 39.34 | 39.38 | 39.41 | N/A | N/A | N/A |
| master-gpu-mtp-off | 100352 | 100353 | off | off | P3 | 39.28 | 39.30 | 39.33 | N/A | N/A | N/A |
| master-gpu-mtp-off | 100352 | 100353 | off | low | P1 | 39.28 | 39.31 | 39.37 | N/A | N/A | N/A |
| master-gpu-mtp-off | 100352 | 100353 | off | low | P2 | 39.32 | 39.33 | 39.34 | N/A | N/A | N/A |
| master-gpu-mtp-off | 100352 | 100353 | off | low | P3 | 39.28 | 39.30 | 39.35 | N/A | N/A | N/A |
| master-gpu-mtp-off | 100352 | 100353 | off | medium | P1 | 39.26 | 39.30 | 39.36 | N/A | N/A | N/A |
| master-gpu-mtp-off | 100352 | 100353 | off | medium | P2 | 39.28 | 39.29 | 39.33 | N/A | N/A | N/A |
| master-gpu-mtp-off | 100352 | 100353 | off | medium | P3 | 39.27 | 39.31 | 39.33 | N/A | N/A | N/A |
| master-gpu-mtp-off | 100352 | 100353 | off | xhigh | P1 | 39.26 | 39.30 | 39.36 | N/A | N/A | N/A |
| master-gpu-mtp-off | 100352 | 100353 | off | xhigh | P2 | 39.27 | 39.31 | 39.34 | N/A | N/A | N/A |
| master-gpu-mtp-off | 100352 | 100353 | off | xhigh | P3 | 39.30 | 39.31 | 39.34 | N/A | N/A | N/A |
| master-gpu-mtp-off | 120832 | 120833 | off | off | P1 | 37.75 | 37.76 | 37.82 | N/A | N/A | N/A |
| master-gpu-mtp-off | 120832 | 120833 | off | off | P2 | 37.79 | 37.81 | 37.82 | N/A | N/A | N/A |
| master-gpu-mtp-off | 120832 | 120833 | off | off | P3 | 37.74 | 37.77 | 37.79 | N/A | N/A | N/A |
| master-gpu-mtp-off | 120832 | 120833 | off | low | P1 | 37.71 | 37.79 | 37.79 | N/A | N/A | N/A |
| master-gpu-mtp-off | 120832 | 120833 | off | low | P2 | 37.71 | 37.74 | 37.78 | N/A | N/A | N/A |
| master-gpu-mtp-off | 120832 | 120833 | off | low | P3 | 37.75 | 37.76 | 37.77 | N/A | N/A | N/A |
| master-gpu-mtp-off | 120832 | 120833 | off | medium | P1 | 37.77 | 37.78 | 37.81 | N/A | N/A | N/A |
| master-gpu-mtp-off | 120832 | 120833 | off | medium | P2 | 37.76 | 37.77 | 37.82 | N/A | N/A | N/A |
| master-gpu-mtp-off | 120832 | 120833 | off | medium | P3 | 37.75 | 37.75 | 37.77 | N/A | N/A | N/A |
| master-gpu-mtp-on | 6144 | 6144 | on | off | P1 | 105.76 | 105.92 | 106.23 | 94.91 | 94.91 | 94.91 |
| master-gpu-mtp-on | 6144 | 6144 | on | off | P2 | 73.46 | 74.16 | 74.45 | 50.81 | 50.81 | 50.81 |
| master-gpu-mtp-on | 6144 | 6144 | on | off | P3 | 94.15 | 95.01 | 95.20 | 79.55 | 79.55 | 79.55 |
| master-gpu-mtp-on | 6144 | 6144 | on | low | P1 | 104.06 | 104.15 | 104.19 | 92.15 | 92.15 | 92.15 |
| master-gpu-mtp-on | 6144 | 6144 | on | low | P2 | 78.23 | 79.36 | 79.62 | 58.74 | 58.74 | 58.74 |
| master-gpu-mtp-on | 6144 | 6144 | on | low | P3 | 95.74 | 96.66 | 96.90 | 83.56 | 83.56 | 83.56 |
| master-gpu-mtp-on | 6144 | 6144 | on | medium | P1 | 95.07 | 95.31 | 97.68 | 83.56 | 83.56 | 83.56 |
| master-gpu-mtp-on | 6144 | 6144 | on | medium | P2 | 78.03 | 78.88 | 78.91 | 57.57 | 57.57 | 57.57 |
| master-gpu-mtp-on | 6144 | 6144 | on | medium | P3 | 96.43 | 97.08 | 97.24 | 82.67 | 82.67 | 82.67 |
| master-gpu-mtp-on | 6144 | 6144 | on | xhigh | P1 | 97.39 | 100.06 | 100.23 | 87.95 | 87.95 | 87.95 |
| master-gpu-mtp-on | 6144 | 6144 | on | xhigh | P2 | 77.91 | 78.54 | 79.57 | 58.82 | 58.82 | 58.82 |
| master-gpu-mtp-on | 6144 | 6144 | on | xhigh | P3 | 90.71 | 90.72 | 90.95 | 74.14 | 74.14 | 74.14 |
| master-gpu-mtp-on | 12288 | 12289 | on | off | P1 | 100.30 | 100.32 | 100.34 | 86.92 | 86.92 | 86.92 |
| master-gpu-mtp-on | 12288 | 12289 | on | off | P2 | 81.81 | 81.81 | 81.83 | 61.69 | 61.69 | 61.69 |
| master-gpu-mtp-on | 12288 | 12289 | on | off | P3 | 90.37 | 90.41 | 90.51 | 73.91 | 73.91 | 73.91 |
| master-gpu-mtp-on | 12288 | 12289 | on | low | P1 | 94.46 | 94.50 | 94.64 | 79.55 | 79.55 | 79.55 |
| master-gpu-mtp-on | 12288 | 12289 | on | low | P2 | 73.07 | 73.21 | 73.28 | 50.51 | 50.51 | 50.51 |
| master-gpu-mtp-on | 12288 | 12289 | on | low | P3 | 90.32 | 90.33 | 90.45 | 73.91 | 73.91 | 73.91 |
| master-gpu-mtp-on | 12288 | 12288 | on | medium | P1 | 96.85 | 97.02 | 97.08 | 83.56 | 83.56 | 83.56 |
| master-gpu-mtp-on | 12288 | 12288 | on | medium | P2 | 84.91 | 85.03 | 85.21 | 66.86 | 66.86 | 66.86 |
| master-gpu-mtp-on | 12288 | 12288 | on | medium | P3 | 94.47 | 94.48 | 94.69 | 79.80 | 79.80 | 79.80 |
| master-gpu-mtp-on | 12288 | 12289 | on | xhigh | P1 | 92.65 | 92.70 | 92.77 | 77.07 | 77.07 | 77.07 |
| master-gpu-mtp-on | 12288 | 12289 | on | xhigh | P2 | 78.54 | 78.62 | 78.69 | 58.15 | 58.15 | 58.15 |
| master-gpu-mtp-on | 12288 | 12289 | on | xhigh | P3 | 85.62 | 85.64 | 85.76 | 67.75 | 67.75 | 67.75 |
| master-gpu-mtp-on | 14336 | 14336 | on | off | P1 | 99.73 | 100.07 | 100.08 | 87.88 | 87.88 | 87.88 |
| master-gpu-mtp-on | 14336 | 14336 | on | off | P2 | 75.00 | 75.08 | 75.37 | 54.49 | 54.49 | 54.49 |
| master-gpu-mtp-on | 14336 | 14336 | on | off | P3 | 91.00 | 91.02 | 91.23 | 76.04 | 76.04 | 76.04 |
| master-gpu-mtp-on | 14336 | 14336 | on | low | P1 | 101.50 | 101.57 | 101.80 | 92.13 | 92.13 | 92.13 |
| master-gpu-mtp-on | 14336 | 14336 | on | low | P2 | 82.07 | 82.11 | 82.35 | 65.29 | 65.29 | 65.29 |
| master-gpu-mtp-on | 14336 | 14336 | on | low | P3 | 89.27 | 89.31 | 89.46 | 74.69 | 74.69 | 74.69 |
| master-gpu-mtp-on | 14336 | 14336 | on | medium | P1 | 97.26 | 97.30 | 97.45 | 84.73 | 84.73 | 84.73 |
| master-gpu-mtp-on | 14336 | 14336 | on | medium | P2 | 79.97 | 80.15 | 80.35 | 61.17 | 61.17 | 61.17 |
| master-gpu-mtp-on | 14336 | 14336 | on | medium | P3 | 89.01 | 89.21 | 89.29 | 73.83 | 73.83 | 73.83 |
| master-gpu-mtp-on | 14336 | 14336 | on | xhigh | P1 | 98.79 | 98.83 | 99.21 | 89.02 | 89.02 | 89.02 |
| master-gpu-mtp-on | 14336 | 14336 | on | xhigh | P2 | 78.37 | 78.42 | 78.60 | 59.50 | 59.50 | 59.50 |
| master-gpu-mtp-on | 14336 | 14336 | on | xhigh | P3 | 78.88 | 84.28 | 84.30 | 59.94 | 67.75 | 67.75 |
| master-gpu-mtp-on | 30000 | 30000 | on | off | P1 | 96.00 | 96.00 | 96.37 | 89.39 | 89.39 | 89.39 |
| master-gpu-mtp-on | 30000 | 30000 | on | off | P2 | 76.44 | 76.48 | 76.51 | 60.00 | 60.00 | 60.00 |
| master-gpu-mtp-on | 30000 | 30000 | on | off | P3 | 90.07 | 90.10 | 90.25 | 80.39 | 80.39 | 80.39 |
| master-gpu-mtp-on | 30000 | 30000 | on | low | P1 | 93.01 | 93.74 | 93.78 | 87.50 | 88.66 | 88.66 |
| master-gpu-mtp-on | 30000 | 30000 | on | low | P2 | 72.83 | 72.86 | 72.87 | 57.31 | 57.31 | 57.31 |
| master-gpu-mtp-on | 30000 | 30000 | on | low | P3 | 85.80 | 85.81 | 85.82 | 76.75 | 76.75 | 76.75 |
| master-gpu-mtp-on | 30000 | 30000 | on | medium | P1 | 92.10 | 92.16 | 92.36 | 83.46 | 83.46 | 83.46 |
| master-gpu-mtp-on | 30000 | 30000 | on | medium | P2 | 74.27 | 74.27 | 74.44 | 57.50 | 57.50 | 57.50 |
| master-gpu-mtp-on | 30000 | 30000 | on | medium | P3 | 86.40 | 86.81 | 86.94 | 76.19 | 76.19 | 76.19 |
| master-gpu-mtp-on | 30000 | 30000 | on | xhigh | P1 | 94.44 | 94.45 | 94.47 | 90.12 | 90.12 | 90.12 |
| master-gpu-mtp-on | 30000 | 30000 | on | xhigh | P2 | 71.14 | 71.15 | 72.14 | 55.10 | 55.10 | 56.55 |
| master-gpu-mtp-on | 30000 | 30000 | on | xhigh | P3 | 72.37 | 72.41 | 72.41 | 56.99 | 56.99 | 56.99 |
| master-gpu-mtp-on | 61440 | 61440 | on | off | P1 | 91.87 | 91.91 | 91.94 | 94.26 | 94.26 | 94.26 |
| master-gpu-mtp-on | 61440 | 61440 | on | off | P2 | 74.43 | 74.48 | 74.48 | 67.57 | 67.57 | 67.57 |
| master-gpu-mtp-on | 61440 | 61440 | on | off | P3 | 82.18 | 82.24 | 82.37 | 79.72 | 79.72 | 79.72 |
| master-gpu-mtp-on | 61440 | 61440 | on | low | P1 | 86.24 | 86.24 | 86.29 | 90.37 | 90.37 | 90.37 |
| master-gpu-mtp-on | 61440 | 61440 | on | low | P2 | 65.69 | 65.70 | 65.70 | 57.20 | 57.20 | 57.20 |
| master-gpu-mtp-on | 61440 | 61440 | on | low | P3 | 78.17 | 78.19 | 78.22 | 77.32 | 77.32 | 77.32 |
| master-gpu-mtp-on | 61440 | 61440 | on | medium | P1 | 86.45 | 86.51 | 86.69 | 86.67 | 86.67 | 86.67 |
| master-gpu-mtp-on | 61440 | 61440 | on | medium | P2 | 69.10 | 69.12 | 69.24 | 59.02 | 59.02 | 59.02 |
| master-gpu-mtp-on | 61440 | 61440 | on | medium | P3 | 80.94 | 81.08 | 81.19 | 78.14 | 78.14 | 78.14 |
| master-gpu-mtp-on | 61440 | 61440 | on | xhigh | P1 | 89.09 | 89.18 | 89.18 | 95.51 | 95.51 | 95.51 |
| master-gpu-mtp-on | 61440 | 61440 | on | xhigh | P2 | 69.05 | 71.06 | 72.23 | 62.61 | 65.89 | 67.55 |
| master-gpu-mtp-on | 61440 | 61440 | on | xhigh | P3 | 77.19 | 77.79 | 80.30 | 75.47 | 76.75 | 80.92 |
| master-gpu-mtp-on | 71680 | 71681 | on | off | P1 | 86.54 | 86.61 | 86.65 | 91.25 | 91.25 | 91.25 |
| master-gpu-mtp-on | 71680 | 71681 | on | off | P2 | 64.52 | 64.65 | 64.71 | 55.06 | 55.06 | 55.06 |
| master-gpu-mtp-on | 71680 | 71681 | on | off | P3 | 74.03 | 78.60 | 78.74 | 70.18 | 78.39 | 78.39 |
| master-gpu-mtp-on | 71680 | 71680 | on | low | P1 | 82.26 | 82.29 | 82.42 | 84.18 | 84.18 | 84.18 |
| master-gpu-mtp-on | 71680 | 71680 | on | low | P2 | 66.99 | 67.00 | 67.24 | 59.34 | 59.34 | 59.34 |
| master-gpu-mtp-on | 71680 | 71680 | on | low | P3 | 74.81 | 74.83 | 74.86 | 71.65 | 71.65 | 71.65 |
| master-gpu-mtp-on | 71680 | 71681 | on | medium | P1 | 82.83 | 82.91 | 83.02 | 84.80 | 84.80 | 84.80 |
| master-gpu-mtp-on | 71680 | 71681 | on | medium | P2 | 65.57 | 65.59 | 65.69 | 56.84 | 56.84 | 56.84 |
| master-gpu-mtp-on | 71680 | 71681 | on | medium | P3 | 75.58 | 75.66 | 75.73 | 73.37 | 73.37 | 73.37 |
| master-gpu-mtp-on | 71680 | 71681 | on | xhigh | P1 | 77.61 | 77.62 | 77.89 | 76.51 | 76.51 | 76.75 |
| master-gpu-mtp-on | 71680 | 71681 | on | xhigh | P2 | 65.21 | 65.21 | 65.40 | 56.42 | 56.42 | 56.42 |
| master-gpu-mtp-on | 71680 | 71681 | on | xhigh | P3 | 80.57 | 80.60 | 80.68 | 81.79 | 81.79 | 81.79 |
| fork-gpu-mtp-off | 6144 | 6145 | off | off | P1 | 49.46 | 49.47 | 49.50 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 6144 | 6145 | off | off | P2 | 49.46 | 49.49 | 49.49 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 6144 | 6145 | off | off | P3 | 49.34 | 49.36 | 49.41 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 6144 | 6145 | off | low | P1 | 49.34 | 49.34 | 49.40 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 6144 | 6145 | off | low | P2 | 49.32 | 49.35 | 49.38 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 6144 | 6145 | off | low | P3 | 49.38 | 49.39 | 49.43 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 6144 | 6145 | off | medium | P1 | 49.37 | 49.37 | 49.41 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 6144 | 6145 | off | medium | P2 | 49.34 | 49.35 | 49.40 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 6144 | 6145 | off | medium | P3 | 49.34 | 49.38 | 49.41 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 6144 | 6145 | off | xhigh | P1 | 49.36 | 49.37 | 49.40 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 6144 | 6145 | off | xhigh | P2 | 49.37 | 49.37 | 49.40 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 6144 | 6145 | off | xhigh | P3 | 49.29 | 49.36 | 49.38 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 12288 | 12289 | off | off | P1 | 48.54 | 48.54 | 48.55 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 12288 | 12289 | off | off | P2 | 48.54 | 48.55 | 48.56 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 12288 | 12289 | off | off | P3 | 48.45 | 48.46 | 48.46 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 12288 | 12289 | off | low | P1 | 48.42 | 48.47 | 48.50 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 12288 | 12289 | off | low | P2 | 48.42 | 48.43 | 48.50 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 12288 | 12289 | off | low | P3 | 48.45 | 48.46 | 48.47 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 12288 | 12288 | off | medium | P1 | 48.43 | 48.43 | 48.51 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 12288 | 12288 | off | medium | P2 | 48.44 | 48.46 | 48.48 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 12288 | 12288 | off | medium | P3 | 48.42 | 48.42 | 48.49 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 12288 | 12289 | off | xhigh | P1 | 48.44 | 48.45 | 48.48 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 12288 | 12289 | off | xhigh | P2 | 48.45 | 48.46 | 48.48 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 12288 | 12289 | off | xhigh | P3 | 48.43 | 48.44 | 48.49 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 28672 | 28673 | off | off | P1 | 46.75 | 46.78 | 46.78 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 28672 | 28673 | off | off | P2 | 46.73 | 46.77 | 46.78 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 28672 | 28673 | off | off | P3 | 46.67 | 46.69 | 46.69 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 28672 | 28673 | off | low | P1 | 46.66 | 46.66 | 46.70 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 28672 | 28673 | off | low | P2 | 46.64 | 46.64 | 46.68 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 28672 | 28673 | off | low | P3 | 46.58 | 46.64 | 46.71 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 28672 | 28673 | off | medium | P1 | 46.65 | 46.66 | 46.68 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 28672 | 28673 | off | medium | P2 | 46.60 | 46.63 | 46.67 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 28672 | 28673 | off | medium | P3 | 46.64 | 46.65 | 46.67 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 28672 | 28673 | off | xhigh | P1 | 46.65 | 46.67 | 46.69 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 28672 | 28673 | off | xhigh | P2 | 46.63 | 46.64 | 46.64 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 28672 | 28673 | off | xhigh | P3 | 46.61 | 46.65 | 46.69 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 62464 | 62465 | off | off | P1 | 42.90 | 42.93 | 43.01 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 62464 | 62465 | off | off | P2 | 43.08 | 43.08 | 43.09 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 62464 | 62465 | off | off | P3 | 43.00 | 43.01 | 43.03 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 62464 | 62465 | off | low | P1 | 42.99 | 42.99 | 43.03 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 62464 | 62465 | off | low | P2 | 42.98 | 42.99 | 43.03 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 62464 | 62465 | off | low | P3 | 42.97 | 42.98 | 43.00 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 62464 | 62465 | off | medium | P1 | 43.00 | 43.00 | 43.00 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 62464 | 62465 | off | medium | P2 | 42.99 | 42.99 | 43.00 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 62464 | 62465 | off | medium | P3 | 42.97 | 42.98 | 43.00 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 62464 | 62464 | off | xhigh | P1 | 42.99 | 43.00 | 43.00 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 62464 | 62464 | off | xhigh | P2 | 42.96 | 42.98 | 43.03 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 62464 | 62464 | off | xhigh | P3 | 42.97 | 42.99 | 42.99 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 78848 | 78849 | off | off | P1 | 41.28 | 41.30 | 41.37 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 78848 | 78849 | off | off | P2 | 41.33 | 41.38 | 41.39 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 78848 | 78849 | off | off | P3 | 41.31 | 41.36 | 41.38 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 78848 | 78849 | off | low | P1 | 41.37 | 41.37 | 41.38 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 78848 | 78849 | off | low | P2 | 41.37 | 41.37 | 41.37 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 78848 | 78849 | off | low | P3 | 41.34 | 41.35 | 41.39 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 78848 | 78849 | off | medium | P1 | 41.34 | 41.36 | 41.39 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 78848 | 78849 | off | medium | P2 | 41.31 | 41.35 | 41.39 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 78848 | 78849 | off | medium | P3 | 41.33 | 41.35 | 41.36 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 78848 | 78849 | off | xhigh | P1 | 41.36 | 41.37 | 41.37 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 78848 | 78849 | off | xhigh | P2 | 41.36 | 41.37 | 41.38 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 78848 | 78849 | off | xhigh | P3 | 41.30 | 41.37 | 41.39 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 88064 | 88065 | off | off | P1 | 40.46 | 40.48 | 40.51 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 88064 | 88065 | off | off | P2 | 40.56 | 40.56 | 40.57 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 88064 | 88065 | off | off | P3 | 40.51 | 40.52 | 40.56 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 88064 | 88065 | off | low | P1 | 40.50 | 40.52 | 40.53 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 88064 | 88065 | off | low | P2 | 40.52 | 40.53 | 40.54 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 88064 | 88065 | off | low | P3 | 40.49 | 40.50 | 40.52 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 88064 | 88065 | off | medium | P1 | 40.51 | 40.52 | 40.52 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 88064 | 88065 | off | medium | P2 | 40.50 | 40.50 | 40.51 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 88064 | 88065 | off | medium | P3 | 40.49 | 40.52 | 40.53 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 88064 | 88065 | off | xhigh | P1 | 40.51 | 40.51 | 40.53 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 88064 | 88065 | off | xhigh | P2 | 40.48 | 40.50 | 40.53 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 88064 | 88065 | off | xhigh | P3 | 40.49 | 40.52 | 40.56 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 100352 | 100353 | off | off | P1 | 39.49 | 39.52 | 39.54 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 100352 | 100353 | off | off | P2 | 39.56 | 39.57 | 39.59 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 100352 | 100353 | off | off | P3 | 39.49 | 39.52 | 39.52 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 100352 | 100353 | off | low | P1 | 39.53 | 39.54 | 39.54 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 100352 | 100353 | off | low | P2 | 39.50 | 39.52 | 39.55 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 100352 | 100353 | off | low | P3 | 39.47 | 39.53 | 39.54 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 100352 | 100353 | off | medium | P1 | 39.47 | 39.53 | 39.56 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 100352 | 100353 | off | medium | P2 | 39.48 | 39.52 | 39.56 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 100352 | 100353 | off | medium | P3 | 39.51 | 39.52 | 39.55 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 100352 | 100353 | off | xhigh | P1 | 39.49 | 39.50 | 39.53 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 100352 | 100353 | off | xhigh | P2 | 39.52 | 39.53 | 39.55 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 100352 | 100353 | off | xhigh | P3 | 39.45 | 39.52 | 39.55 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 112640 | 112640 | off | off | P1 | 38.60 | 38.60 | 38.61 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 112640 | 112640 | off | off | P2 | 38.56 | 38.61 | 38.62 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 112640 | 112640 | off | off | P3 | 38.54 | 38.56 | 38.59 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 116736 | 116737 | off | off | P1 | 38.12 | 38.12 | 38.17 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 116736 | 116737 | off | off | P2 | 38.21 | 38.21 | 38.22 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 116736 | 116737 | off | off | P3 | 38.14 | 38.16 | 38.19 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 120832 | 120833 | off | off | P1 | 37.95 | 37.97 | 37.99 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 120832 | 120833 | off | off | P2 | 38.01 | 38.01 | 38.01 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 120832 | 120833 | off | off | P3 | 37.95 | 37.97 | 37.98 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 120832 | 120833 | off | low | P1 | 37.92 | 37.96 | 37.99 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 120832 | 120833 | off | low | P2 | 37.94 | 37.95 | 37.99 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 120832 | 120833 | off | low | P3 | 37.91 | 37.96 | 37.96 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 120832 | 120833 | off | medium | P1 | 37.91 | 37.95 | 37.97 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 120832 | 120833 | off | medium | P2 | 37.95 | 37.96 | 37.99 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 120832 | 120833 | off | medium | P3 | 37.89 | 37.97 | 37.99 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 120832 | 120833 | off | xhigh | P1 | 37.94 | 37.97 | 38.01 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 120832 | 120833 | off | xhigh | P2 | 37.95 | 37.97 | 38.00 | N/A | N/A | N/A |
| fork-gpu-mtp-off | 120832 | 120833 | off | xhigh | P3 | 37.94 | 37.96 | 37.97 | N/A | N/A | N/A |
| fork-gpu-mtp-on | 6144 | 6144 | on | off | P1 | 101.76 | 102.30 | 102.46 | 89.82 | 90.81 | 90.81 |
| fork-gpu-mtp-on | 6144 | 6144 | on | off | P2 | 73.29 | 74.91 | 75.83 | 51.09 | 53.89 | 54.49 |
| fork-gpu-mtp-on | 6144 | 6144 | on | off | P3 | 87.75 | 87.75 | 87.85 | 71.12 | 71.12 | 71.12 |
| fork-gpu-mtp-on | 6144 | 6144 | on | low | P1 | 99.50 | 100.24 | 101.15 | 86.33 | 87.32 | 88.32 |
| fork-gpu-mtp-on | 6144 | 6144 | on | low | P2 | 74.47 | 75.83 | 77.30 | 52.50 | 54.46 | 56.49 |
| fork-gpu-mtp-on | 6144 | 6144 | on | low | P3 | 88.12 | 88.13 | 88.92 | 71.56 | 71.56 | 73.07 |
| fork-gpu-mtp-on | 6144 | 6144 | on | medium | P1 | 99.32 | 100.00 | 100.21 | 86.94 | 87.89 | 87.89 |
| fork-gpu-mtp-on | 6144 | 6144 | on | medium | P2 | 74.03 | 74.10 | 75.71 | 52.58 | 52.58 | 54.74 |
| fork-gpu-mtp-on | 6144 | 6144 | on | medium | P3 | 87.27 | 89.93 | 96.09 | 70.61 | 73.91 | 82.39 |
| fork-gpu-mtp-on | 6144 | 6144 | on | xhigh | P1 | 97.34 | 97.56 | 98.10 | 83.61 | 83.61 | 84.75 |
| fork-gpu-mtp-on | 6144 | 6144 | on | xhigh | P2 | 71.98 | 72.53 | 73.07 | 49.24 | 50.00 | 50.78 |
| fork-gpu-mtp-on | 6144 | 6144 | on | xhigh | P3 | 79.00 | 79.67 | 80.58 | 59.50 | 60.11 | 61.34 |
| fork-gpu-mtp-on | 12288 | 12289 | on | off | P1 | 99.09 | 99.17 | 99.20 | 86.92 | 86.92 | 86.92 |
| fork-gpu-mtp-on | 12288 | 12289 | on | off | P2 | 83.38 | 83.44 | 83.45 | 66.67 | 66.67 | 66.67 |
| fork-gpu-mtp-on | 12288 | 12289 | on | off | P3 | 90.66 | 90.83 | 90.90 | 76.51 | 76.51 | 76.51 |
| fork-gpu-mtp-on | 12288 | 12289 | on | low | P1 | 94.31 | 94.71 | 94.72 | 81.79 | 81.79 | 81.79 |
| fork-gpu-mtp-on | 12288 | 12289 | on | low | P2 | 76.76 | 76.87 | 77.32 | 57.57 | 57.57 | 58.15 |
| fork-gpu-mtp-on | 12288 | 12289 | on | low | P3 | 81.13 | 81.24 | 81.44 | 63.53 | 63.53 | 63.53 |
| fork-gpu-mtp-on | 12288 | 12288 | on | medium | P1 | 95.57 | 95.81 | 95.90 | 83.28 | 83.28 | 83.28 |
| fork-gpu-mtp-on | 12288 | 12288 | on | medium | P2 | 79.29 | 79.44 | 79.52 | 61.17 | 61.17 | 61.17 |
| fork-gpu-mtp-on | 12288 | 12288 | on | medium | P3 | 92.86 | 93.16 | 93.38 | 79.80 | 79.80 | 79.80 |
| fork-gpu-mtp-on | 12288 | 12289 | on | xhigh | P1 | 92.42 | 92.74 | 92.85 | 79.22 | 79.22 | 79.22 |
| fork-gpu-mtp-on | 12288 | 12289 | on | xhigh | P2 | 69.30 | 69.35 | 69.45 | 47.60 | 47.60 | 47.60 |
| fork-gpu-mtp-on | 12288 | 12289 | on | xhigh | P3 | 84.14 | 84.20 | 84.33 | 67.35 | 67.35 | 67.35 |
| fork-gpu-mtp-on | 30000 | 30000 | on | off | P1 | 94.27 | 95.31 | 95.41 | 90.22 | 91.76 | 91.76 |
| fork-gpu-mtp-on | 30000 | 30000 | on | off | P2 | 73.85 | 74.66 | 77.11 | 60.67 | 61.93 | 65.62 |
| fork-gpu-mtp-on | 30000 | 30000 | on | off | P3 | 86.79 | 86.80 | 87.40 | 80.07 | 80.07 | 80.92 |
| fork-gpu-mtp-on | 30000 | 30000 | on | low | P1 | 93.65 | 93.65 | 93.66 | 90.38 | 90.38 | 90.38 |
| fork-gpu-mtp-on | 30000 | 30000 | on | low | P2 | 72.07 | 72.12 | 72.63 | 58.89 | 58.89 | 59.70 |
| fork-gpu-mtp-on | 30000 | 30000 | on | low | P3 | 85.60 | 85.62 | 86.18 | 78.39 | 78.39 | 79.22 |
| fork-gpu-mtp-on | 30000 | 30000 | on | medium | P1 | 91.70 | 91.70 | 91.71 | 87.24 | 87.24 | 87.24 |
| fork-gpu-mtp-on | 30000 | 30000 | on | medium | P2 | 71.48 | 71.49 | 71.89 | 57.41 | 57.41 | 57.99 |
| fork-gpu-mtp-on | 30000 | 30000 | on | medium | P3 | 88.31 | 88.42 | 88.43 | 82.12 | 82.12 | 82.12 |
| fork-gpu-mtp-on | 30000 | 30000 | on | xhigh | P1 | 89.06 | 89.07 | 90.03 | 83.02 | 83.02 | 84.29 |
| fork-gpu-mtp-on | 30000 | 30000 | on | xhigh | P2 | 73.40 | 73.40 | 73.53 | 60.25 | 60.25 | 60.25 |
| fork-gpu-mtp-on | 30000 | 30000 | on | xhigh | P3 | 78.78 | 78.81 | 78.84 | 68.45 | 68.45 | 68.45 |
| fork-gpu-mtp-on | 61440 | 61440 | on | off | P1 | 78.63 | 79.03 | 80.70 | 80.07 | 80.39 | 83.28 |
| fork-gpu-mtp-on | 61440 | 61440 | on | off | P2 | 57.20 | 57.25 | 57.87 | 44.94 | 46.02 | 46.07 |
| fork-gpu-mtp-on | 61440 | 61440 | on | off | P3 | 69.77 | 70.09 | 71.42 | 65.70 | 66.18 | 68.25 |
| fork-gpu-mtp-on | 61440 | 61440 | on | low | P1 | 78.13 | 78.71 | 79.25 | 78.93 | 79.86 | 80.80 |
| fork-gpu-mtp-on | 61440 | 61440 | on | low | P2 | 58.02 | 58.43 | 59.25 | 46.53 | 47.20 | 48.58 |
| fork-gpu-mtp-on | 61440 | 61440 | on | low | P3 | 70.53 | 72.26 | 73.93 | 67.06 | 69.88 | 72.62 |
| fork-gpu-mtp-on | 61440 | 61440 | on | medium | P1 | 77.07 | 80.25 | 81.35 | 77.56 | 82.67 | 84.46 |
| fork-gpu-mtp-on | 61440 | 61440 | on | medium | P2 | 59.18 | 59.78 | 60.08 | 48.64 | 49.50 | 50.13 |
| fork-gpu-mtp-on | 61440 | 61440 | on | medium | P3 | 71.34 | 72.23 | 74.53 | 68.25 | 69.67 | 73.37 |
| fork-gpu-mtp-on | 61440 | 61440 | on | xhigh | P1 | 71.95 | 72.00 | 73.18 | 69.07 | 69.92 | 71.12 |
| fork-gpu-mtp-on | 61440 | 61440 | on | xhigh | P2 | 58.49 | 58.96 | 59.35 | 47.20 | 47.98 | 48.78 |
| fork-gpu-mtp-on | 61440 | 61440 | on | xhigh | P3 | 53.76 | 54.65 | 56.02 | 39.55 | 41.19 | 43.33 |
| fork-gpu-mtp-on | 78848 | 78848 | on | off | P1 | 83.70 | 83.74 | 83.85 | 91.25 | 91.25 | 91.25 |
| fork-gpu-mtp-on | 78848 | 78848 | on | off | P2 | 65.93 | 65.93 | 66.07 | 61.39 | 61.39 | 61.39 |
| fork-gpu-mtp-on | 78848 | 78848 | on | off | P3 | 75.73 | 75.91 | 77.38 | 77.88 | 77.88 | 80.92 |
| fork-gpu-mtp-on | 78848 | 78849 | on | low | P1 | 77.16 | 77.35 | 77.40 | 80.66 | 80.66 | 80.66 |
| fork-gpu-mtp-on | 78848 | 78849 | on | low | P2 | 63.03 | 63.03 | 63.08 | 56.68 | 56.68 | 56.68 |
| fork-gpu-mtp-on | 78848 | 78849 | on | low | P3 | 66.22 | 66.38 | 66.46 | 62.43 | 62.43 | 62.43 |
| fork-gpu-mtp-on | 78848 | 78849 | on | medium | P1 | 76.53 | 76.87 | 76.90 | 80.07 | 80.07 | 80.07 |
| fork-gpu-mtp-on | 78848 | 78849 | on | medium | P2 | 59.51 | 59.54 | 59.55 | 50.76 | 50.76 | 50.76 |
| fork-gpu-mtp-on | 78848 | 78849 | on | medium | P3 | 70.65 | 70.81 | 70.90 | 69.88 | 69.88 | 69.88 |
| fork-gpu-mtp-on | 78848 | 78849 | on | xhigh | P1 | 75.72 | 75.83 | 75.92 | 78.14 | 78.14 | 78.14 |
| fork-gpu-mtp-on | 78848 | 78849 | on | xhigh | P2 | 63.77 | 63.77 | 63.83 | 57.99 | 57.99 | 57.99 |
| fork-gpu-mtp-on | 78848 | 78849 | on | xhigh | P3 | 71.10 | 71.14 | 71.27 | 70.39 | 70.39 | 70.39 |
| master-cpu-mtp-off | 6144 | 6144 | off | off | P1 | 18.72 | 18.75 | 18.80 | N/A | N/A | N/A |
| master-cpu-mtp-off | 6144 | 6144 | off | off | P2 | 18.76 | 18.77 | 18.82 | N/A | N/A | N/A |
| master-cpu-mtp-off | 6144 | 6144 | off | off | P3 | 18.58 | 18.72 | 18.73 | N/A | N/A | N/A |
| master-cpu-mtp-off | 6144 | 6144 | off | low | P1 | 18.98 | 18.98 | 19.00 | N/A | N/A | N/A |
| master-cpu-mtp-off | 6144 | 6144 | off | low | P2 | 18.85 | 18.86 | 18.88 | N/A | N/A | N/A |
| master-cpu-mtp-off | 6144 | 6144 | off | low | P3 | 18.85 | 18.94 | 18.98 | N/A | N/A | N/A |
| master-cpu-mtp-off | 6144 | 6144 | off | medium | P1 | 18.81 | 18.87 | 18.87 | N/A | N/A | N/A |
| master-cpu-mtp-off | 6144 | 6144 | off | medium | P2 | 18.78 | 18.79 | 18.88 | N/A | N/A | N/A |
| master-cpu-mtp-off | 6144 | 6144 | off | medium | P3 | 18.82 | 18.95 | 18.97 | N/A | N/A | N/A |
| master-cpu-mtp-off | 6144 | 6144 | off | xhigh | P1 | 18.88 | 18.89 | 18.90 | N/A | N/A | N/A |
| master-cpu-mtp-off | 6144 | 6144 | off | xhigh | P2 | 18.85 | 18.98 | 18.98 | N/A | N/A | N/A |
| master-cpu-mtp-off | 6144 | 6144 | off | xhigh | P3 | 18.85 | 18.87 | 18.88 | N/A | N/A | N/A |
| master-cpu-mtp-off | 12288 | 12288 | off | off | P1 | 16.52 | 16.58 | 16.60 | N/A | N/A | N/A |
| master-cpu-mtp-off | 12288 | 12288 | off | off | P2 | 16.53 | 16.56 | 16.64 | N/A | N/A | N/A |
| master-cpu-mtp-off | 12288 | 12288 | off | off | P3 | 16.51 | 16.54 | 16.56 | N/A | N/A | N/A |
| master-cpu-mtp-off | 12288 | 12288 | off | low | P1 | 16.47 | 16.52 | 16.63 | N/A | N/A | N/A |
| master-cpu-mtp-off | 12288 | 12288 | off | low | P2 | 16.46 | 16.50 | 16.54 | N/A | N/A | N/A |
| master-cpu-mtp-off | 12288 | 12288 | off | low | P3 | 16.57 | 16.60 | 16.61 | N/A | N/A | N/A |
| master-cpu-mtp-off | 12288 | 12288 | off | medium | P1 | 16.53 | 16.56 | 16.59 | N/A | N/A | N/A |
| master-cpu-mtp-off | 12288 | 12288 | off | medium | P2 | 16.50 | 16.51 | 16.55 | N/A | N/A | N/A |
| master-cpu-mtp-off | 12288 | 12288 | off | medium | P3 | 16.38 | 16.49 | 16.57 | N/A | N/A | N/A |
| master-cpu-mtp-off | 12288 | 12288 | off | xhigh | P1 | 16.50 | 16.52 | 16.55 | N/A | N/A | N/A |
| master-cpu-mtp-off | 12288 | 12288 | off | xhigh | P2 | 16.46 | 16.46 | 16.61 | N/A | N/A | N/A |
| master-cpu-mtp-off | 12288 | 12288 | off | xhigh | P3 | 16.51 | 16.57 | 16.60 | N/A | N/A | N/A |
| master-cpu-mtp-off | 28672 | 28672 | off | off | P1 | 12.64 | 12.69 | 12.69 | N/A | N/A | N/A |
| master-cpu-mtp-off | 28672 | 28672 | off | off | P2 | 12.42 | 12.64 | 12.69 | N/A | N/A | N/A |
| master-cpu-mtp-off | 28672 | 28672 | off | off | P3 | 11.32 | 12.57 | 12.60 | N/A | N/A | N/A |
| master-cpu-mtp-off | 28672 | 28673 | off | low | P1 | 12.58 | 12.63 | 12.64 | N/A | N/A | N/A |
| master-cpu-mtp-off | 28672 | 28673 | off | low | P2 | 12.58 | 12.66 | 12.68 | N/A | N/A | N/A |
| master-cpu-mtp-off | 28672 | 28673 | off | low | P3 | 12.67 | 12.68 | 12.68 | N/A | N/A | N/A |
| master-cpu-mtp-off | 28672 | 28673 | off | medium | P1 | 12.65 | 12.68 | 12.70 | N/A | N/A | N/A |
| master-cpu-mtp-off | 28672 | 28673 | off | medium | P2 | 12.61 | 12.64 | 12.65 | N/A | N/A | N/A |
| master-cpu-mtp-off | 28672 | 28673 | off | medium | P3 | 12.58 | 12.62 | 12.64 | N/A | N/A | N/A |
| master-cpu-mtp-off | 28672 | 28673 | off | xhigh | P1 | 12.53 | 12.59 | 12.67 | N/A | N/A | N/A |
| master-cpu-mtp-off | 28672 | 28673 | off | xhigh | P2 | 12.51 | 12.62 | 12.63 | N/A | N/A | N/A |
| master-cpu-mtp-off | 28672 | 28673 | off | xhigh | P3 | 12.58 | 12.63 | 12.64 | N/A | N/A | N/A |
| master-cpu-mtp-off | 61440 | 61440 | off | off | P1 | 8.55 | 8.57 | 8.59 | N/A | N/A | N/A |
| master-cpu-mtp-off | 61440 | 61440 | off | off | P2 | 7.11 | 8.60 | 8.62 | N/A | N/A | N/A |
| master-cpu-mtp-off | 61440 | 61440 | off | off | P3 | 8.56 | 8.65 | 8.65 | N/A | N/A | N/A |
| master-cpu-mtp-off | 61440 | 61441 | off | low | P1 | 8.55 | 8.59 | 8.59 | N/A | N/A | N/A |
| master-cpu-mtp-off | 61440 | 61441 | off | low | P2 | 8.59 | 8.61 | 8.61 | N/A | N/A | N/A |
| master-cpu-mtp-off | 61440 | 61441 | off | low | P3 | 8.54 | 8.55 | 8.58 | N/A | N/A | N/A |
| master-cpu-mtp-off | 61440 | 61441 | off | medium | P1 | 8.52 | 8.57 | 8.59 | N/A | N/A | N/A |
| master-cpu-mtp-off | 61440 | 61441 | off | medium | P2 | 8.55 | 8.61 | 8.62 | N/A | N/A | N/A |
| master-cpu-mtp-off | 61440 | 61441 | off | medium | P3 | 8.58 | 8.60 | 8.63 | N/A | N/A | N/A |
| master-cpu-mtp-off | 61440 | 61441 | off | xhigh | P1 | 8.57 | 8.58 | 8.59 | N/A | N/A | N/A |
| master-cpu-mtp-off | 61440 | 61441 | off | xhigh | P2 | 8.58 | 8.59 | 8.61 | N/A | N/A | N/A |
| master-cpu-mtp-off | 61440 | 61441 | off | xhigh | P3 | 8.58 | 8.61 | 8.61 | N/A | N/A | N/A |
| master-cpu-mtp-off | 126976 | 126976 | off | off | P1 | 5.25 | 5.25 | 5.25 | N/A | N/A | N/A |
| master-cpu-mtp-off | 126976 | 126976 | off | off | P2 | 5.28 | 5.29 | 5.29 | N/A | N/A | N/A |
| master-cpu-mtp-off | 126976 | 126976 | off | off | P3 | 5.23 | 5.25 | 5.25 | N/A | N/A | N/A |
| master-cpu-mtp-off | 126976 | 126977 | off | low | P1 | 5.22 | 5.22 | 5.24 | N/A | N/A | N/A |
| master-cpu-mtp-off | 126976 | 126977 | off | low | P2 | 5.24 | 5.25 | 5.26 | N/A | N/A | N/A |
| master-cpu-mtp-off | 126976 | 126977 | off | low | P3 | 5.21 | 5.23 | 5.24 | N/A | N/A | N/A |
| master-cpu-mtp-off | 126976 | 126977 | off | medium | P1 | 5.22 | 5.23 | 5.24 | N/A | N/A | N/A |
| master-cpu-mtp-off | 126976 | 126977 | off | medium | P2 | 5.23 | 5.23 | 5.25 | N/A | N/A | N/A |
| master-cpu-mtp-off | 126976 | 126977 | off | medium | P3 | 5.22 | 5.23 | 5.24 | N/A | N/A | N/A |
| master-cpu-mtp-off | 126976 | 126977 | off | xhigh | P1 | 5.21 | 5.22 | 5.24 | N/A | N/A | N/A |
| master-cpu-mtp-off | 126976 | 126977 | off | xhigh | P2 | 5.25 | 5.25 | 5.25 | N/A | N/A | N/A |
| master-cpu-mtp-off | 126976 | 126977 | off | xhigh | P3 | 5.23 | 5.23 | 5.24 | N/A | N/A | N/A |
| master-cpu-mtp-off | 192512 | 192512 | off | off | P1 | 3.79 | 3.79 | 3.80 | N/A | N/A | N/A |
| master-cpu-mtp-off | 192512 | 192512 | off | off | P2 | 3.80 | 3.81 | 3.81 | N/A | N/A | N/A |
| master-cpu-mtp-off | 192512 | 192512 | off | off | P3 | 3.79 | 3.80 | 3.80 | N/A | N/A | N/A |
| master-cpu-mtp-off | 192512 | 192513 | off | low | P1 | 3.75 | 3.78 | 3.79 | N/A | N/A | N/A |
| master-cpu-mtp-off | 192512 | 192513 | off | low | P2 | 3.78 | 3.79 | 3.79 | N/A | N/A | N/A |
| master-cpu-mtp-off | 192512 | 192513 | off | low | P3 | 3.76 | 3.78 | 3.79 | N/A | N/A | N/A |
| master-cpu-mtp-off | 192512 | 192513 | off | medium | P1 | 3.78 | 3.79 | 3.79 | N/A | N/A | N/A |
| master-cpu-mtp-off | 192512 | 192513 | off | medium | P2 | 3.79 | 3.80 | 3.80 | N/A | N/A | N/A |
| master-cpu-mtp-off | 192512 | 192513 | off | medium | P3 | 3.78 | 3.79 | 3.79 | N/A | N/A | N/A |
| master-cpu-mtp-off | 192512 | 192513 | off | xhigh | P1 | 3.75 | 3.79 | 3.79 | N/A | N/A | N/A |
| master-cpu-mtp-off | 192512 | 192513 | off | xhigh | P2 | 3.79 | 3.80 | 3.80 | N/A | N/A | N/A |
| master-cpu-mtp-off | 192512 | 192513 | off | xhigh | P3 | 3.79 | 3.79 | 3.79 | N/A | N/A | N/A |
| master-cpu-mtp-off | 258048 | 258048 | off | off | P1 | 2.99 | 2.99 | 2.99 | N/A | N/A | N/A |
| master-cpu-mtp-off | 258048 | 258048 | off | off | P2 | 2.99 | 3.00 | 3.00 | N/A | N/A | N/A |
| master-cpu-mtp-off | 258048 | 258048 | off | off | P3 | 2.96 | 2.99 | 2.99 | N/A | N/A | N/A |
| master-cpu-mtp-off | 258048 | 258049 | off | low | P1 | 2.95 | 2.96 | 2.96 | N/A | N/A | N/A |
| master-cpu-mtp-off | 258048 | 258049 | off | low | P2 | 2.96 | 2.96 | 2.97 | N/A | N/A | N/A |
| master-cpu-mtp-off | 258048 | 258049 | off | low | P3 | 2.95 | 2.96 | 2.96 | N/A | N/A | N/A |
| master-cpu-mtp-off | 258048 | 258049 | off | medium | P1 | 2.95 | 2.95 | 2.96 | N/A | N/A | N/A |
| master-cpu-mtp-off | 258048 | 258049 | off | medium | P2 | 2.96 | 2.97 | 2.97 | N/A | N/A | N/A |
| master-cpu-mtp-off | 258048 | 258049 | off | medium | P3 | 2.95 | 2.96 | 2.96 | N/A | N/A | N/A |
| master-cpu-mtp-off | 258048 | 258049 | off | xhigh | P1 | 2.95 | 2.96 | 2.96 | N/A | N/A | N/A |
| master-cpu-mtp-off | 258048 | 258049 | off | xhigh | P2 | 2.96 | 2.96 | 2.97 | N/A | N/A | N/A |
| master-cpu-mtp-off | 258048 | 258049 | off | xhigh | P3 | 2.95 | 2.96 | 2.96 | N/A | N/A | N/A |
| master-cpu-mtp-on | 6144 | 6144 | on | off | P1 | 28.99 | 29.02 | 29.19 | 96.93 | 96.93 | 96.93 |
| master-cpu-mtp-on | 6144 | 6144 | on | off | P2 | 21.57 | 21.78 | 21.94 | 58.65 | 58.65 | 58.65 |
| master-cpu-mtp-on | 6144 | 6144 | on | off | P3 | 22.20 | 24.95 | 25.39 | 61.80 | 79.22 | 79.22 |
| master-cpu-mtp-on | 6144 | 6144 | on | low | P1 | 28.55 | 29.03 | 29.31 | 89.91 | 89.91 | 89.91 |
| master-cpu-mtp-on | 6144 | 6144 | on | low | P2 | 22.25 | 22.41 | 22.55 | 61.17 | 61.17 | 61.17 |
| master-cpu-mtp-on | 6144 | 6144 | on | low | P3 | 25.52 | 25.73 | 25.74 | 76.75 | 76.75 | 76.75 |
| master-cpu-mtp-on | 6144 | 6144 | on | xhigh | P1 | 14.90 | 16.84 | 16.88 | 23.59 | 33.43 | 33.43 |
| master-cpu-mtp-on | 6144 | 6144 | on | xhigh | P2 | 14.89 | 14.99 | 15.12 | 23.44 | 23.44 | 23.44 |
| master-cpu-mtp-on | 6144 | 6144 | on | xhigh | P3 | 14.31 | 22.16 | 22.29 | 20.82 | 59.94 | 59.94 |
| master-cpu-mtp-on | 12288 | 12288 | on | off | P1 | 18.97 | 24.55 | 25.65 | 57.14 | 93.38 | 93.38 |
| master-cpu-mtp-on | 12288 | 12288 | on | off | P2 | 19.75 | 20.07 | 20.38 | 64.62 | 64.62 | 64.62 |
| master-cpu-mtp-on | 12288 | 12288 | on | off | P3 | 22.27 | 22.47 | 22.83 | 76.27 | 76.27 | 76.27 |
| master-cpu-mtp-on | 28672 | 28672 | on | off | P1 | 15.73 | 20.74 | 20.91 | 57.14 | 93.38 | 93.38 |
| master-cpu-mtp-on | 28672 | 28672 | on | off | P2 | 16.06 | 16.25 | 16.50 | 63.16 | 63.16 | 63.16 |
| master-cpu-mtp-on | 28672 | 28672 | on | off | P3 | 11.24 | 11.60 | 16.00 | 29.98 | 29.98 | 63.25 |
| master-cpu-mtp-on | 61440 | 61440 | on | off | P1 | 11.35 | 15.00 | 15.12 | 57.14 | 91.30 | 91.30 |
| master-cpu-mtp-on | 61440 | 61440 | on | off | P2 | 9.65 | 12.22 | 12.26 | 39.90 | 64.10 | 64.10 |
| master-cpu-mtp-on | 61440 | 61440 | on | off | P3 | 11.88 | 13.80 | 13.84 | 60.83 | 77.88 | 77.88 |
| master-cpu-mtp-on | 126976 | 126976 | on | off | P1 | 9.98 | 10.06 | 10.12 | 93.38 | 93.38 | 93.38 |
| master-cpu-mtp-on | 126976 | 126976 | on | off | P2 | 8.13 | 8.16 | 8.27 | 67.24 | 67.24 | 67.24 |
| master-cpu-mtp-on | 126976 | 126976 | on | off | P3 | 9.20 | 9.26 | 9.29 | 81.25 | 81.25 | 81.25 |
| master-cpu-mtp-on | 196608 | 196609 | on | off | P1 | 8.67 | 8.84 | 8.89 | 90.58 | 90.58 | 90.58 |
| master-cpu-mtp-on | 196608 | 196609 | on | off | P2 | 6.79 | 6.79 | 6.80 | 57.74 | 57.74 | 57.74 |
| master-cpu-mtp-on | 196608 | 196609 | on | low | P1 | 8.29 | 8.29 | 8.35 | 83.56 | 83.56 | 83.56 |
| master-cpu-mtp-on | 196608 | 196609 | on | low | P2 | 7.08 | 7.08 | 7.11 | 63.07 | 63.07 | 63.07 |
| master-cpu-mtp-on | 196608 | 196609 | on | low | P3 | 7.17 | 7.17 | 7.20 | 64.00 | 64.00 | 64.00 |
| master-cpu-mtp-on | 196608 | 196609 | on | medium | P1 | 8.24 | 8.32 | 8.33 | 82.67 | 82.67 | 82.67 |
| master-cpu-mtp-on | 196608 | 196609 | on | medium | P2 | 6.81 | 6.81 | 6.84 | 58.74 | 58.74 | 58.74 |
| master-cpu-mtp-on | 196608 | 196609 | on | medium | P3 | 7.85 | 7.87 | 7.87 | 75.16 | 75.16 | 75.16 |
| master-cpu-mtp-on | 196608 | 196609 | on | xhigh | P1 | 7.57 | 7.57 | 7.59 | 70.61 | 70.61 | 70.61 |
| master-cpu-mtp-on | 196608 | 196609 | on | xhigh | P3 | 7.74 | 7.80 | 7.83 | 74.38 | 74.38 | 74.38 |
| fork-cpu-mtp-off | 6144 | 6145 | off | off | P1 | 18.67 | 18.68 | 18.74 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 6144 | 6145 | off | off | P2 | 18.64 | 18.69 | 18.83 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 6144 | 6145 | off | off | P3 | 18.71 | 18.79 | 18.81 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 6144 | 6145 | off | low | P1 | 18.69 | 18.73 | 18.78 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 6144 | 6145 | off | low | P2 | 18.69 | 18.72 | 18.76 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 6144 | 6145 | off | low | P3 | 18.69 | 18.72 | 18.74 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 6144 | 6145 | off | medium | P1 | 18.72 | 18.72 | 18.75 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 6144 | 6145 | off | medium | P2 | 18.67 | 18.69 | 18.76 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 6144 | 6145 | off | medium | P3 | 18.77 | 18.77 | 18.81 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 6144 | 6145 | off | xhigh | P1 | 18.74 | 18.74 | 18.74 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 6144 | 6145 | off | xhigh | P2 | 18.73 | 18.78 | 18.79 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 6144 | 6145 | off | xhigh | P3 | 18.64 | 18.75 | 18.75 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 12288 | 12289 | off | off | P1 | 16.35 | 16.45 | 16.49 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 12288 | 12289 | off | off | P2 | 16.41 | 16.47 | 16.56 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 12288 | 12289 | off | off | P3 | 16.38 | 16.50 | 16.53 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 12288 | 12289 | off | low | P1 | 16.44 | 16.46 | 16.48 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 12288 | 12289 | off | low | P2 | 16.42 | 16.52 | 16.55 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 12288 | 12289 | off | low | P3 | 16.43 | 16.44 | 16.45 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 12288 | 12288 | off | medium | P1 | 16.45 | 16.49 | 16.51 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 12288 | 12288 | off | medium | P2 | 16.46 | 16.47 | 16.49 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 12288 | 12288 | off | medium | P3 | 16.45 | 16.49 | 16.51 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 12288 | 12289 | off | xhigh | P1 | 16.49 | 16.49 | 16.51 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 12288 | 12289 | off | xhigh | P2 | 16.44 | 16.49 | 16.56 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 12288 | 12289 | off | xhigh | P3 | 16.40 | 16.42 | 16.45 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 28672 | 28673 | off | off | P1 | 12.52 | 12.52 | 12.59 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 28672 | 28673 | off | off | P2 | 12.56 | 12.56 | 12.57 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 28672 | 28673 | off | off | P3 | 12.51 | 12.51 | 12.55 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 28672 | 28673 | off | low | P1 | 12.53 | 12.53 | 12.55 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 28672 | 28673 | off | low | P2 | 12.48 | 12.54 | 12.56 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 28672 | 28673 | off | low | P3 | 12.52 | 12.53 | 12.58 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 28672 | 28673 | off | medium | P1 | 12.52 | 12.52 | 12.53 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 28672 | 28673 | off | medium | P2 | 12.52 | 12.53 | 12.54 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 28672 | 28673 | off | medium | P3 | 12.54 | 12.54 | 12.58 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 28672 | 28673 | off | xhigh | P1 | 12.55 | 12.56 | 12.59 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 28672 | 28673 | off | xhigh | P2 | 12.55 | 12.55 | 12.56 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 28672 | 28673 | off | xhigh | P3 | 12.51 | 12.52 | 12.55 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 62464 | 62465 | off | off | P1 | 8.42 | 8.44 | 8.48 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 62464 | 62465 | off | off | P2 | 8.39 | 8.43 | 8.49 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 62464 | 62465 | off | off | P3 | 8.44 | 8.47 | 8.48 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 62464 | 62465 | off | low | P1 | 8.40 | 8.47 | 8.48 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 62464 | 62465 | off | low | P2 | 8.41 | 8.47 | 8.49 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 62464 | 62465 | off | low | P3 | 8.47 | 8.48 | 8.50 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 62464 | 62465 | off | medium | P1 | 8.43 | 8.45 | 8.46 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 62464 | 62465 | off | medium | P2 | 8.50 | 8.51 | 8.53 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 62464 | 62465 | off | medium | P3 | 8.44 | 8.47 | 8.49 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 62464 | 62464 | off | xhigh | P1 | 8.43 | 8.44 | 8.47 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 62464 | 62464 | off | xhigh | P2 | 8.46 | 8.48 | 8.50 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 62464 | 62464 | off | xhigh | P3 | 8.47 | 8.47 | 8.47 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 130048 | 130049 | off | off | P1 | 5.06 | 5.12 | 5.13 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 130048 | 130049 | off | off | P2 | 5.14 | 5.14 | 5.15 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 130048 | 130049 | off | off | P3 | 5.11 | 5.13 | 5.13 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 130048 | 130049 | off | low | P1 | 5.12 | 5.13 | 5.14 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 130048 | 130049 | off | low | P2 | 5.07 | 5.14 | 5.14 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 130048 | 130049 | off | low | P3 | 5.11 | 5.13 | 5.13 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 130048 | 130049 | off | medium | P1 | 5.06 | 5.12 | 5.13 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 130048 | 130049 | off | medium | P2 | 5.12 | 5.13 | 5.16 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 130048 | 130049 | off | medium | P3 | 5.10 | 5.12 | 5.12 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 130048 | 130049 | off | xhigh | P1 | 5.13 | 5.14 | 5.15 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 130048 | 130049 | off | xhigh | P2 | 5.13 | 5.14 | 5.15 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 130048 | 130049 | off | xhigh | P3 | 5.11 | 5.12 | 5.13 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 196608 | 196609 | off | off | P1 | 3.72 | 3.72 | 3.73 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 196608 | 196609 | off | off | P2 | 3.72 | 3.73 | 3.73 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 196608 | 196609 | off | off | P3 | 3.68 | 3.73 | 3.74 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 196608 | 196609 | off | low | P1 | 3.71 | 3.71 | 3.71 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 196608 | 196609 | off | low | P2 | 3.71 | 3.71 | 3.71 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 196608 | 196609 | off | low | P3 | 3.71 | 3.71 | 3.72 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 196608 | 196609 | off | medium | P1 | 3.71 | 3.71 | 3.71 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 196608 | 196609 | off | medium | P2 | 3.68 | 3.71 | 3.71 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 196608 | 196609 | off | medium | P3 | 3.71 | 3.71 | 3.71 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 196608 | 196609 | off | xhigh | P1 | 3.70 | 3.71 | 3.71 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 196608 | 196609 | off | xhigh | P2 | 3.68 | 3.69 | 3.72 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 196608 | 196609 | off | xhigh | P3 | 3.70 | 3.71 | 3.72 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 256000 | 256001 | off | off | P1 | 2.96 | 2.98 | 2.98 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 256000 | 256001 | off | off | P2 | 2.94 | 2.98 | 2.98 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 256000 | 256001 | off | off | P3 | 2.98 | 2.98 | 2.98 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 256000 | 255999 | off | low | P1 | 2.97 | 2.98 | 2.98 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 256000 | 255999 | off | low | P2 | 2.98 | 2.98 | 2.99 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 256000 | 255999 | off | low | P3 | 2.97 | 2.98 | 2.99 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 256000 | 255999 | off | medium | P1 | 2.98 | 2.98 | 2.99 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 256000 | 255999 | off | medium | P2 | 2.98 | 2.99 | 2.99 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 256000 | 255999 | off | medium | P3 | 2.98 | 2.99 | 2.99 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 256000 | 255999 | off | xhigh | P1 | 2.98 | 2.98 | 2.98 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 256000 | 255999 | off | xhigh | P2 | 2.98 | 2.99 | 2.99 | N/A | N/A | N/A |
| fork-cpu-mtp-off | 256000 | 255999 | off | xhigh | P3 | 2.98 | 2.98 | 2.98 | N/A | N/A | N/A |
| fork-cpu-mtp-on | 6144 | 6145 | on | off | P1 | 29.99 | 30.38 | 30.69 | 93.08 | 93.08 | 93.08 |
| fork-cpu-mtp-on | 6144 | 6145 | on | off | P2 | 21.43 | 21.63 | 21.66 | 52.17 | 52.17 | 52.17 |
| fork-cpu-mtp-on | 6144 | 6145 | on | off | P3 | 26.74 | 26.87 | 27.22 | 80.07 | 80.07 | 80.07 |
| fork-cpu-mtp-on | 6144 | 6145 | on | low | P1 | 26.84 | 26.86 | 26.87 | 80.39 | 80.39 | 80.39 |
| fork-cpu-mtp-on | 6144 | 6145 | on | low | P2 | 21.86 | 21.96 | 22.11 | 55.85 | 55.85 | 55.85 |
| fork-cpu-mtp-on | 6144 | 6145 | on | low | P3 | 27.14 | 27.33 | 27.41 | 80.92 | 80.92 | 80.92 |
| fork-cpu-mtp-on | 6144 | 6145 | on | medium | P1 | 27.76 | 28.12 | 28.20 | 85.37 | 85.37 | 85.37 |
| fork-cpu-mtp-on | 6144 | 6145 | on | medium | P2 | 22.19 | 22.22 | 22.56 | 57.23 | 57.23 | 57.23 |
| fork-cpu-mtp-on | 6144 | 6145 | on | medium | P3 | 26.73 | 26.98 | 27.13 | 77.88 | 77.88 | 77.88 |
| fork-cpu-mtp-on | 6144 | 6145 | on | xhigh | P1 | 29.13 | 29.16 | 29.42 | 89.34 | 89.34 | 89.34 |
| fork-cpu-mtp-on | 6144 | 6145 | on | xhigh | P2 | 24.18 | 24.21 | 24.21 | 65.22 | 65.22 | 65.22 |
| fork-cpu-mtp-on | 6144 | 6145 | on | xhigh | P3 | 23.90 | 24.44 | 24.47 | 66.57 | 66.57 | 66.57 |
| fork-cpu-mtp-on | 12288 | 12288 | on | off | P1 | 26.82 | 26.95 | 27.16 | 91.30 | 91.30 | 91.30 |
| fork-cpu-mtp-on | 12288 | 12288 | on | off | P2 | 21.91 | 22.06 | 22.41 | 63.43 | 63.43 | 63.43 |
| fork-cpu-mtp-on | 12288 | 12288 | on | off | P3 | 25.66 | 26.15 | 26.26 | 86.60 | 86.60 | 86.60 |
| fork-cpu-mtp-on | 12288 | 12389–13022 | on | low | P1 | 27.54 | 28.35 | 28.70 | 93.69 | 98.39 | 100.00 |
| fork-cpu-mtp-on | 12288 | 13393–14249 | on | low | P2 | 21.21 | 28.61 | 28.76 | 61.17 | 98.51 | 99.62 |
| fork-cpu-mtp-on | 12288 | 14744–15601 | on | low | P3 | 26.63 | 27.42 | 27.63 | 85.67 | 98.88 | 100.00 |
| fork-cpu-mtp-on | 28672 | 28672 | on | off | P1 | 22.83 | 22.87 | 22.89 | 93.38 | 93.38 | 93.38 |
| fork-cpu-mtp-on | 28672 | 28672 | on | off | P2 | 17.47 | 17.48 | 17.55 | 58.90 | 58.90 | 58.90 |
| fork-cpu-mtp-on | 28672 | 28672 | on | off | P3 | 19.95 | 20.08 | 20.29 | 74.38 | 74.38 | 74.38 |
| fork-cpu-mtp-on | 61440 | 61440 | on | off | P1 | 17.42 | 17.57 | 17.68 | 92.65 | 92.65 | 92.65 |
| fork-cpu-mtp-on | 61440 | 61440 | on | off | P2 | 13.43 | 13.47 | 13.56 | 60.62 | 60.62 | 60.62 |
| fork-cpu-mtp-on | 61440 | 61440 | on | off | P3 | 15.13 | 15.23 | 15.29 | 76.19 | 76.19 | 76.19 |
| fork-cpu-mtp-on | 126976 | 126976 | on | off | P1 | 11.89 | 11.95 | 12.02 | 93.38 | 93.38 | 93.38 |
| fork-cpu-mtp-on | 126976 | 126976 | on | off | P2 | 9.32 | 9.39 | 9.42 | 62.50 | 62.50 | 62.50 |
| fork-cpu-mtp-on | 126976 | 126976 | on | off | P3 | 10.89 | 10.91 | 10.93 | 81.19 | 81.19 | 81.19 |
| fork-cpu-mtp-on | 126976 | 127047–127909 | on | medium | P1 | 11.44 | 12.06 | 12.34 | 89.47 | 96.55 | 100.00 |
| fork-cpu-mtp-on | 196608 | 196609 | on | off | P1 | 8.70 | 8.71 | 8.72 | 88.57 | 88.57 | 88.57 |
| fork-cpu-mtp-on | 196608 | 196609 | on | off | P2 | 6.49 | 6.51 | 6.53 | 54.17 | 54.17 | 54.17 |
| fork-cpu-mtp-on | 196608 | 196609 | on | low | P1 | 8.48 | 8.48 | 8.49 | 84.80 | 84.80 | 84.80 |
| fork-cpu-mtp-on | 196608 | 196609 | on | low | P2 | 7.04 | 7.05 | 7.05 | 62.61 | 62.61 | 62.61 |
| fork-cpu-mtp-on | 196608 | 196609 | on | low | P3 | 7.15 | 7.17 | 7.18 | 64.37 | 64.37 | 64.37 |
| fork-cpu-mtp-on | 196608 | 196609 | on | medium | P1 | 8.29 | 8.33 | 8.41 | 83.00 | 83.00 | 83.00 |
| fork-cpu-mtp-on | 196608 | 196609 | on | medium | P2 | 7.02 | 7.03 | 7.05 | 62.43 | 62.43 | 62.43 |
| fork-cpu-mtp-on | 196608 | 196609 | on | medium | P3 | 7.58 | 7.60 | 7.60 | 71.56 | 71.56 | 71.56 |
| fork-cpu-mtp-on | 196608 | 196609 | on | xhigh | P1 | 8.05 | 8.08 | 8.11 | 79.48 | 79.48 | 79.48 |
| fork-cpu-mtp-on | 196608 | 196609 | on | xhigh | P2 | 6.98 | 7.00 | 7.01 | 61.80 | 61.80 | 61.80 |
| fork-cpu-mtp-on | 196608 | 196609 | on | xhigh | P3 | 8.07 | 8.10 | 8.11 | 79.22 | 79.22 | 79.22 |
| fork-cpu-mtp-on | 256000 | 256001 | on | off | P1 | 7.10 | 7.14 | 7.21 | 88.57 | 88.57 | 88.57 |
| fork-cpu-mtp-on | 256000 | 256001 | on | off | P2 | 5.48 | 5.48 | 5.49 | 56.25 | 56.25 | 56.25 |
| fork-cpu-mtp-on | 256000 | 256001 | on | off | P3 | 6.48 | 6.50 | 6.50 | 75.33 | 75.33 | 75.33 |
| fork-cpu-mtp-on | 256000 | 256001 | on | low | P1 | 6.81 | 6.82 | 6.83 | 81.79 | 81.79 | 81.79 |
| fork-cpu-mtp-on | 256000 | 256001 | on | low | P2 | 5.79 | 5.79 | 5.80 | 61.97 | 61.97 | 61.97 |
| fork-cpu-mtp-on | 256000 | 256001 | on | low | P3 | 6.37 | 6.38 | 6.39 | 73.15 | 73.15 | 73.15 |
| fork-cpu-mtp-on | 256000 | 256001 | on | medium | P1 | 6.80 | 6.84 | 6.87 | 82.67 | 82.67 | 82.67 |
| fork-cpu-mtp-on | 256000 | 256001 | on | medium | P2 | 5.78 | 5.80 | 5.81 | 61.97 | 61.97 | 61.97 |
| fork-cpu-mtp-on | 256000 | 256001 | on | medium | P3 | 6.63 | 6.64 | 6.65 | 78.39 | 78.39 | 78.39 |
| fork-cpu-mtp-on | 256000 | 256001 | on | xhigh | P1 | 6.62 | 6.62 | 6.63 | 78.64 | 78.64 | 78.64 |
| fork-cpu-mtp-on | 256000 | 256001 | on | xhigh | P2 | 5.35 | 5.36 | 5.36 | 53.79 | 53.79 | 53.79 |
| fork-cpu-mtp-on | 256000 | 256001 | on | xhigh | P3 | 6.65 | 6.67 | 6.68 | 78.71 | 78.71 | 78.71 |
| fork-cpu-mtp-on | 258048 | 258048 | on | off | P1 | 7.19 | 7.20 | 7.23 | 91.36 | 91.36 | 91.36 |
| fork-cpu-mtp-on | 258048 | 258048 | on | off | P2 | 5.25 | 5.28 | 5.30 | 53.95 | 53.95 | 53.95 |
| fork-cpu-mtp-on | 258048 | 258048 | on | off | P3 | 6.77 | 6.79 | 6.80 | 83.56 | 83.56 | 83.56 |
| current-aco-mtp-off | 6144 | 6144 | off | off | P1 | 48.97 | 49.04 | 49.05 | N/A | N/A | N/A |
| current-aco-mtp-off | 6144 | 6144 | off | off | P2 | 49.16 | 49.16 | 49.17 | N/A | N/A | N/A |
| current-aco-mtp-off | 6144 | 6144 | off | off | P3 | 49.00 | 49.01 | 49.06 | N/A | N/A | N/A |
| current-aco-mtp-off | 6144 | 6144 | off | low | P1 | 48.95 | 48.95 | 48.98 | N/A | N/A | N/A |
| current-aco-mtp-off | 6144 | 6144 | off | low | P2 | 48.99 | 49.00 | 49.02 | N/A | N/A | N/A |
| current-aco-mtp-off | 6144 | 6144 | off | low | P3 | 48.98 | 49.00 | 49.05 | N/A | N/A | N/A |
| current-aco-mtp-off | 6144 | 6144 | off | medium | P1 | 48.99 | 49.00 | 49.04 | N/A | N/A | N/A |
| current-aco-mtp-off | 6144 | 6144 | off | medium | P2 | 48.97 | 49.00 | 49.03 | N/A | N/A | N/A |
| current-aco-mtp-off | 6144 | 6144 | off | medium | P3 | 48.95 | 48.98 | 49.01 | N/A | N/A | N/A |
| current-aco-mtp-off | 6144 | 6144 | off | xhigh | P1 | 48.96 | 48.97 | 49.00 | N/A | N/A | N/A |
| current-aco-mtp-off | 6144 | 6144 | off | xhigh | P2 | 48.73 | 48.90 | 48.90 | N/A | N/A | N/A |
| current-aco-mtp-off | 6144 | 6144 | off | xhigh | P3 | 48.96 | 48.96 | 48.98 | N/A | N/A | N/A |
| current-aco-mtp-off | 12288 | 12288 | off | off | P1 | 48.03 | 48.04 | 48.10 | N/A | N/A | N/A |
| current-aco-mtp-off | 12288 | 12288 | off | off | P2 | 48.12 | 48.18 | 48.20 | N/A | N/A | N/A |
| current-aco-mtp-off | 12288 | 12288 | off | off | P3 | 48.05 | 48.06 | 48.06 | N/A | N/A | N/A |
| current-aco-mtp-off | 12288 | 12288 | off | low | P1 | 47.98 | 47.98 | 48.01 | N/A | N/A | N/A |
| current-aco-mtp-off | 12288 | 12288 | off | low | P2 | 48.00 | 48.00 | 48.04 | N/A | N/A | N/A |
| current-aco-mtp-off | 12288 | 12288 | off | low | P3 | 48.01 | 48.03 | 48.06 | N/A | N/A | N/A |
| current-aco-mtp-off | 12288 | 12288 | off | medium | P1 | 48.00 | 48.01 | 48.04 | N/A | N/A | N/A |
| current-aco-mtp-off | 12288 | 12288 | off | medium | P2 | 47.97 | 48.04 | 48.07 | N/A | N/A | N/A |
| current-aco-mtp-off | 12288 | 12288 | off | medium | P3 | 48.02 | 48.03 | 48.08 | N/A | N/A | N/A |
| current-aco-mtp-off | 12288 | 12288 | off | xhigh | P1 | 47.94 | 47.96 | 47.97 | N/A | N/A | N/A |
| current-aco-mtp-off | 12288 | 12288 | off | xhigh | P2 | 47.96 | 47.98 | 47.99 | N/A | N/A | N/A |
| current-aco-mtp-off | 12288 | 12288 | off | xhigh | P3 | 48.01 | 48.03 | 48.05 | N/A | N/A | N/A |
| current-aco-mtp-off | 28672 | 28673 | off | off | P1 | 46.01 | 46.09 | 46.11 | N/A | N/A | N/A |
| current-aco-mtp-off | 28672 | 28673 | off | off | P2 | 46.21 | 46.23 | 46.25 | N/A | N/A | N/A |
| current-aco-mtp-off | 28672 | 28673 | off | off | P3 | 46.09 | 46.10 | 46.14 | N/A | N/A | N/A |
| current-aco-mtp-off | 28672 | 28673 | off | low | P1 | 45.95 | 45.96 | 46.02 | N/A | N/A | N/A |
| current-aco-mtp-off | 28672 | 28673 | off | low | P2 | 45.92 | 45.98 | 46.03 | N/A | N/A | N/A |
| current-aco-mtp-off | 28672 | 28673 | off | low | P3 | 45.99 | 46.04 | 46.09 | N/A | N/A | N/A |
| current-aco-mtp-off | 28672 | 28673 | off | medium | P1 | 46.08 | 46.08 | 46.10 | N/A | N/A | N/A |
| current-aco-mtp-off | 28672 | 28673 | off | medium | P2 | 46.08 | 46.08 | 46.11 | N/A | N/A | N/A |
| current-aco-mtp-off | 28672 | 28673 | off | medium | P3 | 46.06 | 46.09 | 46.11 | N/A | N/A | N/A |
| current-aco-mtp-off | 28672 | 28673 | off | xhigh | P1 | 45.94 | 45.95 | 46.00 | N/A | N/A | N/A |
| current-aco-mtp-off | 28672 | 28673 | off | xhigh | P2 | 45.90 | 46.00 | 46.05 | N/A | N/A | N/A |
| current-aco-mtp-off | 28672 | 28673 | off | xhigh | P3 | 46.02 | 46.03 | 46.07 | N/A | N/A | N/A |
| current-aco-mtp-off | 62464 | 62465 | off | off | P1 | 42.29 | 42.33 | 42.33 | N/A | N/A | N/A |
| current-aco-mtp-off | 62464 | 62465 | off | off | P2 | 42.31 | 42.32 | 42.34 | N/A | N/A | N/A |
| current-aco-mtp-off | 62464 | 62465 | off | off | P3 | 42.09 | 42.17 | 42.21 | N/A | N/A | N/A |
| current-aco-mtp-off | 62464 | 62465 | off | low | P1 | 42.03 | 42.05 | 42.08 | N/A | N/A | N/A |
| current-aco-mtp-off | 62464 | 62465 | off | low | P2 | 41.79 | 41.89 | 41.98 | N/A | N/A | N/A |
| current-aco-mtp-off | 62464 | 62465 | off | low | P3 | 42.11 | 42.12 | 42.12 | N/A | N/A | N/A |
| current-aco-mtp-off | 62464 | 62465 | off | medium | P1 | 42.16 | 42.18 | 42.18 | N/A | N/A | N/A |
| current-aco-mtp-off | 62464 | 62465 | off | medium | P2 | 42.15 | 42.18 | 42.19 | N/A | N/A | N/A |
| current-aco-mtp-off | 62464 | 62465 | off | medium | P3 | 42.16 | 42.18 | 42.20 | N/A | N/A | N/A |
| current-aco-mtp-off | 62464 | 62465 | off | xhigh | P1 | 41.98 | 41.99 | 42.00 | N/A | N/A | N/A |
| current-aco-mtp-off | 62464 | 62465 | off | xhigh | P2 | 42.04 | 42.06 | 42.08 | N/A | N/A | N/A |
| current-aco-mtp-off | 62464 | 62465 | off | xhigh | P3 | 42.12 | 42.14 | 42.15 | N/A | N/A | N/A |
| current-aco-mtp-off | 78848 | 78849 | off | off | P1 | 37.49 | 37.51 | 37.54 | N/A | N/A | N/A |
| current-aco-mtp-off | 78848 | 78849 | off | off | P2 | 37.51 | 37.55 | 37.56 | N/A | N/A | N/A |
| current-aco-mtp-off | 78848 | 78849 | off | off | P3 | 37.50 | 37.50 | 37.55 | N/A | N/A | N/A |
| current-aco-mtp-off | 78848 | 78849 | off | low | P1 | 37.43 | 37.45 | 37.48 | N/A | N/A | N/A |
| current-aco-mtp-off | 78848 | 78849 | off | low | P2 | 37.40 | 37.46 | 37.50 | N/A | N/A | N/A |
| current-aco-mtp-off | 78848 | 78849 | off | low | P3 | 37.47 | 37.50 | 37.52 | N/A | N/A | N/A |
| current-aco-mtp-off | 78848 | 78849 | off | medium | P1 | 37.49 | 37.50 | 37.52 | N/A | N/A | N/A |
| current-aco-mtp-off | 78848 | 78849 | off | medium | P2 | 37.49 | 37.49 | 37.50 | N/A | N/A | N/A |
| current-aco-mtp-off | 78848 | 78849 | off | medium | P3 | 37.47 | 37.51 | 37.54 | N/A | N/A | N/A |
| current-aco-mtp-off | 78848 | 78849 | off | xhigh | P1 | 37.38 | 37.40 | 37.49 | N/A | N/A | N/A |
| current-aco-mtp-off | 78848 | 78849 | off | xhigh | P2 | 37.42 | 37.47 | 37.49 | N/A | N/A | N/A |
| current-aco-mtp-off | 78848 | 78849 | off | xhigh | P3 | 37.50 | 37.50 | 37.52 | N/A | N/A | N/A |
| current-aco-mtp-off | 88064 | 88065 | off | off | P1 | 37.49 | 37.49 | 37.53 | N/A | N/A | N/A |
| current-aco-mtp-off | 88064 | 88065 | off | off | P2 | 37.56 | 37.58 | 37.58 | N/A | N/A | N/A |
| current-aco-mtp-off | 88064 | 88065 | off | off | P3 | 37.46 | 37.47 | 37.51 | N/A | N/A | N/A |
| current-aco-mtp-off | 88064 | 88065 | off | low | P1 | 37.46 | 37.46 | 37.50 | N/A | N/A | N/A |
| current-aco-mtp-off | 88064 | 88065 | off | low | P2 | 37.44 | 37.49 | 37.52 | N/A | N/A | N/A |
| current-aco-mtp-off | 88064 | 88065 | off | low | P3 | 37.46 | 37.51 | 37.52 | N/A | N/A | N/A |
| current-aco-mtp-off | 88064 | 88065 | off | medium | P1 | 37.41 | 37.47 | 37.54 | N/A | N/A | N/A |
| current-aco-mtp-off | 88064 | 88065 | off | medium | P2 | 37.47 | 37.50 | 37.52 | N/A | N/A | N/A |
| current-aco-mtp-off | 88064 | 88065 | off | medium | P3 | 37.48 | 37.51 | 37.54 | N/A | N/A | N/A |
| current-aco-mtp-off | 88064 | 88065 | off | xhigh | P1 | 37.36 | 37.39 | 37.50 | N/A | N/A | N/A |
| current-aco-mtp-off | 88064 | 88065 | off | xhigh | P2 | 37.44 | 37.49 | 37.58 | N/A | N/A | N/A |
| current-aco-mtp-off | 88064 | 88065 | off | xhigh | P3 | 37.49 | 37.51 | 37.53 | N/A | N/A | N/A |
| current-aco-mtp-off | 100352 | 100352 | off | off | P1 | 37.46 | 37.50 | 37.53 | N/A | N/A | N/A |
| current-aco-mtp-off | 100352 | 100352 | off | off | P2 | 37.53 | 37.56 | 37.57 | N/A | N/A | N/A |
| current-aco-mtp-off | 100352 | 100352 | off | off | P3 | 37.47 | 37.49 | 37.54 | N/A | N/A | N/A |
| current-aco-mtp-off | 100352 | 100352 | off | low | P1 | 37.41 | 37.41 | 37.48 | N/A | N/A | N/A |
| current-aco-mtp-off | 100352 | 100352 | off | low | P2 | 37.47 | 37.50 | 37.52 | N/A | N/A | N/A |
| current-aco-mtp-off | 100352 | 100352 | off | low | P3 | 37.42 | 37.49 | 37.50 | N/A | N/A | N/A |
| current-aco-mtp-off | 100352 | 100352 | off | medium | P1 | 37.47 | 37.48 | 37.53 | N/A | N/A | N/A |
| current-aco-mtp-off | 100352 | 100352 | off | medium | P2 | 37.49 | 37.49 | 37.50 | N/A | N/A | N/A |
| current-aco-mtp-off | 100352 | 100352 | off | medium | P3 | 37.50 | 37.50 | 37.51 | N/A | N/A | N/A |
| current-aco-mtp-off | 100352 | 100352 | off | xhigh | P1 | 37.37 | 37.47 | 37.48 | N/A | N/A | N/A |
| current-aco-mtp-off | 100352 | 100352 | off | xhigh | P2 | 37.36 | 37.39 | 37.46 | N/A | N/A | N/A |
| current-aco-mtp-off | 100352 | 100352 | off | xhigh | P3 | 37.48 | 37.50 | 37.51 | N/A | N/A | N/A |
| current-aco-mtp-off | 130048 | 130049 | off | off | P1 | 37.48 | 37.50 | 37.50 | N/A | N/A | N/A |
| current-aco-mtp-off | 130048 | 130049 | off | off | P2 | 37.57 | 37.57 | 37.57 | N/A | N/A | N/A |
| current-aco-mtp-off | 130048 | 130049 | off | off | P3 | 37.47 | 37.49 | 37.52 | N/A | N/A | N/A |
| current-aco-mtp-off | 130048 | 130049 | off | low | P1 | 37.42 | 37.43 | 37.47 | N/A | N/A | N/A |
| current-aco-mtp-off | 130048 | 130049 | off | low | P2 | 37.46 | 37.50 | 37.53 | N/A | N/A | N/A |
| current-aco-mtp-off | 130048 | 130049 | off | low | P3 | 37.47 | 37.49 | 37.50 | N/A | N/A | N/A |
| current-aco-mtp-off | 130048 | 130049 | off | medium | P1 | 37.45 | 37.47 | 37.49 | N/A | N/A | N/A |
| current-aco-mtp-off | 130048 | 130049 | off | medium | P2 | 37.44 | 37.47 | 37.50 | N/A | N/A | N/A |
| current-aco-mtp-off | 130048 | 130049 | off | medium | P3 | 37.46 | 37.48 | 37.52 | N/A | N/A | N/A |
| current-aco-mtp-off | 130048 | 130049 | off | xhigh | P1 | 37.34 | 37.45 | 37.47 | N/A | N/A | N/A |
| current-aco-mtp-off | 130048 | 130049 | off | xhigh | P2 | 37.40 | 37.41 | 37.44 | N/A | N/A | N/A |
| current-aco-mtp-off | 130048 | 130049 | off | xhigh | P3 | 37.47 | 37.48 | 37.50 | N/A | N/A | N/A |
| current-aco-mtp-off | 196608 | 196608 | off | off | P1 | 37.48 | 37.49 | 37.52 | N/A | N/A | N/A |
| current-aco-mtp-off | 196608 | 196608 | off | off | P2 | 37.53 | 37.56 | 37.58 | N/A | N/A | N/A |
| current-aco-mtp-off | 196608 | 196608 | off | off | P3 | 37.47 | 37.48 | 37.51 | N/A | N/A | N/A |
| current-aco-mtp-off | 196608 | 196608 | off | low | P1 | 37.36 | 37.39 | 37.42 | N/A | N/A | N/A |
| current-aco-mtp-off | 196608 | 196608 | off | low | P2 | 37.46 | 37.48 | 37.48 | N/A | N/A | N/A |
| current-aco-mtp-off | 196608 | 196608 | off | low | P3 | 37.46 | 37.49 | 37.50 | N/A | N/A | N/A |
| current-aco-mtp-off | 196608 | 196608 | off | medium | P1 | 37.46 | 37.48 | 37.50 | N/A | N/A | N/A |
| current-aco-mtp-off | 196608 | 196608 | off | medium | P2 | 37.43 | 37.49 | 37.50 | N/A | N/A | N/A |
| current-aco-mtp-off | 196608 | 196608 | off | medium | P3 | 37.46 | 37.47 | 37.50 | N/A | N/A | N/A |
| current-aco-mtp-off | 196608 | 196608 | off | xhigh | P1 | 37.42 | 37.44 | 37.48 | N/A | N/A | N/A |
| current-aco-mtp-off | 196608 | 196608 | off | xhigh | P2 | 37.38 | 37.38 | 37.45 | N/A | N/A | N/A |
| current-aco-mtp-off | 196608 | 196608 | off | xhigh | P3 | 37.46 | 37.47 | 37.47 | N/A | N/A | N/A |
| current-aco-mtp-off | 256000 | 256001 | off | off | P1 | 37.42 | 37.42 | 37.51 | N/A | N/A | N/A |
| current-aco-mtp-off | 256000 | 256001 | off | off | P2 | 37.57 | 37.57 | 37.57 | N/A | N/A | N/A |
| current-aco-mtp-off | 256000 | 256001 | off | off | P3 | 37.44 | 37.44 | 37.45 | N/A | N/A | N/A |
| current-aco-mtp-off | 256000 | 256001 | off | low | P1 | 37.40 | 37.42 | 37.45 | N/A | N/A | N/A |
| current-aco-mtp-off | 256000 | 256001 | off | low | P2 | 37.25 | 37.32 | 37.36 | N/A | N/A | N/A |
| current-aco-mtp-off | 256000 | 256001 | off | low | P3 | 37.42 | 37.46 | 37.47 | N/A | N/A | N/A |
| current-aco-mtp-off | 256000 | 256001 | off | medium | P1 | 37.41 | 37.45 | 37.49 | N/A | N/A | N/A |
| current-aco-mtp-off | 256000 | 256001 | off | medium | P2 | 37.44 | 37.45 | 37.46 | N/A | N/A | N/A |
| current-aco-mtp-off | 256000 | 256001 | off | medium | P3 | 37.44 | 37.44 | 37.45 | N/A | N/A | N/A |
| current-aco-mtp-off | 256000 | 256001 | off | xhigh | P1 | 37.37 | 37.41 | 37.45 | N/A | N/A | N/A |
| current-aco-mtp-off | 256000 | 256001 | off | xhigh | P2 | 37.28 | 37.37 | 37.39 | N/A | N/A | N/A |
| current-aco-mtp-off | 256000 | 256001 | off | xhigh | P3 | 37.43 | 37.46 | 37.46 | N/A | N/A | N/A |
| current-aco-mtp-off | 261120 | 261121 | off | off | P1 | 37.48 | 37.55 | 37.59 | N/A | N/A | N/A |
| current-aco-mtp-off | 261120 | 261121 | off | off | P2 | 37.52 | 37.57 | 37.59 | N/A | N/A | N/A |
| current-aco-mtp-off | 261120 | 261121 | off | off | P3 | 37.44 | 37.44 | 37.47 | N/A | N/A | N/A |
| current-aco-mtp-off | 261120 | 261121 | off | low | P1 | 37.37 | 37.38 | 37.43 | N/A | N/A | N/A |
| current-aco-mtp-off | 261120 | 261121 | off | low | P2 | 37.43 | 37.43 | 37.45 | N/A | N/A | N/A |
| current-aco-mtp-off | 261120 | 261121 | off | low | P3 | 37.42 | 37.44 | 37.45 | N/A | N/A | N/A |
| current-aco-mtp-off | 261120 | 261121 | off | medium | P1 | 37.41 | 37.41 | 37.45 | N/A | N/A | N/A |
| current-aco-mtp-off | 261120 | 261121 | off | medium | P2 | 37.44 | 37.46 | 37.46 | N/A | N/A | N/A |
| current-aco-mtp-off | 261120 | 261121 | off | medium | P3 | 37.42 | 37.42 | 37.42 | N/A | N/A | N/A |
| current-aco-mtp-off | 261120 | 261121 | off | xhigh | P1 | 37.38 | 37.38 | 37.43 | N/A | N/A | N/A |
| current-aco-mtp-off | 261120 | 261121 | off | xhigh | P2 | 37.33 | 37.36 | 37.38 | N/A | N/A | N/A |
| current-aco-mtp-off | 261120 | 261121 | off | xhigh | P3 | 37.42 | 37.45 | 37.48 | N/A | N/A | N/A |
| current-aco-mtp-on | 6144 | 6144 | on | off | P1 | 102.23 | 102.42 | 102.48 | 94.20 | 94.20 | 94.20 |
| current-aco-mtp-on | 6144 | 6144 | on | off | P2 | 81.87 | 81.97 | 82.01 | 64.66 | 64.66 | 64.66 |
| current-aco-mtp-on | 6144 | 6144 | on | off | P3 | 92.54 | 92.69 | 92.93 | 80.66 | 80.66 | 80.66 |
| current-aco-mtp-on | 6144 | 6144 | on | low | P1 | 99.93 | 100.11 | 100.63 | 91.20 | 91.20 | 91.20 |
| current-aco-mtp-on | 6144 | 6144 | on | low | P2 | 75.45 | 75.56 | 75.74 | 56.84 | 56.84 | 56.84 |
| current-aco-mtp-on | 6144 | 6144 | on | low | P3 | 92.55 | 92.64 | 92.87 | 80.39 | 80.39 | 80.39 |
| current-aco-mtp-on | 6144 | 6144 | on | medium | P1 | 99.69 | 99.85 | 100.07 | 90.81 | 90.81 | 90.81 |
| current-aco-mtp-on | 6144 | 6144 | on | medium | P2 | 80.42 | 80.45 | 80.47 | 63.53 | 63.53 | 63.53 |
| current-aco-mtp-on | 6144 | 6144 | on | medium | P3 | 90.54 | 90.73 | 90.88 | 77.88 | 77.88 | 77.88 |
| current-aco-mtp-on | 6144 | 6144 | on | xhigh | P1 | 99.82 | 99.90 | 100.23 | 91.07 | 91.07 | 91.07 |
| current-aco-mtp-on | 6144 | 6144 | on | xhigh | P2 | 76.03 | 76.09 | 76.36 | 57.88 | 57.88 | 57.88 |
| current-aco-mtp-on | 6144 | 6144 | on | xhigh | P3 | 79.98 | 80.00 | 80.10 | 62.71 | 62.71 | 62.71 |
| current-aco-mtp-on | 12288 | 12288 | on | off | P1 | 101.53 | 101.63 | 101.72 | 96.69 | 96.69 | 96.69 |
| current-aco-mtp-on | 12288 | 12288 | on | off | P2 | 79.36 | 79.77 | 79.85 | 65.75 | 65.75 | 65.75 |
| current-aco-mtp-on | 12288 | 12288 | on | off | P3 | 90.48 | 90.69 | 91.34 | 82.39 | 82.39 | 82.39 |
| current-aco-mtp-on | 12288 | 12288 | on | low | P1 | 95.05 | 95.20 | 95.36 | 88.10 | 88.10 | 88.10 |
| current-aco-mtp-on | 12288 | 12288 | on | low | P2 | 79.84 | 80.11 | 80.19 | 66.57 | 66.57 | 66.57 |
| current-aco-mtp-on | 12288 | 12288 | on | low | P3 | 88.81 | 88.83 | 89.20 | 79.22 | 79.22 | 79.22 |
| current-aco-mtp-on | 12288 | 12288 | on | medium | P1 | 92.18 | 92.32 | 92.85 | 84.15 | 84.15 | 84.15 |
| current-aco-mtp-on | 12288 | 12288 | on | medium | P2 | 77.53 | 77.53 | 77.84 | 63.25 | 63.25 | 63.25 |
| current-aco-mtp-on | 12288 | 12288 | on | medium | P3 | 89.73 | 89.79 | 90.16 | 80.39 | 80.39 | 80.39 |
| current-aco-mtp-on | 12288 | 12288 | on | xhigh | P1 | 93.17 | 93.50 | 93.66 | 85.38 | 85.38 | 85.38 |
| current-aco-mtp-on | 12288 | 12288 | on | xhigh | P2 | 74.18 | 74.29 | 74.32 | 58.56 | 58.56 | 58.56 |
| current-aco-mtp-on | 12288 | 12288 | on | xhigh | P3 | 77.64 | 77.66 | 77.99 | 63.07 | 63.07 | 63.07 |
| current-aco-mtp-on | 28672 | 28673 | on | off | P1 | 92.07 | 92.17 | 92.32 | 93.31 | 93.31 | 93.31 |
| current-aco-mtp-on | 28672 | 28673 | on | off | P2 | 74.00 | 74.00 | 74.02 | 64.62 | 64.62 | 64.62 |
| current-aco-mtp-on | 28672 | 28673 | on | off | P3 | 81.05 | 81.16 | 81.19 | 76.27 | 76.27 | 76.27 |
| current-aco-mtp-on | 28672 | 28673 | on | low | P1 | 91.89 | 92.01 | 92.43 | 93.20 | 93.20 | 93.20 |
| current-aco-mtp-on | 28672 | 28673 | on | low | P2 | 68.50 | 68.64 | 68.82 | 57.31 | 57.31 | 57.31 |
| current-aco-mtp-on | 28672 | 28673 | on | low | P3 | 85.33 | 85.44 | 85.60 | 83.00 | 83.00 | 83.00 |
| current-aco-mtp-on | 28672 | 28673 | on | medium | P1 | 88.36 | 88.49 | 88.84 | 87.59 | 87.59 | 87.59 |
| current-aco-mtp-on | 28672 | 28673 | on | medium | P2 | 73.27 | 73.45 | 73.48 | 64.84 | 64.84 | 64.84 |
| current-aco-mtp-on | 28672 | 28673 | on | medium | P3 | 82.07 | 82.25 | 83.04 | 78.39 | 78.39 | 79.22 |
| current-aco-mtp-on | 28672 | 28673 | on | xhigh | P1 | 87.62 | 87.97 | 88.19 | 87.36 | 87.36 | 87.36 |
| current-aco-mtp-on | 28672 | 28673 | on | xhigh | P2 | 68.29 | 68.52 | 68.60 | 56.85 | 56.85 | 56.85 |
| current-aco-mtp-on | 28672 | 28673 | on | xhigh | P3 | 79.51 | 79.64 | 79.76 | 73.91 | 73.91 | 73.91 |
| current-aco-mtp-on | 62464 | 62465 | on | off | P1 | 79.22 | 79.26 | 79.69 | 92.31 | 92.31 | 92.31 |
| current-aco-mtp-on | 62464 | 62465 | on | off | P2 | 60.60 | 60.64 | 60.90 | 60.27 | 60.27 | 60.27 |
| current-aco-mtp-on | 62464 | 62465 | on | off | P3 | 69.90 | 70.19 | 70.33 | 77.32 | 77.32 | 77.32 |
| current-aco-mtp-on | 62464 | 62465 | on | low | P1 | 75.83 | 75.86 | 75.94 | 87.50 | 87.50 | 87.50 |
| current-aco-mtp-on | 62464 | 62465 | on | low | P2 | 58.31 | 58.43 | 58.44 | 55.88 | 55.88 | 55.88 |
| current-aco-mtp-on | 62464 | 62465 | on | low | P3 | 73.79 | 73.95 | 74.18 | 83.89 | 83.89 | 83.89 |
| current-aco-mtp-on | 62464 | 62465 | on | medium | P1 | 78.24 | 78.29 | 78.53 | 92.14 | 92.14 | 92.14 |
| current-aco-mtp-on | 62464 | 62465 | on | medium | P2 | 66.12 | 66.26 | 66.33 | 70.18 | 70.18 | 70.18 |
| current-aco-mtp-on | 62464 | 62465 | on | medium | P3 | 70.38 | 70.71 | 70.71 | 78.14 | 78.14 | 78.14 |
| current-aco-mtp-on | 62464 | 62465 | on | xhigh | P1 | 75.06 | 75.31 | 75.71 | 87.06 | 87.06 | 87.06 |
| current-aco-mtp-on | 62464 | 62465 | on | xhigh | P2 | 55.05 | 55.37 | 57.69 | 51.36 | 51.36 | 54.95 |
| current-aco-mtp-on | 62464 | 62465 | on | xhigh | P3 | 64.23 | 64.34 | 64.79 | 67.06 | 67.06 | 67.75 |
| current-aco-mtp-on | 78848 | 78849 | on | off | P1 | 73.85 | 73.93 | 74.20 | 95.62 | 95.62 | 95.62 |
| current-aco-mtp-on | 78848 | 78849 | on | off | P2 | 54.54 | 55.98 | 56.14 | 57.69 | 60.56 | 60.71 |
| current-aco-mtp-on | 78848 | 78849 | on | off | P3 | 64.27 | 66.45 | 66.69 | 77.32 | 82.67 | 82.67 |
| current-aco-mtp-on | 78848 | 78849 | on | low | P1 | 70.35 | 70.39 | 70.52 | 89.35 | 89.81 | 89.91 |
| current-aco-mtp-on | 78848 | 78849 | on | low | P2 | 52.53 | 53.05 | 58.17 | 54.67 | 55.51 | 66.37 |
| current-aco-mtp-on | 78848 | 78849 | on | low | P3 | 68.05 | 68.47 | 68.83 | 85.71 | 86.30 | 86.64 |
| current-aco-mtp-on | 78848 | 78849 | on | medium | P1 | 68.96 | 69.49 | 69.72 | 87.24 | 87.87 | 88.54 |
| current-aco-mtp-on | 78848 | 78849 | on | medium | P2 | 55.77 | 56.10 | 57.07 | 61.00 | 61.80 | 63.71 |
| current-aco-mtp-on | 78848 | 78849 | on | medium | P3 | 61.38 | 61.54 | 64.40 | 72.84 | 72.84 | 78.96 |
| current-aco-mtp-on | 78848 | 78849 | on | xhigh | P1 | 69.43 | 69.66 | 70.09 | 88.24 | 89.17 | 89.83 |
| current-aco-mtp-on | 78848 | 78849 | on | xhigh | P2 | 52.44 | 52.95 | 54.97 | 54.32 | 54.88 | 59.50 |
| current-aco-mtp-on | 78848 | 78849 | on | xhigh | P3 | 53.21 | 58.82 | 65.66 | 56.12 | 67.06 | 80.39 |
| current-aco-mtp-on | 88064 | 88065 | on | off | P1 | 73.04 | 73.12 | 73.31 | 95.62 | 95.62 | 95.62 |
| current-aco-mtp-on | 88064 | 88065 | on | off | P2 | 55.43 | 57.40 | 57.51 | 61.18 | 64.29 | 65.22 |
| current-aco-mtp-on | 88064 | 88065 | on | off | P3 | 63.53 | 65.67 | 66.44 | 78.14 | 82.06 | 83.28 |
| current-aco-mtp-on | 88064 | 88065 | on | low | P1 | 68.64 | 71.02 | 71.19 | 87.39 | 92.92 | 92.92 |
| current-aco-mtp-on | 88064 | 88065 | on | low | P2 | 53.57 | 56.39 | 59.37 | 57.88 | 63.71 | 69.16 |
| current-aco-mtp-on | 88064 | 88065 | on | low | P3 | 66.61 | 67.34 | 67.89 | 83.84 | 85.37 | 86.30 |
| current-aco-mtp-on | 88064 | 88065 | on | medium | P1 | 66.10 | 66.42 | 67.25 | 82.89 | 83.50 | 84.80 |
| current-aco-mtp-on | 88064 | 88065 | on | medium | P2 | 56.11 | 56.62 | 57.18 | 63.35 | 64.18 | 65.32 |
| current-aco-mtp-on | 88064 | 88065 | on | medium | P3 | 62.98 | 63.35 | 65.77 | 77.32 | 77.32 | 82.94 |
| current-aco-mtp-on | 88064 | 88065 | on | xhigh | P1 | 68.12 | 68.34 | 69.94 | 86.93 | 87.19 | 90.68 |
| current-aco-mtp-on | 88064 | 88065 | on | xhigh | P2 | 51.52 | 54.25 | 55.18 | 53.94 | 60.00 | 61.02 |
| current-aco-mtp-on | 88064 | 88065 | on | xhigh | P3 | 51.65 | 53.07 | 53.84 | 54.19 | 57.57 | 58.58 |
| current-aco-mtp-on | 100352 | 100352 | on | off | P1 | 71.04 | 71.83 | 72.01 | 95.62 | 95.62 | 95.62 |
| current-aco-mtp-on | 100352 | 100352 | on | off | P2 | 55.88 | 56.72 | 57.32 | 63.70 | 63.82 | 65.00 |
| current-aco-mtp-on | 100352 | 100352 | on | off | P3 | 63.09 | 63.60 | 64.90 | 77.07 | 78.71 | 81.52 |
| current-aco-mtp-on | 100352 | 100352 | on | low | P1 | 70.71 | 70.95 | 71.29 | 92.78 | 92.78 | 94.27 |
| current-aco-mtp-on | 100352 | 100352 | on | low | P2 | 54.80 | 55.82 | 56.90 | 60.83 | 62.71 | 64.66 |
| current-aco-mtp-on | 100352 | 100352 | on | low | P3 | 61.54 | 62.33 | 62.47 | 74.38 | 75.47 | 76.51 |
| current-aco-mtp-on | 100352 | 100352 | on | medium | P1 | 66.34 | 67.02 | 68.66 | 84.02 | 85.83 | 89.16 |
| current-aco-mtp-on | 100352 | 100352 | on | medium | P2 | 56.75 | 56.98 | 57.25 | 65.03 | 65.51 | 65.99 |
| current-aco-mtp-on | 100352 | 100352 | on | medium | P3 | 62.31 | 62.71 | 63.15 | 76.27 | 76.51 | 77.32 |
| current-aco-mtp-on | 100352 | 100352 | on | xhigh | P1 | 67.01 | 67.35 | 67.77 | 85.39 | 86.05 | 86.93 |
| current-aco-mtp-on | 100352 | 100352 | on | xhigh | P2 | 50.21 | 52.59 | 53.07 | 51.79 | 56.60 | 57.62 |
| current-aco-mtp-on | 100352 | 100352 | on | xhigh | P3 | 50.27 | 54.61 | 54.85 | 51.79 | 60.56 | 61.00 |
| current-aco-mtp-on | 130048 | 130049 | on | off | P1 | 69.08 | 69.13 | 70.61 | 92.50 | 92.50 | 95.62 |
| current-aco-mtp-on | 130048 | 130049 | on | off | P2 | 50.76 | 53.71 | 55.31 | 55.15 | 61.27 | 65.28 |
| current-aco-mtp-on | 130048 | 130049 | on | off | P3 | 62.57 | 63.16 | 64.14 | 80.66 | 81.52 | 83.84 |
| current-aco-mtp-on | 130048 | 130049 | on | low | P1 | 65.47 | 67.16 | 67.53 | 86.81 | 89.84 | 90.09 |
| current-aco-mtp-on | 130048 | 130049 | on | low | P2 | 52.24 | 53.09 | 53.96 | 59.34 | 61.00 | 62.89 |
| current-aco-mtp-on | 130048 | 130049 | on | low | P3 | 61.46 | 62.43 | 62.80 | 78.39 | 80.66 | 80.66 |
| current-aco-mtp-on | 130048 | 130049 | on | medium | P1 | 65.97 | 66.15 | 66.27 | 87.59 | 87.89 | 87.89 |
| current-aco-mtp-on | 130048 | 130049 | on | medium | P2 | 52.28 | 53.43 | 53.77 | 59.62 | 61.62 | 62.71 |
| current-aco-mtp-on | 130048 | 130049 | on | medium | P3 | 60.32 | 61.11 | 61.43 | 75.47 | 78.14 | 78.14 |
| current-aco-mtp-on | 130048 | 130049 | on | xhigh | P1 | 65.63 | 65.77 | 66.35 | 87.50 | 87.71 | 88.14 |
| current-aco-mtp-on | 130048 | 130049 | on | xhigh | P2 | 49.61 | 50.45 | 51.36 | 54.10 | 55.84 | 57.84 |
| current-aco-mtp-on | 130048 | 130049 | on | xhigh | P3 | 50.76 | 51.65 | 53.14 | 56.99 | 58.15 | 61.34 |
| current-aco-mtp-on | 196608 | 196608 | on | off | P1 | 65.12 | 65.39 | 66.52 | 93.86 | 94.20 | 96.69 |
| current-aco-mtp-on | 196608 | 196608 | on | off | P2 | 48.46 | 48.83 | 50.56 | 57.38 | 58.33 | 61.21 |
| current-aco-mtp-on | 196608 | 196608 | on | off | P3 | 59.53 | 60.13 | 60.35 | 82.94 | 84.46 | 85.08 |
| current-aco-mtp-on | 196608 | 196608 | on | low | P1 | 62.23 | 62.40 | 63.05 | 88.32 | 88.94 | 89.81 |
| current-aco-mtp-on | 196608 | 196608 | on | low | P2 | 48.70 | 48.79 | 51.82 | 59.18 | 60.11 | 65.99 |
| current-aco-mtp-on | 196608 | 196608 | on | low | P3 | 56.04 | 56.46 | 57.00 | 74.69 | 76.51 | 77.32 |
| current-aco-mtp-on | 196608 | 196608 | on | medium | P1 | 59.35 | 62.72 | 62.81 | 82.12 | 89.82 | 89.93 |
| current-aco-mtp-on | 196608 | 196608 | on | medium | P2 | 49.34 | 51.53 | 52.23 | 60.72 | 65.32 | 67.06 |
| current-aco-mtp-on | 196608 | 196608 | on | medium | P3 | 57.52 | 58.55 | 58.64 | 78.64 | 80.92 | 80.92 |
| current-aco-mtp-on | 196608 | 196608 | on | xhigh | P1 | 61.03 | 61.99 | 62.35 | 86.54 | 88.19 | 89.83 |
| current-aco-mtp-on | 196608 | 196608 | on | xhigh | P2 | 44.16 | 45.09 | 46.68 | 49.19 | 51.60 | 55.38 |
| current-aco-mtp-on | 196608 | 196608 | on | xhigh | P3 | 44.97 | 48.37 | 52.42 | 52.44 | 58.58 | 68.96 |
| current-aco-mtp-on | 256000 | 256001 | on | off | P1 | 60.62 | 60.81 | 61.13 | 92.31 | 93.86 | 94.20 |
| current-aco-mtp-on | 256000 | 256001 | on | off | P2 | 48.01 | 48.41 | 48.83 | 64.86 | 65.32 | 66.90 |
| current-aco-mtp-on | 256000 | 256001 | on | off | P3 | 51.57 | 55.30 | 55.35 | 73.37 | 82.39 | 83.28 |
| current-aco-mtp-on | 256000 | 256001 | on | low | P1 | 58.13 | 59.09 | 60.61 | 88.93 | 91.23 | 93.56 |
| current-aco-mtp-on | 256000 | 256001 | on | low | P2 | 41.60 | 41.94 | 43.08 | 51.15 | 51.97 | 54.48 |
| current-aco-mtp-on | 256000 | 256001 | on | low | P3 | 52.17 | 53.16 | 53.25 | 76.75 | 77.07 | 78.64 |
| current-aco-mtp-on | 256000 | 256001 | on | medium | P1 | 58.19 | 58.53 | 59.30 | 88.54 | 89.51 | 90.49 |
| current-aco-mtp-on | 256000 | 256001 | on | medium | P2 | 46.00 | 47.86 | 48.28 | 61.45 | 65.70 | 66.67 |
| current-aco-mtp-on | 256000 | 256001 | on | medium | P3 | 51.91 | 53.45 | 55.07 | 74.14 | 78.39 | 83.28 |
| current-aco-mtp-on | 256000 | 256001 | on | xhigh | P1 | 55.31 | 57.86 | 59.06 | 82.67 | 87.90 | 90.83 |
| current-aco-mtp-on | 256000 | 256001 | on | xhigh | P2 | 42.61 | 42.98 | 43.26 | 53.25 | 53.57 | 54.13 |
| current-aco-mtp-on | 256000 | 256001 | on | xhigh | P3 | 43.05 | 45.41 | 47.08 | 54.74 | 60.11 | 64.18 |
| current-aco-mtp-on | 261120 | 261121 | on | off | P1 | 60.35 | 60.83 | 61.23 | 92.31 | 93.85 | 96.88 |
| current-aco-mtp-on | 261120 | 261121 | on | off | P2 | 44.37 | 46.04 | 46.46 | 57.27 | 60.32 | 61.21 |
| current-aco-mtp-on | 261120 | 261121 | on | off | P3 | 53.00 | 54.26 | 54.29 | 79.22 | 80.39 | 82.67 |
| current-aco-mtp-on | 261120 | 261121 | on | low | P1 | 57.86 | 58.22 | 58.79 | 90.00 | 90.00 | 90.00 |
| current-aco-mtp-on | 261120 | 261121 | on | low | P2 | 45.52 | 45.78 | 45.82 | 61.17 | 61.34 | 61.97 |
| current-aco-mtp-on | 261120 | 261121 | on | low | P3 | 54.71 | 55.30 | 56.73 | 81.79 | 82.39 | 86.30 |
| current-aco-mtp-on | 261120 | 261121 | on | medium | P1 | 57.49 | 57.60 | 58.03 | 87.89 | 88.24 | 89.42 |
| current-aco-mtp-on | 261120 | 261121 | on | medium | P2 | 43.64 | 45.27 | 45.65 | 56.27 | 60.83 | 61.00 |
| current-aco-mtp-on | 261120 | 261121 | on | medium | P3 | 50.00 | 50.54 | 51.39 | 71.65 | 72.09 | 74.92 |
| current-aco-mtp-on | 261120 | 261121 | on | xhigh | P1 | 57.30 | 57.75 | 58.03 | 87.30 | 88.66 | 89.83 |
| current-aco-mtp-on | 261120 | 261121 | on | xhigh | P2 | 43.36 | 44.03 | 44.65 | 55.20 | 56.50 | 58.33 |
| current-aco-mtp-on | 261120 | 261121 | on | xhigh | P3 | 42.27 | 44.83 | 49.30 | 53.51 | 59.62 | 69.16 |

## Exact nominal-C ACO/control comparisons

378 pairs match nominal C, MTP state, mode, and prompt. No interpolation or actual-C substitution is used.

| ACO family | Control family | Nominal C | Mode | Prompt | ACO actual C | Control actual C | ACO median tok/s | Control median tok/s | ACO delta % |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| current-aco-mtp-off | fork-cpu-mtp-off | 6144 | off | P1 | 6144 | 6145 | 49.04 | 18.68 | 162.51 |
| current-aco-mtp-off | fork-gpu-mtp-off | 6144 | off | P1 | 6144 | 6145 | 49.04 | 49.47 | -0.87 |
| current-aco-mtp-off | master-cpu-mtp-off | 6144 | off | P1 | 6144 | 6144 | 49.04 | 18.75 | 161.60 |
| current-aco-mtp-off | fork-cpu-mtp-off | 6144 | off | P2 | 6144 | 6145 | 49.16 | 18.69 | 162.99 |
| current-aco-mtp-off | fork-gpu-mtp-off | 6144 | off | P2 | 6144 | 6145 | 49.16 | 49.49 | -0.66 |
| current-aco-mtp-off | master-cpu-mtp-off | 6144 | off | P2 | 6144 | 6144 | 49.16 | 18.77 | 161.87 |
| current-aco-mtp-off | fork-cpu-mtp-off | 6144 | off | P3 | 6144 | 6145 | 49.01 | 18.79 | 160.80 |
| current-aco-mtp-off | fork-gpu-mtp-off | 6144 | off | P3 | 6144 | 6145 | 49.01 | 49.36 | -0.71 |
| current-aco-mtp-off | master-cpu-mtp-off | 6144 | off | P3 | 6144 | 6144 | 49.01 | 18.72 | 161.82 |
| current-aco-mtp-off | fork-cpu-mtp-off | 6144 | low | P1 | 6144 | 6145 | 48.95 | 18.73 | 161.33 |
| current-aco-mtp-off | fork-gpu-mtp-off | 6144 | low | P1 | 6144 | 6145 | 48.95 | 49.34 | -0.79 |
| current-aco-mtp-off | master-cpu-mtp-off | 6144 | low | P1 | 6144 | 6144 | 48.95 | 18.98 | 157.94 |
| current-aco-mtp-off | fork-cpu-mtp-off | 6144 | low | P2 | 6144 | 6145 | 49.00 | 18.72 | 161.84 |
| current-aco-mtp-off | fork-gpu-mtp-off | 6144 | low | P2 | 6144 | 6145 | 49.00 | 49.35 | -0.71 |
| current-aco-mtp-off | master-cpu-mtp-off | 6144 | low | P2 | 6144 | 6144 | 49.00 | 18.86 | 159.83 |
| current-aco-mtp-off | fork-cpu-mtp-off | 6144 | low | P3 | 6144 | 6145 | 49.00 | 18.72 | 161.72 |
| current-aco-mtp-off | fork-gpu-mtp-off | 6144 | low | P3 | 6144 | 6145 | 49.00 | 49.39 | -0.79 |
| current-aco-mtp-off | master-cpu-mtp-off | 6144 | low | P3 | 6144 | 6144 | 49.00 | 18.94 | 158.79 |
| current-aco-mtp-off | fork-cpu-mtp-off | 6144 | medium | P1 | 6144 | 6145 | 49.00 | 18.72 | 161.72 |
| current-aco-mtp-off | fork-gpu-mtp-off | 6144 | medium | P1 | 6144 | 6145 | 49.00 | 49.37 | -0.76 |
| current-aco-mtp-off | master-cpu-mtp-off | 6144 | medium | P1 | 6144 | 6144 | 49.00 | 18.87 | 159.72 |
| current-aco-mtp-off | fork-cpu-mtp-off | 6144 | medium | P2 | 6144 | 6145 | 49.00 | 18.69 | 162.13 |
| current-aco-mtp-off | fork-gpu-mtp-off | 6144 | medium | P2 | 6144 | 6145 | 49.00 | 49.35 | -0.71 |
| current-aco-mtp-off | master-cpu-mtp-off | 6144 | medium | P2 | 6144 | 6144 | 49.00 | 18.79 | 160.76 |
| current-aco-mtp-off | fork-cpu-mtp-off | 6144 | medium | P3 | 6144 | 6145 | 48.98 | 18.77 | 160.89 |
| current-aco-mtp-off | fork-gpu-mtp-off | 6144 | medium | P3 | 6144 | 6145 | 48.98 | 49.38 | -0.82 |
| current-aco-mtp-off | master-cpu-mtp-off | 6144 | medium | P3 | 6144 | 6144 | 48.98 | 18.95 | 158.46 |
| current-aco-mtp-off | fork-cpu-mtp-off | 6144 | xhigh | P1 | 6144 | 6145 | 48.97 | 18.74 | 161.29 |
| current-aco-mtp-off | fork-gpu-mtp-off | 6144 | xhigh | P1 | 6144 | 6145 | 48.97 | 49.37 | -0.82 |
| current-aco-mtp-off | master-cpu-mtp-off | 6144 | xhigh | P1 | 6144 | 6144 | 48.97 | 18.89 | 159.24 |
| current-aco-mtp-off | fork-cpu-mtp-off | 6144 | xhigh | P2 | 6144 | 6145 | 48.90 | 18.78 | 160.42 |
| current-aco-mtp-off | fork-gpu-mtp-off | 6144 | xhigh | P2 | 6144 | 6145 | 48.90 | 49.37 | -0.95 |
| current-aco-mtp-off | master-cpu-mtp-off | 6144 | xhigh | P2 | 6144 | 6144 | 48.90 | 18.98 | 157.70 |
| current-aco-mtp-off | fork-cpu-mtp-off | 6144 | xhigh | P3 | 6144 | 6145 | 48.96 | 18.75 | 161.10 |
| current-aco-mtp-off | fork-gpu-mtp-off | 6144 | xhigh | P3 | 6144 | 6145 | 48.96 | 49.36 | -0.80 |
| current-aco-mtp-off | master-cpu-mtp-off | 6144 | xhigh | P3 | 6144 | 6144 | 48.96 | 18.87 | 159.52 |
| current-aco-mtp-off | fork-cpu-mtp-off | 12288 | off | P1 | 12288 | 12289 | 48.04 | 16.45 | 192.06 |
| current-aco-mtp-off | fork-gpu-mtp-off | 12288 | off | P1 | 12288 | 12289 | 48.04 | 48.54 | -1.03 |
| current-aco-mtp-off | master-cpu-mtp-off | 12288 | off | P1 | 12288 | 12288 | 48.04 | 16.58 | 189.82 |
| current-aco-mtp-off | fork-cpu-mtp-off | 12288 | off | P2 | 12288 | 12289 | 48.18 | 16.47 | 192.54 |
| current-aco-mtp-off | fork-gpu-mtp-off | 12288 | off | P2 | 12288 | 12289 | 48.18 | 48.55 | -0.77 |
| current-aco-mtp-off | master-cpu-mtp-off | 12288 | off | P2 | 12288 | 12288 | 48.18 | 16.56 | 190.94 |
| current-aco-mtp-off | fork-cpu-mtp-off | 12288 | off | P3 | 12288 | 12289 | 48.06 | 16.50 | 191.30 |
| current-aco-mtp-off | fork-gpu-mtp-off | 12288 | off | P3 | 12288 | 12289 | 48.06 | 48.46 | -0.83 |
| current-aco-mtp-off | master-cpu-mtp-off | 12288 | off | P3 | 12288 | 12288 | 48.06 | 16.54 | 190.53 |
| current-aco-mtp-off | fork-cpu-mtp-off | 12288 | low | P1 | 12288 | 12289 | 47.98 | 16.46 | 191.57 |
| current-aco-mtp-off | fork-gpu-mtp-off | 12288 | low | P1 | 12288 | 12289 | 47.98 | 48.47 | -1.02 |
| current-aco-mtp-off | master-cpu-mtp-off | 12288 | low | P1 | 12288 | 12288 | 47.98 | 16.52 | 190.44 |
| current-aco-mtp-off | fork-cpu-mtp-off | 12288 | low | P2 | 12288 | 12289 | 48.00 | 16.52 | 190.55 |
| current-aco-mtp-off | fork-gpu-mtp-off | 12288 | low | P2 | 12288 | 12289 | 48.00 | 48.43 | -0.89 |
| current-aco-mtp-off | master-cpu-mtp-off | 12288 | low | P2 | 12288 | 12288 | 48.00 | 16.50 | 190.89 |
| current-aco-mtp-off | fork-cpu-mtp-off | 12288 | low | P3 | 12288 | 12289 | 48.03 | 16.44 | 192.25 |
| current-aco-mtp-off | fork-gpu-mtp-off | 12288 | low | P3 | 12288 | 12289 | 48.03 | 48.46 | -0.88 |
| current-aco-mtp-off | master-cpu-mtp-off | 12288 | low | P3 | 12288 | 12288 | 48.03 | 16.60 | 189.41 |
| current-aco-mtp-off | fork-cpu-mtp-off | 12288 | medium | P1 | 12288 | 12288 | 48.01 | 16.49 | 191.09 |
| current-aco-mtp-off | fork-gpu-mtp-off | 12288 | medium | P1 | 12288 | 12288 | 48.01 | 48.43 | -0.88 |
| current-aco-mtp-off | master-cpu-mtp-off | 12288 | medium | P1 | 12288 | 12288 | 48.01 | 16.56 | 189.83 |
| current-aco-mtp-off | master-gpu-mtp-off | 12288 | medium | P1 | 12288 | 12288 | 48.01 | 48.41 | -0.84 |
| current-aco-mtp-off | fork-cpu-mtp-off | 12288 | medium | P2 | 12288 | 12288 | 48.04 | 16.47 | 191.65 |
| current-aco-mtp-off | fork-gpu-mtp-off | 12288 | medium | P2 | 12288 | 12288 | 48.04 | 48.46 | -0.86 |
| current-aco-mtp-off | master-cpu-mtp-off | 12288 | medium | P2 | 12288 | 12288 | 48.04 | 16.51 | 190.95 |
| current-aco-mtp-off | fork-cpu-mtp-off | 12288 | medium | P3 | 12288 | 12288 | 48.03 | 16.49 | 191.29 |
| current-aco-mtp-off | fork-gpu-mtp-off | 12288 | medium | P3 | 12288 | 12288 | 48.03 | 48.42 | -0.82 |
| current-aco-mtp-off | master-cpu-mtp-off | 12288 | medium | P3 | 12288 | 12288 | 48.03 | 16.49 | 191.20 |
| current-aco-mtp-off | fork-cpu-mtp-off | 12288 | xhigh | P1 | 12288 | 12289 | 47.96 | 16.49 | 190.83 |
| current-aco-mtp-off | fork-gpu-mtp-off | 12288 | xhigh | P1 | 12288 | 12289 | 47.96 | 48.45 | -1.01 |
| current-aco-mtp-off | master-cpu-mtp-off | 12288 | xhigh | P1 | 12288 | 12288 | 47.96 | 16.52 | 190.28 |
| current-aco-mtp-off | master-gpu-mtp-off | 12288 | xhigh | P1 | 12288 | 12288 | 47.96 | 48.42 | -0.95 |
| current-aco-mtp-off | fork-cpu-mtp-off | 12288 | xhigh | P2 | 12288 | 12289 | 47.98 | 16.49 | 190.95 |
| current-aco-mtp-off | fork-gpu-mtp-off | 12288 | xhigh | P2 | 12288 | 12289 | 47.98 | 48.46 | -1.00 |
| current-aco-mtp-off | master-cpu-mtp-off | 12288 | xhigh | P2 | 12288 | 12288 | 47.98 | 16.46 | 191.53 |
| current-aco-mtp-off | fork-cpu-mtp-off | 12288 | xhigh | P3 | 12288 | 12289 | 48.03 | 16.42 | 192.52 |
| current-aco-mtp-off | fork-gpu-mtp-off | 12288 | xhigh | P3 | 12288 | 12289 | 48.03 | 48.44 | -0.83 |
| current-aco-mtp-off | master-cpu-mtp-off | 12288 | xhigh | P3 | 12288 | 12288 | 48.03 | 16.57 | 189.92 |
| current-aco-mtp-off | fork-cpu-mtp-off | 28672 | off | P1 | 28673 | 28673 | 46.09 | 12.52 | 268.06 |
| current-aco-mtp-off | fork-gpu-mtp-off | 28672 | off | P1 | 28673 | 28673 | 46.09 | 46.78 | -1.47 |
| current-aco-mtp-off | master-cpu-mtp-off | 28672 | off | P1 | 28673 | 28672 | 46.09 | 12.69 | 263.33 |
| current-aco-mtp-off | master-gpu-mtp-off | 28672 | off | P1 | 28673 | 28672 | 46.09 | 46.65 | -1.19 |
| current-aco-mtp-off | fork-cpu-mtp-off | 28672 | off | P2 | 28673 | 28673 | 46.23 | 12.56 | 267.99 |
| current-aco-mtp-off | fork-gpu-mtp-off | 28672 | off | P2 | 28673 | 28673 | 46.23 | 46.77 | -1.14 |
| current-aco-mtp-off | master-cpu-mtp-off | 28672 | off | P2 | 28673 | 28672 | 46.23 | 12.64 | 265.81 |
| current-aco-mtp-off | master-gpu-mtp-off | 28672 | off | P2 | 28673 | 28672 | 46.23 | 46.62 | -0.82 |
| current-aco-mtp-off | fork-cpu-mtp-off | 28672 | off | P3 | 28673 | 28673 | 46.10 | 12.51 | 268.43 |
| current-aco-mtp-off | fork-gpu-mtp-off | 28672 | off | P3 | 28673 | 28673 | 46.10 | 46.69 | -1.26 |
| current-aco-mtp-off | master-cpu-mtp-off | 28672 | off | P3 | 28673 | 28672 | 46.10 | 12.57 | 266.83 |
| current-aco-mtp-off | fork-cpu-mtp-off | 28672 | low | P1 | 28673 | 28673 | 45.96 | 12.53 | 266.90 |
| current-aco-mtp-off | fork-gpu-mtp-off | 28672 | low | P1 | 28673 | 28673 | 45.96 | 46.66 | -1.50 |
| current-aco-mtp-off | master-cpu-mtp-off | 28672 | low | P1 | 28673 | 28673 | 45.96 | 12.63 | 263.83 |
| current-aco-mtp-off | fork-cpu-mtp-off | 28672 | low | P2 | 28673 | 28673 | 45.98 | 12.54 | 266.72 |
| current-aco-mtp-off | fork-gpu-mtp-off | 28672 | low | P2 | 28673 | 28673 | 45.98 | 46.64 | -1.42 |
| current-aco-mtp-off | master-cpu-mtp-off | 28672 | low | P2 | 28673 | 28673 | 45.98 | 12.66 | 263.17 |
| current-aco-mtp-off | fork-cpu-mtp-off | 28672 | low | P3 | 28673 | 28673 | 46.04 | 12.53 | 267.59 |
| current-aco-mtp-off | fork-gpu-mtp-off | 28672 | low | P3 | 28673 | 28673 | 46.04 | 46.64 | -1.28 |
| current-aco-mtp-off | master-cpu-mtp-off | 28672 | low | P3 | 28673 | 28673 | 46.04 | 12.68 | 263.19 |
| current-aco-mtp-off | fork-cpu-mtp-off | 28672 | medium | P1 | 28673 | 28673 | 46.08 | 12.52 | 268.11 |
| current-aco-mtp-off | fork-gpu-mtp-off | 28672 | medium | P1 | 28673 | 28673 | 46.08 | 46.66 | -1.23 |
| current-aco-mtp-off | master-cpu-mtp-off | 28672 | medium | P1 | 28673 | 28673 | 46.08 | 12.68 | 263.54 |
| current-aco-mtp-off | fork-cpu-mtp-off | 28672 | medium | P2 | 28673 | 28673 | 46.08 | 12.53 | 267.74 |
| current-aco-mtp-off | fork-gpu-mtp-off | 28672 | medium | P2 | 28673 | 28673 | 46.08 | 46.63 | -1.18 |
| current-aco-mtp-off | master-cpu-mtp-off | 28672 | medium | P2 | 28673 | 28673 | 46.08 | 12.64 | 264.70 |
| current-aco-mtp-off | fork-cpu-mtp-off | 28672 | medium | P3 | 28673 | 28673 | 46.09 | 12.54 | 267.53 |
| current-aco-mtp-off | fork-gpu-mtp-off | 28672 | medium | P3 | 28673 | 28673 | 46.09 | 46.65 | -1.21 |
| current-aco-mtp-off | master-cpu-mtp-off | 28672 | medium | P3 | 28673 | 28673 | 46.09 | 12.62 | 265.08 |
| current-aco-mtp-off | fork-cpu-mtp-off | 28672 | xhigh | P1 | 28673 | 28673 | 45.95 | 12.56 | 265.91 |
| current-aco-mtp-off | fork-gpu-mtp-off | 28672 | xhigh | P1 | 28673 | 28673 | 45.95 | 46.67 | -1.53 |
| current-aco-mtp-off | master-cpu-mtp-off | 28672 | xhigh | P1 | 28673 | 28673 | 45.95 | 12.59 | 264.96 |
| current-aco-mtp-off | fork-cpu-mtp-off | 28672 | xhigh | P2 | 28673 | 28673 | 46.00 | 12.55 | 266.50 |
| current-aco-mtp-off | fork-gpu-mtp-off | 28672 | xhigh | P2 | 28673 | 28673 | 46.00 | 46.64 | -1.37 |
| current-aco-mtp-off | master-cpu-mtp-off | 28672 | xhigh | P2 | 28673 | 28673 | 46.00 | 12.62 | 264.43 |
| current-aco-mtp-off | fork-cpu-mtp-off | 28672 | xhigh | P3 | 28673 | 28673 | 46.03 | 12.52 | 267.71 |
| current-aco-mtp-off | fork-gpu-mtp-off | 28672 | xhigh | P3 | 28673 | 28673 | 46.03 | 46.65 | -1.33 |
| current-aco-mtp-off | master-cpu-mtp-off | 28672 | xhigh | P3 | 28673 | 28673 | 46.03 | 12.63 | 264.52 |
| current-aco-mtp-off | fork-cpu-mtp-off | 62464 | off | P1 | 62465 | 62465 | 42.33 | 8.44 | 401.41 |
| current-aco-mtp-off | fork-gpu-mtp-off | 62464 | off | P1 | 62465 | 62465 | 42.33 | 42.93 | -1.41 |
| current-aco-mtp-off | master-gpu-mtp-off | 62464 | off | P1 | 62465 | 62465 | 42.33 | 42.97 | -1.50 |
| current-aco-mtp-off | fork-cpu-mtp-off | 62464 | off | P2 | 62465 | 62465 | 42.32 | 8.43 | 402.09 |
| current-aco-mtp-off | fork-gpu-mtp-off | 62464 | off | P2 | 62465 | 62465 | 42.32 | 43.08 | -1.77 |
| current-aco-mtp-off | master-gpu-mtp-off | 62464 | off | P2 | 62465 | 62465 | 42.32 | 42.95 | -1.47 |
| current-aco-mtp-off | fork-cpu-mtp-off | 62464 | off | P3 | 62465 | 62465 | 42.17 | 8.47 | 397.60 |
| current-aco-mtp-off | fork-gpu-mtp-off | 62464 | off | P3 | 62465 | 62465 | 42.17 | 43.01 | -1.96 |
| current-aco-mtp-off | master-gpu-mtp-off | 62464 | off | P3 | 62465 | 62465 | 42.17 | 42.90 | -1.72 |
| current-aco-mtp-off | fork-cpu-mtp-off | 62464 | low | P1 | 62465 | 62465 | 42.05 | 8.47 | 396.40 |
| current-aco-mtp-off | fork-gpu-mtp-off | 62464 | low | P1 | 62465 | 62465 | 42.05 | 42.99 | -2.18 |
| current-aco-mtp-off | master-gpu-mtp-off | 62464 | low | P1 | 62465 | 62464 | 42.05 | 42.89 | -1.95 |
| current-aco-mtp-off | fork-cpu-mtp-off | 62464 | low | P2 | 62465 | 62465 | 41.89 | 8.47 | 394.70 |
| current-aco-mtp-off | fork-gpu-mtp-off | 62464 | low | P2 | 62465 | 62465 | 41.89 | 42.99 | -2.57 |
| current-aco-mtp-off | master-gpu-mtp-off | 62464 | low | P2 | 62465 | 62464 | 41.89 | 42.88 | -2.32 |
| current-aco-mtp-off | fork-cpu-mtp-off | 62464 | low | P3 | 62465 | 62465 | 42.12 | 8.48 | 396.55 |
| current-aco-mtp-off | fork-gpu-mtp-off | 62464 | low | P3 | 62465 | 62465 | 42.12 | 42.98 | -2.00 |
| current-aco-mtp-off | fork-cpu-mtp-off | 62464 | medium | P1 | 62465 | 62465 | 42.18 | 8.45 | 399.16 |
| current-aco-mtp-off | fork-gpu-mtp-off | 62464 | medium | P1 | 62465 | 62465 | 42.18 | 43.00 | -1.91 |
| current-aco-mtp-off | master-gpu-mtp-off | 62464 | medium | P1 | 62465 | 62465 | 42.18 | 42.87 | -1.62 |
| current-aco-mtp-off | fork-cpu-mtp-off | 62464 | medium | P2 | 62465 | 62465 | 42.18 | 8.51 | 395.36 |
| current-aco-mtp-off | fork-gpu-mtp-off | 62464 | medium | P2 | 62465 | 62465 | 42.18 | 42.99 | -1.89 |
| current-aco-mtp-off | master-gpu-mtp-off | 62464 | medium | P2 | 62465 | 62465 | 42.18 | 42.90 | -1.67 |
| current-aco-mtp-off | fork-cpu-mtp-off | 62464 | medium | P3 | 62465 | 62465 | 42.18 | 8.47 | 398.09 |
| current-aco-mtp-off | fork-gpu-mtp-off | 62464 | medium | P3 | 62465 | 62465 | 42.18 | 42.98 | -1.86 |
| current-aco-mtp-off | master-gpu-mtp-off | 62464 | medium | P3 | 62465 | 62465 | 42.18 | 42.88 | -1.61 |
| current-aco-mtp-off | fork-cpu-mtp-off | 62464 | xhigh | P1 | 62465 | 62464 | 41.99 | 8.44 | 397.45 |
| current-aco-mtp-off | fork-gpu-mtp-off | 62464 | xhigh | P1 | 62465 | 62464 | 41.99 | 43.00 | -2.34 |
| current-aco-mtp-off | master-gpu-mtp-off | 62464 | xhigh | P1 | 62465 | 62464 | 41.99 | 42.91 | -2.14 |
| current-aco-mtp-off | fork-cpu-mtp-off | 62464 | xhigh | P2 | 62465 | 62464 | 42.06 | 8.48 | 395.87 |
| current-aco-mtp-off | fork-gpu-mtp-off | 62464 | xhigh | P2 | 62465 | 62464 | 42.06 | 42.98 | -2.13 |
| current-aco-mtp-off | master-gpu-mtp-off | 62464 | xhigh | P2 | 62465 | 62464 | 42.06 | 42.87 | -1.88 |
| current-aco-mtp-off | fork-cpu-mtp-off | 62464 | xhigh | P3 | 62465 | 62464 | 42.14 | 8.47 | 397.56 |
| current-aco-mtp-off | fork-gpu-mtp-off | 62464 | xhigh | P3 | 62465 | 62464 | 42.14 | 42.99 | -1.97 |
| current-aco-mtp-off | master-gpu-mtp-off | 62464 | xhigh | P3 | 62465 | 62464 | 42.14 | 42.89 | -1.75 |
| current-aco-mtp-off | fork-gpu-mtp-off | 78848 | off | P1 | 78849 | 78849 | 37.51 | 41.30 | -9.19 |
| current-aco-mtp-off | master-gpu-mtp-off | 78848 | off | P1 | 78849 | 78849 | 37.51 | 41.15 | -8.87 |
| current-aco-mtp-off | fork-gpu-mtp-off | 78848 | off | P2 | 78849 | 78849 | 37.55 | 41.38 | -9.26 |
| current-aco-mtp-off | master-gpu-mtp-off | 78848 | off | P2 | 78849 | 78849 | 37.55 | 41.28 | -9.03 |
| current-aco-mtp-off | fork-gpu-mtp-off | 78848 | off | P3 | 78849 | 78849 | 37.50 | 41.36 | -9.32 |
| current-aco-mtp-off | master-gpu-mtp-off | 78848 | off | P3 | 78849 | 78849 | 37.50 | 41.20 | -8.98 |
| current-aco-mtp-off | fork-gpu-mtp-off | 78848 | low | P1 | 78849 | 78849 | 37.45 | 41.37 | -9.47 |
| current-aco-mtp-off | master-gpu-mtp-off | 78848 | low | P1 | 78849 | 78849 | 37.45 | 41.19 | -9.06 |
| current-aco-mtp-off | fork-gpu-mtp-off | 78848 | low | P2 | 78849 | 78849 | 37.46 | 41.37 | -9.46 |
| current-aco-mtp-off | master-gpu-mtp-off | 78848 | low | P2 | 78849 | 78849 | 37.46 | 41.18 | -9.05 |
| current-aco-mtp-off | fork-gpu-mtp-off | 78848 | low | P3 | 78849 | 78849 | 37.50 | 41.35 | -9.32 |
| current-aco-mtp-off | master-gpu-mtp-off | 78848 | low | P3 | 78849 | 78849 | 37.50 | 41.21 | -9.02 |
| current-aco-mtp-off | fork-gpu-mtp-off | 78848 | medium | P1 | 78849 | 78849 | 37.50 | 41.36 | -9.33 |
| current-aco-mtp-off | master-gpu-mtp-off | 78848 | medium | P1 | 78849 | 78849 | 37.50 | 41.17 | -8.91 |
| current-aco-mtp-off | fork-gpu-mtp-off | 78848 | medium | P2 | 78849 | 78849 | 37.49 | 41.35 | -9.34 |
| current-aco-mtp-off | master-gpu-mtp-off | 78848 | medium | P2 | 78849 | 78849 | 37.49 | 41.19 | -8.99 |
| current-aco-mtp-off | fork-gpu-mtp-off | 78848 | medium | P3 | 78849 | 78849 | 37.51 | 41.35 | -9.29 |
| current-aco-mtp-off | master-gpu-mtp-off | 78848 | medium | P3 | 78849 | 78849 | 37.51 | 41.17 | -8.89 |
| current-aco-mtp-off | fork-gpu-mtp-off | 78848 | xhigh | P1 | 78849 | 78849 | 37.40 | 41.37 | -9.60 |
| current-aco-mtp-off | master-gpu-mtp-off | 78848 | xhigh | P1 | 78849 | 78849 | 37.40 | 41.20 | -9.23 |
| current-aco-mtp-off | fork-gpu-mtp-off | 78848 | xhigh | P2 | 78849 | 78849 | 37.47 | 41.37 | -9.42 |
| current-aco-mtp-off | master-gpu-mtp-off | 78848 | xhigh | P2 | 78849 | 78849 | 37.47 | 41.21 | -9.07 |
| current-aco-mtp-off | fork-gpu-mtp-off | 78848 | xhigh | P3 | 78849 | 78849 | 37.50 | 41.37 | -9.34 |
| current-aco-mtp-off | master-gpu-mtp-off | 78848 | xhigh | P3 | 78849 | 78849 | 37.50 | 41.17 | -8.91 |
| current-aco-mtp-off | fork-gpu-mtp-off | 88064 | off | P1 | 88065 | 88065 | 37.49 | 40.48 | -7.39 |
| current-aco-mtp-off | fork-gpu-mtp-off | 88064 | off | P2 | 88065 | 88065 | 37.58 | 40.56 | -7.36 |
| current-aco-mtp-off | fork-gpu-mtp-off | 88064 | off | P3 | 88065 | 88065 | 37.47 | 40.52 | -7.54 |
| current-aco-mtp-off | fork-gpu-mtp-off | 88064 | low | P1 | 88065 | 88065 | 37.46 | 40.52 | -7.56 |
| current-aco-mtp-off | fork-gpu-mtp-off | 88064 | low | P2 | 88065 | 88065 | 37.49 | 40.53 | -7.49 |
| current-aco-mtp-off | fork-gpu-mtp-off | 88064 | low | P3 | 88065 | 88065 | 37.51 | 40.50 | -7.40 |
| current-aco-mtp-off | fork-gpu-mtp-off | 88064 | medium | P1 | 88065 | 88065 | 37.47 | 40.52 | -7.52 |
| current-aco-mtp-off | master-gpu-mtp-off | 88064 | medium | P1 | 88065 | 88065 | 37.47 | 40.35 | -7.13 |
| current-aco-mtp-off | fork-gpu-mtp-off | 88064 | medium | P2 | 88065 | 88065 | 37.50 | 40.50 | -7.41 |
| current-aco-mtp-off | master-gpu-mtp-off | 88064 | medium | P2 | 88065 | 88065 | 37.50 | 40.31 | -6.98 |
| current-aco-mtp-off | fork-gpu-mtp-off | 88064 | medium | P3 | 88065 | 88065 | 37.51 | 40.52 | -7.43 |
| current-aco-mtp-off | master-gpu-mtp-off | 88064 | medium | P3 | 88065 | 88065 | 37.51 | 40.32 | -6.98 |
| current-aco-mtp-off | fork-gpu-mtp-off | 88064 | xhigh | P1 | 88065 | 88065 | 37.39 | 40.51 | -7.71 |
| current-aco-mtp-off | master-gpu-mtp-off | 88064 | xhigh | P1 | 88065 | 88065 | 37.39 | 40.32 | -7.27 |
| current-aco-mtp-off | fork-gpu-mtp-off | 88064 | xhigh | P2 | 88065 | 88065 | 37.49 | 40.50 | -7.45 |
| current-aco-mtp-off | master-gpu-mtp-off | 88064 | xhigh | P2 | 88065 | 88065 | 37.49 | 40.33 | -7.04 |
| current-aco-mtp-off | fork-gpu-mtp-off | 88064 | xhigh | P3 | 88065 | 88065 | 37.51 | 40.52 | -7.43 |
| current-aco-mtp-off | master-gpu-mtp-off | 88064 | xhigh | P3 | 88065 | 88065 | 37.51 | 40.35 | -7.04 |
| current-aco-mtp-off | fork-gpu-mtp-off | 100352 | off | P1 | 100352 | 100353 | 37.50 | 39.52 | -5.09 |
| current-aco-mtp-off | master-gpu-mtp-off | 100352 | off | P1 | 100352 | 100353 | 37.50 | 39.29 | -4.55 |
| current-aco-mtp-off | fork-gpu-mtp-off | 100352 | off | P2 | 100352 | 100353 | 37.56 | 39.57 | -5.08 |
| current-aco-mtp-off | master-gpu-mtp-off | 100352 | off | P2 | 100352 | 100353 | 37.56 | 39.38 | -4.60 |
| current-aco-mtp-off | fork-gpu-mtp-off | 100352 | off | P3 | 100352 | 100353 | 37.49 | 39.52 | -5.14 |
| current-aco-mtp-off | master-gpu-mtp-off | 100352 | off | P3 | 100352 | 100353 | 37.49 | 39.30 | -4.62 |
| current-aco-mtp-off | fork-gpu-mtp-off | 100352 | low | P1 | 100352 | 100353 | 37.41 | 39.54 | -5.37 |
| current-aco-mtp-off | master-gpu-mtp-off | 100352 | low | P1 | 100352 | 100353 | 37.41 | 39.31 | -4.83 |
| current-aco-mtp-off | fork-gpu-mtp-off | 100352 | low | P2 | 100352 | 100353 | 37.50 | 39.52 | -5.11 |
| current-aco-mtp-off | master-gpu-mtp-off | 100352 | low | P2 | 100352 | 100353 | 37.50 | 39.33 | -4.65 |
| current-aco-mtp-off | fork-gpu-mtp-off | 100352 | low | P3 | 100352 | 100353 | 37.49 | 39.53 | -5.16 |
| current-aco-mtp-off | master-gpu-mtp-off | 100352 | low | P3 | 100352 | 100353 | 37.49 | 39.30 | -4.60 |
| current-aco-mtp-off | fork-gpu-mtp-off | 100352 | medium | P1 | 100352 | 100353 | 37.48 | 39.53 | -5.19 |
| current-aco-mtp-off | master-gpu-mtp-off | 100352 | medium | P1 | 100352 | 100353 | 37.48 | 39.30 | -4.63 |
| current-aco-mtp-off | fork-gpu-mtp-off | 100352 | medium | P2 | 100352 | 100353 | 37.49 | 39.52 | -5.12 |
| current-aco-mtp-off | master-gpu-mtp-off | 100352 | medium | P2 | 100352 | 100353 | 37.49 | 39.29 | -4.58 |
| current-aco-mtp-off | fork-gpu-mtp-off | 100352 | medium | P3 | 100352 | 100353 | 37.50 | 39.52 | -5.12 |
| current-aco-mtp-off | master-gpu-mtp-off | 100352 | medium | P3 | 100352 | 100353 | 37.50 | 39.31 | -4.62 |
| current-aco-mtp-off | fork-gpu-mtp-off | 100352 | xhigh | P1 | 100352 | 100353 | 37.47 | 39.50 | -5.16 |
| current-aco-mtp-off | master-gpu-mtp-off | 100352 | xhigh | P1 | 100352 | 100353 | 37.47 | 39.30 | -4.67 |
| current-aco-mtp-off | fork-gpu-mtp-off | 100352 | xhigh | P2 | 100352 | 100353 | 37.39 | 39.53 | -5.42 |
| current-aco-mtp-off | master-gpu-mtp-off | 100352 | xhigh | P2 | 100352 | 100353 | 37.39 | 39.31 | -4.90 |
| current-aco-mtp-off | fork-gpu-mtp-off | 100352 | xhigh | P3 | 100352 | 100353 | 37.50 | 39.52 | -5.12 |
| current-aco-mtp-off | master-gpu-mtp-off | 100352 | xhigh | P3 | 100352 | 100353 | 37.50 | 39.31 | -4.60 |
| current-aco-mtp-off | fork-cpu-mtp-off | 130048 | off | P1 | 130049 | 130049 | 37.50 | 5.12 | 631.84 |
| current-aco-mtp-off | fork-cpu-mtp-off | 130048 | off | P2 | 130049 | 130049 | 37.57 | 5.14 | 630.70 |
| current-aco-mtp-off | fork-cpu-mtp-off | 130048 | off | P3 | 130049 | 130049 | 37.49 | 5.13 | 630.87 |
| current-aco-mtp-off | fork-cpu-mtp-off | 130048 | low | P1 | 130049 | 130049 | 37.43 | 5.13 | 629.94 |
| current-aco-mtp-off | fork-cpu-mtp-off | 130048 | low | P2 | 130049 | 130049 | 37.50 | 5.14 | 630.31 |
| current-aco-mtp-off | fork-cpu-mtp-off | 130048 | low | P3 | 130049 | 130049 | 37.49 | 5.13 | 630.89 |
| current-aco-mtp-off | fork-cpu-mtp-off | 130048 | medium | P1 | 130049 | 130049 | 37.47 | 5.12 | 631.44 |
| current-aco-mtp-off | fork-cpu-mtp-off | 130048 | medium | P2 | 130049 | 130049 | 37.47 | 5.13 | 629.75 |
| current-aco-mtp-off | fork-cpu-mtp-off | 130048 | medium | P3 | 130049 | 130049 | 37.48 | 5.12 | 632.12 |
| current-aco-mtp-off | fork-cpu-mtp-off | 130048 | xhigh | P1 | 130049 | 130049 | 37.45 | 5.14 | 627.97 |
| current-aco-mtp-off | fork-cpu-mtp-off | 130048 | xhigh | P2 | 130049 | 130049 | 37.41 | 5.14 | 627.97 |
| current-aco-mtp-off | fork-cpu-mtp-off | 130048 | xhigh | P3 | 130049 | 130049 | 37.48 | 5.12 | 631.88 |
| current-aco-mtp-off | fork-cpu-mtp-off | 196608 | off | P1 | 196608 | 196609 | 37.49 | 3.72 | 907.39 |
| current-aco-mtp-off | fork-cpu-mtp-off | 196608 | off | P2 | 196608 | 196609 | 37.56 | 3.73 | 908.22 |
| current-aco-mtp-off | fork-cpu-mtp-off | 196608 | off | P3 | 196608 | 196609 | 37.48 | 3.73 | 904.45 |
| current-aco-mtp-off | fork-cpu-mtp-off | 196608 | low | P1 | 196608 | 196609 | 37.39 | 3.71 | 908.72 |
| current-aco-mtp-off | fork-cpu-mtp-off | 196608 | low | P2 | 196608 | 196609 | 37.48 | 3.71 | 909.80 |
| current-aco-mtp-off | fork-cpu-mtp-off | 196608 | low | P3 | 196608 | 196609 | 37.49 | 3.71 | 909.71 |
| current-aco-mtp-off | fork-cpu-mtp-off | 196608 | medium | P1 | 196608 | 196609 | 37.48 | 3.71 | 910.74 |
| current-aco-mtp-off | fork-cpu-mtp-off | 196608 | medium | P2 | 196608 | 196609 | 37.49 | 3.71 | 909.91 |
| current-aco-mtp-off | fork-cpu-mtp-off | 196608 | medium | P3 | 196608 | 196609 | 37.47 | 3.71 | 908.77 |
| current-aco-mtp-off | fork-cpu-mtp-off | 196608 | xhigh | P1 | 196608 | 196609 | 37.44 | 3.71 | 909.54 |
| current-aco-mtp-off | fork-cpu-mtp-off | 196608 | xhigh | P2 | 196608 | 196609 | 37.38 | 3.69 | 912.99 |
| current-aco-mtp-off | fork-cpu-mtp-off | 196608 | xhigh | P3 | 196608 | 196609 | 37.47 | 3.71 | 909.74 |
| current-aco-mtp-off | fork-cpu-mtp-off | 256000 | off | P1 | 256001 | 256001 | 37.42 | 2.98 | 1157.69 |
| current-aco-mtp-off | fork-cpu-mtp-off | 256000 | off | P2 | 256001 | 256001 | 37.57 | 2.98 | 1162.23 |
| current-aco-mtp-off | fork-cpu-mtp-off | 256000 | off | P3 | 256001 | 256001 | 37.44 | 2.98 | 1156.16 |
| current-aco-mtp-off | fork-cpu-mtp-off | 256000 | low | P1 | 256001 | 255999 | 37.42 | 2.98 | 1157.66 |
| current-aco-mtp-off | fork-cpu-mtp-off | 256000 | low | P2 | 256001 | 255999 | 37.32 | 2.98 | 1150.19 |
| current-aco-mtp-off | fork-cpu-mtp-off | 256000 | low | P3 | 256001 | 255999 | 37.46 | 2.98 | 1156.61 |
| current-aco-mtp-off | fork-cpu-mtp-off | 256000 | medium | P1 | 256001 | 255999 | 37.45 | 2.98 | 1155.29 |
| current-aco-mtp-off | fork-cpu-mtp-off | 256000 | medium | P2 | 256001 | 255999 | 37.45 | 2.99 | 1153.66 |
| current-aco-mtp-off | fork-cpu-mtp-off | 256000 | medium | P3 | 256001 | 255999 | 37.44 | 2.99 | 1153.23 |
| current-aco-mtp-off | fork-cpu-mtp-off | 256000 | xhigh | P1 | 256001 | 255999 | 37.41 | 2.98 | 1157.06 |
| current-aco-mtp-off | fork-cpu-mtp-off | 256000 | xhigh | P2 | 256001 | 255999 | 37.37 | 2.99 | 1151.52 |
| current-aco-mtp-off | fork-cpu-mtp-off | 256000 | xhigh | P3 | 256001 | 255999 | 37.46 | 2.98 | 1157.56 |
| current-aco-mtp-on | fork-cpu-mtp-on | 6144 | off | P1 | 6144 | 6145 | 102.42 | 30.38 | 237.15 |
| current-aco-mtp-on | fork-gpu-mtp-on | 6144 | off | P1 | 6144 | 6144 | 102.42 | 102.30 | 0.11 |
| current-aco-mtp-on | master-cpu-mtp-on | 6144 | off | P1 | 6144 | 6144 | 102.42 | 29.02 | 252.89 |
| current-aco-mtp-on | master-gpu-mtp-on | 6144 | off | P1 | 6144 | 6144 | 102.42 | 105.92 | -3.31 |
| current-aco-mtp-on | fork-cpu-mtp-on | 6144 | off | P2 | 6144 | 6145 | 81.97 | 21.63 | 278.94 |
| current-aco-mtp-on | fork-gpu-mtp-on | 6144 | off | P2 | 6144 | 6144 | 81.97 | 74.91 | 9.42 |
| current-aco-mtp-on | master-cpu-mtp-on | 6144 | off | P2 | 6144 | 6144 | 81.97 | 21.78 | 276.29 |
| current-aco-mtp-on | master-gpu-mtp-on | 6144 | off | P2 | 6144 | 6144 | 81.97 | 74.16 | 10.53 |
| current-aco-mtp-on | fork-cpu-mtp-on | 6144 | off | P3 | 6144 | 6145 | 92.69 | 26.87 | 244.95 |
| current-aco-mtp-on | fork-gpu-mtp-on | 6144 | off | P3 | 6144 | 6144 | 92.69 | 87.75 | 5.63 |
| current-aco-mtp-on | master-cpu-mtp-on | 6144 | off | P3 | 6144 | 6144 | 92.69 | 24.95 | 271.55 |
| current-aco-mtp-on | master-gpu-mtp-on | 6144 | off | P3 | 6144 | 6144 | 92.69 | 95.01 | -2.44 |
| current-aco-mtp-on | fork-cpu-mtp-on | 6144 | low | P1 | 6144 | 6145 | 100.11 | 26.86 | 272.73 |
| current-aco-mtp-on | fork-gpu-mtp-on | 6144 | low | P1 | 6144 | 6144 | 100.11 | 100.24 | -0.13 |
| current-aco-mtp-on | master-cpu-mtp-on | 6144 | low | P1 | 6144 | 6144 | 100.11 | 29.03 | 244.89 |
| current-aco-mtp-on | master-gpu-mtp-on | 6144 | low | P1 | 6144 | 6144 | 100.11 | 104.15 | -3.88 |
| current-aco-mtp-on | fork-cpu-mtp-on | 6144 | low | P2 | 6144 | 6145 | 75.56 | 21.96 | 244.13 |
| current-aco-mtp-on | fork-gpu-mtp-on | 6144 | low | P2 | 6144 | 6144 | 75.56 | 75.83 | -0.36 |
| current-aco-mtp-on | master-cpu-mtp-on | 6144 | low | P2 | 6144 | 6144 | 75.56 | 22.41 | 237.12 |
| current-aco-mtp-on | master-gpu-mtp-on | 6144 | low | P2 | 6144 | 6144 | 75.56 | 79.36 | -4.79 |
| current-aco-mtp-on | fork-cpu-mtp-on | 6144 | low | P3 | 6144 | 6145 | 92.64 | 27.33 | 239.01 |
| current-aco-mtp-on | fork-gpu-mtp-on | 6144 | low | P3 | 6144 | 6144 | 92.64 | 88.13 | 5.11 |
| current-aco-mtp-on | master-cpu-mtp-on | 6144 | low | P3 | 6144 | 6144 | 92.64 | 25.73 | 259.98 |
| current-aco-mtp-on | master-gpu-mtp-on | 6144 | low | P3 | 6144 | 6144 | 92.64 | 96.66 | -4.16 |
| current-aco-mtp-on | fork-cpu-mtp-on | 6144 | medium | P1 | 6144 | 6145 | 99.85 | 28.12 | 255.05 |
| current-aco-mtp-on | fork-gpu-mtp-on | 6144 | medium | P1 | 6144 | 6144 | 99.85 | 100.00 | -0.15 |
| current-aco-mtp-on | master-gpu-mtp-on | 6144 | medium | P1 | 6144 | 6144 | 99.85 | 95.31 | 4.77 |
| current-aco-mtp-on | fork-cpu-mtp-on | 6144 | medium | P2 | 6144 | 6145 | 80.45 | 22.22 | 261.98 |
| current-aco-mtp-on | fork-gpu-mtp-on | 6144 | medium | P2 | 6144 | 6144 | 80.45 | 74.10 | 8.56 |
| current-aco-mtp-on | master-gpu-mtp-on | 6144 | medium | P2 | 6144 | 6144 | 80.45 | 78.88 | 1.99 |
| current-aco-mtp-on | fork-cpu-mtp-on | 6144 | medium | P3 | 6144 | 6145 | 90.73 | 26.98 | 236.33 |
| current-aco-mtp-on | fork-gpu-mtp-on | 6144 | medium | P3 | 6144 | 6144 | 90.73 | 89.93 | 0.89 |
| current-aco-mtp-on | master-gpu-mtp-on | 6144 | medium | P3 | 6144 | 6144 | 90.73 | 97.08 | -6.54 |
| current-aco-mtp-on | fork-cpu-mtp-on | 6144 | xhigh | P1 | 6144 | 6145 | 99.90 | 29.16 | 242.62 |
| current-aco-mtp-on | fork-gpu-mtp-on | 6144 | xhigh | P1 | 6144 | 6144 | 99.90 | 97.56 | 2.39 |
| current-aco-mtp-on | master-cpu-mtp-on | 6144 | xhigh | P1 | 6144 | 6144 | 99.90 | 16.84 | 493.22 |
| current-aco-mtp-on | master-gpu-mtp-on | 6144 | xhigh | P1 | 6144 | 6144 | 99.90 | 100.06 | -0.16 |
| current-aco-mtp-on | fork-cpu-mtp-on | 6144 | xhigh | P2 | 6144 | 6145 | 76.09 | 24.21 | 214.29 |
| current-aco-mtp-on | fork-gpu-mtp-on | 6144 | xhigh | P2 | 6144 | 6144 | 76.09 | 72.53 | 4.91 |
| current-aco-mtp-on | master-cpu-mtp-on | 6144 | xhigh | P2 | 6144 | 6144 | 76.09 | 14.99 | 407.55 |
| current-aco-mtp-on | master-gpu-mtp-on | 6144 | xhigh | P2 | 6144 | 6144 | 76.09 | 78.54 | -3.12 |
| current-aco-mtp-on | fork-cpu-mtp-on | 6144 | xhigh | P3 | 6144 | 6145 | 80.00 | 24.44 | 227.38 |
| current-aco-mtp-on | fork-gpu-mtp-on | 6144 | xhigh | P3 | 6144 | 6144 | 80.00 | 79.67 | 0.42 |
| current-aco-mtp-on | master-cpu-mtp-on | 6144 | xhigh | P3 | 6144 | 6144 | 80.00 | 22.16 | 260.94 |
| current-aco-mtp-on | master-gpu-mtp-on | 6144 | xhigh | P3 | 6144 | 6144 | 80.00 | 90.72 | -11.82 |
| current-aco-mtp-on | fork-cpu-mtp-on | 12288 | off | P1 | 12288 | 12288 | 101.63 | 26.95 | 277.08 |
| current-aco-mtp-on | fork-gpu-mtp-on | 12288 | off | P1 | 12288 | 12289 | 101.63 | 99.17 | 2.48 |
| current-aco-mtp-on | master-cpu-mtp-on | 12288 | off | P1 | 12288 | 12288 | 101.63 | 24.55 | 314.03 |
| current-aco-mtp-on | master-gpu-mtp-on | 12288 | off | P1 | 12288 | 12289 | 101.63 | 100.32 | 1.31 |
| current-aco-mtp-on | fork-cpu-mtp-on | 12288 | off | P2 | 12288 | 12288 | 79.77 | 22.06 | 261.66 |
| current-aco-mtp-on | fork-gpu-mtp-on | 12288 | off | P2 | 12288 | 12289 | 79.77 | 83.44 | -4.39 |
| current-aco-mtp-on | master-cpu-mtp-on | 12288 | off | P2 | 12288 | 12288 | 79.77 | 20.07 | 297.42 |
| current-aco-mtp-on | master-gpu-mtp-on | 12288 | off | P2 | 12288 | 12289 | 79.77 | 81.81 | -2.50 |
| current-aco-mtp-on | fork-cpu-mtp-on | 12288 | off | P3 | 12288 | 12288 | 90.69 | 26.15 | 246.78 |
| current-aco-mtp-on | fork-gpu-mtp-on | 12288 | off | P3 | 12288 | 12289 | 90.69 | 90.83 | -0.16 |
| current-aco-mtp-on | master-cpu-mtp-on | 12288 | off | P3 | 12288 | 12288 | 90.69 | 22.47 | 303.67 |
| current-aco-mtp-on | master-gpu-mtp-on | 12288 | off | P3 | 12288 | 12289 | 90.69 | 90.41 | 0.31 |
| current-aco-mtp-on | fork-cpu-mtp-on | 12288 | low | P1 | 12288 | 12717 | 95.20 | 28.35 | 235.79 |
| current-aco-mtp-on | fork-gpu-mtp-on | 12288 | low | P1 | 12288 | 12289 | 95.20 | 94.71 | 0.52 |
| current-aco-mtp-on | master-gpu-mtp-on | 12288 | low | P1 | 12288 | 12289 | 95.20 | 94.50 | 0.74 |
| current-aco-mtp-on | fork-cpu-mtp-on | 12288 | low | P2 | 12288 | 13822 | 80.11 | 28.61 | 179.99 |
| current-aco-mtp-on | fork-gpu-mtp-on | 12288 | low | P2 | 12288 | 12289 | 80.11 | 76.87 | 4.21 |
| current-aco-mtp-on | master-gpu-mtp-on | 12288 | low | P2 | 12288 | 12289 | 80.11 | 73.21 | 9.43 |
| current-aco-mtp-on | fork-cpu-mtp-on | 12288 | low | P3 | 12288 | 15174 | 88.83 | 27.42 | 223.88 |
| current-aco-mtp-on | fork-gpu-mtp-on | 12288 | low | P3 | 12288 | 12289 | 88.83 | 81.24 | 9.33 |
| current-aco-mtp-on | master-gpu-mtp-on | 12288 | low | P3 | 12288 | 12289 | 88.83 | 90.33 | -1.66 |
| current-aco-mtp-on | fork-gpu-mtp-on | 12288 | medium | P1 | 12288 | 12288 | 92.32 | 95.81 | -3.64 |
| current-aco-mtp-on | master-gpu-mtp-on | 12288 | medium | P1 | 12288 | 12288 | 92.32 | 97.02 | -4.84 |
| current-aco-mtp-on | fork-gpu-mtp-on | 12288 | medium | P2 | 12288 | 12288 | 77.53 | 79.44 | -2.40 |
| current-aco-mtp-on | master-gpu-mtp-on | 12288 | medium | P2 | 12288 | 12288 | 77.53 | 85.03 | -8.82 |
| current-aco-mtp-on | fork-gpu-mtp-on | 12288 | medium | P3 | 12288 | 12288 | 89.79 | 93.16 | -3.62 |
| current-aco-mtp-on | master-gpu-mtp-on | 12288 | medium | P3 | 12288 | 12288 | 89.79 | 94.48 | -4.96 |
| current-aco-mtp-on | fork-gpu-mtp-on | 12288 | xhigh | P1 | 12288 | 12289 | 93.50 | 92.74 | 0.82 |
| current-aco-mtp-on | master-gpu-mtp-on | 12288 | xhigh | P1 | 12288 | 12289 | 93.50 | 92.70 | 0.86 |
| current-aco-mtp-on | fork-gpu-mtp-on | 12288 | xhigh | P2 | 12288 | 12289 | 74.29 | 69.35 | 7.13 |
| current-aco-mtp-on | master-gpu-mtp-on | 12288 | xhigh | P2 | 12288 | 12289 | 74.29 | 78.62 | -5.51 |
| current-aco-mtp-on | fork-gpu-mtp-on | 12288 | xhigh | P3 | 12288 | 12289 | 77.66 | 84.20 | -7.77 |
| current-aco-mtp-on | master-gpu-mtp-on | 12288 | xhigh | P3 | 12288 | 12289 | 77.66 | 85.64 | -9.32 |
| current-aco-mtp-on | fork-cpu-mtp-on | 28672 | off | P1 | 28673 | 28672 | 92.17 | 22.87 | 302.98 |
| current-aco-mtp-on | master-cpu-mtp-on | 28672 | off | P1 | 28673 | 28672 | 92.17 | 20.74 | 344.51 |
| current-aco-mtp-on | fork-cpu-mtp-on | 28672 | off | P2 | 28673 | 28672 | 74.00 | 17.48 | 323.36 |
| current-aco-mtp-on | master-cpu-mtp-on | 28672 | off | P2 | 28673 | 28672 | 74.00 | 16.25 | 355.47 |
| current-aco-mtp-on | fork-cpu-mtp-on | 28672 | off | P3 | 28673 | 28672 | 81.16 | 20.08 | 304.16 |
| current-aco-mtp-on | master-cpu-mtp-on | 28672 | off | P3 | 28673 | 28672 | 81.16 | 11.60 | 599.49 |
| current-aco-mtp-on | fork-gpu-mtp-on | 78848 | off | P1 | 78849 | 78848 | 73.93 | 83.74 | -11.71 |
| current-aco-mtp-on | fork-gpu-mtp-on | 78848 | off | P2 | 78849 | 78848 | 55.98 | 65.93 | -15.09 |
| current-aco-mtp-on | fork-gpu-mtp-on | 78848 | off | P3 | 78849 | 78848 | 66.45 | 75.91 | -12.46 |
| current-aco-mtp-on | fork-gpu-mtp-on | 78848 | low | P1 | 78849 | 78849 | 70.39 | 77.35 | -8.99 |
| current-aco-mtp-on | fork-gpu-mtp-on | 78848 | low | P2 | 78849 | 78849 | 53.05 | 63.03 | -15.83 |
| current-aco-mtp-on | fork-gpu-mtp-on | 78848 | low | P3 | 78849 | 78849 | 68.47 | 66.38 | 3.15 |
| current-aco-mtp-on | fork-gpu-mtp-on | 78848 | medium | P1 | 78849 | 78849 | 69.49 | 76.87 | -9.60 |
| current-aco-mtp-on | fork-gpu-mtp-on | 78848 | medium | P2 | 78849 | 78849 | 56.10 | 59.54 | -5.77 |
| current-aco-mtp-on | fork-gpu-mtp-on | 78848 | medium | P3 | 78849 | 78849 | 61.54 | 70.81 | -13.08 |
| current-aco-mtp-on | fork-gpu-mtp-on | 78848 | xhigh | P1 | 78849 | 78849 | 69.66 | 75.83 | -8.14 |
| current-aco-mtp-on | fork-gpu-mtp-on | 78848 | xhigh | P2 | 78849 | 78849 | 52.95 | 63.77 | -16.97 |
| current-aco-mtp-on | fork-gpu-mtp-on | 78848 | xhigh | P3 | 78849 | 78849 | 58.82 | 71.14 | -17.31 |
| current-aco-mtp-on | fork-cpu-mtp-on | 196608 | off | P1 | 196608 | 196609 | 65.39 | 8.71 | 651.21 |
| current-aco-mtp-on | master-cpu-mtp-on | 196608 | off | P1 | 196608 | 196609 | 65.39 | 8.84 | 640.17 |
| current-aco-mtp-on | fork-cpu-mtp-on | 196608 | off | P2 | 196608 | 196609 | 48.83 | 6.51 | 649.98 |
| current-aco-mtp-on | master-cpu-mtp-on | 196608 | off | P2 | 196608 | 196609 | 48.83 | 6.79 | 618.86 |
| current-aco-mtp-on | fork-cpu-mtp-on | 196608 | low | P1 | 196608 | 196609 | 62.40 | 8.48 | 635.57 |
| current-aco-mtp-on | master-cpu-mtp-on | 196608 | low | P1 | 196608 | 196609 | 62.40 | 8.29 | 652.70 |
| current-aco-mtp-on | fork-cpu-mtp-on | 196608 | low | P2 | 196608 | 196609 | 48.79 | 7.05 | 592.45 |
| current-aco-mtp-on | master-cpu-mtp-on | 196608 | low | P2 | 196608 | 196609 | 48.79 | 7.08 | 588.90 |
| current-aco-mtp-on | fork-cpu-mtp-on | 196608 | low | P3 | 196608 | 196609 | 56.46 | 7.17 | 687.42 |
| current-aco-mtp-on | master-cpu-mtp-on | 196608 | low | P3 | 196608 | 196609 | 56.46 | 7.17 | 687.30 |
| current-aco-mtp-on | fork-cpu-mtp-on | 196608 | medium | P1 | 196608 | 196609 | 62.72 | 8.33 | 652.53 |
| current-aco-mtp-on | master-cpu-mtp-on | 196608 | medium | P1 | 196608 | 196609 | 62.72 | 8.32 | 653.74 |
| current-aco-mtp-on | fork-cpu-mtp-on | 196608 | medium | P2 | 196608 | 196609 | 51.53 | 7.03 | 633.22 |
| current-aco-mtp-on | master-cpu-mtp-on | 196608 | medium | P2 | 196608 | 196609 | 51.53 | 6.81 | 656.96 |
| current-aco-mtp-on | fork-cpu-mtp-on | 196608 | medium | P3 | 196608 | 196609 | 58.55 | 7.60 | 670.04 |
| current-aco-mtp-on | master-cpu-mtp-on | 196608 | medium | P3 | 196608 | 196609 | 58.55 | 7.87 | 644.07 |
| current-aco-mtp-on | fork-cpu-mtp-on | 196608 | xhigh | P1 | 196608 | 196609 | 61.99 | 8.08 | 666.94 |
| current-aco-mtp-on | master-cpu-mtp-on | 196608 | xhigh | P1 | 196608 | 196609 | 61.99 | 7.57 | 718.55 |
| current-aco-mtp-on | fork-cpu-mtp-on | 196608 | xhigh | P2 | 196608 | 196609 | 45.09 | 7.00 | 544.45 |
| current-aco-mtp-on | fork-cpu-mtp-on | 196608 | xhigh | P3 | 196608 | 196609 | 48.37 | 8.10 | 497.12 |
| current-aco-mtp-on | master-cpu-mtp-on | 196608 | xhigh | P3 | 196608 | 196609 | 48.37 | 7.80 | 519.95 |
| current-aco-mtp-on | fork-cpu-mtp-on | 256000 | off | P1 | 256001 | 256001 | 60.81 | 7.14 | 751.67 |
| current-aco-mtp-on | fork-cpu-mtp-on | 256000 | off | P2 | 256001 | 256001 | 48.41 | 5.48 | 782.85 |
| current-aco-mtp-on | fork-cpu-mtp-on | 256000 | off | P3 | 256001 | 256001 | 55.30 | 6.50 | 750.26 |
| current-aco-mtp-on | fork-cpu-mtp-on | 256000 | low | P1 | 256001 | 256001 | 59.09 | 6.82 | 766.65 |
| current-aco-mtp-on | fork-cpu-mtp-on | 256000 | low | P2 | 256001 | 256001 | 41.94 | 5.79 | 623.89 |
| current-aco-mtp-on | fork-cpu-mtp-on | 256000 | low | P3 | 256001 | 256001 | 53.16 | 6.38 | 733.02 |
| current-aco-mtp-on | fork-cpu-mtp-on | 256000 | medium | P1 | 256001 | 256001 | 58.53 | 6.84 | 755.52 |
| current-aco-mtp-on | fork-cpu-mtp-on | 256000 | medium | P2 | 256001 | 256001 | 47.86 | 5.80 | 725.70 |
| current-aco-mtp-on | fork-cpu-mtp-on | 256000 | medium | P3 | 256001 | 256001 | 53.45 | 6.64 | 704.65 |
| current-aco-mtp-on | fork-cpu-mtp-on | 256000 | xhigh | P1 | 256001 | 256001 | 57.86 | 6.62 | 773.59 |
| current-aco-mtp-on | fork-cpu-mtp-on | 256000 | xhigh | P2 | 256001 | 256001 | 42.98 | 5.36 | 702.37 |
| current-aco-mtp-on | fork-cpu-mtp-on | 256000 | xhigh | P3 | 256001 | 256001 | 45.41 | 6.67 | 580.45 |

## Mapped-fill interval rates by checkpoint

Rates are interval-weighted from net-new input since the preceding checkpoint, including recorded tails. Full precision and source series are in `mapped-prefill.csv`.

| Family | Fill series | Nominal C | Actual C start–end | Appends | Backend tok/s | Net-new tok/s | HTTP-wall tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| master-gpu-mtp-off | `attempt-001-L131072` | 6144 | 0–6144 | 1 | 1971.28 | 1971.28 | 1960.10 |
| master-gpu-mtp-off | `attempt-001-L122880` | 6144 | 0–6144 | 1 | 1960.61 | 1960.61 | 1949.86 |
| master-gpu-mtp-off | `attempt-001-L122880` | 6144 | 1–6145 | 1 | 1966.25 | 1966.25 | 1896.07 |
| master-gpu-mtp-off | `attempt-001-L131072` | 12288 | 6144–12288 | 1 | 1809.15 | 1809.15 | 1800.83 |
| master-gpu-mtp-off | `attempt-001-L131072` | 28672 | 12288–28673 | 1 | 1631.22 | 1631.22 | 1625.91 |
| master-gpu-mtp-off | `attempt-001-L131072` | 62464 | 28673–62465 | 2 | 1263.92 | 1263.92 | 1258.15 |
| master-gpu-mtp-off | `attempt-001-L131072` | 78848 | 62465–78849 | 1 | 1045.62 | 1045.62 | 1041.08 |
| master-gpu-mtp-off | `attempt-001-L131072` | 88064 | 78849–88065 | 1 | 955.54 | 955.54 | 947.53 |
| master-gpu-mtp-off | `attempt-001-L131072` | 100352 | 88065–100352 | 1 | 895.14 | 895.14 | 889.11 |
| master-gpu-mtp-off | `attempt-001-L131072` | 120832 | 100352–120833 | 1 | 821.88 | 821.88 | 818.30 |
| master-gpu-mtp-on | `attempt-001-L65536` | 6144 | 0–6144 | 1 | 1783.13 | 1783.13 | 1774.01 |
| master-gpu-mtp-on | `attempt-001-L65536` | 12288 | 6144–12288 | 1 | 1673.76 | 1673.76 | 1664.15 |
| master-gpu-mtp-on | `attempt-001-L65536` | 28672 | 12288–28673 | 1 | 1536.71 | 1536.71 | 1531.55 |
| master-gpu-mtp-on | `attempt-001-L65536` | 62464 | 28673–62465 | 2 | 1181.57 | 1181.57 | 1176.95 |
| fork-gpu-mtp-off | `attempt-001-L131072` | 6144 | 0–6144 | 1 | 1771.56 | 1771.56 | 1762.82 |
| fork-gpu-mtp-off | `attempt-001-L122880` | 6144 | 1–6145 | 1 | 1769.08 | 1769.08 | 1760.65 |
| fork-gpu-mtp-off | `attempt-001-L131072` | 12288 | 6144–12288 | 1 | 1456.99 | 1456.99 | 1448.80 |
| fork-gpu-mtp-off | `attempt-001-L122880` | 12288 | 6146–12289 | 1 | 1452.56 | 1452.32 | 1444.68 |
| fork-gpu-mtp-off | `attempt-001-L131072` | 28672 | 12288–28673 | 1 | 1193.02 | 1193.02 | 1190.08 |
| fork-gpu-mtp-off | `attempt-001-L122880` | 28672 | 12290–28673 | 1 | 1191.62 | 1191.55 | 1188.68 |
| fork-gpu-mtp-off | `attempt-001-L131072` | 62464 | 28673–62465 | 2 | 781.61 | 781.61 | 779.30 |
| fork-gpu-mtp-off | `attempt-001-L122880` | 62464 | 28674–62465 | 2 | 781.60 | 781.56 | 779.32 |
| fork-gpu-mtp-off | `attempt-001-L131072` | 78848 | 62465–78849 | 1 | 531.52 | 531.52 | 530.20 |
| fork-gpu-mtp-off | `attempt-001-L122880` | 78848 | 62466–78848 | 1 | 531.93 | 531.90 | 530.56 |
| fork-gpu-mtp-off | `attempt-001-L131072` | 88064 | 78849–88065 | 1 | 457.09 | 457.09 | 455.24 |
| fork-gpu-mtp-off | `attempt-001-L122880` | 88064 | 78849–88065 | 1 | 457.41 | 457.36 | 455.53 |
| fork-gpu-mtp-off | `attempt-001-L131072` | 100352 | 88065–100352 | 1 | 412.94 | 412.94 | 411.74 |
| fork-gpu-mtp-off | `attempt-001-L122880` | 100352 | 88066–100351 | 1 | 413.55 | 413.52 | 412.25 |
| fork-gpu-mtp-off | `attempt-001-L122880` | 120832 | 100352–120833 | 1 | 362.21 | 362.20 | 361.51 |
| fork-gpu-mtp-on | `attempt-001-L65536` | 6144 | 0–6144 | 1 | 1648.48 | 1648.48 | 1641.48 |
| fork-gpu-mtp-on | `attempt-001-L65536` | 12288 | 6144–12288 | 1 | 1374.20 | 1374.20 | 1367.28 |
| fork-gpu-mtp-on | `attempt-001-L65536` | 28672 | 12288–28673 | 1 | 1128.23 | 1128.23 | 1125.79 |
| fork-gpu-mtp-on | `attempt-001-L65536` | 62464 | 28673–62465 | 2 | 736.54 | 736.54 | 734.62 |
| master-cpu-mtp-off | `attempt-001-L262144` | 6144 | 0–6144 | 1 | 1305.05 | 1305.05 | 1300.13 |
| master-cpu-mtp-off | `attempt-001-L262144` | 6144 | 0–6144 | 1 | 1314.14 | 1314.14 | 1309.31 |
| master-cpu-mtp-off | `attempt-001-L262144` | 12288 | 6144–12288 | 1 | 1197.06 | 1197.06 | 1192.03 |
| master-cpu-mtp-off | `attempt-001-L262144` | 12288 | 6144–12288 | 1 | 1208.91 | 1208.91 | 1203.94 |
| master-cpu-mtp-off | `attempt-001-L262144` | 28672 | 12288–28673 | 1 | 1099.29 | 1099.29 | 1096.96 |
| master-cpu-mtp-off | `attempt-001-L262144` | 28672 | 12288–28673 | 1 | 1106.90 | 1106.90 | 1104.42 |
| master-cpu-mtp-off | `attempt-001-L262144` | 61440 | 28673–61441 | 1 | 876.44 | 876.44 | 875.11 |
| master-cpu-mtp-off | `attempt-001-L262144` | 126976 | 61441–126977 | 2 | 619.64 | 619.64 | 618.45 |
| master-cpu-mtp-on | `attempt-001-L200704` | 6144 | 0–6144 | 1 | 930.02 | 930.02 | 928.57 |
| master-cpu-mtp-on | `attempt-001-L200704` | 6144 | 0–6144 | 1 | 928.73 | 928.73 | 926.52 |
| master-cpu-mtp-on | `attempt-001-L32768-series-a` | 6144 | 1–6145 | 1 | 934.75 | 934.75 | 933.10 |
| master-cpu-mtp-on | `attempt-001-L32768-series-b` | 6144 | 1–6145 | 1 | 940.67 | 940.67 | 938.08 |
| master-cpu-mtp-on | `attempt-001-L32768-series-c` | 6144 | 1–6145 | 1 | 940.14 | 940.14 | 938.78 |
| master-cpu-mtp-on | `attempt-001-L200704` | 6144 | 1–6145 | 1 | 925.39 | 925.39 | 922.91 |
| master-cpu-mtp-on | `attempt-001-L200704` | 12288 | 6144–12288 | 1 | 884.46 | 883.88 | 881.16 |
| master-cpu-mtp-on | `attempt-001-L200704` | 12288 | 6144–12288 | 1 | 882.98 | 882.98 | 880.25 |
| master-cpu-mtp-on | `attempt-001-L32768-series-a` | 12288 | 6146–12289 | 1 | 881.79 | 881.65 | 878.68 |
| master-cpu-mtp-on | `attempt-001-L32768-series-b` | 12288 | 6146–12289 | 1 | 879.41 | 879.27 | 876.54 |
| master-cpu-mtp-on | `attempt-001-L32768-series-c` | 12288 | 6146–12289 | 1 | 880.72 | 880.57 | 877.88 |
| master-cpu-mtp-on | `attempt-001-L200704` | 28672 | 12288–28673 | 1 | 824.23 | 824.23 | 822.89 |
| master-cpu-mtp-on | `attempt-001-L32768-series-a` | 28672 | 12290–28673 | 1 | 822.28 | 822.23 | 820.83 |
| master-cpu-mtp-on | `attempt-001-L32768-series-b` | 28672 | 12290–28673 | 1 | 822.30 | 822.25 | 820.82 |
| master-cpu-mtp-on | `attempt-001-L32768-series-c` | 28672 | 12290–28673 | 1 | 820.94 | 820.89 | 819.55 |
| master-cpu-mtp-on | `attempt-001-L200704` | 61440 | 28673–61441 | 1 | 677.13 | 677.13 | 676.30 |
| master-cpu-mtp-on | `attempt-001-L200704` | 126976 | 61441–126977 | 2 | 500.26 | 500.26 | 499.45 |
| master-cpu-mtp-on | `attempt-001-L200704` | 196608 | 126977–196609 | 3 | 365.67 | 365.67 | 364.74 |
| fork-cpu-mtp-off | `attempt-001-L262144` | 6144 | 0–6144 | 1 | 1253.11 | 1253.11 | 1248.66 |
| fork-cpu-mtp-off | `attempt-001-L262144` | 12288 | 6144–12288 | 1 | 1128.86 | 1128.86 | 1125.70 |
| fork-cpu-mtp-off | `attempt-001-L262144` | 28672 | 12288–28673 | 1 | 1085.33 | 1085.33 | 1082.92 |
| fork-cpu-mtp-off | `attempt-001-L262144` | 62464 | 28673–62465 | 2 | 848.47 | 848.47 | 845.98 |
| fork-cpu-mtp-off | `attempt-001-L262144` | 130048 | 62465–130049 | 3 | 604.80 | 604.80 | 603.06 |
| fork-cpu-mtp-off | `attempt-001-L262144` | 256000 | 196609–256001 | 2 | 347.31 | 347.31 | 346.49 |
| fork-cpu-mtp-on | `attempt-001-L262144` | 6144 | 0–6144 | 1 | 911.60 | 911.60 | 909.17 |
| fork-cpu-mtp-on | `attempt-001-L262144` | 6144 | 0–6144 | 1 | 907.00 | 907.00 | 904.60 |
| fork-cpu-mtp-on | `attempt-001-L262144` | 6144 | 1–6145 | 1 | 913.82 | 913.82 | 911.39 |
| fork-cpu-mtp-on | `attempt-001-L262144` | 6144 | 1–6145 | 1 | 905.91 | 905.91 | 904.60 |
| fork-cpu-mtp-on | `attempt-001-L262144` | 12288 | 6144–12288 | 1 | 864.03 | 864.03 | 861.35 |
| fork-cpu-mtp-on | `attempt-001-L262144` | 12288 | 6144–12288 | 1 | 859.43 | 859.43 | 857.52 |
| fork-cpu-mtp-on | `attempt-001-L262144` | 12288 | 6146–12289 | 1 | 860.10 | 859.96 | 857.42 |
| fork-cpu-mtp-on | `attempt-001-L262144` | 28672 | 12288–28673 | 1 | 816.86 | 816.86 | 815.45 |
| fork-cpu-mtp-on | `attempt-001-L262144` | 28672 | 12290–28673 | 1 | 816.23 | 816.18 | 814.97 |
| fork-cpu-mtp-on | `attempt-001-L262144` | 61440 | 28673–61441 | 1 | 672.69 | 672.69 | 671.84 |
| fork-cpu-mtp-on | `attempt-001-L262144` | 61440 | 28674–61441 | 1 | 670.20 | 670.18 | 669.38 |
| fork-cpu-mtp-on | `attempt-001-L262144` | 126976 | 61441–126977 | 2 | 490.06 | 490.06 | 489.31 |
| fork-cpu-mtp-on | `attempt-001-L262144` | 126976 | 63183–126977 | 2 | 518.16 | 338.58 | 338.06 |
| fork-cpu-mtp-on | `attempt-001-L262144` | 126976 | 61442–126977 | 2 | 488.55 | 488.53 | 487.80 |
| fork-cpu-mtp-on | `attempt-001-L262144` | 196608 | 126977–196609 | 3 | 355.09 | 355.09 | 354.22 |
| fork-cpu-mtp-on | `attempt-001-L262144` | 256000 | 196609–256001 | 2 | 285.15 | 285.15 | 284.58 |
| current-aco-mtp-off | `candidate-full256k-aco-mtp-off-20261104T115523300080953Z` | 6144 | 0–6144 | 1 | 1674.96 | 1674.96 | 1666.11 |
| current-aco-mtp-off | `candidate-full256k-aco-mtp-off-20261104T115523300080953Z` | 12288 | 6144–12288 | 1 | 1405.16 | 1405.85 | 1399.17 |
| current-aco-mtp-off | `candidate-full256k-aco-mtp-off-20261104T115523300080953Z` | 28672 | 12288–28673 | 1 | 1123.04 | 1123.25 | 1120.64 |
| current-aco-mtp-off | `candidate-full256k-aco-mtp-off-20261104T115523300080953Z` | 62464 | 28673–62465 | 2 | 738.50 | 738.57 | 736.56 |
| current-aco-mtp-off | `candidate-full256k-aco-mtp-off-20261104T115523300080953Z` | 78848 | 62465–78849 | 1 | 976.95 | 977.13 | 972.75 |
| current-aco-mtp-off | `candidate-full256k-aco-mtp-off-20261104T115523300080953Z` | 88064 | 78849–88065 | 1 | 922.42 | 922.72 | 914.92 |
| current-aco-mtp-off | `candidate-full256k-aco-mtp-off-20261104T115523300080953Z` | 100352 | 88065–100352 | 1 | 945.61 | 945.85 | 938.90 |
| current-aco-mtp-off | `candidate-full256k-aco-mtp-off-20261104T115523300080953Z` | 130048 | 100352–130049 | 1 | 995.27 | 995.37 | 991.59 |
| current-aco-mtp-off | `candidate-full256k-aco-mtp-off-20261104T115523300080953Z` | 196608 | 130049–196608 | 3 | 943.03 | 929.38 | 923.00 |
| current-aco-mtp-off | `candidate-full256k-aco-mtp-off-20261104T115523300080953Z` | 256000 | 196608–256001 | 2 | 953.94 | 953.99 | 947.57 |
| current-aco-mtp-off | `candidate-full256k-aco-mtp-off-20261104T115523300080953Z` | 261120 | 256001–261121 | 1 | 797.38 | 797.85 | 767.40 |
| current-aco-mtp-on | `candidate-full256k-aco-mtp-on-20261104T105341378940038Z` | 6144 | 0–6144 | 1 | 1475.37 | 1475.37 | 1469.77 |
| current-aco-mtp-on | `candidate-full256k-aco-mtp-on-20261104T105341378940038Z` | 12288 | 6144–12288 | 1 | 1267.83 | 1268.45 | 1263.00 |
| current-aco-mtp-on | `candidate-full256k-aco-mtp-on-20261104T105341378940038Z` | 28672 | 12288–28673 | 1 | 1006.17 | 1006.35 | 1004.21 |
| current-aco-mtp-on | `candidate-full256k-aco-mtp-on-20261104T105341378940038Z` | 62464 | 28673–62465 | 2 | 662.01 | 662.07 | 660.62 |
| current-aco-mtp-on | `candidate-full256k-aco-mtp-on-20261104T105341378940038Z` | 78848 | 62465–78849 | 1 | 888.88 | 889.04 | 885.39 |
| current-aco-mtp-on | `candidate-full256k-aco-mtp-on-20261104T105341378940038Z` | 88064 | 78849–88065 | 1 | 826.17 | 826.44 | 820.43 |
| current-aco-mtp-on | `candidate-full256k-aco-mtp-on-20261104T105341378940038Z` | 100352 | 88065–100352 | 1 | 833.87 | 834.07 | 828.88 |
| current-aco-mtp-on | `candidate-full256k-aco-mtp-on-20261104T105341378940038Z` | 130048 | 100352–130049 | 1 | 849.45 | 849.53 | 846.51 |
| current-aco-mtp-on | `candidate-full256k-aco-mtp-on-20261104T105341378940038Z` | 196608 | 130049–196608 | 3 | 766.27 | 755.18 | 750.73 |
| current-aco-mtp-on | `candidate-full256k-aco-mtp-on-20261104T105341378940038Z` | 256000 | 196608–256001 | 2 | 726.23 | 726.27 | 722.64 |
| current-aco-mtp-on | `candidate-full256k-aco-mtp-on-20261104T105341378940038Z` | 261120 | 256001–261121 | 1 | 605.26 | 605.62 | 590.53 |

## Weighted independent fill cohorts

Each rate is summed tokens divided by summed matching time; independent series and protocols are not pooled. Detailed append groups remain in the external full report.

| Family | Series | Protocol | Status | Appends | Nominal contexts | Final actual C | Net-new tokens | Backend tok/s | Net-new tok/s | HTTP-wall tok/s |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|
| master-gpu-mtp-off | `attempt-001-L131072` | resident_prefix_reuse | retained partial series; source-level completion not asserted | 9 | 6144, 12288, 28672, 62464, 78848, 88064, 100352, 120832 | 120833 | 120833.0 | 1126.61 | 1126.61 | 1120.96 |
| master-gpu-mtp-off | `attempt-001-L122880` | resident_prefix_reuse | retained partial series; source-level completion not asserted | 1 | 6144 | 6144 | 6144.0 | 1960.61 | 1960.61 | 1949.86 |
| master-gpu-mtp-off | `attempt-001-L122880` | resident_cumulative_conversation | partial planned fill series | 1 | 6144 | 6145 | 6144.0 | 1966.25 | 1966.25 | 1896.07 |
| master-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 38 | 100352 | 100353 | 100353.0 | 1141.59 | 1141.59 | 1113.97 |
| master-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 38 | 100352 | 100353 | 100353.0 | 1143.25 | 1143.25 | 1114.11 |
| master-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 38 | 100352 | 100353 | 100353.0 | 1149.36 | 1149.36 | 1121.47 |
| master-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 38 | 100352 | 100353 | 100353.0 | 1142.15 | 1142.15 | 1114.40 |
| master-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 44 | 120832 | 120833 | 120833.0 | 1058.99 | 1058.99 | 1031.85 |
| master-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 44 | 120832 | 120833 | 120833.0 | 1061.12 | 1061.12 | 1033.66 |
| master-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 44 | 120832 | 120833 | 120833.0 | 1068.81 | 1068.81 | 1042.07 |
| master-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 25 | 62464 | 62465 | 62465.0 | 1341.09 | 1341.09 | 1311.44 |
| master-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 25 | 62464 | 62465 | 62465.0 | 1345.52 | 1345.52 | 1317.75 |
| master-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 25 | 62464 | 62464 | 62464.0 | 1338.24 | 1338.24 | 1310.04 |
| master-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 32 | 78848 | 78849 | 78849.0 | 1241.71 | 1241.71 | 1209.43 |
| master-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 32 | 78848 | 78849 | 78849.0 | 1244.81 | 1244.81 | 1210.60 |
| master-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 32 | 78848 | 78849 | 78849.0 | 1251.16 | 1251.16 | 1222.11 |
| master-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 32 | 78848 | 78849 | 78849.0 | 1240.85 | 1240.85 | 1206.44 |
| master-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 34 | 88064 | 88065 | 88065.0 | 1201.13 | 1201.13 | 1169.19 |
| master-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 34 | 88064 | 88065 | 88065.0 | 1197.36 | 1197.36 | 1169.13 |
| master-gpu-mtp-on | `attempt-001-L65536` | resident_prefix_reuse | retained partial series; source-level completion not asserted | 5 | 6144, 12288, 28672, 62464 | 62465 | 62465.0 | 1346.87 | 1346.87 | 1341.45 |
| master-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12289 | 12289.0 | 1461.48 | 1461.48 | 1432.85 |
| master-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12288 | 12288.0 | 1476.58 | 1476.58 | 1445.43 |
| master-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12289 | 12289.0 | 1483.33 | 1483.33 | 1451.47 |
| master-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12289 | 12289.0 | 1455.99 | 1455.99 | 1426.60 |
| master-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 28 | 71680 | 71680 | 71680.0 | 1143.58 | 1143.58 | 1121.68 |
| master-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 28 | 71680 | 71681 | 71681.0 | 1146.17 | 1146.17 | 1124.73 |
| master-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 28 | 71680 | 71681 | 71681.0 | 1155.47 | 1155.47 | 1132.73 |
| master-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 28 | 71680 | 71681 | 71681.0 | 1144.68 | 1144.68 | 1123.12 |
| master-gpu-mtp-on | `MASTER106_GPU.json` | MASTER106 GPU fresh-fill curve; legacy protocol | complete source fill record(s); historical cohort only | 50 | 6144, 14336, 30000, 61440 | 61440 | 111920.0 | 1297.40 | 1297.40 | 1273.37 |
| fork-gpu-mtp-off | `attempt-001-L131072` | resident_prefix_reuse | retained partial series; source-level completion not asserted | 8 | 6144, 12288, 28672, 62464, 78848, 88064, 100352 | 100352 | 100352.0 | 690.20 | 690.20 | 688.08 |
| fork-gpu-mtp-off | `attempt-001-L122880` | resident_cumulative_conversation | complete planned fill series | 9 | 6144, 12288, 28672, 62464, 78848, 88064, 100352, 120832 | 120833 | 120824.0 | 598.50 | 598.46 | 596.82 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 38 | 100352 | 100353 | 100353.0 | 651.17 | 651.17 | 641.90 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 38 | 100352 | 100353 | 100353.0 | 651.09 | 651.09 | 642.04 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 38 | 100352 | 100353 | 100353.0 | 652.30 | 652.30 | 643.51 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 38 | 100352 | 100353 | 100353.0 | 650.87 | 650.87 | 642.12 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 42 | 112640 | 112640 | 112640.0 | 600.19 | 600.19 | 592.15 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 43 | 116736 | 116737 | 116737.0 | 584.37 | 584.37 | 576.70 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 44 | 120832 | 120833 | 120833.0 | 569.62 | 569.62 | 562.00 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 44 | 120832 | 120833 | 120833.0 | 570.04 | 570.04 | 562.67 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 44 | 120832 | 120833 | 120833.0 | 569.94 | 569.94 | 562.51 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 44 | 120832 | 120833 | 120833.0 | 569.48 | 569.48 | 562.08 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12289 | 12289.0 | 1314.28 | 1314.28 | 1287.40 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12288 | 12288.0 | 1329.36 | 1329.36 | 1304.64 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12289 | 12289.0 | 1328.40 | 1328.40 | 1303.66 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12289 | 12289.0 | 1312.82 | 1312.82 | 1287.81 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 13 | 28672 | 28673 | 28673.0 | 1166.71 | 1166.71 | 1149.92 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 13 | 28672 | 28673 | 28673.0 | 1172.96 | 1172.96 | 1155.99 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 13 | 28672 | 28673 | 28673.0 | 1172.01 | 1172.01 | 1154.84 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 13 | 28672 | 28673 | 28673.0 | 1165.84 | 1165.84 | 1144.42 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 4 | 6144 | 6145 | 6145.0 | 1371.34 | 1371.34 | 1338.26 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 4 | 6144 | 6145 | 6145.0 | 1396.86 | 1396.86 | 1363.71 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 4 | 6144 | 6145 | 6145.0 | 1399.95 | 1399.95 | 1363.95 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 4 | 6144 | 6145 | 6145.0 | 1368.14 | 1368.14 | 1333.69 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 25 | 62464 | 62465 | 62465.0 | 886.00 | 886.00 | 873.44 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 25 | 62464 | 62465 | 62465.0 | 887.35 | 887.35 | 874.67 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 25 | 62464 | 62465 | 62465.0 | 889.04 | 889.04 | 876.47 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 25 | 62464 | 62464 | 62464.0 | 886.03 | 886.03 | 873.87 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 32 | 78848 | 78849 | 78849.0 | 764.57 | 764.57 | 752.62 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 32 | 78848 | 78849 | 78849.0 | 765.41 | 765.41 | 753.74 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 32 | 78848 | 78849 | 78849.0 | 766.86 | 766.86 | 754.53 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 32 | 78848 | 78849 | 78849.0 | 764.30 | 764.30 | 753.07 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 34 | 88064 | 88065 | 88065.0 | 712.77 | 712.77 | 702.88 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 34 | 88064 | 88065 | 88065.0 | 713.45 | 713.45 | 703.76 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 34 | 88064 | 88065 | 88065.0 | 714.49 | 714.49 | 704.70 |
| fork-gpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 34 | 88064 | 88065 | 88065.0 | 712.73 | 712.73 | 702.64 |
| fork-gpu-mtp-on | `attempt-001-L65536` | resident_prefix_reuse | retained partial series; source-level completion not asserted | 5 | 6144, 12288, 28672, 62464 | 62465 | 62465.0 | 910.57 | 910.57 | 908.05 |
| fork-gpu-mtp-on | `MATRIX105_105-10.json` | MATRIX105 fresh-prefill request; isolated legacy cohort | complete source fill record(s); historical cohort only | 4 | 6144 | 6144 | 23024.0 | 1595.69 | 1595.69 | N/A |
| fork-gpu-mtp-on | `MATRIX105_105-13.json` | MATRIX105 fresh-prefill request; isolated legacy cohort | complete source fill record(s); historical cohort only | 8 | 30000 | 30000 | 146366.0 | 1359.48 | 1359.48 | 1315.49 |
| fork-gpu-mtp-on | `MATRIX105_105-16.json` | MATRIX105 fresh-prefill request; isolated legacy cohort | complete source fill record(s); historical cohort only | 18 | 61440 | 61440 | 113570.0 | 884.71 | 884.71 | 820.82 |
| fork-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12288 | 12288.0 | 1251.40 | 1251.40 | 1230.89 |
| fork-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12289 | 12289.0 | 1235.78 | 1235.78 | 1215.00 |
| fork-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12288 | 12288.0 | 1250.22 | 1250.22 | 1227.83 |
| fork-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12289 | 12289.0 | 1252.78 | 1252.78 | 1230.40 |
| fork-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12289 | 12289.0 | 1235.51 | 1235.51 | 1214.43 |
| fork-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 32 | 78848 | 78848 | 78848.0 | 707.14 | 707.14 | 697.65 |
| fork-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 32 | 78848 | 78849 | 78849.0 | 703.95 | 703.95 | 694.54 |
| fork-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 32 | 78848 | 78849 | 78849.0 | 705.27 | 705.27 | 695.61 |
| fork-gpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 32 | 78848 | 78849 | 78849.0 | 703.09 | 703.09 | 693.51 |
| master-cpu-mtp-off | `attempt-001-L262144` | disk_snapshot_restore | complete source fill record(s); historical cohort only | 3 | 6144, 12288, 28672 | 28673 | 28673.0 | 1142.54 | 1142.54 | 1139.28 |
| master-cpu-mtp-off | `attempt-001-L262144` | resident_prefix_reuse | retained partial series; source-level completion not asserted | 6 | 6144, 12288, 28672, 61440, 126976 | 126977 | 126977.0 | 755.70 | 755.70 | 754.20 |
| master-cpu-mtp-off | `attempt-001-L262144` | resident_prefix_reuse | retained series approved by plan; historical completeness kept separate from active step | 10 | 6144, 12288, 28672, 61440, 126976, 192512, 258048 | 258049 | 258049.0 | 514.63 | 514.63 | 513.58 |
| master-cpu-mtp-on | `attempt-001-L200704` | resident_prefix_reuse | retained partial series; source-level completion not asserted | 2 | 6144, 12288 | 12288 | 12288.0 | 906.66 | 906.37 | 904.24 |
| master-cpu-mtp-on | `attempt-001-L200704` | resident_prefix_reuse | retained partial series; source-level completion not asserted | 9 | 6144, 12288, 28672, 61440, 126976, 196608 | 196609 | 196609.0 | 487.54 | 487.54 | 486.53 |
| master-cpu-mtp-on | `attempt-001-L32768-series-a` | resident_cumulative_conversation | complete planned fill series | 3 | 6144, 12288, 28672 | 28673 | 28670.0 | 856.76 | 856.70 | 854.94 |
| master-cpu-mtp-on | `attempt-001-L32768-series-b` | resident_cumulative_conversation | complete planned fill series | 3 | 6144, 12288, 28672 | 28673 | 28670.0 | 857.35 | 857.29 | 855.39 |
| master-cpu-mtp-on | `attempt-001-L32768-series-c` | resident_cumulative_conversation | complete planned fill series | 3 | 6144, 12288, 28672 | 28673 | 28670.0 | 856.68 | 856.62 | 854.99 |
| master-cpu-mtp-on | `attempt-001-L200704` | resident_cumulative_conversation | partial planned fill series | 1 | 6144 | 6145 | 6144.0 | 925.39 | 925.39 | 922.91 |
| master-cpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 75 | 196608 | 196609 | 196609.0 | 422.65 | 422.65 | 415.89 |
| fork-cpu-mtp-off | `attempt-001-L262144` | disk_snapshot_restore | retained series approved by plan; historical completeness kept separate from active step | 13 | 6144, 12288, 28672, 62464, 130048, 196608, 256000 | 256001 | 256001.0 | 510.84 | 510.84 | 509.42 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12289 | 12289.0 | 956.47 | 956.47 | 942.98 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12288 | 12288.0 | 974.37 | 974.37 | 960.59 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12289 | 12289.0 | 969.28 | 969.28 | 950.49 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 7 | 12288 | 12289 | 12289.0 | 963.13 | 963.13 | 950.55 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 49 | 130048 | 130049 | 130049.0 | 631.78 | 631.78 | 621.34 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 49 | 130048 | 130049 | 130049.0 | 630.50 | 630.50 | 620.17 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 49 | 130048 | 130049 | 130049.0 | 632.41 | 632.41 | 622.05 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 49 | 130048 | 130049 | 130049.0 | 628.76 | 628.76 | 617.64 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 75 | 196608 | 196609 | 196609.0 | 511.31 | 511.31 | 501.27 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 75 | 196608 | 196609 | 196609.0 | 511.10 | 511.10 | 501.21 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 75 | 196608 | 196609 | 196609.0 | 512.67 | 512.67 | 502.80 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 75 | 196608 | 196609 | 196609.0 | 511.28 | 511.28 | 501.30 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 105 | 256000 | 255999 | 255999.0 | 430.17 | 430.17 | 419.77 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 105 | 256000 | 255999 | 255999.0 | 430.47 | 430.47 | 420.19 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 105 | 256000 | 256001 | 256001.0 | 431.24 | 431.24 | 420.57 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 105 | 256000 | 255999 | 255999.0 | 430.35 | 430.35 | 420.14 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 13 | 28672 | 28673 | 28673.0 | 937.90 | 937.90 | 923.83 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 13 | 28672 | 28673 | 28673.0 | 941.59 | 941.59 | 930.46 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 13 | 28672 | 28673 | 28673.0 | 941.37 | 941.37 | 930.50 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 13 | 28672 | 28673 | 28673.0 | 933.30 | 933.30 | 922.28 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 4 | 6144 | 6145 | 6145.0 | 957.84 | 957.84 | 941.77 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 4 | 6144 | 6145 | 6145.0 | 977.73 | 977.73 | 960.23 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 4 | 6144 | 6145 | 6145.0 | 977.28 | 977.28 | 958.18 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 4 | 6144 | 6145 | 6145.0 | 960.23 | 960.23 | 943.09 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 25 | 62464 | 62465 | 62465.0 | 815.33 | 815.33 | 804.84 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 25 | 62464 | 62465 | 62465.0 | 816.19 | 816.19 | 805.72 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 25 | 62464 | 62465 | 62465.0 | 814.56 | 814.56 | 804.30 |
| fork-cpu-mtp-off | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 25 | 62464 | 62464 | 62464.0 | 815.38 | 815.38 | 804.56 |
| fork-cpu-mtp-on | `attempt-001-L262144` | resident_prefix_reuse | retained partial series; source-level completion not asserted | 2 | 6144, 12288 | 12288 | 12288.0 | 887.18 | 887.18 | 884.62 |
| fork-cpu-mtp-on | `attempt-001-L262144` | resident_prefix_reuse | retained partial series; source-level completion not asserted | 11 | 6144, 12288, 28672, 61440, 126976, 196608, 256000 | 256001 | 256001.0 | 412.37 | 412.37 | 411.55 |
| fork-cpu-mtp-on | `attempt-001-L262144` | resident_cumulative_conversation | retained partial series; source-level completion not asserted | 8 | 6144, 12288, 28672, 61440, 126976 | 126977 | 118404.0 | 577.47 | 443.44 | 442.46 |
| fork-cpu-mtp-on | `attempt-001-L262144` | resident_cumulative_conversation | partial planned fill series | 6 | 6144, 12288, 28672, 61440, 126976 | 126977 | 126971.0 | 585.05 | 585.03 | 584.16 |
| fork-cpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 75 | 196608 | 196609 | 196609.0 | 404.68 | 404.68 | 398.37 |
| fork-cpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 75 | 196608 | 196609 | 196609.0 | 404.39 | 404.39 | 398.21 |
| fork-cpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 75 | 196608 | 196609 | 196609.0 | 405.50 | 405.50 | 399.30 |
| fork-cpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 75 | 196608 | 196609 | 196609.0 | 404.59 | 404.59 | 397.93 |
| fork-cpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 105 | 256000 | 256001 | 256001.0 | 342.64 | 342.64 | 336.04 |
| fork-cpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 105 | 256000 | 256001 | 256001.0 | 342.74 | 342.74 | 336.25 |
| fork-cpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 105 | 256000 | 256001 | 256001.0 | 343.43 | 343.43 | 337.02 |
| fork-cpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 105 | 256000 | 256001 | 256001.0 | 342.53 | 342.53 | 336.10 |
| fork-cpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 4 | 6144 | 6145 | 6145.0 | 710.04 | 710.04 | 701.24 |
| fork-cpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 4 | 6144 | 6145 | 6145.0 | 663.51 | 663.51 | 655.36 |
| fork-cpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 4 | 6144 | 6145 | 6145.0 | 725.12 | 725.12 | 715.18 |
| fork-cpu-mtp-on | `schedule.jsonl` | historical schedule | complete source fill record(s); historical cohort only | 4 | 6144 | 6145 | 6145.0 | 715.88 | 715.88 | 706.63 |
| current-aco-mtp-off | `candidate-full256k-aco-mtp-off-20261104T115523300080953Z` | persistent_32k_map_with_checkpoint_tails | complete ACO fill curve | 15 | 6144, 12288, 28672, 62464, 78848, 88064, 100352, 130048, 196608, 256000, 261120 | 261121 | 261121.0 | 941.85 | 938.44 | 932.60 |
| current-aco-mtp-on | `candidate-full256k-aco-mtp-on-20261104T105341378940038Z` | persistent_32k_map_with_checkpoint_tails | complete ACO fill curve | 15 | 6144, 12288, 28672, 62464, 78848, 88064, 100352, 130048, 196608, 256000, 261120 | 261121 | 261121.0 | 784.72 | 781.88 | 777.90 |

## Retained resource observations

Resource peaks are shown at their recorded scope. A series peak is not assigned to an individual context. Full paths/source hashes remain in the external JSON.

| Family | Series | Configured L | H ref | Actual final C | Status | Metrics |
|---|---|---:|---:|---:|---|---|
| current-aco-mtp-off | `current-aco-mtp-off|source cohort` | None | 65536 | 261121 | valid complete cells retained | No resource telemetry found in retained source files |
| current-aco-mtp-on | `current-aco-mtp-on|source cohort` | None | 65536 | 261121 | valid complete cells retained | No resource telemetry found in retained source files |
| fork-cpu-mtp-off | `campaign` | 8192 | None | 6145 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2775.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13171.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321060 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5938876 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 43864540 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 16384 | None | 12289 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2478.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13468.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321100 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5947560 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 43723460 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 30720 | None | 28673 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2627.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13319.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320700 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5947592 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 43207708 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 64512 | None | 62465 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2457.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13489.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320896 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5949176 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42758704 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 132096 | None | 130049 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2087.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13859.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320936 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5950024 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41506788 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 198656 | None | 196609 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1721.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14225.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321004 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6027592 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 40005800 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 262144 | None | 255999 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1364.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14582.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321188 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6048876 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 39036376 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 8192 | None | 6145 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2775.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13171.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320988 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5946696 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 43818480 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 16384 | None | 12288 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2723.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13223.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320732 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5947588 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 43544816 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 30720 | None | 28673 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2627.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13319.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320296 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5948056 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 43389532 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 64512 | None | 62465 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2457.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13489.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321260 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5949400 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42777024 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 132096 | None | 130049 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2087.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13859.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320908 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5953320 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41499712 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 198656 | None | 196609 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1721.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14225.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320464 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6029888 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 40299464 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 262144 | None | 255999 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1369.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14577.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320932 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6048908 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 39031648 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 8192 | None | 6145 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2775.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13171.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14319912 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5938840 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 43685024 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 16384 | None | 12289 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2723.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13223.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321140 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5947436 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 43699972 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 30720 | None | 28673 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2627.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13319.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320828 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5947592 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 43424780 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 64512 | None | 62465 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2457.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13489.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320288 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5949152 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42789644 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 132096 | None | 130049 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2087.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13859.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321052 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5949920 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41473108 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 198656 | None | 196609 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1721.0 MiB (n=2), resource_peaks.gpu_used_mib peak 14225.0 MiB (n=2); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320672 bytes (n=2); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6020456 bytes (n=2); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 40253704 bytes (n=2) |
| fork-cpu-mtp-off | `attempt-001-L262144` | 262144 | None | 196609 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1373.0 MiB (n=2), resource_peaks.gpu_used_mib peak 14573.0 MiB (n=2); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14323920 bytes (n=2); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5783208 bytes (n=2); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44574804 bytes (n=2) |
| fork-cpu-mtp-off | `campaign` | 262144 | None | 256001 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1371.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14575.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320772 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6048632 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 38828788 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 8192 | None | 6145 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2775.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13171.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320856 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5947120 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 43825724 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 16384 | None | 12289 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2723.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13223.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320508 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5947592 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 43739812 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 30720 | None | 28673 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2627.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13319.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320448 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5949108 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 43425232 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 64512 | None | 62464 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2443.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13504.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320448 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5949476 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42570980 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 132096 | None | 130049 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2087.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13859.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320780 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5964184 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41437644 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 198656 | None | 196609 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1721.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14225.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321148 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6031112 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 40287084 bytes (n=3) |
| fork-cpu-mtp-off | `campaign` | 262144 | None | 255999 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1371.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14575.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320880 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6049108 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 39001892 bytes (n=3) |
| fork-cpu-mtp-on | `campaign` | 8192 | None | 6145 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2311.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13635.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321684 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6285404 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42735112 bytes (n=3) |
| fork-cpu-mtp-on | `attempt-001-L262144` | 262144 | None | 15174 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 559.0 MiB (n=6), resource_peaks.gpu_used_mib peak 15387.0 MiB (n=6); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 19966528 bytes (n=6); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5686892 bytes (n=6); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 34894644 bytes (n=6) |
| fork-cpu-mtp-on | `campaign` | 198656 | None | 196609 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1005.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14941.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320568 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6287712 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 38943304 bytes (n=3) |
| fork-cpu-mtp-on | `campaign` | 262144 | None | 256001 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 559.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15387.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320200 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6122300 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 38506396 bytes (n=3) |
| fork-cpu-mtp-on | `campaign` | 8192 | None | 6145 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2311.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13635.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321672 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6282260 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42692004 bytes (n=3) |
| fork-cpu-mtp-on | `attempt-001-L262144` | 262144 | None | 127480 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 559.0 MiB (n=2), resource_peaks.gpu_used_mib peak 15387.0 MiB (n=2); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 20990700 bytes (n=2); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5677536 bytes (n=2); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 34146552 bytes (n=2) |
| fork-cpu-mtp-on | `campaign` | 198656 | None | 196609 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1005.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14941.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321052 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6287496 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 38894016 bytes (n=3) |
| fork-cpu-mtp-on | `campaign` | 262144 | None | 256001 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 545.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15402.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320840 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6066920 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 38730540 bytes (n=3) |
| fork-cpu-mtp-on | `campaign` | 8192 | None | 6145 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2311.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13635.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14319976 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6304772 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42403764 bytes (n=3) |
| fork-cpu-mtp-on | `L16384-C12288` | None | None | 12288 | valid complete cells retained | No resource telemetry found in retained source files |
| fork-cpu-mtp-on | `L32768-C28672` | None | None | 28672 | valid complete cells retained | No resource telemetry found in retained source files |
| fork-cpu-mtp-on | `L65536-C61440` | None | None | 61440 | valid complete cells retained | No resource telemetry found in retained source files |
| fork-cpu-mtp-on | `L131072-C126976` | None | None | 126976 | valid complete cells retained | No resource telemetry found in retained source files |
| fork-cpu-mtp-on | `campaign` | 198656 | None | 196609 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1005.0 MiB (n=2), resource_peaks.gpu_used_mib peak 14941.0 MiB (n=2); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321156 bytes (n=2); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6268352 bytes (n=2); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 39075404 bytes (n=2) |
| fork-cpu-mtp-on | `campaign` | 262144 | None | 256001 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 559.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15387.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321304 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6273852 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 37667724 bytes (n=3) |
| fork-cpu-mtp-on | `L262144-C258048` | None | None | 258048 | valid complete cells retained | No resource telemetry found in retained source files |
| fork-cpu-mtp-on | `campaign` | 8192 | None | 6145 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2311.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13635.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321616 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6279624 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42437680 bytes (n=3) |
| fork-cpu-mtp-on | `campaign` | 198656 | None | 196609 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1005.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14941.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321984 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6288100 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 38863956 bytes (n=3) |
| fork-cpu-mtp-on | `campaign` | 262144 | None | 256001 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 559.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15387.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14315620 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6014240 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 38760960 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 8192 | None | 6145 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2395.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13551.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321320 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387572 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42131172 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 16384 | None | 12289 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2235.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13711.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321480 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387472 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42266988 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 30720 | None | 28673 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1933.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14013.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321304 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387472 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42298104 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 64512 | None | 62465 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1241.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14705.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321008 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387468 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42201524 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 80896 | None | 78849 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 923.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15023.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321684 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8376568 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41926240 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 90112 | None | 88065 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 715.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15231.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320668 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6144260 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44367300 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 102400 | None | 100353 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 481.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15465.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321156 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6091676 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44385268 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 122880 | None | 120833 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 61.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15885.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321712 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6089540 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44271948 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 8192 | None | 6145 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2397.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13549.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321388 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387568 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42265104 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 16384 | None | 12288 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2235.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13711.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321636 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387472 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42042912 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 30720 | None | 28673 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1933.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14013.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321412 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387468 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42079892 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 64512 | None | 62465 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1241.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14705.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321332 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387468 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42161148 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 80896 | None | 78849 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 923.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15023.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321616 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 7688028 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42642776 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 90112 | None | 88065 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 733.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15213.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321788 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6142612 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44434620 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 102400 | None | 100353 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 481.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15465.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321236 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6093140 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44348188 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 122880 | None | 120833 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 61.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15885.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320328 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6089560 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44396252 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 8192 | None | 6145 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2397.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13549.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321328 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387572 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41963376 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 16384 | None | 12289 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2235.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13711.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321784 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387472 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41950588 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 30720 | None | 28673 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1933.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14013.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320672 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387472 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42189260 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 64512 | None | 62465 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1241.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14705.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321836 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387468 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42134876 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 80896 | None | 78849 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 905.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15041.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321724 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387468 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42040568 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 90112 | None | 88065 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 715.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15231.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321712 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6862704 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 43328652 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 102400 | None | 100353 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 463.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15483.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321368 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6090716 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44308580 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 114688 | None | 112640 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 211.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15735.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321472 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6097704 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44362252 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 118784 | None | 116737 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 127.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15819.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320848 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6095512 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44286132 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 122880 | None | 120833 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 43.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15903.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320976 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6093416 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44412028 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 8192 | None | 6145 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2397.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13549.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321864 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387544 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42277012 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 16384 | None | 12289 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2235.0 MiB (n=3), resource_peaks.gpu_used_mib peak 13711.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321728 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387472 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42293236 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 30720 | None | 28673 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1933.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14013.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321124 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387468 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42246724 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 64512 | None | 62464 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1241.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14705.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321812 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387468 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42274208 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 80896 | None | 78849 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 923.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15023.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320884 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 7643024 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42940340 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 90112 | None | 88065 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 733.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15213.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321308 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6090500 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44413372 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 102400 | None | 100353 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 463.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15483.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321864 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6088472 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44387936 bytes (n=3) |
| fork-gpu-mtp-off | `campaign` | 122880 | None | 120833 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 61.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15885.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14319900 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6102068 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44204996 bytes (n=3) |
| fork-gpu-mtp-on | `attempt-01` | None | None | 6144 | valid complete cells retained | No resource telemetry found in retained source files |
| fork-gpu-mtp-on | `campaign` | 16384 | None | 12289 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1447.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14499.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321224 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8388604 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41526076 bytes (n=3) |
| fork-gpu-mtp-on | `attempt-01` | None | None | 30000 | valid complete cells retained | No resource telemetry found in retained source files |
| fork-gpu-mtp-on | `attempt-02` | None | None | 61440 | valid complete cells retained | No resource telemetry found in retained source files |
| fork-gpu-mtp-on | `campaign` | 80896 | None | 78849 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 35.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15911.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320540 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8388596 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41618924 bytes (n=3) |
| fork-gpu-mtp-on | `campaign` | 16384 | None | 12288 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1445.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14501.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321612 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8388604 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41421904 bytes (n=3) |
| fork-gpu-mtp-on | `campaign` | 80896 | None | 78849 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 39.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15907.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321540 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8388288 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41646680 bytes (n=3) |
| fork-gpu-mtp-on | `campaign` | 16384 | None | 12289 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1463.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14483.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321644 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8388604 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41438700 bytes (n=3) |
| fork-gpu-mtp-on | `campaign` | 80896 | None | 78848 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 39.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15907.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321504 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8388604 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41485636 bytes (n=3) |
| fork-gpu-mtp-on | `campaign` | 16384 | None | 12289 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1447.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14499.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321396 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8388604 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41409220 bytes (n=3) |
| fork-gpu-mtp-on | `campaign` | 80896 | None | 78849 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 37.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15909.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321732 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8387572 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41698892 bytes (n=3) |
| master-cpu-mtp-off | `attempt-001-L262144` | 262144 | None | 28673 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1377.0 MiB (n=45), resource_peaks.gpu_used_mib peak 14569.0 MiB (n=45); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14322732 bytes (n=45); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5762820 bytes (n=45); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 47861160 bytes (n=45) |
| master-cpu-mtp-off | `attempt-001-L262144` | 262144 | None | 61441 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1377.0 MiB (n=18), resource_peaks.gpu_used_mib peak 14569.0 MiB (n=18); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14322556 bytes (n=18); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5758964 bytes (n=18); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 43550672 bytes (n=18) |
| master-cpu-mtp-off | `attempt-001-L262144` | 262144 | None | 258049 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1377.0 MiB (n=54), resource_peaks.gpu_used_mib peak 14569.0 MiB (n=54); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 21438676 bytes (n=54); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5683308 bytes (n=54); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 33845488 bytes (n=54) |
| master-cpu-mtp-off | `L8192-C6144` | None | None | 6144 | valid complete cells retained | No resource telemetry found in retained source files |
| master-cpu-mtp-off | `L16384-C12288` | None | None | 12288 | valid complete cells retained | No resource telemetry found in retained source files |
| master-cpu-mtp-off | `L32768-C28672` | None | None | 28672 | valid complete cells retained | No resource telemetry found in retained source files |
| master-cpu-mtp-off | `L65536-C61440` | None | None | 61440 | valid complete cells retained | No resource telemetry found in retained source files |
| master-cpu-mtp-off | `L131072-C126976` | None | None | 126976 | valid complete cells retained | No resource telemetry found in retained source files |
| master-cpu-mtp-off | `L196608-C192512` | None | None | 192512 | valid complete cells retained | No resource telemetry found in retained source files |
| master-cpu-mtp-off | `L262144-C258048` | None | None | 258048 | valid complete cells retained | No resource telemetry found in retained source files |
| master-cpu-mtp-on | `attempt-001-L200704` | 200704 | None | 6144 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 31.0 MiB (n=12), resource_peaks.gpu_used_mib peak 15915.0 MiB (n=12); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14322368 bytes (n=12); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5762520 bytes (n=12); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 47483432 bytes (n=12) |
| master-cpu-mtp-on | `campaign` | 198656 | None | 196609 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 51.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15895.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320864 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6093040 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 38368208 bytes (n=3) |
| master-cpu-mtp-on | `campaign` | 198656 | None | 196609 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 49.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15897.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321448 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6148844 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 38316680 bytes (n=3) |
| master-cpu-mtp-on | `L8192-C6144` | None | None | 6144 | valid complete cells retained | No resource telemetry found in retained source files |
| master-cpu-mtp-on | `L16384-C12288` | None | None | 12288 | valid complete cells retained | No resource telemetry found in retained source files |
| master-cpu-mtp-on | `L32768-C28672` | None | None | 28672 | valid complete cells retained | No resource telemetry found in retained source files |
| master-cpu-mtp-on | `L65536-C61440` | None | None | 61440 | valid complete cells retained | No resource telemetry found in retained source files |
| master-cpu-mtp-on | `L131072-C126976` | None | None | 126976 | valid complete cells retained | No resource telemetry found in retained source files |
| master-cpu-mtp-on | `campaign` | 198656 | None | 196609 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 51.0 MiB (n=2), resource_peaks.gpu_used_mib peak 15895.0 MiB (n=2); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321348 bytes (n=2); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6098852 bytes (n=2); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 39668012 bytes (n=2) |
| master-cpu-mtp-on | `campaign` | 198656 | None | 196609 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 51.0 MiB (n=2), resource_peaks.gpu_used_mib peak 15895.0 MiB (n=2); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321300 bytes (n=2); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6156060 bytes (n=2); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 38221716 bytes (n=2) |
| master-gpu-mtp-off | `campaign` | 64512 | None | 62464 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1259.0 MiB (n=2), resource_peaks.gpu_used_mib peak 14687.0 MiB (n=2); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320928 bytes (n=2); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5941680 bytes (n=2); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44196304 bytes (n=2) |
| master-gpu-mtp-off | `campaign` | 80896 | None | 78849 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 923.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15023.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321032 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5941656 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44243816 bytes (n=3) |
| master-gpu-mtp-off | `campaign` | 102400 | None | 100353 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 501.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15445.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320580 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5934224 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44125424 bytes (n=3) |
| master-gpu-mtp-off | `campaign` | 122880 | None | 120833 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 71.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15876.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321068 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5934444 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 43751620 bytes (n=3) |
| master-gpu-mtp-off | `campaign` | 16384 | None | 12288 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2255.0 MiB (n=1), resource_peaks.gpu_used_mib peak 13691.0 MiB (n=1); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321344 bytes (n=1); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6023692 bytes (n=1); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44005512 bytes (n=1) |
| master-gpu-mtp-off | `campaign` | 64512 | None | 62465 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1261.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14685.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320920 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5941644 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44288008 bytes (n=3) |
| master-gpu-mtp-off | `campaign` | 80896 | None | 78849 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 925.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15021.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321132 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5941684 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44092752 bytes (n=3) |
| master-gpu-mtp-off | `campaign` | 90112 | None | 88065 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 753.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15193.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320868 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5932732 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44117728 bytes (n=3) |
| master-gpu-mtp-off | `campaign` | 102400 | None | 100353 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 501.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15445.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320856 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5934288 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44123120 bytes (n=3) |
| master-gpu-mtp-off | `campaign` | 122880 | None | 120833 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 81.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15865.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321160 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5938184 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44086200 bytes (n=3) |
| master-gpu-mtp-off | `campaign` | 30720 | None | 28672 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1953.0 MiB (n=2), resource_peaks.gpu_used_mib peak 13993.0 MiB (n=2); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320876 bytes (n=2); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5942032 bytes (n=2); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44210796 bytes (n=2) |
| master-gpu-mtp-off | `campaign` | 64512 | None | 62465 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1261.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14685.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320528 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5941672 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44201108 bytes (n=3) |
| master-gpu-mtp-off | `campaign` | 80896 | None | 78849 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 925.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15021.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320864 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5941668 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44092964 bytes (n=3) |
| master-gpu-mtp-off | `campaign` | 102400 | None | 100353 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 483.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15463.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321056 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5934172 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44066368 bytes (n=3) |
| master-gpu-mtp-off | `campaign` | 122880 | None | 120833 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 63.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15883.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320564 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5934392 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44074560 bytes (n=3) |
| master-gpu-mtp-off | `campaign` | 16384 | None | 12288 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 2255.0 MiB (n=1), resource_peaks.gpu_used_mib peak 13691.0 MiB (n=1); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320936 bytes (n=1); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6023692 bytes (n=1); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44009100 bytes (n=1) |
| master-gpu-mtp-off | `campaign` | 64512 | None | 62464 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1259.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14687.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321088 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5941664 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44127784 bytes (n=3) |
| master-gpu-mtp-off | `campaign` | 80896 | None | 78849 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 941.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15005.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320388 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5941700 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44176920 bytes (n=3) |
| master-gpu-mtp-off | `campaign` | 90112 | None | 88065 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 751.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15195.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320436 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5934156 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44091248 bytes (n=3) |
| master-gpu-mtp-off | `campaign` | 102400 | None | 100353 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 499.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15447.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14320528 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5934344 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 44013104 bytes (n=3) |
| master-gpu-mtp-on | `gpu-8192-6144-r2` | None | None | 6144 | valid complete cells retained | No resource telemetry found in retained source files |
| master-gpu-mtp-on | `campaign` | 16384 | None | 12289 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1445.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14501.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321384 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8388604 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41547468 bytes (n=3) |
| master-gpu-mtp-on | `gpu-16384-14336` | None | None | 14336 | valid complete cells retained | No resource telemetry found in retained source files |
| master-gpu-mtp-on | `gpu-32768-30000-r2` | None | None | 30000 | valid complete cells retained | No resource telemetry found in retained source files |
| master-gpu-mtp-on | `gpu-65536-61440-r4` | None | None | 61440 | valid complete cells retained | No resource telemetry found in retained source files |
| master-gpu-mtp-on | `campaign` | 73728 | None | 71680 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 13.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15933.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321472 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6350424 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42536320 bytes (n=3) |
| master-gpu-mtp-on | `campaign` | 16384 | None | 12288 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1451.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14495.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321440 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8388600 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41647268 bytes (n=3) |
| master-gpu-mtp-on | `campaign` | 73728 | None | 71681 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 9.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15937.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321672 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6350476 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42684924 bytes (n=3) |
| master-gpu-mtp-on | `campaign` | 16384 | None | 12289 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1443.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14503.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321380 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8388604 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41464688 bytes (n=3) |
| master-gpu-mtp-on | `campaign` | 73728 | None | 71681 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 9.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15937.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321192 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6350468 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42555536 bytes (n=3) |
| master-gpu-mtp-on | `campaign` | 16384 | None | 12289 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 1447.0 MiB (n=3), resource_peaks.gpu_used_mib peak 14499.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321740 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 8388600 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 41682036 bytes (n=3) |
| master-gpu-mtp-on | `campaign` | 73728 | None | 71681 | valid complete cells retained | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 9.0 MiB (n=3), resource_peaks.gpu_used_mib peak 15937.0 MiB (n=3); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14321620 bytes (n=3); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 6350500 bytes (n=3); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 42696580 bytes (n=3) |
| current-aco-mtp-on | `candidate-full256k-aco-mtp-on-20261104T105341378940038Z` | None | 65536 | None | complete | GPU_global_used_free: resource_peaks.peaks.gpu_used_mib peak 15713.0 MiB (n=11); other_memory_measure: resource_peaks.peaks.system_memory_used_kib peak 17352584 reported (n=11); server_process_VmSwap: resource_peaks.peaks.server_vm_swap_kib peak 543148 bytes (n=11), resource_peaks.peaks.system_swap_used_kib peak 3597480 bytes (n=11) |
| current-aco-mtp-off | `candidate-full256k-aco-mtp-off-20261104T115523300080953Z` | None | 65536 | None | complete | GPU_global_used_free: resource_peaks.peaks.gpu_used_mib peak 14611.0 MiB (n=11); other_memory_measure: resource_peaks.peaks.system_memory_used_kib peak 16048792 reported (n=11); server_process_VmSwap: resource_peaks.peaks.server_vm_swap_kib peak 0 bytes (n=11), resource_peaks.peaks.system_swap_used_kib peak 3035044 bytes (n=11) |
| master-cpu-mtp-on | `known capacity result` | 262144 | None | None | known clean_startup_cuda_oom (not a missing mandatory decode cell) | No resource telemetry found in retained failure receipt |
| master-gpu-mtp-off | `known runtime-limit result` | 131072 | None | None | known reproduced_cuda_illegal_memory_access; not classified as allocation OOM | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 463.0 MiB (n=1), resource_peaks.gpu_used_mib peak 15483.0 MiB (n=1); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14322356 bytes (n=1); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5781040 bytes (n=1); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 53966216 bytes (n=1) |
| fork-cpu-mtp-on | `fork-cpu-mtp-on` | 262144 | None | 256001 | quarantined; excluded from primary comparison | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 561.0 MiB (n=8), resource_peaks.gpu_used_mib peak 15385.0 MiB (n=8); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14323824 bytes (n=8); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5757532 bytes (n=8); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 43099280 bytes (n=8) |
| master-cpu-mtp-on | `master-cpu-mtp-on` | 200704 | None | 28673 | quarantined; excluded from primary comparison | GPU_global_used_free: resource_peaks.gpu_free_min_mib peak 31.0 MiB (n=4), resource_peaks.gpu_used_mib peak 15915.0 MiB (n=4); server_process_VmHWM: resource_peaks.server_vmhwm_max_kib peak 14322272 bytes (n=4); server_process_VmSwap: resource_peaks.system_swapfree_min_kib peak 5757532 bytes (n=4); whole_host_MemAvailable: resource_peaks.system_memavailable_min_kib peak 46638336 bytes (n=4) |

## Separate empty-prefix controls

These controls are not points in stepped filled-context curves.

| Family | Configured L | Label | Source report |
|---|---:|---|---|
| master-gpu-mtp-on | 77824 | gpu-empty-77824 | `MASTER106_GPU.json` |
| fork-gpu-mtp-on | 77824 | 105-06 GPU-only Turbo4 | `MATRIX105_105-06.json` |
