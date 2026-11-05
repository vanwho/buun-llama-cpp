# Comparison digest — audited snapshot, 2026-11-05

R9 finished with findings. The quality-filtered dataset contains **805 usable complete cells / 2415 measured repetitions** across ten families.

A comparison cell is one nominal context × mode × prompt with three distinct passing, complete measured repetitions. Raw response audit excludes a whole triplet only when a hash-verified stream has at least 64 nonspace characters, no alphanumeric visible content or reasoning, and one punctuation character forms at least 80% of the response. Short EOS and reasoning-only outputs remain eligible. This narrow check does not establish semantic accuracy. 189 source triplets with missing raw SSE remain unassessed.

## Family coverage

“Selected gap-fill cells” count requirements from the completion plan; they are not the total historical coverage. The usable-cell column includes retained historical measurements.

| Family | Usable cells | Selected gap-fill recovered / total | Feasible missing | Capacity N/A |
|---|---:|---:|---:|---:|
| master-gpu-mtp-off | 54 | 0/54 | 42 | 12 |
| master-gpu-mtp-on | 72 | 0/0 | 0 | 0 |
| fork-gpu-mtp-off | 102 | 0/12 | 0 | 12 |
| fork-gpu-mtp-on | 60 | 0/0 | 0 | 0 |
| master-cpu-mtp-off | 84 | 63/63 | 0 | 0 |
| master-cpu-mtp-on | 31 | 7/48 | 41 | 0 |
| fork-cpu-mtp-off | 84 | 0/0 | 0 | 0 |
| fork-cpu-mtp-on | 54 | 4/37 | 33 | 0 |
| current-aco-mtp-off | 132 | 0/0 | 0 | 0 |
| current-aco-mtp-on | 132 | 0/0 | 0 | 0 |

## Feasible gaps by nominal checkpoint

| Family | Nominal C | Missing cells | Detail |
|---|---:|---:|---|
| master-gpu-mtp-off | 6144 | 12 | not measured or blocked by runtime fault |
| master-gpu-mtp-off | 12288 | 10 | not measured or blocked by runtime fault |
| master-gpu-mtp-off | 28672 | 10 | not measured or blocked by runtime fault |
| master-gpu-mtp-off | 62464 | 1 | not measured or blocked by runtime fault |
| master-gpu-mtp-off | 88064 | 6 | not measured or blocked by runtime fault |
| master-gpu-mtp-off | 120832 | 3 | not measured or blocked by runtime fault |
| master-cpu-mtp-on | 6144 | 3 | not measured or blocked by runtime fault |
| master-cpu-mtp-on | 12288 | 9 | not measured or blocked by runtime fault |
| master-cpu-mtp-on | 28672 | 9 | not measured or blocked by runtime fault |
| master-cpu-mtp-on | 61440 | 9 | not measured or blocked by runtime fault |
| master-cpu-mtp-on | 126976 | 9 | not measured or blocked by runtime fault |
| master-cpu-mtp-on | 196608 | 2 | not measured or blocked by runtime fault |
| fork-cpu-mtp-on | 12288 | 6 | 6 quality-rejected; 0 not completed |
| fork-cpu-mtp-on | 28672 | 9 | 9 quality-rejected; 0 not completed |
| fork-cpu-mtp-on | 61440 | 9 | 9 quality-rejected; 0 not completed |
| fork-cpu-mtp-on | 126976 | 8 | 7 quality-rejected; 1 not completed |
| fork-cpu-mtp-on | 196608 | 1 | not measured or blocked by runtime fault |

## Separate capacity limits

| Family | Nominal C | Cells | Status |
|---|---:|---:|---|
| master-gpu-mtp-off | 130048 | 12 | capacity_limit at R6 L131072; R7 L122880 excludes this nominal context |
| fork-gpu-mtp-off | 130048 | 12 | capacity_limit at R6 L131072; not retried; R7 only fills through L122880 |

## Findings and limits

- The fork-GPU/MTP-off R9 mapped fill completed nine append groups through nominal C=120,832 (actual final C=120,833); it produced no decode cells.
- Master-GPU/MTP-off still has 42 feasible gaps after a CUDA illegal access on the first normal appended question; the separate 12-cell C=130,048 endpoint is capacity N/A.
- Master-CPU/MTP-on has 41 feasible gaps. A clean L=262,144 startup check failed before readiness with a 1,730 MiB GPU allocation request, 15,900 MiB free, and no other compute owner; do not synthesize 256K timings.
- Three independent master-CPU/MTP-on C=28,672/off/P3 sessions each retained two measured repetitions; the third failed with the DFlash target argmax row lookup fault, so no triplet is reported.
- R4 target-only slot-restore CPU/MTP-on cohorts remain quarantined. R8’s 30 protocol-complete fork-CPU/MTP-on triplets yielded only 3 quality-eligible cells; R9 added one more eligible cell.
- Full sanitized source inventory and charts remain external. Raw SSE, answer text, binaries, and model files are not copied into this snapshot.

## Snapshot files

- [Per-cell decode and acceptance min/median/max tables](comparison-tables.md)
- [Decode triplet CSV with source paths and hashes](decode-summary.csv)
- [Mapped-prefill interval CSV](mapped-prefill.csv)
- [Response-quality failure inventory](response-quality-failures.csv)
- [Manifest with source/output hashes, coverage, and limitations](manifest.json)

The exhaustive sanitized report JSON and three chart previews remain at `/srv/ai/paged-kv/results/comparison-final-20261104-quality-audit/`; their full paths and hashes are in `manifest.json`.
