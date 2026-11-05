# Supplemental comparison ledger — 2026-11-05

This addendum leaves the retained audited snapshot unchanged. [`decode-summary.csv`](decode-summary.csv), [`comparison-tables.md`](comparison-tables.md), [`comparison-digest.md`](comparison-digest.md), and [`manifest.json`](manifest.json) remain the original quality-audited cohort. This file contains additional, explicitly attributed cells that were absent from or quarantined by that snapshot. It stores aggregate metrics and provenance only; no responses, SSE, request bodies, or model files are copied.

Cell inclusion requires at least two measured repetitions with `status=pass`, a positive completion count, and no saved output meeting the snapshot repeated-punctuation rejection rule. Invalid repetitions are not included in max or median. Decode summary uses the maximum of the eligible repetitions; the median and individual speed/token values are retained for audit. Cells with fewer than two eligible repetitions are gaps and are not labelled usable.

## Supplemental cohorts

| Cohort | Family | Usable cells | Fill/method interpretation |
|---|---|---:|---|
| `fork-cpu-mtp-on-r4-restored-base` | `fork-cpu-mtp-on` | 37 | Historical R4, disk snapshot restore/resume; retained separately from cumulative runs. |
| `master-cpu-mtp-off-192k-independent-base` | `master-cpu-mtp-off` | 1 | Independent questions reuse one frozen base; not cumulative conversation. |
| `master-cpu-mtp-off-256k-independent-base` | `master-cpu-mtp-off` | 12 | Independent questions reuse one frozen base; not cumulative conversation. |
| `master-cpu-mtp-off-phase6-cumulative` | `master-cpu-mtp-off` | 12 | Pristine master, resident cumulative conversation; invalid output repetitions excluded. |
| `master-cpu-mtp-on-phase7-cumulative` | `master-cpu-mtp-on` | 2 | Pristine master 192K run, resident cumulative conversation. |
| `master-gpu-mtp-off-120k-independent-base` | `master-gpu-mtp-off` | 3 | GPU target; independent questions reuse one frozen base; not cumulative conversation. |

R4 contributed 37 eligible fork CPU/MTP-on keys: 33 cover selected cells absent from the retained snapshot and 4 overlap same family/context/mode/prompt keys already retained. These are supplemental source rows, not pooled repetitions. Thus the snapshot’s 33 selected fork CPU/MTP-on cells are covered by R4, with the restored-base caveat; the snapshot itself is unchanged.

## Exact high-context additions and remaining gaps

- **Fork CPU/MTP-on:** The selected 37-cell R4 set is represented with 3 eligible repetitions per cell. R4 uses `disk_snapshot_restore`/resume; do not treat its prefill or lifecycle as cumulative-conversation evidence.
- **Master CPU/MTP-on 192K:** phase7 provides the two covered cumulative cells (off/P3 and xhigh/P2), each with 3 eligible reps. Their measured nominal target is `C=195664` at `L=200704`; the graph label is 192K.
- **Master CPU/MTP-on 256K:** 12 cells at nominal C=256000 remain unmeasured. Clean pristine-master startup failed before readiness on a 1,730 MiB GPU allocation request with 15,900 MiB free; do not synthesize timings or relabel it as success.
- **Master CPU/MTP-off cumulative 192K:** phase6 has 11 usable cells at measured `C=195664`. Medium/P2 has 0 valid reps (all three measured records are `output_invalid`), so it remains a gap for this cumulative cohort.
- **Master CPU/MTP-off cumulative near 256K:** phase6 has one usable cell at C=255488 (off/P1; two valid reps); off/P2 and off/P3 have zero valid reps and remain gaps in that phase6 subset.
- **Master CPU/MTP-off independent-base:** the 192K medium/P2 cell (measured `C=195664`) has 3 eligible reps; C=255488 has 12 cells (off/low/medium/xhigh × P1–P3), each 3 eligible reps. These independent-query results are not pooled with the cumulative phase6 measurements.
- **Master GPU/MTP-off independent-base:** C=120832 xhigh/P1–P3 has 3 eligible reps per cell. The retained snapshot has the other 9 cells at this coordinate; the new xhigh data fills the remaining three for independent-base queries, but not for a cumulative-conversation cohort. The retained master GPU/MTP-off total of 42 feasible cumulative gaps is unchanged by this separate-method evidence.

## Retained snapshot

The original audited snapshot files and their hashes remain listed in `manifest.json`; the original counts and quality exclusions are not rewritten here. The additive rows above preserve their own source record path and SHA-256, base occupancy, context length, runtime family, binary/model identity, mode, prompt, per-repetition speeds and token counts, maximum eligible speed, and method caveat.
