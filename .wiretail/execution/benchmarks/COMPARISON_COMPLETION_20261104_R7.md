# Comparison completion R7 — active, 2026-11-04

**Active and frozen after root review.** This plan carries forward the exact missing-cell set from the R6 preview and retains every valid complete cell already available.

- Primary plan: **600 requests** across **150 selected measured-cell groups** (147 distinct missing cells plus three independent C28672/off/P3 rechecks).
- Startup fallbacks replace a failed profile before measurements; they are alternatives, not448 additional requests. The automatic queue budget is at most600 canonical requests.
- Fill targets: **45 primary** and **61** when conditional fallback profiles are included.
- All ordinary profiles use normal server prefix caching with a fixed repository base and `context_checkpoints=32`; the approved fork GPU MTP-off ingest-only L122880 fallback keeps 8.
- No benchmark-owned slot save/restore/erase is used. Batch, ubatch, quantization, MTP mode, threads, model, build, template, and mapped source remain unchanged.
- Requested decode requirements remain the full R6 historical union; gaps excluded by the GPU L122880 bound remain explicitly missing/capacity N/A.
- R6 C130048 GPU capacity limits remain source outcomes; R7 does not repeat those infeasible endpoints. R7 retries master GPU-off C6144 once under 32 checkpoints using the resolved R1 normal-prefix receipt.
- CPU 32-checkpoint capacity note: prior normal-cache lookup had no usable checkpoint after C61440; budget increase is at most 3.51 GiB host against roughly 50 GiB free. Verify first repeated-request cache hit before continuing.

## Ordered steps

| # | Family / allocation | Work | Primary decode cells | Conditional fallback |
|---:|---|---|---:|---|
| 1 | master-cpu-mtp-off · L262144 (32 CP) | R6 completed the first four context checkpoints as fills. Decode only the 27 recorded missing low/medium/xhigh cells at 126976, 192512, and 258048; keep C6144/12288/28672/61440 ingest-only. | 27 | none |
| 2 | master-gpu-mtp-off · L122880 (32 CP) | Run only the exact missing GPU MTP-off cells that fit at L122880 with 32 checkpoints; all R6 lower-profile fill targets remain included. C130048 stays capacity N/A. | 42 | none |
| 3 | fork-cpu-mtp-on · L262144 (32 CP), L200704 (32 CP), L131072 (32 CP) | Collect the 37 missing fork CPU MTP-on cells at the primary L262144 profile. Use only the frozen L200704/L131072 profiles after startup capacity failure; each fallback selection is clipped to its actual L-400 context bound. | 37 | L200704: 37 cells; L131072: 36 cells |
| 4 | master-cpu-mtp-on · L200704 (32 CP), L131072 (32 CP) | Collect the 41 missing master CPU MTP-on cells at L200704. The known master L262144 startup OOM remains an explicit historical bound; the L131072 fallback runs only its 39 feasible selected cells after startup capacity failure. | 41 | L131072: 39 cells |
| 5 | master-cpu-mtp-on · L32768 (32 CP) | Independent session A: one mapped fill per series; only C28672/off/P3 gets warmup plus three measured requests. Keep all three series as robustness evidence; no result cherry-picking. | 1 | none |
| 6 | master-cpu-mtp-on · L32768 (32 CP) | Independent session B: one mapped fill per series; only C28672/off/P3 gets warmup plus three measured requests. Keep all three series as robustness evidence; no result cherry-picking. | 1 | none |
| 7 | master-cpu-mtp-on · L32768 (32 CP) | Independent session C: one mapped fill per series; only C28672/off/P3 gets warmup plus three measured requests. Keep all three series as robustness evidence; no result cherry-picking. | 1 | none |
| 8 | fork-gpu-mtp-off · L122880 (8 CP) | After the earlier R6 endpoint capacity limit, retain the already approved L122880 fallback strictly for mapped ingest. Do not repeat historical decode cells; C130048 remains capacity N/A. | 0 | none |

## Missing-cell coverage and non-completion notes

R6 preview had 171 missing declared cells across all families. R7 primary selections include 147 feasible gap cells; the 24 GPU C130048 cells remain capacity N/A. Any fallback-omitted cells and any uncompleted R7 selections stay missing until valid triplets are recorded.

The three outlier series each keep a separate mapped fill history and one C28672/off/P3 cell. They are not merged or cherry-picked. Existing R4/R5/R6 retained sources are explicit paths; the two R4 CPU MTP-on target-only restore cohorts remain quarantined.

The previous master GPU-off C6144 normal-prefix receipt is linked by source path and SHA-256 in `reused_decode_outcomes`; it remains a missing result until the R7 retry passes. The R6 C130048 outcomes are capacity-limited observations, not measurements at C130048 under R7.

## Validation summary

Generated from the now-archived R6 revision6; its SHA-256 is pinned in JSON. Root reviewed and activated R7. The runner command targets the canonical plan path so R7 stays draft-only until activation. R7 primary request count: 600; selected primary decode cells: 150; total fill targets across all profiles: 61. All profile server argv were regenerated from `matrix.argv_for_row` using the explicit per-profile checkpoint count. Build, model, DSO, template, and chunk-map hashes remain inherited; light driver hashes were refreshed.

## Canonical request shape

Each canonical request is the filled repo base plus the current question. Earlier benchmark questions and answers do not accumulate. New disjoint repo chunks alone grow the ingest frontier. No benchmark KV disk save/restore/erase is performed. New ordinary series use32 native checkpoints; the existing fork GPU-off ingest-only lower-allocation profile keeps8. Native checkpoint images are server-managed host state, not benchmark slot files.

The first128K repeated question must retain its recent prefix. If it falls back to a much older checkpoint, stop before repeating that cost; preserve any valid completed cells and diagnose the actual capture/retention predicate. The observed150 MiB target checkpoint estimate excludes any additional native-MTP companion state; monitor existing host resource samples without inventing a telemetry gate.
