# Comparison completion — active working handoff

User confirmation on 2026-11-04: **normal server prefix caching, no benchmark
KV disk saves/restores, same context-size comparisons**. Repository ingestion
appends only fresh disjoint mapped repo chunks. Canonical questions reuse that
filled base without accumulating prior benchmark answers in the base.

## Active frozen plan

- Branch: `plan/attention-aware-kv-paging`.
- Plan: `COMPARISON_COMPLETION_20261104.json`, revision 6; output
  `/srv/ai/paged-kv/results/comparison-completion-20261104-r6`.
- R6 is running durably. Steps1/2 GPU-MTP-on mapped ingest completed through
  actual C62465; master backend/net-new1346.87 tok/s, fork910.57 tok/s. Step3
  fork GPU/MTP-off is active; query live queue progress for its latest frontier.
- Runtime owner: `/root/comparison_runtime_prep`, Luna High. One model and
  `/tmp/ai-pager-benchmark.lock`; protect Jellyfin, 8091, 8092.
- Driver fix `095a1fc05` keeps every occupied control on ordinary prefix reuse;
  no benchmark `/slots` save/restore/erase calls. 28 focused tests passed.
- No inference source, binary, DSO, model, quant, B1024/U256, t16/tb16,
  T0/seed42, 40-token warmup or 3x400 measured setting changes.
- R6 has 204 primary canonical requests at most; all valid historical results
  and the two complete current ACO curves remain. Most R6 steps are ingest-only.
- Execution order: GPU-MTP-on master/fork mapped ingest curves; fork GPU-off
  conditional 127K endpoint; master GPU-off mapped ingest; fork/master CPU-MTP-on
  mapped ingest; remaining master CPU-MTP-off non-off mode cells; three clean
  independent sessions of the one historical master CPU-MTP-on Bash outlier.
- All argv are predetermined in JSON/Markdown. Startup-OOM fallbacks retain
  B/U, threads and quants. Runtime faults never become allocation-limit claims.
- `requested_decode_requirements` preserves the earlier requested but currently
  unmeasured decode cells. Ingest-only completion does not erase decode gaps.

## Preserved runs / failures

- Current ACO on/off curves: `evidence/ACO_FULL_CURVES_20261104.json`, 264 cells,
  792 measured plus 264 warmups, all passing. Do not rerun them.
- R4 fork CPU-Q4/MTP-off completed all seven points; repaired 192K Bash
  decode min/median/max 3.6813 / 3.7312 / 3.7395 tok/s, all 400 tokens.
- R4 fork/master CPU-MTP-on target-file-restored data is quarantined. It drops
  native draft prefix/carry and is not a normal full-prefix MTP comparison.
- R5 normal resident fork CPU-MTP-on faulted at C12288 after 31 passing records.
  Master CPU-MTP-on produced invalid target-argmax rows at C6144. Retain only
  independently valid complete triplets, with partial-series status explicit.
- R5 master CPU-MTP-off C6144/C12288/C28672 each completed 36/36 requests.
  Root deliberately held the idle series driver after C28672 to remove manual
  file restores. Supervisor exit143 was an operator hold, not a model failure.
- R5 JSON/MD are archived as `COMPARISON_COMPLETION_20261104_R5.*` for audit
  only. Do not run them: their older driver hashes/protocol are superseded.
- Pristine master GPU-off repeated-context checkpoint faults reproduced at
  C6144 under multiple L values; this is not an established memory ceiling.
  R6 does not repeat that failed decode path again just to obtain another fault.
- Exact source findings: `engineering-notes/benchmark-prefix-state-failures-20261104.md`.

## Outlier and reporting policy

- Coverage index: `COMPARISON_COVERAGE_20261104.json`; retain valid historical
  whole-cell triplets, including CPU-Q4 records whose null mode was verified off.
- Old master CPU-MTP-on C28672/off/Bash had 15.9977/11.2388/11.6034 tok/s and
  63.25/29.98/29.98% acceptance. All three had identical target text and full
  fresh prompt processing (cache_n=0); restore/carry was not an evidenced cause.
  Acceptance tracks the speed spread, but its differing value remains unresolved.
  R6 schedules three independent normal-prefix rechecks, without cherry-picking.
- No >30% repeat spread or >35% adjacent decode jump in current ACO curves.
  Small-tail ingest rows remain visible, but mapped interval curves weight
  all chunks in each interval as sum(tokens)/sum(seconds).
- Primary prefill graphs: mapped repo chunk schedules only; separate backend,
  net-new/backend-time, and net-new/HTTP-wall rates. Canonical question-suffix
  prefill is a separate diagnostic, never the bulk ingestion curve.
- Preserve actual C and nominal target separately, min/median/max, source hashes,
  independent cohorts, and explicit failures. Never interpolate missing cells.
- Report generator: `report-comparisons-20261104.py`; docs owner
  `/root/aco_docs_refresh_20261104` completed R5 partial-series whitelists and
  R6 outlier aliases in commit a85dab5a6; synthetic checks passed. No broad
  import of quarantined external roots. Preview export awaits a quiet window.
- Matplotlib environment: `/srv/ai/paged-kv/venvs/comparison-report`; use Agg and
  OMP/OpenBLAS/MKL threads1. Render/export only outside timed CPU runs.
- Preview-r5 is stale and not publication output. Final report needs fresh
  generation, image inspection, completeness review, and compact tracked exports.
- Raw logs/binaries remain external. Benchmark-tool changes stay separate from
  portable inference commits. Commit/push compact reports and final graphs.

The systematic GPU-MTP-on ingest difference is documented in
`engineering-notes/dense-turbo4-prefill-cohort-difference-20261104.md`: differing
CUDA attention dispatch predicates are a source-grounded candidate, not confirmed
per-kernel timing attribution. Do not change measured binaries to investigate it.
