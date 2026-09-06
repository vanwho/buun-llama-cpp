# Phase 21 harness evidence

Task 21-04 adds a bounded preflight and fail-fast boundary to the quality
corpus runner. The standalone preflight plan is exactly three requests:
`warm`, `cold_needle`, and `selected_all`. A campaign can be gated with
`--preflight`; `--mode preflight` runs only that diagnostic gate.

Every live case now carries the effective argv, client timestamps, server
snapshot timestamp, actual occupied prompt tokens, hot-page budget, target and
MTP placement, route, selected/physical/logical page counts, host-valid rows,
movement/fault/eviction counters, and queue/copy/wait timings. The shared
contract rejects absent telemetry and preserves measured zero values.

`--max-cases`, `--case-id`, `--case-index`, `--stop-after-failures`, and
`--fail-fast-capability` are explicit controls. Remaining cases are durable
`not_run` records, with an atomic `campaign-checkpoint.json`; resume only
reuses matching completed case keys. `report.json` and `report.md` classify
setup failure, capability refusal, quality mismatch, timeout, valid measurement,
and not-run cases separately.

## Verification

- `python3 -m unittest discover -s tools/server/bench -p 'test_*.py'` — 48 passed.
- `python3 -m py_compile tools/server/bench/pager_benchmark_contract.py tools/server/bench/run-quality-corpus.py` — passed.
- `python3 tools/server/bench/run-quality-corpus.py --help` — passed; all task controls are exposed.
- `git diff --check` — passed.

The preflight and resume regressions use deterministic fake server responses
and complete telemetry envelopes. No full corpus was run, and no managed
service or port 8092 was touched.

## Deferred verification

An authenticated live preflight against the pager-capable managed service is
deferred to the next bounded live task. Later setup should run the standalone
`--mode preflight` command with the intended endpoint, model, API-key file,
and candidate profile, then inspect `preflight.json` and the complete request
envelopes before starting any corpus campaign. No live hardware, quality, or
speed result is claimed by this receipt.
