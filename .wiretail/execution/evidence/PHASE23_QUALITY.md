# Phase 23 quality acceptance

Task 23-05 is locally complete as a deferred quality campaign. No model
quality score is claimed.

The frozen identity is source commit `e981d13e`, the Qwen3.8 model resolved at
`/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf`, and the
`pager-corpus-v4` fixture with a SHA-256 of
`cece9edd2043c8489c1dc2f2f11c1d5a0e70542bb8f1bee6a540854ce599a0bd` and a
derived 22,016-token context ceiling. Sampling is T=0, seed 42, thinking off,
maximum output 8, one trial except ambiguous rows.

The planned matrix contains the 24 calibration/held-out corpus records in
exact, selected-all, and selective labels: 72 requests total. Zero requests
were executed and every score has denominator zero because the repaired local
binary was not deployed. The existing port-8080 candidate is a preserved
historical selected-reference runtime; port 8092 is occupied by another server
and was not contacted. Missing route, page, checksum, movement, timing,
placement, hot-budget, and MTP telemetry is therefore recorded as unavailable,
not zero.

The frozen corpus has no multi-hop category or case ID. That family is
explicitly excluded and assigned to the corpus maintainer rather than silently
treated as a passing or failing row.

## Verification

- `python3 -m unittest discover -s tools/server/bench -p 'test*.py' -v` — 48/48 passed.
- `python3 -m json.tool .wiretail/execution/evidence/PHASE23_QUALITY.json` — passed.
- `PYTHONPATH=tools/server/bench python3 -c 'import json; from pager_benchmark_contract import validate_evidence; value=json.load(open(".wiretail/execution/evidence/PHASE23_QUALITY.json")); errors=validate_evidence(value); assert not errors, errors'` — passed.
- `PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp python3 /srv/wiretail/task_state.py validate` — passed before completion.
- `git diff --check` — passed.

## Deferred verification

After a controlled lifecycle transition, deploy the repaired binary and run the
three-request telemetry preflight before scoring. Then run the exact,
selected-all, and selective matrix with the same frozen model/template/config,
recording exact numerators/denominators, route/page/movement telemetry,
host-valid rows, cold identity/checksum/promotion, timings, occupied tokens,
budget-derived hot pages, and GPU Turbo4 MTP. Add a multi-hop corpus case and
rehash the corpus before claiming that family’s coverage.
