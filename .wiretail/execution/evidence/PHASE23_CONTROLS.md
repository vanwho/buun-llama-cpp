# Phase 23 matched controls and bottleneck decomposition — task 23-08

## Result

No release-matched control was run. The dependency curve receipt contains zero
completed PHASE23 target rows, and the repaired executable is not deployed as
one immutable executable-plus-DSO bundle. Port 8080 serves the preserved 20-07
runtime, so its historical observations remain diagnostics only. No CPU-KV,
dense all-GPU, observe, or MTP-off result is paired or used as a denominator.

The machine-readable receipt is [`PHASE23_CONTROLS.json`](PHASE23_CONTROLS.json).
It contains empty paired raw-ID arrays, null ratios and component timings, and
explicit status/reasons for every nominal context and representative point.

## Matching contract

Controls would be admitted only when the target and control share the immutable
bundle and loaded DSOs, model/tokenizer/template, exact prompt/request hash,
actual occupied tokens, logical context, sampling, cache condition, K/V codec,
and MTP placement. Because the target campaign has zero completed rows, the
receipt reports zero paired control rows and no ratios.

| Control | Status | Paired raw IDs | Ratios | Reason |
| --- | --- | --- | --- | --- |
| CPU-KV + GPU MTP | unavailable for pairing | `[]` | null | no completed target row |
| Dense all-GPU Turbo4 + MTP | unavailable for pairing | `[]` | null | shorter/historical rows would not match |
| Observe | not measured | `[]` | null | separate ablation; no target row |
| MTP-off | not measured | `[]` | null | separate ablation; no target row |

## Representative decomposition

The requested short/middle/long points are 20,000, 60,000, and 262,144
tokens. Model compute, paged attention, draft/verification, routing, H2D,
D2H, graph launch/reuse, synchronization, and memory are all null for each
point. The blocker ranking in the JSON is a ranking of measurement blockers,
not a measured performance bottleneck ranking.

Historical 3x/5x/70% comparisons are retained only as annotations in the
protocol context; they are not recomputed, used as denominators, or treated as
gates.

## Verification

- `python3 -m json.tool .wiretail/execution/evidence/PHASE23_CONTROLS.json` — passed after creation.
- `PYTHONPATH=tools/server/bench python3 -c 'import json; from pager_benchmark_contract import validate_evidence; value=json.load(open(".wiretail/execution/evidence/PHASE23_CONTROLS.json")); errors=validate_evidence(value); assert not errors, errors'` — passed after creation.
- `PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp python3 /srv/wiretail/task_state.py validate` — pending final state update.
- `git diff --check` — passed after creation.

## Deferred verification

After a controlled lifecycle transition, deploy the repaired executable and all
loaded project DSOs together, pass the 23-06 full-context and telemetry
preflight, complete the six-point target curve, then run matched CPU-KV/GPU-MTP
and safely fitting dense all-GPU Turbo4/MTP controls using one discarded
40-token warmup and three measured 400-token streamed trials. Join only exact
matching completed case IDs. Run observe and MTP-off separately, and rank only
component bottlenecks supported by paired counters and memory peaks.
