# Phase 23 full 256K acceptance

Task 23-06 is locally complete as a deferred full-context gate. The base goal
is explicitly **not demonstrated** (`base_goal_demonstrated: false`) because no
fresh production run using the immutable 23-05 release was available.

The required contract remains unchanged: automatic budget-derived allocation at
262,144 context tokens, 1,024 logical 256-token pages, exactly 262,144
GPU-resident native-MTP Turbo4 rows, canonical host backing, and a separate
near-full tokenized population with eight tokens of generation headroom. That
must be followed by early/middle/late/recent and focus-shift retrieval,
continuation, physical movement, and selected-all/exact parity correlation.

The deterministic pager geometry fixture passes the 262,144/1,024/256 and
native-MTP-row arithmetic checks, but it does not substitute for production
allocation, occupied population, or GPU placement evidence. The prior full-256K
receipt remains a negative control and is not relabeled as this task’s result.

## Verification

- Focused pager, residency, policy, routing, attention, cache-budget, and CUDA tests — 12/12 passed.
- `python3 -m unittest discover -s tools/server/bench -p 'test*.py' -v` — 48/48 passed.
- `python3 -m json.tool .wiretail/execution/evidence/PHASE23_FULL256K.json` — passed.
- `PYTHONPATH=tools/server/bench python3 -c 'import json; from pager_benchmark_contract import validate_evidence; value=json.load(open(".wiretail/execution/evidence/PHASE23_FULL256K.json")); errors=validate_evidence(value); assert not errors, errors'` — passed.
- `PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp python3 /srv/wiretail/task_state.py validate` — passed before completion.
- `git diff --check` — passed.

## Deferred verification

The 262,144-token automatic startup, near-full occupancy, canonical host-valid
rows, MTP GPU placement, cold retrieval/promotion/fence/slot correlation,
focus-shift continuation, and selected-all/exact parity were not run. The
existing 8080 server is a preserved historical candidate and port 8092 is owned
by another server; neither was changed. No full-context or physical-paging
claim is made. Resume only after a controlled lifecycle transition, using
resumable checkpoints and preserving the complete 262,144 logical context even
if a smaller occupied diagnosis is required.
