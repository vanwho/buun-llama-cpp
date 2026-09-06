# Phase 23 repaired-release Turbo4 context curve — task 23-07

## Result

The six-point curve is **not measured**. The 23-06 full-context gate remains
`not_measured`, and the repaired local CUDA candidate was not deployed as an
immutable executable-plus-DSO bundle. Port 8080 serves the preserved 20-07
bundle (`d534fd99…`), while the local repaired candidate is
`build-cuda/bin/llama-server` (`7968b152…`). No curve request was sent to the
unlike runtime. No speed, quality, route, movement, MTP, or memory result is
claimed.

The machine-readable receipt is [`PHASE23_CURVE.json`](PHASE23_CURVE.json).
It records all six nominal contexts, all three unchanged questions, one planned
warmup and three planned measured case IDs per question, and null metrics with
the reason they were not run. Planned case IDs are not raw results.

## Planned protocol

| Item | Contract |
| --- | --- |
| Contexts | 20,000; 40,000; 60,000; 100,000; 175,000; 262,144 |
| Sampling | T=0, seed 42, thinking off |
| Warmup | one discarded 40-token streamed request per question |
| Measured trials | three streamed requests per question, up to 400 output tokens |
| Prompt | token-counted neutral material before each unchanged original question |
| Target | selective Turbo4 target KV with canonical host Turbo4 backing |
| Native MTP | Turbo4 on GPU at each resolved context |
| Scheduling | one context job at a time, checkpoint after each case |

## Point classification

| Requested context | Startup | Speed point | Per-question medians/dispersion | Occupancy/telemetry |
| ---: | --- | --- | --- | --- |
| 20,000 | not measured | not measured | null | null |
| 40,000 | not measured | not measured | null | null |
| 60,000 | not measured | not measured | null | null |
| 100,000 | not measured | not measured | null | null |
| 175,000 | not measured | not measured | null | null |
| 262,144 | not measured | not measured | null | null |

There are zero completed trials and zero valid points. Missing values are null;
the receipt does not reinterpret the preserved server's startup counters as
curve measurements or as zero-transfer evidence.

## Runtime identity

The locally available repaired executable and CUDA DSOs are recorded in the
receipt, but are not a deployed release. The live 8080 process is PID 3761038,
started from the historical 20-07 bundle. The separate 8092 `llama-server` was
not contacted or changed. The model and pager-corpus-v4 hashes are frozen in the
receipt.

## Verification

- `sudo -n true` — passed.
- Named service/process/listener inspection — passed; 8080 identity mismatch confirmed and 8092 preserved.
- `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader` — RTX 4080, driver 595.84, 16376 MiB.
- `python3 -m json.tool .wiretail/execution/evidence/PHASE23_CURVE.json` — passed.
- `PYTHONPATH=tools/server/bench python3 -c 'import json; from pager_benchmark_contract import validate_evidence; value=json.load(open(".wiretail/execution/evidence/PHASE23_CURVE.json")); errors=validate_evidence(value); assert not errors, errors'` — passed after creation.
- `git diff --check` — passed after creation.

## Deferred verification

After a controlled lifecycle transition, deploy the repaired binary together
with its loaded project DSOs, verify model/template/config/MTP identity, pass
the 23-06 full-context and three-request telemetry preflight, then run the
resumable six-point campaign. Retain the exact per-question trial records and
report medians, dispersion, actual occupancy, budget-derived hot pages, route
fractions, movement, MTP acceptance, timing decomposition, graph reuse, and
memory peaks. Do not substitute the preserved 20-07 runtime or a shorter point.
