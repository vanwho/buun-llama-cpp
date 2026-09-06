# Phase 23 physical-pressure and lifecycle soak — task 23-09

## Result

The soak was **not measured**. The repaired target bundle is not deployed, the
23-06 full-context gate remains not measured, and the dependent 23-07/23-08
receipts contain no completed target rows. Port 8080 still serves the
preserved 20-07 runtime; port 8092 is an unrelated CPU server and was not
touched. No physical paging, lifecycle, resource-plateau, or state-preservation
claim is made.

The machine-readable receipt is [`PHASE23_SOAK.json`](PHASE23_SOAK.json). Each
segment is independently `not_measured`, has planned case IDs, an empty
completed-case list, and a resumable classification. Missing movement and
resource values are null; no preserved-runtime counter is imported as zero.

## Segment classification

| Segment | Status | Planned cases | Completed cases | Correlation/resources |
| --- | --- | --- | --- | --- |
| Warm focus | not measured | 2 | 0 | null |
| Known cold-page promotion | not measured | 3 | 0 | page/checksum/fence/slot null |
| Early/middle/late focus shift | not measured | 3 | 0 | answers/counters null |
| Sustained churn | not measured | 4 | 0 | movement/plateau null |
| Cancellation and drain | not measured | 2 | 0 | drain/follow-up null |
| Checkpoint save/restore | not measured | 3 | 0 | state/checkpoint null |
| Restart | not measured | 2 | 0 | restart/identity/state null |
| Slot reuse | not measured | 2 | 0 | generation/parity/plateau null |

No occupied-page pressure was asserted because the repaired runtime’s actual
budget-derived hot capacity was unavailable. No shorter diagnostic or
historical run was promoted into this receipt.

## Verification

- Named process/listener inspection — passed; 8080 identity mismatch confirmed and 8092 preserved.
- `python3 -m json.tool .wiretail/execution/evidence/PHASE23_SOAK.json` — passed after creation.
- `PYTHONPATH=tools/server/bench python3 -c 'import json; from pager_benchmark_contract import validate_evidence; value=json.load(open(".wiretail/execution/evidence/PHASE23_SOAK.json")); errors=validate_evidence(value); assert not errors, errors'` — passed after creation.
- `git diff --check` — passed after creation.

## Deferred verification

After a controlled lifecycle transition, deploy the repaired executable and all
loaded project DSOs, pass the full-context and telemetry preflight, establish
the measured hot capacity, and run the segments one at a time with append-only
case records and atomic progress checkpoints. For every cold result retain
logical identity, canonical checksum, transfer completion event/fence, physical
slot/generation, selected route, answer, and post-request counters. Sample all
host/pinned/VRAM/event/descriptor/page-table/queue/RSS resources at before,
high-water, and after points. Never touch port 8092 or unowned processes.
