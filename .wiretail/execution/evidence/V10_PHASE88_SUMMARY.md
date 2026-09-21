# Phase-88 compact summary

Revision: `hotpath-v10-20260914`; amendment: `repair87-20260921`.
Scope: phase88 raw records and manifests only. Nulls remain explicit where a
measurement was invalid or a claim was not established.

## Promotion and semantic quality

The repaired organic native GPU Turbo4 run completed cold candidate use,
8,650,752 useful H2D bytes, publication, and target-graph consumption. The
same logical page 0 was published at epoch 2992 into physical slot 2, with
target-use epochs 4439, 4573, and 4651. This is an organic result; no
controlled promotion fixture was run in phase88.

The exact sequence was `ACK-A`, `ACK-B`, `A-ARCHIVE-MARKER-914`, `ACK-C`.
Draft restoration was replayed and target restoration trimmed, with positive
restore epochs 1, 2, and 3. A separate long-run probe remains a finding:
`A-again` returned `A-ARCHIVE-META`, so it is not converted into a success.

## MTP and speed

The repaired organic request-scoped MTP sum is 1 accepted of 16 attempts
(6.25%): A `0/2`, B `0/2`, A-again `1/10`, terminal `0/2`.

The matched direct-GPU fresh row measured 276.869 evaluated prompt tok/s,
29.236 committed decode tok/s, and 202.262 ms TTFT. Cached-append speed is
`null` because the server reported no cache; the context-bearing row is also
`null` for cold-promotion speed because no promotion or H2D completion was
observed. Their attribution counters remain retained separately in the JSON.

## Allocation versus occupied frontier

| Quantity | Result |
| --- | ---: |
| Full-L allocation | 262,144 tokens |
| Target / physical-pool allocation | 69,206,016 / 69,206,016 bytes |
| H / A / B / U | 4096 / 2048 / 128 / 64 tokens |
| Committed frontier sequence | 1200 → 5292 → 9384 → 13476 → 17568 → 21660 → 25752 |
| Final committed C | 25,752 tokens |
| Resident / host pages | 16 / 15 |
| Target / host valid rows | 3992 / 3840 |
| Full capacity occupied | `false` |

High-water scratch was 8,331,264 bytes for prefill, draft, and verify;
rollback was not exercised and is `null`.

The structured source of truth is `V10_PHASE88_SUMMARY.json`. Raw phase88
receipts and record hashes are listed there; no phase87 record is used.
