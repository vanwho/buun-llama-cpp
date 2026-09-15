# Phase-72 benchmark summary

Revision: `hotpath-v10-20260914`. This summary aggregates only the phase-72
receipts and raw manifests. Requested, observed, and immutable identities stay
separate.

## Result

Phase 72 produced a controlled-model C6144 promotion fixture, an 8K/H8192
bounded occupancy frontier, and a matched post-repair q0/q1/q2 benchmark. The
full-L allocation passed, but occupied C262144 did not.

| Row | Status | Finding |
| --- | --- | --- |
| Verified controlled C6144 coordinate | measured | L8192/C6144/H4096/A2048/B128/U64; cold page reached completed target-graph use |
| 8K occupancy proof | measured | L262144/H8192 reached durable C40001 and live C40014 in 29 turns |
| L262144 allocation | measured | Target 138,412,032 bytes; draft 276,955,136 bytes; admission passed |
| Occupied C262144 | failed | Bounded run stopped at C40001/C40014; full-L allocation is not occupancy |
| Matched cold prefill | measured | Three measured trials per prompt; selected/native observed C39–42, controls C27–30 |
| Matched committed decode | measured | Three measured trials per prompt; 128 output tokens |
| Cached append 64/256 | not_run | C6143 prefix request did not commit a live frontier |
| Native MTP | measured | 2,229 drafted and 15 accepted across nine positive request-scoped denominators; 0.67294751% |
| Controlled target-graph use | measured | 72-01 cold logical page 0 promoted and consumed by the target graph |
| Organic target-graph use | measured | 72-04 selected/native natural cold candidate recorded completed target-graph use |
| Answer quality | not_run | No phase-72 quality campaign; correctness was not inferred |

The requested C6144 coordinate is a verified controlled fixture, while the
benchmark rows are short-prompt observations and are not C6144 rates. The 8K
proof and the L262144 allocation do not establish occupied C262144. Cached
append rates are absent rather than estimated from the failed-to-commit prefix.

Placement, byte details, request-scoped MTP denominators, pilot boundaries,
and raw source paths are recorded in
[`V10_SUMMARY_72.json`](V10_SUMMARY_72.json).
