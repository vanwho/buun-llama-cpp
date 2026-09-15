# Phase-70 benchmark summary

Revision: `hotpath-v10-20260914`. This summary aggregates only phase-70
receipts and raw manifests. Requested, observed, and immutable identities are
kept separate.

## Result

Phase 70 measured repaired A/B/A answer quality, organic physical promotion,
full-L allocation/forward progress, and the matched three-prompt benchmark.
It did not establish occupied `C262144`. The benchmark requested
`L8192/C6144/H4096/A2048/B128/U64`, but its clean rows observed only
`C=27, 28, 30`; its rates therefore are not C6144 claims.

| Row | Status | Finding |
| --- | --- | --- |
| A/B/A answer quality | measured | `AURORA-CEDAR-161`, `EMBER-OAK-204`, and A-again were all correct |
| Organic physical promotion | measured | Cold logical page 0 reached H2D completion and completed target-graph use |
| Matched cold prefill | measured | Nine measured rows per mode at observed C27–30; selected/native median 673.215978 tok/s |
| Matched committed decode | measured | Nine measured rows per mode; selected/native median 33.2694229 tok/s |
| Native MTP | measured | 2,247 drafted and 6 accepted across 9 request-scoped denominators; 0.26702225% |
| Cached-append rate | not_run | No isolated 64/256-token cached-append timing row was captured in phase 70 |
| L262144 allocation/startup | measured | H8192 allocation passed with 138,412,032 target bytes and 276,955,136 draft bytes |
| Occupied C262144 | failed | Bounded run stopped at durable C29297/live C30336 under wall budget |
| Controlled model query | not_run | Phase 70 used the organic A/B/A path, not a separate controlled captured-query run |

The phase-68 C30-versus-requested-C6144 limitation remains explicit: phase 70
did not prove the requested occupied coordinate. Allocation, a correct answer,
or a published mapping is not substituted for occupied context proof.

Raw sources and immutable identity details are in
[`V10_SUMMARY_70.json`](V10_SUMMARY_70.json).
