# Phase-68 benchmark summary

Revision: `hotpath-v10-20260914`. This summary aggregates only the phase-68
receipts and raw manifests. Requested, observed, and immutable identities are
kept separate.

## Result

Phase 68 measured repaired organic physical promotion, restored cached append
reuse, the matched three-prompt benchmark, and bounded full-L allocation and
forward progress. It did not establish useful nonce answer quality or occupied
`C262144`.

The matched benchmark coordinate was requested as
`L8192/C6144/H4096/A2048/B128/U64`, with 256-token pages. The actual clean
short benchmark rows observed `C=30`, so the rate rows are not an occupied
`C6144` claim. The full-L pilot used `L262144`, Turbo4 target K/V on CUDA,
Turbo4 native-MTP K/V on GPU, and a 262144-row draft allocation.

## Findings

| Row | Status | Finding |
| --- | --- | --- |
| Controlled physical promotion | measured | Cold page reached H2D completion, mapping publication, and target-graph use |
| Organic physical promotion | measured | A/B/A page was cold, promoted, and consumed by the target graph |
| Answer quality | failed | `AURORA-CEDAR-161` was absent; responses were `////////////////` |
| Cached append 64/256 | measured | Both reused restored `cache_n=6087`; rates and MTP denominators are recorded |
| Matched benchmark | measured | 9 measured rows per mode; feature-off controls remain outside native acceptance |
| Full-L allocation | measured | `L262144` startup passed; allocation is not occupancy proof |
| Occupied `C262144` | failed | Final bounded frontier was durable `C12000`, live `C12127` |

Selected/native MTP had 2,243 drafted and 8 accepted tokens across 9 rows,
for a request-scoped acceptance rate of 0.35666518%. Append probes had valid
denominators of 251 drafted and zero accepted tokens each. CPU-main-KV and
all-GPU rows were explicit feature-off controls.

Rates are recomputed from phase-68 raw records. Median cold-prefill rates were
649.476 tok/s selected/native, 295.004 tok/s CPU-main-KV/GPU-draft, and
652.174 tok/s all-GPU/GPU-draft. Median committed-decode rates were 33.557,
17.960, and 49.778 tok/s respectively. Repaired append rates were 72.278
tok/s for 64 requested tokens and 82.758 tok/s for 256 requested tokens, using
observed new prompt tokens over prompt wall time.

Full-L startup allocated 423,886,848 target bytes and 276,955,136 draft bytes.
The bounded H8192 recovery reported 138,412,032 target bytes, 201,326,592
bytes headroom, 2,068,443,264 reserved bytes, and durable/live occupancy
`12000/12127`. The larger H25088 path stopped at C1200 on packed selected
attention allocation failure. None of these allocation or reserve figures is
substituted for occupied-context proof.

Raw paths and receipt references are listed in `V10_SUMMARY_68.json`. No
phase-66 rate or historical result is imported.
