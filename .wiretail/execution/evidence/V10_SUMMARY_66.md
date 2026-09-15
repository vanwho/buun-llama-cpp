# Phase-66 benchmark summary

Result: phase 66 measured the repaired controlled promotion chain, matched
small-path benchmark rows, and bounded full-L forward progress. Organic
target-graph consumption, cached append reuse, useful nonce answer quality,
and occupied C262144 remain failed boundaries.

The matched coordinate is L8192/C6144/H4096/A2048/B128/U64 with 256-token
pages. The observed target and native-MTP K/V are Turbo4 on CUDA/GPU. The
resolved model is Qwen3.8-27B UD-IQ4_XS, SHA-256
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
Requested, observed, and immutable identities are separate in the JSON.

## Findings

| Row | Status | Boundary |
| --- | --- | --- |
| Controlled model-query promotion | measured | Selector, H2D, mapping, and target-graph use passed |
| Organic T3 promotion | failed | A/B/A transport passed, but target graph did not consume the promoted page |
| Answer quality | failed | Expected `AURORA-CEDAR-161` was absent; observed `////////////////` |
| Cached append 64/256 | failed | HTTP 200, but both probes reported `cache_n=0` |
| Full-L allocation | measured | L262144 allocation/startup completed; this is not occupancy |
| Occupied L262144, H16384 | failed | Packed selected-attention allocation failed after durable C4002 |
| Occupied L262144, H8192 | measured | Seven turns; durable C9606 and live snapshot C9733 |
| Occupied C262144 | failed | The required occupied frontier was not reached |

The matched campaign completed nine rows per mode. Median cold-prefill rates
were 116.807 tok/s selected native, 449.099 tok/s CPU-main-KV/GPU-draft, and
1248.228 tok/s all-GPU/GPU-draft. Median committed-decode rates were 13.098,
10.298, and 37.084 tok/s respectively. Native rows had 2,259 drafted and
zero accepted tokens using request-scoped draft-counter deltas; controls are
feature-off and do not provide native-MTP denominators.

The controlled model-query row used logical page 12 with 4,325,376 useful and
aligned H2D bytes. The organic candidate was cold and reached H2D completion
and mapping publication, but `target_graph_used` was false. Allocation,
allocation bytes, a correct transport response, and a published mapping are
not substituted for occupied-context or target-consumption proof.

Raw phase-66 manifests and predecessor receipts are listed in
`V10_SUMMARY_66.json`; no phase-64 rate is imported.
