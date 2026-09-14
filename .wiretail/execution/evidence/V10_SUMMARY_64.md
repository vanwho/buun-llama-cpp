# Phase-64 benchmark summary

Result: phase-64 produced measured T0 promotion, a measured organic promotion
edge, matched cold-prefill/decode rates, cached append probes, scale pilots,
and full-L allocation. The controlled model-query seam, complete organic A/B/A
round trip, useful nonce answer, and occupied C262144 remain failed or unproven.

The selected small path is L8192/C6144/H4096/A2048/B128/U64 with 256-token
pages. Observed target and native-MTP K/V are Turbo4 on GPU. The resolved model
is Qwen3.8-27B UD-IQ4_XS, SHA-256
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
Requested, observed, and immutable identities are separate in the JSON.

## Findings

| Row | Status | Boundary |
| --- | --- | --- |
| T0 production promotion chain | measured | Pass; mature-FA parity and real transfer chain |
| Controlled model query | failed | Two attempts segfaulted before proof JSON |
| Organic promotion | measured/failed round trip | Cold page reached target-graph use; first A response failed HTTP 500 |
| Answer quality | failed | Expected nonce absent; response was `////////////` |
| L32768 pilot | failed | Prefill deadline at observed C14336 frontier |
| L128K pilot | measured | Bounded C1024 forward progress only |
| Full L262144 allocation | measured | Allocation/startup plus bounded C1200 request |
| Occupied C262144 | failed | Only C1200 was observed; allocation is not occupancy |

The matched campaign measured 27 rows at L8192/C6144/H4096/A2048/B128/U64.
Selected native committed decode median was 13.1402 tok/s; CPU-main-KV and
all-GPU feature-off controls were 10.0811 and 36.7867 tok/s. Cached append
probes observed cache_n 6271: requested 64 gave 90.9554 prompt and 21.4492
committed-decode tok/s with 6/2 draft/accepted; requested 256 gave 89.8575
prompt and 42.8517 committed-decode tok/s with 88/82 draft/accepted.

No phase-63 rates are imported. Allocation, a correct answer, or a published
table is not treated as physical promotion or occupied-context proof.
