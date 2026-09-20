# Phase-85 targeted revalidation summary

Revision: `hotpath-v10-20260914`. Source: task 85-05 focused revalidation at
`L8192/C6143/H4096/A4096/B128/U128`, with 256-token pages. This is a bounded
diagnostic summary, not an overall capability or performance acceptance.

## Identity

The immutable candidate source was `d0472faa0e837c00a7e10983dcb32c2e4e146ba5`
and the endpoint-owned binary was
`/srv/ai/paged-kv/results/v10/85-05/candidate-bundle-wdJFg7/bin/llama-server`
with SHA-256
`83611b093c8031baaa3faefab89e75c93ae38765ab970cc5073e7af0b2063f89`.
The model was Qwen3.8-27B UD-IQ4_XS at
`/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf`, SHA-256
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
Target and native draft K/V were Turbo4/Turbo4; the native draft had 8192 rows
and 8,781,824 bytes on GPU/CUDA0.

## Gate results

| Gate | Result | Evidence |
| --- | --- | --- |
| All-GPU native-MTP positive denominator | Pass for denominator only | 507 drafted, 0 accepted, 0.0%; this is not a positive-acceptance claim. |
| Automatic multi-page direct route and completed target use | Failed | Selected telemetry saw prefill dense/reference/direct/packed `0/49/1/0`; no completed selected target graph use was established. |
| Removal of selected prefill host-bound maintenance | Failed | 46 seals, 47 pages scanned, 23 changed, 2208 host-seal D2H calls and 99,483,648 bytes remained. |

The selected request reached all 6143 prompt tokens and emitted four tokens, then
stalled. It ended with 52 graph submissions versus 51 completions. H2D useful
and aligned bytes were both 233,570,304; faults and evictions were both 54.
Table epoch changes were 51, and catalogue/published epochs were 8421/8551.
These counters show activity and publication, not completed target consumption.

## Separate claims

Physical promotion is **not established**: H2D completion and table publication
were observed, but a completed target graph consuming the promoted mapping was
not. The MTP first-divergence receipt retains `first_divergent_token=null`;
the selected diagnostic reported target/draft state not restored before
verification. The all-GPU and CPU-main-KV controls both measured 507 drafts and
0 accepts.

Answer quality is **not run** as a selected-paging oracle. The feature-off
cache control completed its matched 6143-token seed and 64-token continuation
(6139 cached rows of 6143), but that is not a paging-quality result. Speed
ratios are **null/not run**. GPU utilization sampling was **not run** because
the canonical runner did not include a concurrent sampler. A healthy restored
process, startup placement, or correct answer would not elevate any of these
rows into a capability claim.

## Stop gates for later work

- Explain and rerun native-MTP acceptance with positive request-scoped
  denominators and accepted-token behavior before claiming native acceptance.
- Keep automatic multi-page direct/packed promotion disabled until a completed
  target graph consumes the mapping past the selected stall/illegal-memory
  boundary.
- Reduce and independently measure selected prefill host-seal D2H maintenance
  before claiming its removal.
- Do not run broad speed ratios, occupancy scaling, or full-L occupancy claims
  while the selected target-use and native-MTP gates remain unresolved.
- Keep physical promotion, answer quality, and speed as separate claims.

Raw artifacts remain under `/srv/ai/paged-kv/results/v10/85-05/`; the structured
summary is `V10_SUMMARY_85.json`.
