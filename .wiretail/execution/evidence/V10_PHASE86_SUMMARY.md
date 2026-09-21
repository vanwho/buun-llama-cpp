# Phase86 compact summary

- Result: **current_findings**
- Scope: `phase86 raw records and manifests only`
- Geometry: L=8192, H=8 pages, page=256, B=128, U=64
- Candidate: `/srv/ai/paged-kv/candidates/86-02-cuda`; binary `61fb0f5feb24fa74661e00fcb11a611cba593a618e1a6cd231c9227c96505643`
- Model: `/srv/ai/models/text/current.gguf`; SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`

## Measured short rows

| rate | value | samples | reason |
|---|---:|---:|---|
| fresh_pp | None | 0 | canonical short rows used prompt cache |
| cached_new_token_pp | 103.15127133941927 | 9 | None |
| committed_tg | 34.91117046625808 | 9 | None |

- Native MTP sum: attempted `555`, accepted `3`, acceptance `0.005405405405405406`.
- Fresh prompt rate and TTFT remain null because these rows used the canonical prompt cache and TTFT was not exported.

## Promotion and capacity

- Natural promotion: null; no rank→copy→publication→target-use chain was measured.
- Occupancy frontier: committed C `17568`, live snapshot C `21282`; full 256K occupancy is `False`.
- 256K allocation/startup and occupied context are reported separately; 32K/128K requests did not complete within their bounds.

Raw paths and checksums are in the JSON `raw_pointers` object.
