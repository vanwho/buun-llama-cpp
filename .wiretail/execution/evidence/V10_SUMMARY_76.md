# Phase-76 benchmark summary

Revision: `hotpath-v10-20260914`. This summary aggregates only the phase-76
receipts and their raw manifests. Requested, observed, and immutable identities
remain separate.

## Result

Phase 76 repaired and measured cached append reuse, reran answer quality, advanced
the L262144 occupancy frontier, and ran matched controls. The selected/native
matched benchmark still failed at the selected packed route, so its rates are
null rather than inferred.

| Requested row | Status | Finding |
| --- | --- | --- |
| Answer quality | measured / failed row | A and A-again correct; B returned `COBALT-ELM-117` instead of `COBALT-MAPLE-731` |
| Cached append 64 | measured | Non-MTP reused C6143 and processed 64 new tokens; native-MTP row did not reuse the prefix |
| Cached append 256 | measured | Non-MTP reused C6143 and processed 256 new tokens; native-MTP row did not reuse the prefix |
| Cold prefill | not_run | No separate phase-76 cold-prefill campaign was requested after the matched capture |
| Committed decode | not_run | No separate phase-wide decode campaign; selected/native matched rows failed before decode |
| Native MTP denominator | measured | Cached-append rows: 4 drafted / 0 accepted across four request-scoped deltas |
| Physical promotion | measured | Cold logical page 0 was queued, completed, published, and used by the A-again target graph |
| Target-graph use | measured | Completed target graph recorded the published mapping |
| L262144 allocation | measured | Target allocation 138,412,032 bytes; native-MTP allocation 276,955,136 bytes |
| Occupied C262144 | failed | Bounded frontier stopped at durable C13034/live C13049 at the operator wall-budget boundary |

## Geometry, placement, and rates

The cached-append capture observed L8192/H8192/A2048/B128/U64 with 256-token
pages and 4,325,376-byte pages. The occupancy pilot observed
L262144/H8192/A4096/B128/U64. It measured 69,206,016 bytes packed storage,
138,412,032 bytes packed workspace, 15,965,452,416 bytes charged,
2,068,443,264 reserved bytes,
201,326,592 bytes headroom, and 449,839,104 useful/aligned H2D bytes.

Controls completed the matched three-prompt benchmark with three measured trials
per prompt. CPU-main-KV/GPU-draft prompt medians were 312.324/285.981/289.771
tok/s and decode medians 17.898/17.910/17.878 tok/s. All-GPU/GPU-draft prompt
medians were 682.424/651.246/673.692 tok/s and decode medians
49.364/49.341/49.357 tok/s. Selected/native rows failed with HTTP 500 from the
selected packed route and have no rate claim.

## Boundaries and provenance

The resolved model was `Qwen3.8-27B-UD-IQ4_XS.gguf`, SHA-256
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
Physical promotion and answer correctness are separate: the B mismatch does not
erase target-graph use, and a correct A-again answer alone is not the promotion
proof. Full-L allocation is not occupied-C262144 evidence. All raw roots and
receipt paths are listed in `V10_SUMMARY_76.json`.
