# Phase-79 benchmark summary

Revision: `hotpath-v10-20260914`. This summary aggregates the phase-79
receipts and raw manifests. Requested, observed, and immutable identities are
kept separate.

## Result

Phase 79 proved both controlled and organic physical promotion through the
normal selector/transfer/target-graph chain, and the organic round trip got
all three expected nonce answers correct. Full-L allocation and packed forward
progress reached durable C33952/live C33967, but occupied C262144 was not
established. The fresh C6143 benchmark rerun timed out; retained short-prefix
controls remain labeled C27–C30 and are not relabeled as C6144.

| Requested row | Status | Finding |
| --- | --- | --- |
| Controlled physical promotion | measured | Cold logical page 0 was queued, completed, published, and consumed by the target graph; 250871808 useful H2D bytes |
| Organic physical promotion | measured | A became cold before A-again; page 0 was promoted and used by the target graph |
| Target-graph use | measured | `target_graph_used=true` in both natural proofs |
| Answer quality | measured | A, B, and A-again nonce answers: 3/3 correct |
| Cached append 64 | measured | C6207, cache_n6143, 64 new tokens; 75.334 tok/s |
| Cached append 256 | measured | C6399, cache_n6146, 253 tokenizer-realized new tokens; 90.523 tok/s |
| Cold prefill | failed | Fresh C6143 prefill exceeded the 300-second bound; no rate fabricated |
| Committed decode | not_run | Verified-coordinate prefill timed out before decode; retained short-prefix controls are separate |
| Native MTP denominator | measured | Request-scoped deltas retained for matched rows and packed occupancy routes |
| Full-L allocation | measured | L262144 admitted with 138412032 target and 276955136 MTP bytes |
| Occupied C262144 | failed | Operator wall bound stopped at durable C33952/live C33967 |

## Identity, geometry, placement, and bytes

The requested coordinate was L8192/C6144/H4096/A2048/B128/U64 with 256-token
pages. The observed verified coordinate was L8192/H8192/A4096/B128/U64 with
C6143, C6207, and C6399 frontiers. The resolved model was
`Qwen3.8-27B-UD-IQ4_XS.gguf`, SHA-256
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
Selected-native and full-L used CUDA Turbo4/Turbo4 target KV and GPU Turbo4/
Turbo4 native draft KV. The 256-token page occupied 4325376 bytes.

Full-L allocation measured target 138412032, MTP 276955136, packed storage
4325376 at startup and 69206016 at final, packed workspace 138412032, packed
dequant 16777216, charged 15965452416, reserved 2068443264, and headroom
201326592 bytes. The occupancy route recorded 546 prefill-packed, 7
decode-packed, and 98 MTP-verify-packed routes, with 651 accepted route
overrides and 38928384 useful/aligned H2D bytes.

## Promotion and answer quality

The controlled model-query proof used 6144 tokens, 32 logical pages, a
16-page physical pool, 15 final host pages, 250871808 useful H2D bytes, and
5568 H2D submissions/completions. Its cold page 0 proof had selector rank 0,
host-ready, published mapping, completed H2D, and target-graph use.

The organic two-document run used A=1332 tokens and B=3954 tokens with
disjoint nonces. A/B/A-again were observed at C1370/C5366/C5414; page 0 was
cold and host-backed before A-again, then selected and consumed after
promotion. The canonical transport used no client page ID, no force-promotion
API, and no context rebuild. Expected values `ASTRAL-PINE-482`,
`COBALT-MAPLE-731`, and `ASTRAL-PINE-482` were all returned correctly.

## Rates, cache reuse, and MTP

Cached append reused 6143 tokens for append-64 and 6146 for append-256. The
measured rates were 75.334 and 90.523 tok/s respectively. The retained
matched matrix has 27 rows, three measured trials per prompt, and short-prefix
C27–C30 controls: selected-native prefill medians 101.965/108.716/104.499
tok/s and decode medians 32.784/33.046/32.486; CPU-main-KV/GPU-draft
prefill/decode medians 36.222/10.060, 36.719/10.063, 36.063/10.006; and
all-GPU/GPU-draft medians 103.864/36.776, 110.229/36.782, 109.999/36.777.

Native MTP used request-scoped Prometheus deltas. Retained selected-native
rows totaled 5337 drafted and 9 accepted tokens (0.1686%); feature-off
controls totaled 7155 drafted and 0 accepted and are diagnostic controls, not
native-MTP capability evidence. The fresh verified-coordinate run timed out
before it could supply a new MTP denominator.

## Occupancy boundary and raw evidence

Seven cache-preserving packed requests passed. Durable C stopped at 33952 and
live C was 33967, so the full occupied C262144 claim remains failed despite
successful full-L allocation. The JSON summary lists every receipt and raw
manifest used, including the explicitly retained phase-78 matrix and cached
coordinate artifacts referenced by the phase-79 validator.
