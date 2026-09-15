# Phase-77 benchmark summary

Revision: `hotpath-v10-20260914`. This summary aggregates only the phase-77
receipts and raw manifests. Requested, observed, and immutable identities are
kept separate.

## Result

Phase 77 repaired the selected route and native-MTP cached-reuse boundary,
repaired the organic query boundary, measured a clean A/B/A quality result,
advanced the occupied frontier, and completed the matched 27-row benchmark.
Full occupied C262144 remains unestablished.

| Requested row | Status | Finding |
| --- | --- | --- |
| Answer quality | measured | A, B, and A-again all returned their exact nonces |
| Controlled physical promotion | not_run | The route/cache repair coordinate did not execute a deliberate controlled physical-promotion scenario |
| Organic physical promotion | measured | Cold logical page 0 was queued, completed, published, and used by the A-again target graph |
| Target-graph use | measured | Published logical page 0 was consumed after H2D completion |
| Cached append 64 | measured | Repair coordinate and retained predecessor speed capture used cache_n 6143; observed repair row also measured |
| Cached append 256 | measured | Repair coordinate used cache_n 6146; retained predecessor speed capture used cache_n 6143 |
| Cold prefill | measured | 9 rows per campaign in the 27-row matched matrix |
| Committed decode | measured | 9 rows per campaign in the 27-row matched matrix |
| Native MTP denominator | measured | Selected native: 1 accepted / 751 drafted; request-scoped rows retained |
| Full-L allocation | measured | L262144 admitted with 138412032 target and 276955136 draft bytes |
| Occupied C262144 | failed | Wall-bounded frontier stopped at durable C32762/live C32773 |
| Cached-append rates | not_run | No new phase-wide cached-append rate campaign was run |

## Identity, geometry, placement, and bytes

The requested coordinate was L8192/C6144/H4096/A2048/B128/U64 with 256-token
pages. Observed quality prompt C values were 2034, 4080, and 4126; the matched
benchmark observed C30 with L8192/H8192/A2048/B128/U64. The occupancy pilot
used L262144/H8192/A4096/B128/U64. The resolved model was
`Qwen3.8-27B-UD-IQ4_XS.gguf`, SHA-256
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.

Target K/V was CUDA Turbo4/Turbo4; native MTP K/V was GPU Turbo4/Turbo4.
The full-L allocation measured target 138412032 bytes, draft 276955136 bytes,
packed storage 0 bytes, packed workspace 138412032 bytes, charged
15965452416 bytes, reserved 2068443264 bytes, and headroom 201326592 bytes.
Each 256-token page occupied 4325376 bytes.

## Quality, promotion, and MTP

The canonical A/B/A rows returned `ASTRAL-PINE-482`, `COBALT-MAPLE-731`, and
`ASTRAL-PINE-482`; occupied slot counts were 2042, 4090, and 4134. This answer
claim is separate from paging. For the A-again request, logical page 0 was
cold, host-ready, queued and completed through H2D, published at physical slot
15, and selected in the completed target graph. Useful and aligned H2D bytes
were both 4325376.

The selected-native matched rows measured 751 drafted and 1 accepted token
(1/249 on prompt 0; 0/251 on prompts 1 and 2). Feature-off controls measured
1506 drafted and 0 accepted and are not native-MTP capability evidence. The
cached-append repair coordinates measured 3 drafted and 0 accepted on each
append row.

## Rates and boundaries

The matched matrix used three trials for each of three prompts in each campaign.
Selected-native cold-prefill medians were 102.585, 106.792, and 109.505 tok/s;
committed-decode medians were 33.544, 33.258, and 33.257 tok/s. CPU-main-KV /
GPU-draft medians were 35.842, 35.956, and 36.385 tok/s for prefill and 9.925,
9.936, and 10.024 tok/s for decode. All-GPU/GPU-draft medians were 103.437,
110.296, and 110.442 tok/s for prefill and 36.737, 36.874, and 36.869 tok/s
for decode.

The L262144 allocation and seven successful forward requests are measured, but
the requested occupied-C262144 claim failed at the explicit operator wall
boundary after durable C32762/live C32773. No result relabels allocation as
occupancy, or correctness as physical promotion. All raw roots and receipts
are listed in `V10_SUMMARY_77.json`.
