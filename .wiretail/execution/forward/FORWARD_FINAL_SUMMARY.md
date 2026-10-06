# Forward final measured summary (refreshed 103-04)

**Decision:** primary architecture remains unaccepted. The 262,144-row allocation was configured, while the largest executable occupied frontier was C=250,572 (gap 1,572); exact C=262,144 is unproven.

## Geometry and capacity

| L | C | H | R | G | A | B/U | Result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 131,072 | 120,000 | 51,200 | 50,944* | 256* | 0** | 1024/256 | 200/200 pages; occupied run passed |
| 262,144 | 250,572 | 51,200 | 50,944* | 256* | 0** | 1024/256 | near-full frontier; exact-full unproven |

* R/G are derived from H and the 256-token generation-tail invariant (R+G=H), not separately exported counters. **A is additional packed-owner allocation (measured high-water 0); 54,067,200 B graph scratch is separate.

## Canonical 8K/4K same-candidate rerun (103-04)

Main candidate `build-102-06/bin/llama-server` SHA256 `a24b26ba1b55066d81eb1af795d3d58889099a4b1ed087a4d3e6f74f3fe375de`
(model SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`), L=8192/H=4096, page=256,
B=1024/U=256, native GPU Turbo4 MTP. Three measured trials per prompt used
the same frozen prefix and cleared slot; each row rendered 4,112–4,115 tokens
with zero cached tokens. Rows and identity are under `/srv/ai/paged-kv/results/forward/103-04/attempt-01`.

| Arm | Fresh prefill medians p1/p2/p3 (tok/s) | Decode medians p1/p2/p3 (tok/s) | MTP acceptance p1/p2/p3 | Answer quality |
|---|---:|---:|---:|---|
| Selected pager, GPU target KV | 1503.74 / 1506.10 / 1505.13 | 36.82 / 36.80 / 36.78 | 0 / 0 / 0% | Slash filler on all prompts (semantic miss) |
| CPU-main-KV, pager off, GPU MTP | 671.77 / 678.15 / 671.50 | 25.14 / 19.12 / 21.43 | 80.37 / 50.00 / 65.03% | Coherent useful answers |
| Feature off, all-GPU target KV and MTP | 1633.61 / 1631.29 / 1633.80 | 84.30 / 62.85 / 71.33 | 78.20 / 51.61 / 63.32% | Coherent useful answers |

The selected route reported `selected packed`; route override was `auto`. Fresh
prompts had zero query replay time/count. Selector/retrieval cost was not
separately reported. The selected pager snapshot measured 138,412,032 B target
allocation, 8,781,824 B full-L MTP KV, 14,877,589,504 B device used and
1,842,610,176 B free. CPU and pager-off controls did not expose comparable
pager allocator memory counters. Independent answer scoring was not run.

The CPU-main-KV control must disable pager mode because this candidate rejects
CPU target KV with selective pager enabled (`bounded KV pager requires GPU
target cache storage`). An initial 30-token short probe and that invalid
configuration remain recorded, excluded from canonical rows.

104-08's ranking experiment verdict remains **inconclusive**; no experimental
source or binary is adopted and no ranking win/loss is inferred. The reported
L=262,144 allocation and C=250,572/C=254,393 occupied frontiers remain
candidate-bound partial findings; exact C=L is unproven. 103-03 remains deferred.

## Historical occupancy speed findings

At C=120,000, 128K post-load rows completed three 400-token trials per prompt, but MTP acceptance was 0% on all prompts. Each reused the occupied prefix; about 42 prompt tokens were newly processed, so these are not fresh-prefill comparisons. At 256K, no canonical speed matrix ran. The 249,921-token recall generated 400 tokens with 398 drafts, 0 accepted, and slash filler.

The occupancy-fill journals do contain incremental fresh-input prefill curves,
which were previously omitted from this compact summary. In the 128K run,
server-reported prompt rate declined from 1,117 tok/s at C=3,890 to 668 tok/s
at C=119,357 (multi-thousand-token fresh chunks; short 112/140-token tail
requests excluded). In the separate 256K build/run it declined from 1,131
tok/s at C=3,890 to 585 tok/s at C=250,037 (the final point processed 7,754
fresh tokens). These are occupancy-fill measurements, not the three-prompt
canonical benchmark, not paired A/B, and from different candidate binaries;
they must not be attributed to a specific code change. Complete point tables,
fresh-token counts, candidate hashes, and raw journal SHA256 values are recorded
in the 102-05/102-06 handoffs and V10 receipts.

## Retrieval, replay, memory and quality

A prior 101-18 candidate passed natural cold promotion (page 5, 34,603,008 useful H2D bytes, mapping published, target/draft consumed). Final 256K replay restored target and draft, kept 200/200 pages resident and frozen-history generation at 0, advanced slot generation 36→37, and observed zero historical H2D in that interval. It failed semantic file retrieval. Selector/replay/promotion/view timings, final 256K route comparison, generation rollback count and exact full-C commit are not measured.

At 128K: target pool 865,075,200 B; target compute 193,974,912 B; full-L GPU Turbo4 MTP 138,543,104 B (131,072 rows); MTP compute 102,768,768 B; MTP dequant scratch 536,870,912 B; target dequant and packed live/peak/draining high-water 0 B; minimum sampled free VRAM 232 MiB over 110 samples. Host-valid/pinned bytes and separate graph scratch/headroom are unknown in this receipt, not zero.

At 256K: target pool 865,075,200 B; target compute 244,929,152 B; full-L GPU Turbo4 MTP 276,955,136 B (262,144 rows); MTP compute 169,877,632 B; graph scratch 54,067,200 B; target/MTP dequant high-water 0 B; packed live/draining high-water 0 B; host pageable 4,233,660,000 B, host valid 816,617,000 B, pinned 0 B; minimum sampled free VRAM 502 MiB. Headroom (201,326,592 B), catalogue (100,876,288 B), graph (198,435,072 B), and external (771,728,128 B) are reservations, not observed allocations.

## Goal verdict

All 15 items are listed in `FORWARD_FINAL_SUMMARY.json`. Core misses are occupied-context MTP acceptance, final semantic retrieval, exact-full C, final route/scale attribution and feature-off reassessment. The 8K selected-prefill floor and CPU-main-KV comparison pass. Earlier replay, freeze, promotion and ring proofs remain scoped to their executable evidence. The full canonical three-prompt benchmark at each context and a paired before/after context-speed comparison remain unmeasured; the incremental occupancy-fill curves above are findings, not acceptance gates. Receipt-backed raw roots and hashes are listed in the JSON.
