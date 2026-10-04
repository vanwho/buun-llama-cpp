# Forward final measured summary (102-07)

**Decision:** primary architecture remains unaccepted. The 262,144-row allocation was configured, while the largest executable occupied frontier was C=250,572 (gap 1,572); exact C=262,144 is unproven.

## Geometry and capacity

| L | C | H | R | G | A | B/U | Result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 131,072 | 120,000 | 51,200 | 50,944* | 256* | 0** | 1024/256 | 200/200 pages; occupied run passed |
| 262,144 | 250,572 | 51,200 | 50,944* | 256* | 0** | 1024/256 | near-full frontier; exact-full unproven |

* R/G are derived from H and the 256-token generation-tail invariant (R+G=H), not separately exported counters. **A is additional packed-owner allocation (measured high-water 0); 54,067,200 B graph scratch is separate.

## Canonical speeds and MTP

The 8K/4K selected run used three fresh-input trials per prompt. Selected prefill medians were 1518.75, 1530.69, and 1517.47 tok/s (prompts 1–3); decode medians were 54.95, 59.24, and 68.59 tok/s. MTP medians were 39.69%, 49.37%, and 65.26%; prompt 1 missed the 40% floor by 0.31 points. CPU-main-KV prefill medians were 672.07, 675.21, and 671.73 tok/s; selected prefill was 2.26x CPU on all prompts. Dense GPU comparison is in the JSON. Selected, CPU-main-KV, and dense GPU controls all used Turbo4 target K/V and GPU Turbo4 MTP; target placement distinguishes the controls.

At C=120,000, 128K post-load rows completed three 400-token trials per prompt, but MTP acceptance was 0% on all prompts. Each reused the occupied prefix; about 42 prompt tokens were newly processed, so these are not fresh-prefill comparisons. At 256K, no canonical speed matrix ran. The 249,921-token recall generated 400 tokens with 398 drafts, 0 accepted, and slash filler.

## Retrieval, replay, memory and quality

A prior 101-18 candidate passed natural cold promotion (page 5, 34,603,008 useful H2D bytes, mapping published, target/draft consumed). Final 256K replay restored target and draft, kept 200/200 pages resident and frozen-history generation at 0, advanced slot generation 36→37, and observed zero historical H2D in that interval. It failed semantic file retrieval. Selector/replay/promotion/view timings, final 256K route comparison, generation rollback count and exact full-C commit are not measured.

At 128K: target pool 865,075,200 B; target compute 193,974,912 B; full-L GPU Turbo4 MTP 138,543,104 B (131,072 rows); MTP compute 102,768,768 B; MTP dequant scratch 536,870,912 B; target dequant and packed live/peak/draining high-water 0 B; minimum sampled free VRAM 232 MiB over 110 samples. Host-valid/pinned bytes and separate graph scratch/headroom are unknown in this receipt, not zero.

At 256K: target pool 865,075,200 B; target compute 244,929,152 B; full-L GPU Turbo4 MTP 276,955,136 B (262,144 rows); MTP compute 169,877,632 B; graph scratch 54,067,200 B; target/MTP dequant high-water 0 B; packed live/draining high-water 0 B; host pageable 4,233,660,000 B, host valid 816,617,000 B, pinned 0 B; minimum sampled free VRAM 502 MiB. Headroom (201,326,592 B), catalogue (100,876,288 B), graph (198,435,072 B), and external (771,728,128 B) are reservations, not observed allocations.

## Goal verdict

All 15 items are listed in `FORWARD_FINAL_SUMMARY.json`. Core misses are occupied-context MTP acceptance, final semantic retrieval, exact-full C, final route/scale attribution and feature-off reassessment. The 8K selected-prefill floor and CPU-main-KV comparison pass. Earlier replay, freeze, promotion and ring proofs remain scoped to their executable evidence. The final context curve is deferred until the architecture is accepted. Receipt-backed raw roots and hashes are listed in the JSON.
