# Ranking104 experiment decision

**Decision: inconclusive.** The same binary and model were used in the selected paired campaigns, but the legacy failures and probe finalization/degenerate outcomes leave no complete paired quality result. The evidence does not support adoption or rejection of the ranking algorithm.

## Outcome by independent target

| Target | Legacy | Probe-rerank | Quality conclusion |
|---|---|---|---|
| PY_MERGE_03 | HTTP 0 before natural recall; tensor-buffer assertion | HTTP 500 at query replay finalization | shortlist and answer outcome unknown |
| PY_MERGE_01 | HTTP 0 before natural recall; tensor-buffer assertion | HTTP 500 at query replay finalization | shortlist and answer outcome unknown |
| BASH_WATCH_01 | HTTP 0 before natural recall; tensor-buffer assertion | HTTP 200, 400-token degenerate reply | no usable paired answer comparison |

Dense controls were diagnostic mismatches for both Python targets and transport-incomplete for BASH. The [paired recall report](/srv/ai/paged-kv/results/ranking104/104-06d/attempt-01/RANKING104_RECALL.md) records these final selected outcomes. HTTP 200 is not treated as answer quality.

## Quality and cost findings

- **Coarse shortlist:** unknown end-to-end; legacy did not reach usable query ranking and probe did not complete paired natural outcomes.
- **Exact-key mass:** the encoded Turbo4 numerical oracle passed CPU/CUDA at 64, 256, and 1024 candidates, including resident/cold candidates. This validates the mass calculation, not natural shortlist recall.
- **Promotion budget:** unknown because no completed paired query exposes a budget omission.
- **Publication:** probe Python targets failed query replay finalization with HTTP 500. This is an execution/publication failure, not a measured ranking miss.
- **Answer quality:** the probe BASH result was degenerate; no useful recall improvement is demonstrated. Fresh same-input MTP sanity controls in 104-06b returned coherent code/prose, bounding the fresh-generation anomaly.
- **Speed and memory:** per-prompt pp/tg ratios, retrieval/replay time, paired GPU/pinned/pageable memory, ranker scratch, transfer bytes, and a paired MTP regression comparison are unknown. 104-07's exact L131072/H51200 legacy request hit the backend tensor-buffer assertion before a comparison; probe-rerank was not attempted for that row.

The 104-07 machine result is [RANKING104_RESULTS.json](RANKING104_RESULTS.json), with the [human summary](RANKING104_RESULTS.md). Its outcome remains inconclusive; no scores or missing counters are represented as zero.

## Decision and next diagnostic

Adoption criteria remain unmeasured: useful recall improvement across targets, the paired throughput limit, absence of recurring MTP regression, and the bounded byte ledger. The single missing fact is why the legacy query-accumulation tensor has no backend buffer in `llama_kv_cache_context::set_kv_query_accumulate_inputs`. The next technical action is a deterministic L8192/B1024/U256 regression/diagnostic for that exact tensor, repair its missing buffer assignment, and verify one canonical legacy request. This decision does not call for another full corpus or occupancy campaign.

No experiment code was merged or cherry-picked and main default routing was not changed. The experiment branch remains at `83e5cbfdaa76e57ccc7716131e0f1b2a55202e20` with tracked diff SHA256 `a6c68dab97b3f543a5ec629c3f2cf8d2d070f068f7298844d1238df67c62124b` and combined dirty snapshot SHA256 `ee22810f9da38722a01ccc70d48e77cba903d43214ef9106d90dbb2fc2f23bd1`. The current main coordinator HEAD is `b73b8f1a05ba7a8a2b200af42e43126a8594c4e1`.

## Kept managed service

The active service passed health and identity checks. It runs the experiment candidate binary with probe-rerank, L8192/H4096, B1024/U256, hot-pages 16 (4096 hot tokens), Turbo4 target/draft GPU MTP. See the hashed service snapshot in the decision JSON.

## Hashed evidence

The decision JSON lists SHA256 references for the 104-06d recall report, 104-07 result/receipt/raw diagnostic, experiment source snapshot, and current service identity.
