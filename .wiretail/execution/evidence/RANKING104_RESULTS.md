# Ranking104 paired speed, memory, and scale findings

**Outcome: inconclusive.** The final candidate is identical across the recorded legacy and probe-rerank modes. The legacy baseline hit a fatal backend assertion before a complete canonical speed row set or occupied-context recall result could be collected. There are no fabricated performance or ranking scores.

## Configuration and observed result

- Candidate: experiment `83e5cbfdaa76e57ccc7716131e0f1b2a55202e20`, base `699eaf1f5dabc3c6ffbb89670622b25d48c8d8ff`; one binary and DSO set, identified in the JSON receipt. Model SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
- Speed setup was L8192/H4096/B1024/U256 with reasoning off and Turbo4 target/draft MTP on GPU. The runner had zero complete paired prompt rows; no ratios, TTFT, MTP acceptance, GPU delta, or memory delta are reported.
- The bounded occupancy setup was L131072/H51200/B1024/U256. The model loaded and reserved 131072 MTP rows (138543104 bytes) on CUDA0. The frozen source schedule approached C59392 while remaining below L and above H; its first actual legacy request failed at frontier 3630. No recall answer or cross-mode memory comparison exists.
- Both failures show `GGML_ASSERT(buf != NULL && "tensor buffer not set")` in `ggml_backend_tensor_set`, via `llama_kv_cache_context::set_kv_query_accumulate_inputs`. The loaded high-context model and recorded GPU state make this a fatal runtime failure, not an observed VRAM admission failure or ranking miss. Probe-rerank was not attempted after the fatal legacy peer failure; it was restored and verified at L8192 with zero generation requests.

## Router boundary

Probe-rerank dispatches PAGE_RANK in `llama-kv-cache.cpp`; `llama_kv_router_job::execute_query` creates PAGE_RERANK and PAGE_MASS work and reaches the final-user query publication boundary. Legacy dispatches PAGE_SELECT and does not execute that probe ranking/job publication path. These are code-path findings; no query timing was reached or measured.

## Raw evidence and validation

All raw files and the candidate build receipt are SHA256-indexed in `RANKING104_RESULTS.json`. The machine-readable record carries the exact failure request, both runtime assertion logs, frozen schedule/preflight evidence, build output, and the restored probe-rerank identity evidence.
