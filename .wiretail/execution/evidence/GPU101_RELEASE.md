# GPU101 release decision — 101-12l retake

**Decision: goal_miss. Scale remains gated.** The immutable candidate-v6 canonical retake completed all three matched routes. Selected prefill clears both 500 and 750 tok/s targets on all prompts, and selected decode beats CPU-main-KV on all prompts. Prompt 1 selected MTP acceptance is 39.69%, below its 75% floor; prompts 2 and 3 clear their 40% and 60% floors.

## Candidate and protocol

- Candidate: `/srv/ai/paged-kv/results/forward/101-12k/attempt-01/candidate-bundle-prefix-score-v6/llama-server` (SHA-256 `1b01e3f0d568abbb136da95f7c7f6a22d53964852d13583042d81e6ebf0f61a2`), source `e56ee54fb9fea563b70230c2e5c832a3e6bd8ace`.
- Model SHA-256: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`; frozen prefix SHA-256: `8218b0f427cc931fa38d08f1c29f91ec7d82e12ac309ff7743eb7a60da678d20`.
- Geometry: L8192/H4096/P256/B1024/U256; Turbo4 K/V; native GPU Turbo4 MTP, n-max 2; temperature 0, seed 42, reasoning off.
- Each route: one 40-token warmup and three measured requests per prompt, 400-token cap, fresh slot clear for every row. Total: 36 rows, all HTTP 200, no errors.
- Runtime PIDs: cpu_main_kv_gpu_mtp `80557`, pager_off_all_gpu `77307`, selected `54083`.

## Per-prompt medians

| Prompt | Selected prefill tok/s | Selected decode tok/s | Selected MTP | Floor | Pager-off prefill/decode | CPU-main-KV prefill/decode | Selected/CPU prefill | Selected/CPU decode | Selected/pager-off prefill/decode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1518.75 | 54.95 | 39.69% | 75% | 1633.70 / 84.13 | 672.07 / 24.94 | 2.26x | 2.20x | 0.93x / 0.65x |
| 2 | 1530.69 | 59.24 | 49.37% | 40% | 1630.44 / 62.61 | 675.21 / 18.46 | 2.27x | 3.21x | 0.94x / 0.95x |
| 3 | 1517.47 | 68.59 | 65.26% | 60% | 1632.02 / 71.15 | 671.73 / 21.65 | 2.26x | 3.17x | 0.93x / 0.96x |

## Counts, TTFT, and telemetry limits

Selected prompt token counts are 4115, 4112, and 4113 for prompts 1–3, with zero cached input tokens on every measured row. Selected output token counts are recorded in the JSON release evidence; each request used the 400-token generation cap and the endpoint may stop earlier. Canonical TTFT is null, and first-content TTFT was not captured by this streaming runner. MTP percentages are recomputed from native Prometheus accepted/drafted counter deltas.

This campaign did not measure CUDA driver graph capture/replay, CUDA event timings, per-kernel timing, GPU memory peaks, passive utilization, or copy/overlap timing. Pager logical graph counters are not used as CUDA graph proof.

## Gate and next owner

The low prompt-1 acceptance remains unexplained. The previous same-prefix probe recorded same top-1 but different top-8 score ordering, verify-visible selected page 16, and no committed selected-history IDs; backend mask bits were not captured, so this does not establish a dropped-history defect. Task 101-12m owns that narrow prompt-1 attribution. Task 102-01 remains dependent on 101-12m and must not begin while the release is a goal miss.

Raw rows, run configs, lifecycle identity, summaries, and runner logs are hashed in `GPU101_RELEASE.json`. Independent row and identity verification: `/srv/ai/paged-kv/results/forward/101-12l/attempt-03/retake-verification.json`. Pre-retake report bytes are preserved under `.wiretail/execution/evidence/provenance/101-12l/`.
