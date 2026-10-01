# GPU101 release decision — 101-18

Decision: **goal_miss**. Candidate source `a0086f056f20efa519100718f010bc15c2efb830`; binary SHA256 `300dc8cf420abbd0b9121ffafd9ed90e4caf299a797280186ac84426a10ade54`.

## Candidate and protocol

- Model SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`; runtime PIDs: selected 2037909, pager-off 2061092, CPU-main-KV 2064564; retained selected candidate PID after the TTFT probe 2077995.
- L8192/H4096/P256/B1024/U256; target and draft Turbo4; native GPU MTP nmax2; prefix SHA256 `8218b0f427cc931fa38d08f1c29f91ec7d82e12ac309ff7743eb7a60da678d20`.
- Canonical campaign: one 40-token warmup then three 400-token requests per prompt, temperature 0, reasoning off, fresh slot each row. All three routes completed with zero errors.

## Selected and matched controls

| Prompt | Selected prefill tok/s | Selected decode tok/s | Selected MTP median | CPU prefill tok/s | CPU decode tok/s | Pager-off prefill tok/s | Pager-off decode tok/s | Supplemental TTFT ms | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| prompt_1 | 1452.08 | 60.05 | 48.56% | 660.72 | 25.50 | 1556.16 | 84.24 | 3134.88 | measured_goal_miss |

| prompt_2 | 1463.84 | 58.45 | 47.58% | 664.31 | 18.54 | 1556.18 | 62.89 | 298.77 | pass |

| prompt_3 | 1451.19 | 65.41 | 59.79% | 660.74 | 21.09 | 1557.64 | 71.25 | 298.43 | measured_goal_miss |

Raw drafted/accepted counts by selected prompt:
- prompt_1: drafted `[313, 326, 307]`, accepted `[152, 128, 157]`.
- prompt_2: drafted `[106, 128, 124]`, accepted `[59, 58, 59]`.
- prompt_3: drafted `[301, 291, 282]`, accepted `[158, 174, 207]`.

Selected prefill exceeds the 500 tok/s minimum on all prompts but does not reach the 750 tok/s preference. MTP misses prompt 1 at 48.56% and prompt 3 at 59.79%; prompt 2 passes at 47.58%. Selected decode beats CPU-main-KV by prompt_1: 2.36x, prompt_2: 3.15x, prompt_3: 3.10x. Selected prefill is prompt_1: 2.20x CPU, prompt_2: 2.20x CPU, prompt_3: 2.20x CPU, below 3x and 5x; it is prompt_1: 0.93x pager-off GPU, prompt_2: 0.94x pager-off GPU, prompt_3: 0.93x pager-off GPU.

The CPU control is ordinary CPU-main-KV via `--no-kv-offload`; its Turbo4 GPU MTP remains enabled. Supplemental TTFT uses one 32-token request per prompt after the canonical rows; it is not the canonical three-measurement median.

## Promotion, route, and runtime diagnostics

- Natural old-file recall: Python03 logical page 5/generation 7/content version 256 was cold and host-backed, naturally nominated at rank 4/5, admitted, H2D-completed (34,603,008 useful/aligned bytes), mapped, and consumed by target and draft. Query replay and frozen-history Turbo4 MTP passed (75 accepted / 128 drafted).
- Selected sparse post-request snapshot: 16/16 physical resident capacity, 15 host-backed pages, 32 logical pages; page inventory had 16 resident and 17 host-backed pages. Host pageable bytes 73531392; pinned 0; headroom 201326592 bytes.
- Summary telemetry snapshot: 2112 builds / 71368704 payload bytes; 0 reads / 0 bytes. Sparse cumulative waits 250 / 7815207 us; copy 0 us; queue 1532278 us; overlap 0 us; transfer backpressure 0.
- Pager logical graph counters in the same snapshot: captures 153, replays 97, rebuilds 153. Driver CUDA events were not enabled in the timing receipt. The model-backed replay log separately contains 12 actual CUDA graph reuse log lines; these are not the pager counters.
- Treat these passive snapshot counters as cumulative diagnostics, not per-prompt attribution.

## Raw evidence and preserved history

Current raw roots are under `/srv/ai/paged-kv/results/forward/101-18/attempt-01/`; JSON contains hashed references for each route config, summary, records and runner log. Old frozen release bytes and their previous raw-run roots are preserved at `.wiretail/execution/evidence/provenance/101-18/` and referenced from the current JSON. Final review successor on this measured miss: **101-12h**.
