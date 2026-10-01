# GPU101 canonical release decision — 101-12b

**Decision: `goal_miss`. Scale remains gated.** The corrected H4096 candidate completed the selected suite and both matched controls, with no request errors. Every selected prefill median missed 500 tok/s; selected MTP missed prompts 1 and 3. Selected decode beat CPU-KV on all prompts.

Candidate: source tree `8914d9e1b2dc86637e1d4d07770ffc5fa0dd988a`, runtime binary SHA-256 `a464e9099821b424fe8e637b4bc325c38d6d2a7453ec22832310b67f4fba1fba`, model SHA-256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`. Geometry L8192/H4096/P256/B1024/U256; frozen prefix SHA-256 `8218b0f427cc931fa38d08f1c29f91ec7d82e12ac309ff7743eb7a60da678d20`; selected mode automatic selective pager; target/draft GPU Turbo4 MTP nmax2.

## Per-prompt medians

| Prompt | Selected prefill tok/s | Selected decode tok/s | Selected MTP accepted/drafted median % | CPU-KV decode tok/s | Dense-GPU decode tok/s | Selected/CPU decode | Selected/dense decode | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| prompt_1 | 332.39 | 37.00 | 56.15 | 25.59 | 84.26 | 1.45x | 0.44x | miss |
| prompt_2 | 388.48 | 33.46 | 45.00 | 18.62 | 62.85 | 1.80x | 0.53x | miss |
| prompt_3 | 332.17 | 34.91 | 51.00 | 21.41 | 71.37 | 1.63x | 0.49x | miss |

Each selected prompt has three 400-token measured rows after one 40-token warmup, fresh slot erase before every row, 0 cached input tokens, and exact input counts 4115/4112/4113. Raw MTP drafted and accepted arrays are in `GPU101_RELEASE.json`.

## Matched controls

Pager-off all-GPU medians: prefill 1568.90, 1566.08, 1569.55 tok/s; decode 84.26, 62.85, 71.37 tok/s.
Ordinary CPU-main-KV/GPU-MTP medians: prefill 659.02, 664.28, 653.40 tok/s; decode 25.59, 18.62, 21.41 tok/s.

Both controls completed 12 rows with zero errors and share the selected candidate, frozen prefix, L/B/U and output protocol. All run configs, summaries and complete rows are SHA-256 referenced in the JSON receipt.

## Successors and limits

Runnable tasks 101-12c/101-12d/101-12e now own one selected-path repair, the canonical retest and repeated final release review. 102-01 depends on 101-12e and cannot proceed until `goal_status=pass`.



| Prompt | Supplementary TTFT s | Direct prefill subroutes | Packed prefill subroutes | Packed MTP verify subroutes | Wait us | Queue us | Graph replays | H2D useful bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prompt_1 | 10.597 | 15 | 5 | 163 | 6896840 | 1474020 | 70 | 0 |
| prompt_2 | 12.443 | 15 | 5 | 57 | 3622570 | 706950 | 36 | 0 |
| prompt_3 | 10.588 | 15 | 5 | 144 | 6027000 | 1098440 | 87 | 0 |

A separate same-candidate selected telemetry probe ran one fresh 400-token-capped request per prompt after the matched matrices. TTFT was 10.597/12.443/10.588 seconds; exact before/after pager, wait, graph replay, summary, transfer and H2D counters are retained in the raw JSONL. The supplementary selected process peaks were GPU memory 14,357 MiB, process RSS 2,412,648 KiB, VmPin 0 KiB, VmData 5,776,844 KiB; pager target allocation 69,206,000 bytes, live allocation and scratch high-water values are in `runtime_peaks.selected`. These are labeled supplementary telemetry, not full-matrix maxima. No occupancy campaign ran.

Raw results: `/srv/ai/paged-kv/results/forward/101-12b/attempt-01/`.
