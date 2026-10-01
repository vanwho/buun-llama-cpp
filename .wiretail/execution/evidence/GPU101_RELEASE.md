# GPU101 release decision — review 101-12j

Decision: **goal_miss**. Candidate source `a0086f056f20efa519100718f010bc15c2efb830`; binary SHA256 `300dc8cf420abbd0b9121ffafd9ed90e4caf299a797280186ac84426a10ade54`; model SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`; frozen prefix SHA256 `8218b0f427cc931fa38d08f1c29f91ec7d82e12ac309ff7743eb7a60da678d20`. This is a fresh canonical campaign on the verified 101-12i candidate.

## Candidate and protocol

- L8192/H4096/P256/B1024/U256; target/draft Turbo4; native GPU MTP nmax2; temperature 0; reasoning off. Each route used one 40-token warmup and three 400-cap measured requests for each exact prompt, clearing the slot before each row.

- All 36 canonical requests returned HTTP 200. Each route has three warmups and nine measured rows; cached input was zero for all measured rows. Actual output tokens are recorded below and may be shorter than 400 at EOG.

## Recalculated per-prompt medians

| Prompt | Selected prefill/decode | Selected drafted → accepted (per row) | Selected MTP median | CPU prefill/decode | Pager-off prefill/decode | Prefill ratios selected/CPU, selected/off | Decode ratios selected/CPU, selected/off | Result |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Python sorted merge | 1519.71 / 81.92 tok/s | 279/223, 304/165, 274/234 | 79.93% | 672.75 / 24.97 | 1632.47 / 75.09 | 2.26x, 0.93x | 3.28x, 1.09x | pass |
| mmap versus read | 1535.38 / 58.77 tok/s | 128/58, 146/72, 104/51 | 49.04% | 676.96 / 18.54 | 1631.09 / 62.90 | 2.27x, 0.94x | 3.17x, 0.93x | pass |
| Bash directory watcher | 1519.69 / 58.85 tok/s | 304/146, 313/135, 280/212 | 48.03% | 671.78 / 21.64 | 1633.04 / 71.34 | 2.26x, 0.93x | 2.72x, 0.82x | MTP miss |

Selected measured output counts by prompt were `400,400,400 / 168,196,138 / 400,400,400`; fresh inputs were `4115,4115,4115 / 4112,4112,4112 / 4113,4113,4113` tokens. MTP drafted/accepted counts above are raw per-row counts; medians use the three independently calculated row acceptance percentages.

## TTFT and telemetry scope

Canonical run records do not expose TTFT, so canonical TTFT is recorded as unavailable. A separate client-side streaming probe measured time to the first non-empty content delta once per prompt at 32 requested tokens: `3040.87 / 228.11 / 232.92 ms. This is supplemental, not a three-row median or a release gate.

Canonical telemetry includes harness prompt/generation timers and MTP Prometheus counter deltas. No CUDA driver event timing, per-kernel timing, passive utilization samples, or per-row memory peaks were captured; these are not inferred. Raw records, configs, summaries, runner logs, recalculation, and TTFT probe are under `/srv/ai/paged-kv/results/forward/101-12j/attempt-01`.

## Decision and next owner

Selected prefill clears both 500 tok/s and 750 tok/s on all prompts, and selected decode beats CPU-main-KV decode on all prompts. Prompt 3 MTP is 48.03%, below its unchanged 60% floor; therefore the result is `goal_miss` and scale remains gated. 101-12i did not isolate a single source owner. Schedule `101-12k` for an identical-prefix target/draft score, mask/page visibility, carry, and state comparison; then `101-12l` owns candidate-specific final review. `102-01` now depends on `101-12l`.

Prior frozen report copies and raw roots remain preserved. The pre-101-12j report bytes are copied to `.wiretail/execution/evidence/provenance/101-12j/`; earlier snapshots under `provenance/101-12h/` and `provenance/101-18/` were not modified.
