# Short path post-merge decision

Decision: **goal_miss**. Measurement completed on the 100-03h candidate with the exact frozen prefix, L8192/H4096, page256, B1024/U256, selected GPU Turbo4 target, native full-L GPU Turbo4 MTP nmax2, temperature 0 and reasoning off. The three rendered prompt inputs were 4329, 4326 and 4327 tokens, each with zero cached tokens; each exceeds H=4096; the corresponding slot snapshots report host-backed page inventory entries. The requests use shared prefix SHA256 `ad528d87807f615a95b0694c0b0dd097e3facd24be14b71914a5ecf9694f4067`.

## Stage 1 individual rows

| Prompt | Fresh prefill tok/s | Fresh tokens | Prefill ms | Output tokens / finish | Decode tok/s | MTP accepted / drafted | Acceptance |
|---|---:|---:|---:|---|---:|---:|---:|
| 1 | 210.540 | 4329 | 20561.456 | 181 / stop | 37.502 | 78 / 134 | 58.21% |
| 2 | 210.614 | 4326 | 20539.912 | 400 / length | 22.112 | 0 / 398 | 0% |
| 3 | 210.511 | 4327 | 20554.787 | 400 / length | 22.086 | 0 / 398 | 0% |

Each row is an individual Stage 1 observation. Stage 2 was not run, and per-prompt medians and placement ratios are null. All three rows miss the 500 tok/s minimum (750 preferred), so selected/CPU and selected/dense comparisons were not run. TTFT is not exposed by the runner. The canonical runner does not emit outbound HTTP bodies; the saved compact request JSON was reconstructed from its payload template, prompts, and frozen prefix, and is hashed in the request manifest. The canonical record reports total fresh-prefill timing; request-local substage wall times and per-row peak VRAM/RAM, power and utilization are not exposed.

The request journals show five live-rewind checkpoint creations per measured prompt, with reported image sizes around 151–154 MiB. The sum is an image-size signal, not measured physical copy traffic or time. Fresh-prefill time remains the dominant measured stage at about 20.5 s per row, but its internal cause is not isolated. This supports a focused repair hypothesis for repeated near-end checkpoint materialization in `server-context.cpp`; the scheduled packet requires the focused fixture and one short retake.

Pager counters are cumulative snapshots and are not presented as per-row deltas. The per-request raw MTP counter snapshots and journal excerpts are indexed below. No CPU or dense-GPU rows were run. A focused successor, `100-03j`, is scheduled for the observed live-rewind checkpoint churn, and `100-04` now depends on it.

Raw root: `/srv/ai/paged-kv/results/forward/100-03i/attempt-01/`. Artifact index: `/srv/ai/paged-kv/results/forward/100-03i/attempt-01/artifact-index.json` (SHA256 `64f68d25ddcbbd687d7a8f734700225dc96abc102942b31650f523535dc6f6ed`). Request-body manifest: `/srv/ai/paged-kv/results/forward/100-03i/attempt-01/selected-stage1/requests/manifest.json` (SHA256 `8f7a2fba44c2255be2ccd4fa43b2dac9665051b8fb980c4d708d3f776be46a34`). The machine-readable summary is [SHORT_PATH_POSTMERGE.json](SHORT_PATH_POSTMERGE.json).
