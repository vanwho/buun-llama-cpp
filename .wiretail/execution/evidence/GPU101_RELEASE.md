# GPU101 canonical retest — 101-12d

Outcome: **goal_miss**. The complete selected matrix and both matched controls used the repaired candidate. All 3x400-cap measured requests per matrix completed with HTTP 200 and no errors. The selected fresh-prefill medians remain below the 500 tok/s gate; final release review is owned by 101-12e. No long occupancy ran.

## Identity and protocol

- Source commit: `8a5a3091f598c8952f13a904f44b7dce467685e7`; server SHA-256 `d6242076cd05d65289939ea251549367350fe4845cbe936c6b518d6d92ff797d`; libllama SHA-256 `67ea4b07a275af63278816333af180c9b13d0df1d8b912bfd0b60f0d2fe93103`.
- Model: `/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf` SHA-256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
- L8192/H4096/P256/B1024/U256; automatic selective pager; target/draft Turbo4; native GPU MTP nmax2; temperature 0; reasoning off.
- Frozen shared-prefix SHA-256 `8218b0f427cc931fa38d08f1c29f91ec7d82e12ac309ff7743eb7a60da678d20`; each row erased slot 0; production cache mode, no isolated clean-cache run.
- Prompts: Python sorted merge, mmap/read paragraph, Bash directory watcher. Each matrix contains 3 warmups and 9 measured rows; selected prompt inputs were 4115/4112/4113 tokens with zero cached input tokens.

## Selected measurements

| Prompt | Fresh prefill tok/s median | Decode tok/s median | MTP acceptance median | Supplemental TTFT s | Disposition |
|---:|---:|---:|---:|---:|---|
| 1 | 331.28 | 34.69 | 48.88% | 10.732 | measured_goal_miss |
| 2 | 386.66 | 34.30 | 48.44% | 12.447 | measured_goal_miss |
| 3 | 331.21 | 37.11 | 57.05% | 10.964 | measured_goal_miss |

## Matched controls

| Prompt | Pager-off all-GPU prefill | Pager-off all-GPU decode | CPU-main-KV prefill | CPU-main-KV decode |
|---:|---:|---:|---:|---:|
| 1 | 1566.34 | 84.13 | 659.75 | 25.06 |
| 2 | 1562.65 | 61.70 | 661.72 | 18.60 |
| 3 | 1564.67 | 71.12 | 662.30 | 21.50 |

## Telemetry and memory

The separate three-request selected probe recorded exact Prometheus before/after deltas for route, replay, query, transfer, H2D, wait, and scratch counters. H2D useful-byte deltas: 0.0, 0.0, 0.0. Full request deltas are in `selected.request_local_metrics` and the raw probe rows. Passive samplers recorded RAM RSS/data/pinned and GPU VRAM peaks for all three matrices; selected pager allocation, pinned-host, Turbo4 scratch, and scratch high-water peaks are in `runtime_peaks.selected.kv_pager_metrics_peak`.

## Raw evidence

Raw root: `/srv/ai/paged-kv/results/forward/101-12d/attempt-01`. The JSON receipt indexes and hashes each run configuration, summary, record stream, runtime monitor, runner log, and selected telemetry probe. The two invalid setup attempts remain preserved but are excluded.

## Gate and next owner

The candidate missed the selected fresh-prefill gate on all prompts and also missed MTP acceptance floors. 101-12e owns the repeated release review and must schedule another concrete repair/retest chain before 102-01 if the miss remains. The GPU101 scale gate remains closed until a complete passing release.
