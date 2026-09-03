# Phase 00 baseline results

Captured 2026-09-03 against the pinned Qwen3.8-27B/qwen35 model and Buun
server build from `reference.json`. The established benchmark uses greedy
sampling (`T=0`, seed 42) and clean-isolated profile startup.

## Valid benchmark runs

| Run | Configuration | Result | Artifacts |
| --- | --- | --- | --- |
| Fast/default | 77,824 context, Turbo4 target and MTP, thinking off, 1 warmup plus 3 measured 400-token requests for each of 3 prompts | 9/9 passed; decode medians 101.35, 77.14, and 93.56 tok/s; prompt medians 710.09, 653.37, and 672.38 tok/s | [`run-config.json`](file:///srv/ai/paged-kv/results/profile-benchmark-default-fast-20260903-055158/run-config.json), [`records.jsonl`](file:///srv/ai/paged-kv/results/profile-benchmark-default-fast-20260903-055158/records.jsonl), [`summary.json`](file:///srv/ai/paged-kv/results/profile-benchmark-default-fast-20260903-055158/summary.json) |
| Fast/large | 77,824 context, Turbo4 target and MTP, 44K input, thinking off, 2 warmups plus 5 measured 512-token requests | 7/7 passed; prompt median 1457.70 tok/s; decode median 89.35 tok/s; MTP acceptance median 100.00% | [`run-config.json`](file:///srv/ai/paged-kv/results/profile-benchmark-large-fast-20260903-055411/run-config.json), [`records.jsonl`](file:///srv/ai/paged-kv/results/profile-benchmark-large-fast-20260903-055411/records.jsonl), [`summary.json`](file:///srv/ai/paged-kv/results/profile-benchmark-large-fast-20260903-055411/summary.json) |

The measured MTP acceptance in the short run was 93.697%, 58.571%, and
83.654% by prompt (the third prompt also varied to 82.381% in one trial).
Both manifests record model, build, GPU, profile, sampling, and exact request
settings.

## Controls not completed

The temporary full-context ordinary CPU-KV launch loaded the pinned model at
262,144 tokens with explicit `--no-kv-offload` and logged the CPU-bound KV
path, but its standalone process was reclaimed by the service manager before
the first request. No CPU-KV throughput or quality number is reported. Its
launch log and failed control setup are retained at
`/srv/ai/paged-kv/results/qwen38-cpu-kv-launch-20260903.log` and
`/srv/ai/paged-kv/results/ordinary-cpu-kv-full-20260903/`.

A same-corpus greedy spec-off token comparison was not run after the failed
temporary launch. The valid Fast run is the spec-on/MTP control; no spec-off
acceptance or quality result is claimed.

## Historical comparison

The dated Section 12.4 anchors remain unchanged: 44,018-token Fast validation
at 77,824 passed at 1,411.64 prompt and 37.43 decode tok/s; recorded large
Fast/off median was 1,326.38 prompt and 71.05 decode tok/s; and the prior
short Turbo4/F16 MTP acceptance was 81.591%/80.974%. The new Fast results use
the same model/profile/build family and harness contract, but are faster on
decode and prefill; this is run-to-run variance and/or runtime state, not a
replacement of the dated history. The new large run uses 512-token outputs
and reports 89.35 decode tok/s, so it is not directly interchangeable with
the dated 71.05 figure without preserving those configuration differences.

The controlled restore left `qwen38-big` active and Qwen `/health` on port
8080 healthy. The harness and activation workflow also require the separate
8091 health endpoint, which was unavailable during restore; this remains the
task blocker recorded in the handoff.
