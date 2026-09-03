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

## Controls

The full-context ordinary CPU-KV control used a transient systemd unit with
the pinned model/build, 262,144-token context, Turbo4 K/V, and explicit
`--no-kv-offload`. Three requests passed: prompt median 1049.49 tok/s and
decode median 8.96 tok/s. Artifacts, including the launch manifest, records,
summary, HTTP metadata, and raw responses, are under
`/srv/ai/paged-kv/results/ordinary-cpu-kv-full-20260903/`.

The canonical Big profile supplied the same-build greedy spec-off control:
9/9 short requests passed, with decode medians 49.08, 48.98, and 48.93 tok/s
and prompt medians 736.79, 671.84, and 692.45 tok/s. No MTP acceptance is
expected for this profile. Its manifest, records, and summaries are under
`/srv/ai/paged-kv/results/profile-benchmark-default-big-20260903/`. The Fast
MTP run is the spec-on control above; these profiles differ in context and
MTP placement, so the comparison is a rollback/performance control rather
than a paired quality claim.

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

The controlled restore left `qwen38-big` active with both Qwen `/health` on
8080 and the configured long-memory `/health` on 8091 healthy.
