# Cluster forward106-master-baselines — ACO promotion-cost prelude and upstream-master controls

Revision: `aco-promotion-cost-20261012`.

Purpose: first quantify the request-level cost of natural cold-page selection
and promotion on the frozen Phase-105 ACO candidate, then compare the exact
current `origin/master` source against the existing ACO results using the
original three-prompt speed schedule and matched fresh repository-context
prefill. This is a benchmark-only campaign: do not modify llama.cpp production
code, codecs, batch sizes, sampling, prompts, or model settings to improve a
result.

## Ordered prelude: task 106-00

Task 106-00 runs before any master fetch/build. It reuses the exact Phase-105
ACO scratch-fix candidate and measures one identical, source-grounded question
while its answer page is hot versus definitely cold across the original
77,824/65,536 profile and L/H pairs 8K/4K, 16K/8K, 32K/16K, 64K/32K,
128K/64K, 192K/64K and 256K/64K. Its packet is the complete protocol. It owns
its frozen repo fill, fresh-prefill curves, initial-hot results, matched-C
hot/cold pairs, request-local counter deltas and total TTFT/wall-time deltas.
Do not use cumulative `transfer_time_us` as a proxy for end-to-end promotion
cost. Task 106-01 begins only after 106-00 records and validates its truthful
result; 106-00 does not alter the later master binary or baseline schedules.

## Source and binary identity

Task 106-01 fetches `origin/master` once and records its full 40-character SHA.
Build that detached source in a separate worktree and a clean, SHA-named CUDA
build directory; do not build the active plan branch or reuse `build-cuda`.
Keep that exact source worktree, build directory, executable, and DSOs for
106-02. 106-02 must verify their hashes against the committed
`MASTER106_FREEZE.json`; it must not fetch, rebuild, or silently substitute a
binary. Rebuild only if the recorded build is objectively unusable, and then
restart both comparison tasks under a new explicitly named cohort.

The one-model lifecycle is mandatory. Take `/tmp/ai-pager-benchmark.lock`,
stop and verify the prior managed Qwen process is gone before every new
`-c`/KV-placement configuration, and wait for its CUDA allocations to be
released. Start only the exact frozen master binary, never a second model.
Preserve 8091 and do not touch 8092. Restore the previously active managed
profile after each task and verify its identity and health.

## Fixed benchmark contract

- Model: `/srv/ai/models/text/current.gguf`; record SHA-256 once and require it
  unchanged for every row.
- GPU: CUDA0, all model layers on GPU (`-ngl 999`), Flash Attention enabled,
  fit disabled, one sequence, context shifting disabled.
- Target and draft KV for GPU-only: Turbo4 on GPU.
- Target KV for CPU offload: ordinary F16 CPU K/V using `-ctk f16 -ctv f16
  --no-kv-offload`; model weights remain on CUDA0 and draft-MTP K/V remain
  Turbo4 on CUDA0. This is not CPU MTP and is not ACO.
- MTP in every row: `--spec-type draft-mtp --spec-draft-n-max 2
  --spec-draft-kv-device gpu --spec-draft-type-k turbo4
  --spec-draft-type-v turbo4`.
- Batch/micro-batch: `-b 1024 -ub 256`; parallelism `-np 1`.
- No ACO/pager options: on upstream master the baseline is ordinary KV
  placement. Do not add selected/observe/reference routes or ACO-only flags.
- Three exact prompts, unchanged from the original campaign:
  1. `write a python function that merges two sorted lists into one sorted list, with docstring.`
  2. `explain the difference between mmap and read for loading large files, one paragraph.`
  3. `write a bash script that watches a directory and prints new files as they appear.`
- Request settings: SSE, temperature 0, seed 42, reasoning modes `off`, `low`,
  `medium`, `xhigh`; one warmup capped at 40 generated tokens and three
  measured runs capped at 400 tokens for each prompt/mode. Preserve native EOS;
  400 is a cap, not a forced response length. Each row therefore has 48
  requests (12 warmups + 36 measured). Calculate medians from measured runs
  only. Report MTP acceptance from request-local drafted/accepted counters;
  missing counters are `unavailable`, never zero or an automatic performance
  failure.
- For each configuration, emit separate tables for median decode tok/s,
  response-local MTP acceptance %, and fresh-prefill tok/s, with rows for the
  four reasoning modes and columns for the three prompts.
- Prefill-curve samples are uncached repository-source append chunks at
  cumulative occupancies nearest 8,192, 16,384, 32,768, 65,536, and 131,072
  tokens when that row reaches them. Record actual cumulative token count,
  newly processed token count, per-append time/tok/s, cache status, and model/
  binary identity. Never infer fresh prefill speed from cached prompt reuse.
- For filled rows, start from an empty slot and load deterministic tracked
  repository source using the existing repo-content fixture/chunk selector.
  Freeze the selected source ranges, hashes, rendered prompt bytes, and token
  spans once in `MASTER106_FREEZE.json`; use the same source schedule for GPU
  and CPU rows. Do not manufacture filler or mutate contents between rows.
  Reach the exact occupancy target C before canonical probes; maintain at least
  1,536 tokens of safety reserve after accounting for the next prompt,
  400-token generation ceiling, and MTP replay. The schedule leaves 2,048
  tokens at L=8,192/16,384, 2,768 at L=32,768, and 4,096 at
  L=65,536/131,072. Restore the exact filled base snapshot before
  each of the 48 canonical requests so rows are independent.
- No speed, MTP, or semantic threshold gates this measurement campaign. Repair
  a concrete setup/harness mismatch in the current task and resume the same
  frozen row; do not count a zero-request preflight as a measurement. A
  completed low-speed result is data, not grounds for repeated identical runs.

## Benchmark driver requirements

The existing `run-occupancy-frontier.py` is not a drop-in driver: it is
selected-pager oriented and its post-load suite is only 12 off-reasoning
requests. Do not invoke it unchanged or claim its result is this matrix. The
existing `run-final-curve.py` owns useful request/SSE and MTP parsing helpers,
and `repo_context.py` owns tracked-file selection, stable source-range hashes,
rendering, and tokenizer-aware bounded appends. In 106-01, add a small
`tools/server/bench/run-master-context-matrix.py` adapter only if the existing
helpers cannot express this protocol. Reuse those functions rather than
copying their tokenizer, SSE parser, or source selection logic.

The adapter must accept explicit `--context-tokens L`, `--target-tokens C`
(omit only for the empty baseline), `--placement gpu|cpu`, `--output`, and
`--api-key-file`; reject unsupported geometries before contacting the live
server. It must use the exact server-rendered tokenizer for preflight; create
and persist an append-only schedule and source manifest; chunk repository
ingestion to at most 4,096 fresh tokens; log per-append timings; and save the
occupied base once. It then runs 48 canonical branches from that identical
base, separated by all 4 reasoning modes and three prompts, using a verified
slot snapshot/checkpoint restore between branches. If the server's snapshot
API cannot restore both target and draft state, stop before the 48-row suite
and repair the harness/lifecycle within this task; do not silently continue
with contaminated sequential conversations. Each case writes request and raw
SSE artifacts once (unique attempt names; never overwrite), parsed token
counts, request-local MTP drafted/accepted counters or explicit `unavailable`,
prefill/decode rates, status, and source/base identity. Resume skips only
rows that have complete raw request/response plus validated parsed records
with the same frozen fingerprint.

Add the smallest deterministic regression set: (1) exact 48-row schedule and
warmup/measurement classification, (2) fixed L/C/placement-to-argv mapping,
including no pager flags and CPU target/GPU Turbo4 draft, and (3) resume/base
fingerprint refuses changed source or server geometry and never overwrites
raw rows. Run these tests before the live campaign. Do not add instrumentation
to llama.cpp runtime code or alter the server implementation for this
benchmark.

## Exact context schedule

The first baseline is the original empty-slot test at L=77,824 (no context
fill). Then both placements run the matched filled rows below. The label
“8K/4K” refers to the prior ACO geometry; these GPU-only and CPU-KV controls
have no hot-cache boundary, so fill to the same paired C=6,144 only.

| Label | Exact server L | Exact occupied C | Placement rows |
|---|---:|---:|---|
| original baseline | 77,824 | empty | GPU-only, CPU-F16-KV |
| 8K/4K paired occupancy | 8,192 | 6,144 | GPU-only, CPU-F16-KV |
| 16K | 16,384 | 14,336 | GPU-only, CPU-F16-KV |
| 32K | 32,768 | 30,000 | GPU-only, CPU-F16-KV |
| 64K | 65,536 | 61,440 | GPU-only, CPU-F16-KV |
| 128K | 131,072 | 126,976 | CPU-F16-KV only |

K means 1,024 tokens. For every row, validate the exact rendered prompt,
output, MTP replay, and 1,536-token safety reserve fit before sending requests. If a
particular upstream-master build has a verified runtime capacity limit, record
the exact failed configuration and stop only that row; do not silently reduce
L/C, batch, KV type, or MTP settings. The requested measurements must not be
mislabelled as successful if a row cannot run.

## Artifacts and completion behavior

Keep bulky request/SSE logs, build output, and memory samples under
`/srv/ai/paged-kv/results/forward/106-master/`; do not commit raw evidence.
Commit only compact handoffs, summary JSON/Markdown, freeze manifests, and V10
receipts. Preserve previous results. The 106-03 report joins 106-01 and 106-02
by master SHA, model SHA, prompts, settings, and exact L/C; it must flag any
identity mismatch and present the old ACO measurements as a separate cohort.
Neither task completion nor a poor/unavailable row implies that the overall
ACO goal has passed or failed.
