# Phase 93 — Turbo4 MTP and attention-pager diagnostics

This is the compact, current phase context. Task packets own their exact
protocols and acceptance criteria. Historical V9/phase-85 plans, old failed
receipts, and prior raw transcripts are not startup context; consult only a
named artifact when a current task has a specific unresolved question.

## Project direction

Keep Qwen3.8-27B UD-IQ4_XS as the target. The intended runtime is Turbo4 target
K/V with CPU-RAM backing, an attention-aware GPU hot set, and native MTP whose
draft K/V remains GPU Turbo4. Prefer the direct/packed selected path. A
`selected_reference` route is diagnostic fallback, not acceptable performance
evidence; do not silently label it as selected-direct or a speed result.

## Models and task segmentation

- Use `gpt-6-luna` High for phase-93 task work and recovery assessments.
- Only explicitly designated upstream merge/incoming-change tasks 93-11i and
  93-11j may use `gpt-6-luna` XHigh.
- Keep tasks in a cluster while source/contracts/context remain cohesive; do
  not load unrelated phase history. The current 93-11g task has an explicit
  small context list and a compact current handoff.

## Candidate and managed-service rules

- Use one immutable candidate binary, the canonical
  `CANONICAL_BENCHMARK_RUNNER`, one slot, and the existing managed Qwen process.
- Before each live campaign, compare PID/start time, executable realpath/hash,
  model realpath/hash, and exact relevant command-line settings. Reuse only
  when identity matches and the prior lifecycle receipt says
  `continue_loaded`; otherwise load the expected candidate using the managed
  lifecycle, then verify identity before sending requests.
- Never launch a second Qwen3.8-27B CUDA process or touch port 8092. Use
  noninteractive `sudo -n`; repair invocation/configuration issues instead of
  recording them as unavailable authorization.
- A healthy but mismatched endpoint is a setup failure, not candidate evidence.
  Preserve raw output, exact request/config/model hashes, lifecycle record,
  and service health. Do not claim a live gate without a candidate-bound
  request.

## Performance contract for speed tasks

- Keep MTP enabled with GPU Turbo4 draft K/V for every speed measurement.
  MTP-off is allowed only for a named functional control.
- The canonical speed test is the three original prompts: Python sorted-list
  merge with docstring; one-paragraph mmap-vs-read explanation; Bash directory
  watcher. Use one 40-token warmup, then three 400-token measured generations
  per prompt, reasoning off, B=1024/U=256.
- Report each prompt's MTP acceptance separately; minimum medians are P1 75%,
  P2 40%, P3 60%. Do not average prompts/configurations together.
- Selected prefill minimum is 500 tok/s median per prompt, with 750 tok/s the
  target. Measure only exact rendered-input tokens (maximum 16,384), and
  reassess direct route, cache reuse, candidate, timings, waits/transfers,
  CPU time, and coarse GPU use after each result.
- No test may reserve more than 49,152 target hot tokens. Request only the
  smaller byte budget needed; allocator and scratch headroom remain
  authoritative.

## 93-11g functional-promotion exception

The detailed protocol is in `tasks/93-11g.md`. It replaces the old one-file
8K A/B test with exactly three same-slot requests: five complete Python merge
fixtures and an operational allocation-efficiency filename question; five
complete Bash watcher fixtures and an operational watcher-efficiency filename
question; then the identical Python question again. The intended Python fact
is `PY_MERGE_03`; the intended Bash fact is `BASH_WATCH_01`.

Ten fixtures alone total 10,240 tokens, so this functional test uses a 16,384
server context and 4,096 hot tokens (not 8K/4K). Keep 256-token pages,
B=128/U=64, one slot, selective Turbo4 target K/V, GPU Turbo4 native MTP, and
reasoning off. Preflight the complete rendered conversation; send all three
requests even if answers are wrong. Never force eviction, select a page, or
change route. Record the `PY_MERGE_03` page's coldness before the repeated
question and correlate every page overlapping the Python fixture, especially
the answer-bearing page, through natural selector, H2D, publication, target,
and draft events. Report whole-fixture residency only if all of its pages are
resident. Decode speed may be recorded as a secondary matched observation,
but cannot prove which page was resident or promoted; answer quality and
physical promotion are independent findings.

The old 8K single-file protocol and its incomplete results are historical only.
Keep their receipts/raw roots immutable; do not make them acceptance criteria
for the new sequence.

## Forward task order and evidence discipline

After 93-11g, retain the declared order: 93-11i local upstream integration and
conflict resolution; 93-11j assessment of incoming changes for useful
performance approaches; 93-11k lock/validate the benchmark contract; then
93-12 paired cold-context performance work. Do not sync upstream as part of
93-11g. Report null/unmeasured values as unknown; never infer success from a
route label, aggregate counter, or prose answer alone.

Keep current handoffs as concise state snapshots. Raw JSONL and benchmark
records are append-only forensic artifacts, not task prompt context. Use a
final message/handoff first and extract only one targeted raw event if a
specific fact remains unresolved.
