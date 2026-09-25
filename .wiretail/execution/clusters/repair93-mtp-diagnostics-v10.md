# Phase 93 — Turbo4 MTP and attention-pager diagnostics

Revision: `hotpath-v10-20260914`. Amendment: `repair93-selector-promotion-20260925`.

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
  not load unrelated phase history. The 93-11g live attempt is deferred; its
  compact packet/handoff are historical only. The executable replacement is
  93-11l through 93-11n in `repair93-selector-promotion-v10.md`.

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

## Natural selector and promotion correction

Attempt 08 of 93-11g established a cold, host-backed page before request 3
but no completed promotion. Its empty success-only `natural_proof` does not
prove the selector emitted no nomination; raw selector output and page-level
eligibility/policy reasons were not captured. A concrete code weakness is
that graph construction selects Q row 0 and pairs it with `ubatch.pos[0]`,
which can be unrelated to the final causal query in a prompt microbatch.

Do not rerun the same live sequence until tasks 93-11l through 93-11n complete:
93-11l uses the last valid causal Q row and tests row/position pairing;
93-11m adds opt-in bounded stage diagnostics and deterministic selector-to-
policy/transfer tests; 93-11n runs one corrected candidate-bound natural
promotion campaign. Full code directions and test limits are in
`clusters/repair93-selector-promotion-v10.md`. Keep attempt-08 receipt/raw
artifacts immutable and keep answer scoring independent of page movement.

## Forward task order and evidence discipline

After the deferred 93-11g disposition, run 93-11i local upstream integration,
93-11j assessment of incoming changes, and 93-11k benchmark-contract lock.
Then run 93-11l/11m/11n in order before 93-12 paired cold-context performance.
Report null/unmeasured values as unknown; never infer nomination from a
success-only proof, or promotion from a route label, aggregate counter, or
prose answer alone.

Keep current handoffs as concise state snapshots. Raw JSONL and benchmark
records are append-only forensic artifacts, not task prompt context. Use a
final message/handoff first and extract only one targeted raw event if a
specific fact remains unresolved.
