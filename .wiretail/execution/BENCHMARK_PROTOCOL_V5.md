# Benchmark protocol V5: meaningful runs, not repeated setup failures

Applies to post-17 tasks. Read alongside `POST17_IMPLEMENTATION_STRATEGY.md`.
Site operations live in `/srv/ai`; generic harness/engine changes live in Buun.
Never change a running campaign, silently rescore historical data, or edit
the three original benchmark questions to make quality/speed look better.

## A. Runtime identity and safe lifecycle

1. Verify `sudo -n true`. Use `sudo -n systemctl ...` for privileged lifecycle
   actions; an unprivileged systemctl error is not evidence sudo is unavailable.
2. Snapshot exact named service MainPID, executable and loaded shared-library
   identities, model realpath/hash, full effective non-secret argv, profile and
   transient benchmark overrides. Inspect `/proc/<MainPID>/exe` and maps; never
   select a server using broad `pgrep` or the unrelated service on 8092.
3. Build incrementally during diagnosis. Freeze an immutable runnable bundle
   of executable **and loaded project DSOs** before a campaign. A binary hash
   alone is insufficient if shared libraries are rebuilt underneath it. Record
   source/tree-diff hash, compiler/CMake flags, build IDs and GPU/driver.
4. One site-level lifecycle owner holds a bounded lock and owns launch, signal,
   timeout and restore. Use measured wall-clock deadlines, not loop counts
   whose individual curl calls can multiply timeout by minutes. Stop a detected
   restart storm; don't launch the full matrix while readiness flickers.
5. Authenticated health, props, metrics and one generation must agree on model,
   bundle, target/draft codec, MTP rows/GPU placement, n_ctx, hot budget and mode.
   A profile name or HTTP 200 is not identity. Pass key-file paths through the
   established environment; never print/copy secret values.
6. Keep a healthy tested candidate loaded across tasks/retries. For an explicit
   control, change only declared variables. On failed startup, restore the exact
   last healthy runtime/overrides, not just a profile name that expands to a
   different context/binary. Verify identity plus 8080/8091 health; never touch
   8092. Preserve crash artifacts before cleanup.
7. Never kill processes by generic `python3`, `/dev/fd/3`, server basename or
   broad process-group guesses. Capture PID, start time, command and ownership
   when launching each benchmark child; only terminate that verified child/tree.

## B. Three separate sizes

### Choose context for the test, not from the final-results curve

The six 20K/40K/60K/100K/175K/256K coordinates are **only the final results
campaign in 21-02 and its matched final controls**, after the functionality
works. They are not a development-test matrix, phase gate or default context
ladder. Earlier tasks choose and record the smallest useful context/occupancy
and explicit test hot budget for the behavior under investigation:

- arithmetic/metadata: synthetic sizes and boundary values, no model allocation;
- transform/logit parity: a short prefix, then enough tokens for the relevant
  page/query/ubatch boundary, usually far below a long-context benchmark;
- residency: a few more occupied pages than a deliberately small test hot
  capacity, not tens of thousands of tokens just to cause a fault;
- MTP/rollback: enough history and verify tokens to exercise the failing state;
- throughput optimization: a representative measured bottleneck shape, not
  six complete end-to-end runs for every patch;
- full-context admission/population: 262,144 only in the explicit capacity
  tests (19-01/19-08) and final proof. Allocation and populated-history proof
  are distinct operations with different runtime costs.

A small context is a legitimate targeted diagnostic; the error is claiming
it demonstrates the overall 256K goal. Keep MTP rows matched to whatever
target context that particular test resolves, and preserve Turbo4 everywhere.
Reproduce a failed setting when needed; smaller regression cases are allowed
without changing the promised final target.

Record `n_ctx_configured`, `n_ctx_resolved_per_sequence`, `prompt_tokens_actual`,
`history_tokens_committed`, `generation_budget`, `mtp_rows_actual`,
`hot_pages_capacity`, `resident_valid_tokens`, `host_valid_tokens`, and per-layer
selected rows. Bytes must never be stored in a tokens field.

Use the target tokenizer on the **actual rendered chat template**, including
system/tool prefixes, suffix question, BOS/special tokens and prior messages.
Prefer the server's template/tokenization APIs or the same model's tokenizer
offline. Verify one preflight against server-reported usage. Fit by token IDs
or tokenizer-guided search; never assume tokens per word/character. Do not
double-apply the template or submit already-rendered text as another chat body.
Token trimming must preserve inserted facts and the exact final question.

Require prompt+history+generation and any separately required verify lookahead
to fit the actual context contract. Identify whether lookahead is already
included by the server before subtracting it twice. Boundary tests: non-ASCII,
punctuation-heavy markers, tools, page-edge tails, 16K/22,016 and 262,144.
An HTTP 400 context overflow is a sizing bug, not slow inference or quality loss.

For the explicit full-capacity milestone and final release campaign, run an
allocation-only 262,144 startup and short generation, then a resumable,
near-full **occupied** context proof with explicit generation headroom.
This is not a prerequisite for unrelated earlier unit/parity/transfer tests.
A sparse request at `-c 262144` is only an allocation test.
For pressure use occupied pages greater than actual hot capacity; otherwise
set an explicit smaller **test-only** hot-byte budget without shrinking logical
context or GPU MTP. Warm no-fault timing may legitimately have zero transfers;
a separate cold-promotion segment must prove physical movement.

## C. Progressive tests and resumability

Use this order for each new binary/configuration family:

1. Local regression and allocation ledger; supported-capability probe once.
2. One short all-fit generation and authenticated telemetry.
3. A two/three-case correctness sentinel: page boundary, distant fact and MTP
   rejection. On systematic mismatch, preserve one minimized reproducer and
   repair before running 24 doomed cases. Capability 501 skips that mode's
   campaign as `not_implemented` with its implementation task, not 24 wrong answers.
4. One token-counted pressure request, then full quality/performance/soak.

Implement a durable case manifest and append-only per-case records. Key a result
by bundle/model/tokenizer/template/corpus/config/prompt hash, mode, context,
sampling, warm/cold cache condition and repetition. Flush/fsync at case
completion, atomic checkpoint after it; resume only matching **completed** keys.
Mark interrupted attempts separately, never duplicate them in medians. Preserve
prior raw answers; a corpus repair creates a new version and fresh controls.

Separate connect/startup, prefill-progress, decode-idle and total-run deadlines.
Estimate prefill time from same-mode lower-context measurements, with generous
growth allowance (attention may be superlinear); print ETA/progress regularly.
Do not wrap a long 256K campaign or CUDA build in a 180/240/300-second blanket
timeout. Keep a configurable hard wall limit, but checkpoint on it and classify
the row `incomplete_timeout` with elapsed work, not zero tok/s or a wrong answer.
If prefill reports no SSE tokens yet, use server progress/slot metrics, not lack
of client output alone, to decide whether it is hung. Reconnect after dropped
display does not prove the workload stopped.

After timeout/cancel, verify the request is no longer occupying its slot before
starting another. Save request ID, slot/generation, progress and server counters;
do not let an orphaned request turn subsequent tests into queue-time measurements.
Run long clients as named resumable tools, not anonymous shell heredoc loops.
Poll their durable progress rather than resubmitting or rereading giant logs.

## D. Canonical speed curve

This section defines **final results collection**, not the test protocol for
every development task. Run the six-point curve only after working storage,
transfers, attention, native MTP, correctness sentinels and full-capacity
functionality have been demonstrated. If those prerequisites fail, preserve
the minimized failure and schedule its repair; do not spend the full curve
budget collecting repeated setup errors. There is no numeric speed gate.

Exact final user questions from the existing `/srv/ai/benchmarks` suite:

1. `write a python function that merges two sorted lists into one sorted list, with docstring.`
2. `explain the difference between mmap and read for loading large files, one paragraph.`
3. `write a bash script that watches a directory and prints new files as they appear.`

Keep the original short benchmark as an all-fit regression probe. For the long
curve add reproducible contextual material **before** each unchanged question;
record the actual rendered request/token hash. Use deterministic, non-adversarial
context without hidden answer leakage; add separately labeled coding/retrieval
corpus runs so repetitive padding is not mistaken for representative quality.

Curve nominal context capacities: **20,000; 40,000; 60,000; 100,000; 175,000;
262,144**. These are experimental coordinates, not production defaults. At
each coordinate fill close to capacity with declared generation headroom,
record exact occupied size, and set native MTP to that resolved capacity. Also
retain one continuing full-capacity 262,144 session with growing occupancy to
separate MTP allocation changes from effects of history length. A short live
smoke proves 256K startup before the ramp; near-full population is separately
required before claiming the complete 256K result.

Primary protocol: T=0, seed=42, thinking off, SSE usage/timings, one discarded
40-token warmup per question/config, three measured trials with up to 400
output tokens, preserving real EOS/output count. Do not silently substitute
16-token nonstream timings or ten differently prompted trials. Extend selected
points to ten repetitions when needed for variability; label the extension.

Define clean-prefill versus prefix-reuse explicitly. Warm decode means the
working set is prepared, not that a hidden cache bypassed the input work.
Saved prefix/checkpoint reuse is a separate mode with provenance and MTP state.
Report per-question metrics first; never pool three prompt distributions into
one unexplained median. For summaries use explicit aggregation and paired
ratios, with sample counts/dispersion; tiny output samples are diagnostics only.

Record server pp/tg separately from wall-clock TTFT and completion latency.
Measure TTFT from client monotonic send to first generated token/SSE content;
server prompt-eval time is not TTFT. SSE chunks may contain multiple tokens,
so label chunk gaps honestly unless per-token timestamps are available. Use
server committed output count, not proposed MTP tokens, in user decode speed.

## E. Denominators, quality and actual paging

Primary: selective Turbo4 target + canonical CPU backing + native GPU Turbo4
MTP. Controls: ordinary CPU **KV** with GPU model/recurrent compute and GPU
Turbo4 MTP, dense all-GPU Turbo4+MTP where it fits, observe, and exact reference.
MTP-off is an explicit ablation only. Verify backend support for ordinary CPU
Turbo4 KV; if absent, implement/diagnose it or label that control unsupported.
Never silently swap codecs, move all weights to CPU or call MTP-off all-GPU a
same-MTP denominator. Don't infer long-context speed from a shorter control.

Match source bundle, prompt token IDs, context, model, sampling, output limit
and cache state for ratios. Keep slow CPU controls resumable with independent
runtime estimates; if a paired segment is incomplete, ratio is null with a
reason. Offload and no-offload differences in backend execution are recorded.
Report 3x/5x/70% comparisons as findings; none ends work for missing a threshold.

Quality separates: valid corpus facts; teacher-forced dense parity; all-selected
parity; native-MTP equivalence; sparse recall; sparse-prefill accumulation.
Dense failures remain visible and are adjudicated from raw prompts, not removed
to lift the pager score. Freeze calibration versus held-out IDs before tuning.
Measure cold distant needles, focus shifts and multi-hop facts at several depths.
Don't answer a constrained-marker grammar with a known literal and count that
as retrieval; grammar checks framing/isolation only.

Pressure receipts must correlate real host page checksums/bytes, CUDA event or
profiler copy evidence, H2D completed useful/aligned bytes, resident slot reuse,
evictions, page-table generations and continued valid responses. Show a formerly
cold chosen page entering attention, not just counters increasing. Expect D2H
for sealing new writes but zero eviction D2H for already-clean pages. Zero
traffic during a no-fault warm segment is good, not grounds to manufacture churn.

Single-sequence Qwen is the initial supported production scope. Soak sequential
requests, queued overlapping clients/cancel, reuse, MTP reject, checkpoint/restore
and restart with resource plateaus. Unsupported simultaneous multi-slot paging
is a separately labeled limitation, not a requirement to turn this single-slot
project into multi-tenancy before 256K can be assessed.

## F. Evidence product and review

Each campaign emits small summary JSON/Markdown and durable raw pointers. Required
row states distinguish pass, quality mismatch, runtime fault, invalid setup,
unsupported, incomplete and not measured. Missing/NaN metrics are null plus
reason, never zero. All hashes include a stable bundle identity; dirty builds
can be diagnosed but cannot masquerade as a clean reproducible release.

The phase-21 consolidated summary contains a provenance table, 256K allocation
and occupancy proof, per-context/per-question speed curve, matched CPU/GPU ratios,
quality/reference results, MTP rows/bytes/backend, direct-route fraction,
host/resident/selected page evidence, transfer/wait breakdown, physical soak,
and ranked measured bottlenecks. It lists exact commands and restart/resume
instructions with secrets excluded. Review only this summary and targeted raw
pointers, not every historical acceptance gate. If the goal remains unmet,
create implementation tasks for actual gaps, repeat these benchmark tasks,
and schedule the same benchmark-only review for that new phase.
