# Benchmark protocol V6 — short iterations, honest cold-prefill and hot-pager results

> Historical phase25 reference. For unfinished tasks26-01 onward, use
> PHASE26_INTERACTIVE_KV_STRATEGY.md and BENCHMARK_PROTOCOL_V7.md. They replace
> this document's256K-first/six-coordinate campaign and ambiguous B notation.
> Current campaign maximum is131072; primary speed fixture8192/4096.


Effective 20 26-09-11 for tasks 25-02 onward; supersedes V5 where scope/cadence
conflicts. All site-specific settings remain under /srv/ai or execution
metadata. This file is a contract to implement, not a claim that every new
driver flag already exists. Read the current task and --help first.

## A. One reusable driver and four small suites

Extend tools/server/bench/run-final-curve.py and reuse prompt_sizing.py,
pager_benchmark_contract.py and the profile adapter. Do not create disposable
here-doc Python loops for every task. The current curve runner buffers the
whole SSE response, unconditionally does a full-length warmup per question,
defaults to three measured trials, and does not actually skip completed
matching cases on resume. Fix these before expensive campaigns.

Proposed interface (25-02 implements and tests; names may change once, record
the final command in handoff and update this document):

- --suite micro|pressure|curve; micro is the explicit development choice.
- --prompt-tokens N separately from --context N; --question-index 0|1|2
  repeatable; pressure uses explicit test-only hot-budget configuration.
- --warmups 0 by default for model prompt runs, --trials 1 initially;
  one separate <=256-token kernel warmup per new configuration is sufficient.
- --max-tokens 128 initially, up to 256 for final curve; reserve at least 512
  context tokens plus any documented extra lookahead, and measure EOS honestly.
- --resume, explicit case manifest/keys, append+flush partial SSE and records.
- --startup-timeout, --progress-idle-timeout, --decode-idle-timeout,
  --total-timeout with progress-aware extension only under a declared budget.
  A no-total-limit setting must be explicit; never infinite silent waits.
- --cache-condition cold-prefill|live-continuation. Cold-prefill guarantees
  zero reused prompt rows; clear/recreate just the owned slot, not the service.
- --prefill-policy and selected/hot budget are recorded from actual runtime,
  not inferred solely from requested flags.

The final curve coordinates are context capacities 20480, 40960, 61440,
102400, 179200, 262144 (the historical shorthand 20K/40K/60K/100K/175K/256K).
Record exact prompt occupancy, which is slightly smaller due to generation
reserve and template fitting. Do not claim 262144 input tokens plus output fit
in a 262144-total-token context.

Suites/cost:

| Purpose | Initial work | Expand only when |
| --- | --- | --- |
| Micro | One original question, 2K/4K/8K occupied at <=16K logical, 128 output; first use 2K pair | Need a slope/noise decision, not every source patch |
| Pressure | 4–8K occupied, test H about 2K rows or another budget that leaves write/transfer slack; <=16K logical + matching GPU MTP | Need more pages to isolate an actual transfer defect |
| Implementation checkpoint | One all-fit and one genuine pressure request, changed source hypothesis | A measured regression requires one minimized repro |
| Final curve | Three original questions x six capacities x ONE cold-prefill request each | Final reports need uncertainty on a particular noisy point |
| CPU control | One short matched all-fit and one short matched pressure point | Longer control is affordable and materially changes a conclusion |

A 2K H is a test fixture, not a production hot-count default. If recent/append
pins cannot fit it, increase this diagnostic H just enough and record why.
Do not shrink the final logical target to solve a paging defect.

Default development request budget: roughly 120 seconds initially, progress
report every 5–10 seconds. If the baseline is extremely slow, record a
partial/censored 2K result and profile a few page waves; do not require that
slow run to finish before implementing the source-proven fix. Extend a useful
compilation/process wait with tool polling; don't rerun the command.

## B. Placement and same-runtime identity

Use /srv/ai/benchmarks/run-profile-benchmark.sh via the existing adapter for
site launch. Before first activation, verify sudo -n true. MainPID must come
from the named service, not broad pgrep. Freeze executable and its mapped
project DSOs together, using the 25-01 bundle helpers. Match the model's
realpath/hash, tokenizer and template, effective configuration and source diff.
Hash the large model once per unchanged identity, cache validated fingerprints;
do not read many GB before each request or mix binaries between A/B rows.

Pass existing LLAMA_API_KEY_FILE, BENCH_ENDPOINT, LLAMA_ACTIVE_PROFILE,
CANONICAL_BENCHMARK_RUNNER and BENCH_SERVER_BIN boundaries. Do not print key
values or environment dumps. The active service on revision day used an older
binary with MTP off; re-read current identity. A healthy label is not the target.

Keep the successful candidate loaded. One owner controls service transition,
startup, request cancellation and explicit failed-start recovery. Verify 8080
and 8091; leave unrelated 8092 alone. Retain stderr/core paths before cleanup.
Stop systemd restart storms; never kill a process by /dev/fd/3, Python basename
or an unvalidated PID. A client timeout must cancel the owned request or keep
its durable ownership record, not leave a mystery GPU workload.

A single fresh bundle per changed candidate/configuration family is enough.
Do not recreate/hash/deploy it between compatible request variants. Reuse model
weights or controlled context recreation where supported. Account for startup,
tokenization and packing separately from measured model prefill.

## C. Mandatory small receipt; optional research telemetry

Use one new `pager-speed-v6` experiment schema with schema_version=1.
Extend validators with a speed-specific entrypoint instead of requiring the old
quality/soak envelope. Historical V5 rows remain immutable and valid as history.

Required to accept a speed result:

- task/experiment/case ID, timestamps, complete/partial/error status;
- source commit + dirty-diff fingerprint, executable/DSO bundle, model/template/
  tokenizer/config hashes, GPU/driver/build and effective runtime placement;
- actual L, prompt tokens admitted/processed, cached/reused rows, effective B
  and CUDA tile, generated and committed output tokens, MTP proposal/acceptance;
- target and MTP K/V types Turbo4, target GPU compute, MTP GPU residency and
  actual allocated rows equal resolved context; actual H and per-layer A;
- target GPU allocation bytes, host committed rows/bytes, pinned/ring bytes;
- wall prefill and decode timing/count definitions; end-to-end latency/TTFT;
- real host publication/movement evidence for pressure claims.

Optional diagnostic fields: actual CUDA capture/update/launch counters,
stage CPU durations, CUDA events/NVTX, layer microseconds, PCIe copy calls and
bytes, resource series, individual token latency. Missing optional profiling
does NOT erase valid measured throughput. It is null with reason, never zero.
Missing codec/placement/request identity means invalid for the claimed target,
but retain the diagnostic result. Do not accept hot_tokens containing bytes.

Distinguish target ordinary CPU-KV, target paged + GPU MTP, feature-off GPU,
observe, packed-Turbo and direct-Turbo routes. Observe is instrumentation on
ordinary attention, not proof of bounded GPU storage or sparse page use.
A/B paired inference must use the same model, token IDs, occupancy and settings
except the declared variable; different H/A configurations are a tradeoff, not
an apples-to-apples kernel comparison.

## D. Timing definition and profiling overhead

Instrument the actual stages, not ambiguous counter names:

1. CPU admission, dirty sealing/publication, summary update, routing and
   graph-build/launch spans.
2. CUDA target matmul/GDN, target attention, target KV write, D2H, H2D,
   pack/relocation, summary scoring, native-MTP prefill/draft/verify.
3. API boundaries: tokenize start/end, request sent, first output token,
   stream complete. Source of each timing must be explicit.

Use CUDA events or a SHORT nsys CUDA/NVTX trace (installed); do not
cudaDeviceSynchronize after every measured node. Sample ncu only for one
kernel/shape; do not profile whole long runs. Cross-stream durations overlap:
report critical-path wait separately, never sum all stage durations into wall
time. Backend-neutral graph_rebuild counters retain their real names/scope;
actual CUDA graph counts are sampled at the ggml CUDA backend capture/update/
launch sites. Disable tracing for the headline run.

TTFT is request-sent to first actual output token (including reported
reasoning if enabled); no buffering entire SSE before timestamping. SSE
events may contain multiple tokens, especially under MTP. If per-token IDs/
timestamps are unavailable, report chunk gaps as chunk gaps, not token p99.
Server tg/s and end-to-end committed tok/s both matter; exclude proposals and
draft rejects from committed throughput. Prompt pp/s includes target+native
draft catch-up when reporting whole-request prefill; report target-only time
separately if the native server counter omits draft work.

Do not label warm prefix reuse “prefill throughput.” Cold-prefill cases must
show zero cache reuse. A live followup is a separate incremental decode case.
An interrupted 11K population sample is not a completed 256K pp/s result.
Preserve actual processed tokens and time as partial slope/ETA only.

## E. Dynamic sizing, OOM and deadlines

Three independent quantities are logical L, actual occupied tokens and hot
memory H (plus attended A). Native MTP follows L, NOT H. Reserve weights,
GDN, full-context draft KV, actual graph/scratch, ring/packed buffer and safety
headroom before admitting H. If a full-L launch OOMs, reduce H, B or transient
scratch first while holding L and MTP fixed; don't automatically fall back
to 16K and call it the milestone. Discover safe B with short generations.
Memory estimates must include both model and native-MTP graph buffers.

Token fitting uses the model's exact chat renderer/tokenizer. Preserve the
question, template and final generation reserve; do not tokenize rendered text
then wrap it again. Reconcile request usage to preflight, allow only documented
special-token differences, never HTTP400 as a speed sample.

Separate readiness timeout, prefill-progress timeout, decode idle and campaign
wall budget. A 262K run at 1000 tok/s already needs ~262 seconds before
generation; a fixed 180/240-second wrapper is invalid. Derive a generous ETA
from recent processed-token slopes and the selected algorithm; set and record
an appropriate budget. Re-estimate at chunk checkpoints.

During final population inspect monotonic progress every 10 seconds, not
blocking full-body read. If the next point projects beyond the experiment's
budget or shows a large unexplained slope collapse, stop after a recorded
page-wave boundary, retain stage evidence and minimize the new defect.
This is `incomplete_budget`/`performance_regression`, not a valid zero speed
or an accuracy failure. Do not reset the timeout and retry the identical
unmodified multi-hour request three times.

Readiness at 262K is a short allocation smoke; occupied proof is an actual
request near capacity with at least 512 generation-reserve tokens. One
completed prompt + useful generation is sufficient for this prototype.
Full-MTP occupancy must be verified too, allowing documented one-row pending
boundary semantics, not merely a 262K allocation with an empty draft cache.

## F. Resume contract and final campaign

Case key includes exact input IDs/prompt hash, all provenance, L/H/A/B,
mode/route, MTP configuration, sample settings, cache condition, trial and
suite revision. Append and flush raw SSE/progress/results; update the
manifest atomically. Resume skips only matching completed keys. Partial
requests are not in medians. If a request is still running, reattach by its
verified ownership or cancel before resubmission.

Raw checkpoints do not magically resume compute. Persisting target KV alone
is insufficient for this hybrid model: GDN state, GPU MTP KV/frontier and
pending hidden activation must match. Production checkpoint recovery is out
of this revision's critical path. Prefer keeping the live request/process
owned while monitoring from later agent turns. If interrupted beyond recovery,
restart only the affected case, not the whole suite; say explicitly that its
prefill work restarted.

Phase 26:
- 26-01 runs the 262144 allocation smoke, then original question 0 at near-full
  occupancy as a single genuine pressure-capable long result.
- 26-02 measures two bounded paired control points, not a six-point CPU matrix.
- 26-03 records final six-coordinate/three-question curve. Reuse 26-01's
  question-0 result only with an identical case key; do not prefill it again
  simply because the evidence filename/task ID changed.
- Initial complete curve has 18 results, no compulsory long warmups or ten
  repetitions. Different questions have different outputs; display all three
  plus range/median, not a confidence interval from three unrelated samples.
  Repeat one point only if noise materially changes a design decision.
- Final experiments are staged sequentially with one request/slot. Synthetic
  kernel repetitions are cheap and separate from model prompt repetitions.
- Once a run works, keep its candidate loaded and retain a concise next command.

## G. Minimal correctness retained; quality options deferred

For a changed attention kernel: raw Turbo4 roundtrip/domain, causal tail/native
position, GQA, bounded numerical comparison with the existing oracle; then a
small live Qwen model check. Byte transfers require exact equality. Dense
versus sparse free-running answers need not match; selected-all equivalent
math should agree within pre-established numerical tolerances.

For pressure: first deterministic known-cold transfer with page ID + content
version + host bytes + H2D fence + destination generation + consumed route.
Then one natural query-driven recall after 25-11. Answer content is diagnostic,
not an exact-string gate. Counted transfer traffic must not be just metadata.

Disable extensive page-mass output, per-token checksum/logging, checkpoint
snapshots and idle-slot caches for headline speed. Record these settings.
Brief diagnostic toggles may isolate costs, but a headline result must retain
real Turbo4 host backing, GPU hot attention and full native-MTP operation.
Do not silently speed up by dropping MTP, evicting draft history, not storing
host KV, ignoring all old pages, or moving weights to CPU.
