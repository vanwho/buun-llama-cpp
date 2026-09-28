# Minimal testing and honest performance measurement

Revision: `hotpath-v10-20260914`. Amendment: `forward-turn-retrieval-20260927`.

## Setup once, verify cheaply, keep loaded

Use the existing incremental `build-cuda` tree and build only named affected
targets. Record source base SHA + dirty-diff hash, build flags, binary and loaded
DSO hashes, model SHA, endpoint owner PID/start time, effective argv, L/H/G/A,
B/U, codecs and MTP placement. Reuse the managed 8080 service only when these
match; otherwise reload through the current managed lifecycle. One Qwen process
at a time. Never touch 8092 or use `pgrep | head` to choose identity.

Known local bindings: `CANONICAL_BENCHMARK_RUNNER=/srv/ai/benchmarks/run-profile-benchmark.sh`,
`LLAMA_API_KEY_FILE=/srv/ai/config/llama/api-keys`, model resolved from
`/srv/ai/models/text/current.gguf`, and candidate via `BENCH_SERVER_BIN`.
Use existing credential-file readers and lifecycle lock; never print keys.
Use `sudo -n` for service actions. If it fails, inspect `id`, `sudo -n -l`,
executable command paths and process privilege restrictions and repair the
authorized launch path within the task. Do not switch to a CPU-only 27B run.
Retain a successful candidate loaded for the next task.

## Geometry

- Small implementation tests: real CUDA fixtures with 1–3 pages; synthetic
  tiny page sizes belong only in deterministic tests. CPU float math is an
  oracle, not CPU Turbo4 inference. A CPU fixture refusing TurboQuant execution
  is a known backend boundary, not a host-storage failure.
- First corrected live semantics: L=8192, H=4096, page=256, one slot;
  R/G derive from each invariant. Ring test may choose G=256 with R<=H-G.
- Use B=1024/U=256 for all new comparable speeds, target/draft Turbo4, native
  GPU MTP, draft-n-max=2, temperature=0, reasoning/thinking off. These are test
  settings, not production constants. The legacy 93-11n campaign is deferred;
  its B128/U64 results are diagnostics, not the new comparable speed baseline.
  Do not rerun it as a prerequisite. Phases 94 and 95-01/02 are complete;
  95-03's partial changed/unchanged query proof is historical evidence, not a
  gate on 96-01 through 100-03. Final integrated replay/cancellation/promotion
  is owned by 100-04 after the short performance iteration and before scaling.
- Keep fresh input speed probes <=16K tokens. For local tests and scaling,
  physical target hot capacity H <=49,152; L may grow to 128K/256K independently.
  Product code derives H/G from model/backend/memory and remains tunable.
- Reserve full-L draft, weights, GDN, catalogue, transfer slabs, graph/scratch,
  peak verify/replay usage and measured safety headroom before H. If OOM, stop
  only the owned restart loop, ensure one model is loaded, reduce H/R or packed
  workspace first, and keep L/full-L draft/B/U fixed. A changed geometry must
  be relabeled and controls rerun. Do not silently shrink an L-labeled result.

## Fast iteration

Each code edit gets only its named deterministic regression and affected
CUDA fixture. Do not launch, reload, or benchmark the managed Qwen service
unless the task's named proof explicitly requires a live request. A task whose
proof is only a unit/CUDA fixture completes from that executable proof; the
generic service-identity boilerplate is a safety rule for any live run, not an
instruction to create one. When a live proof is required, use at most the
smallest named request/campaign for that task. Do not repeat all prior gates,
24/48-case matrices, ten-trial campaigns, 256K fills or the final context curve
per implementation. Builds that make progress are allowed to finish; avoid a
universal 240s bound. Long occupancy work is resumable with
candidate-identity-bound checkpoints and per-request journals, never an
in-memory loop whose partial work is discarded.

### Build/test recovery ladder

When a build or test fails, preserve its complete command, exit status and raw
output, then classify it before retrying. Do not repeat the same command with
no changed cause.

1. **Configuration or target discovery:** inspect the existing build cache,
   generator, CUDA architecture selection, and `ctest -N`/build target list.
   Repair the local build configuration or target invocation; do not add a
   server-specific architecture to production CMake or expand to a full build.
2. **Wrong executable or live identity:** compare candidate binary, loaded
   DSO, model, process start time, command line and cache/MTP flags. Correct
   the launch/managed lifecycle and repeat only the affected short check.
3. **Fixture/backend mismatch:** use the backend that supports the behavior
   being tested. In particular, a CPU fixture declining TurboQuant execution
   does not fail CUDA Turbo4 behavior or CPU host-storage correctness. Keep a
   CPU math oracle separate from TurboQuant inference and run the named CUDA
   fixture for Turbo4 kernels.
4. **Source assertion or parity failure:** identify the first failing state or
   numerical boundary, add/fix the smallest regression at that owner, build
   only affected targets, and rerun that fixture. A passing receipt/schema
   validator never substitutes for the executable assertion.
5. **Resource/capacity failure:** preserve the actual frontier and allocation
   ledger. For a short implementation fixture, reduce only its scratch/page
   fixture geometry if the production invariant remains covered; for live
   Qwen, keep L, Turbo4 and MTP fixed and reduce admitted H/workspace according
   to the memory ledger. Relabel changed geometry and retry only the failed
   proof. Never convert allocation or partial occupancy into a pass.

After a materially new diagnostic/repair path, either produce the named proof
or leave the task resumable with the precise unresolved source boundary and
the next command/action. Do not mark a required proof `done`, `deferred`, or
`blocked` merely because the first build, fixture configuration, server
identity, or request failed. If the underlying behavior is a real defect,
repair it in scope or insert one source-directed dependency task and minimal
retest before consumers that require it; do not stop unrelated implementation
tasks that do not depend on that behavior. Keep the current task pointer equal
to the first unfinished task and run both state and V10 receipt validation
after any task/dependency insertion.

Use polling of forward progress for long ingest, but a slow progressing loop
is a diagnostic failure worth stopping after one representative chunk. Compare
fresh token count, route/kernel attribution, GPU utilization/power, copy waits
and CPU time; repair the cause before another long frontier. No indefinite
same-profile reruns. Config/identity/auth/missing-runner errors are repaired and
the affected row retried, not passed to the next task as not_measured.

## Canonical performance workload

Keep the exact final user prompts:

1. `write a python function that merges two sorted lists into one sorted list, with docstring.`
2. `explain the difference between mmap and read for loading large files, one paragraph.`
3. `write a bash script that watches a directory and prints new files as they appear.`

One warmup with 40 requested output tokens, then three measured requests per
prompt with 400 requested output tokens. EOG can shorten actual output; record
actual tokens and elapsed time. Never require exactly 400 generated tokens or
strict answer formatting. B=1024/U=256. MTP acceptance median goals are 75%,
40%,60% respectively; report drafted/accepted raw counts independently.

The final paired short campaign uses fixture-backed actual C>H, same rendered
prefix/final prompts, one binary/model and one B/U. Separate cold fresh ingest
from cached conversation reuse; cached input is never counted as fresh prefill.
Clear/reset via existing slot lifecycle between fresh rows. Measure retrieval,
promotion, checkpoint/replay, view preparation, kernel time, decode and
generation sealing separately. Record overhead without adding synchronization
to production: CUDA events in explicit profiling runs, async/readiness counters
in normal runs. Compare direct and packed using identical page IDs/bytes.

Minimum selected fresh-prefill goal is 500 tok/s for each canonical prompt;
750 is preferred. A completed measurement below goal stays a failed goal
finding. Phase 100-03 must schedule a concrete source repair and minimal retest
before capacity work. There is no universal number-of-attempts optimization
law, no shortening facts to fit a 16-token cap, and no repeated audit task.

If 100-03 inserts repair tasks, use task IDs `100-03a`, `100-03b`, etc. in
the state-list order immediately after `100-03`; set each dependency to the
previous repair task; preserve every existing task's usage ledger and proof
metadata; make `100-04` depend on the last repair/retest; and leave
`101-01 -> 100-04` unchanged. Add each packet, cluster membership and V10
proof requirement before completing 100-03. Keep `current_task` at the first
unfinished item (the runner/task-state completion transition advances it).
Run `PROJECT_ROOT="$PWD" python3 /srv/wiretail/task_state.py validate` and
`python3 .wiretail/execution/v10/validate.py --task 100-03 --receipt
.wiretail/execution/evidence/V10_100-03.json`; fix metadata/receipt structure
errors in 100-03 rather than leaving the next run to discover them. These
repair packets must contain one measured cause, one owner/symbol, one focused
regression and one minimal retest—not another full benchmark loop.

Ordinary CPU-main-KV and dense-GPU controls must report actual codec/route.
Keep Turbo4 wherever the implemented control supports it; if ordinary CPU
Turbo4 attention is unsupported, report that typed capability boundary and
measure the existing original supported CPU-offload control with its codec
explicitly labeled as a non-codec-matched baseline. Do not implement a new
CPU TurboQuant kernel for a benchmark, invent a Turbo4 speed, or relabel exact
host-streaming as ordinary CPU offload. Production selected target/draft remain
Turbo4. Phase 100-01 must resolve/control this before the campaign.

The 95-03 attempt-38 live run is useful partial evidence for a changed query
replay, a following unchanged query, and natural promotion. Do not rerun that
full campaign now. The matched one-pass parity, cancellation and final
selected-route check moves to 100-04, after the initial short performance
iteration and before any context scaling. Short-path speed results before
100-04 are diagnostic; unfinished parity must not block phases96-100-03.

## Scaling/final findings

Prove 32K/16K useful speed before 128K, then 256K. Allocation and occupied C
are different proofs. Grow C using cached incremental input chunks <=16K;
fresh-token accounting excludes prior history. Keep enough logical space for
query/replay/output; use a final no-output commit probe to measure full-L
occupancy if supported by the existing driver, and report the last actual C.
Do not claim C=262144 from allocation or a partial frontier.

The 20K/40K/60K/100K/175K/256K speed curve is a final reporting experiment
after the architecture works, not an implementation gate. Prefer samples from
the single scaling frontier; do not refill L six times. H remains admitted,
never equals occupied C by assumption. YaRN beyond 256K remains a stretch goal.
