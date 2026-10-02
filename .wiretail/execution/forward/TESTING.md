# Minimal testing and honest performance measurement

Revision: `hotpath-v10-20260914`. Amendment: `gpu-execution-101-20260930`.
Current next implementation procedure is PREFILL101 and tasks101-13–18 plus101-12h after
the101-12g measurement. Completed phase100/101 packets are provenance, not startup
context.101-18 owns the next canonical measurement,101-12h its review;102 owns
capacity only after the accepted release or its explicitly scheduled successor.

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
- Historical corrected live-semantics coordinate: L=8192, H=4096, page=256,
  one slot. Phase 102's repo-content benchmark instead uses its explicit
  L=16384/H=8192 fixture;
  R/G derive from each invariant. Ring test may choose G=256 with R<=H-G.
- Use B=1024/U=256 for all new comparable speeds, target/draft Turbo4, native
  GPU MTP, draft-n-max=2, temperature=0, reasoning/thinking off. These are test
  settings, not production constants. The legacy 93-11n campaign is deferred;
  its B128/U64 results are diagnostics, not the new comparable speed baseline.
  Do not rerun it as a prerequisite. Final integrated replay/cancellation is
  owned by101-10; natural promotion by101-11, after GPU execution repairs.
  Missing100-04a receipt is not a gate on these implementation tasks.
- Keep fresh input speed probes <=16K tokens. In phase 102, use L/H=16K/8K,
  32K/16K, 128K/<=60K, then 256K/<=60K. The 60K value is only the upper
  bound for the two high-context test hotsets, not a production default;
  product code derives H/G from model/backend/memory and remains tunable.
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
smallest named request/campaign for that task. Most implementation probes use
256–1024 fresh tokens and <=80 output; the4K attribution row is reserved for
101-01 and materially affected retakes. Do not repeat all prior gates,
24/48-case matrices, ten-trial campaigns, 256K fills or the final context curve
per implementation. Builds that make progress are allowed to finish; avoid a
universal 240s bound. Long occupancy work is resumable with
candidate-identity-bound checkpoints and per-request journals, never an
in-memory loop whose partial work is discarded.

### Live benchmark setup: fail fast, do not manufacture false failures

Apply these rules to every live CUDA/MTP/pager benchmark, including all phase
102 capacity rows:

1. Resolve device arguments from the exact candidate executable before
   touching the managed service. `llama-server --list-devices` gives the
   identifiers accepted by `--device`; use an exact identifier such as
   `CUDA0`, or `auto` followed by a runtime-identity check. Never pass a generic
   backend word such as `cuda` as if it were a device ID. Validate the chosen
   identifier locally and fail before writing service overrides/restarting if
   it is not listed.
2. A one-token generated response is not an MTP activity test: it may finish
   without proposing any draft tokens. Do not fail setup or restore/reload the
   model because its draft counter delta is zero. Establish startup readiness
   from health plus candidate identity and startup allocation evidence for GPU
   Turbo4 MTP. Measure MTP draft/accepted counts on the task's real benchmark
   generations at the stated output budget. An optional readiness request may
   validate only HTTP/identity and is not a benchmark sample or acceptance
   gate.
3. Token/context preflight must model the entire exact request being sent,
   including all already-committed conversation turns, fixed fixture payload,
   selected tracked-repo chunks, chat-template overhead, output budget, and
   query/replay/MTP reserve. For a scaled repository workload, choose and
   freeze the deterministic scale selection during a zero-generation preflight
   before sending A1; preflighting only the unscaled A/B/A fixture is
   insufficient. If the full workload does not fit, adjust the deterministic
   scale selection to the largest safe prefix before any generation; do not
   discover this after sending a partial campaign.
4. A shared frozen selection is a resumable artifact. Re-running preflight
   with the same source/candidate/fixture identity must validate and reuse it
   idempotently. It must not fail just because the file exists or overwrite a
   different selection. A mismatched source, prompt, inventory, tokenizer,
   geometry, or candidate identity must fail before generation with an explicit
   mismatch and a new named attempt path. Add a regression that runs preflight
   twice against the same selection, verifies identical selected ranges and
   zero generated requests, then proves that an identity mismatch is rejected
   without modifying the saved selection.
5. Separate setup validity from performance evidence. Fix a malformed device,
   wrong profile, missing runner, or stale candidate identity at the setup
   boundary and repeat only the zero-generation validation / affected short
   request. Do not spend substantive retries on a configuration typo, a
   zero-token MTP denominator, or a preflight that omitted the actual scaled
   prompt. Once setup passes, freeze source/config and run the paired benchmark
   rows against that same candidate.

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
strict answer formatting. B=1024/U=256. Current MTP acceptance median floor is
40% for each of the three prompts; report drafted/accepted raw counts
independently. A sub-40% observed median remains an honest MTP goal miss, but
does not by itself invalidate a run or block the next capacity/frontier
benchmark after the scheduled release review. Continue to require verified
GPU Turbo4 draft placement, correct target/draft state behavior, and report
the MTP miss without relabeling it as a pass. Setup, identity, or correctness
failures remain blockers.

The final paired short campaign uses fixture-backed actual C>H, same rendered
prefix/final prompts, one binary/model and one B/U. Separate cold fresh ingest
from cached conversation reuse; cached input is never counted as fresh prefill.
Clear/reset via existing slot lifecycle between fresh rows. Measure retrieval,
promotion, checkpoint/replay, view preparation, kernel time, decode and
generation sealing separately. Record overhead without adding synchronization
to production: CUDA events in explicit profiling runs, async/readiness counters
in normal runs. Compare direct and packed using identical page IDs/bytes.

Minimum selected fresh-prefill goal is500 tok/s for each canonical prompt;
750 is preferred. A completed measurement below goal stays a failed finding.
101-12 must insert specific101-12a/b... repair/retest/review successors before
102-01 on a miss and update102-01 to depend on the actual new release task.
Preserve usage ledgers; create packet/cluster/proofs/contexts and validate
ordering before completing the measurement. A truthful goal_miss receipt can
complete the benchmark only if those runnable successors exist. It cannot
authorize scale. No universal optimization-attempt count or unchanged audit.
Each successor has one measured cause, explicit code owner/algorithm, focused
regression and minimum retake. Never reopen the exhausted100-04a task.

Ordinary CPU-main-KV and dense-GPU controls must report actual codec/route.
Keep Turbo4 wherever the implemented control supports it; if ordinary CPU
Turbo4 attention is unsupported, report that typed capability boundary and
measure the existing original supported CPU-offload control with its codec
explicitly labeled as a non-codec-matched baseline. Do not implement a new
CPU TurboQuant kernel for a benchmark, invent a Turbo4 speed, or relabel exact
host-streaming as ordinary CPU offload. Production selected target/draft remain
Turbo4. Phase 100-01 must resolve/control this before the campaign.

Prior live promotion is useful provenance, not an integrated parity proof.
101-10 tests the real server checkpoint/target/draft path with one loaded
model;101-11 observes natural promotion on the final implementation. The
oracle starts from the same pre-query recurrent state and final historical
map, not history recomputed under a different map. These proofs precede102.

## Scaling/final findings

Phase 102 order is fixed: first a regression-only occupancy-driver limit task;
then the repo-backed 16K/8K A→B→A baseline with GPU-resident, host-resident
target-KV, and selected placements; then 32K/16K; then L=128K/C=120,000 and
L=256K/C=250,000. Every
stage uses deterministic tracked Buun source/docs as input, with file hashes,
line ranges and rendered Qwen token counts. The 16K/8K placement comparison is
the only host-resident target-KV row; all rows keep model compute and full-L
Turbo4 MTP on GPU, and higher-context stages compare GPU-resident control only
when memory admission safely permits it, plus selected mode. Do not run CPU
target-KV at higher contexts.

For occupancy, extend one same-slot history from repository files in cached
incremental input chunks no larger than 16K fresh rendered tokens. Stop before
the measured reserve for the final query, output, replay and MTP verification;
record the exact reserve and last committed C. Do not allow context shift,
truncation or compaction. Allocation and occupied C are different proofs; do
not claim C=262144 from allocation or a partial frontier. At 128K and 256K,
target C=120,000 and C=250,000 respectively; reserve 11,072 and 12,144 tokens
for measured final query/output/replay/MTP requirements. Start selected mode
with H<=60,000 tokens (page aligned). If startup, request,
OOM or scratch admission fails, lower only H in page-aligned steps and retry;
do not change L, B/U, batch, ubatch, model, MTP, codec, or other server
settings to make the test pass. The final 20K/40K/60K/100K/175K/256K speed
curve remains a later findings-only experiment, after the architecture works.

The 20K/40K/60K/100K/175K/256K speed curve is a final reporting experiment
after the architecture works, not an implementation gate. Prefer samples from
the single scaling frontier; do not refill L six times. H remains admitted,
never equals occupied C by assumption. YaRN beyond 256K remains a stretch goal.

## Current repair and receipt protocol

100-02 was prematurely marked done without a receipt; it is deliberately
deferred, preserving its source work and usage. Current owners are100-02a/b
admission/ring,100-02c hot-path costs,100-02d MTP,100-02e natural promotion,
100-02f paired speeds.100-03 reads those results;100-04 releases scaling.
The original36-request ceiling is retired.36 is the number of VALID rows in
a completed three-placement campaign, not a lifetime recovery budget.
Use one short selected smoke before completing selected12 and control12+12;
only failed/missing rows retry within an identical candidate. EOG is normal.

For each new task, receipt path is `evidence/V10_<task>.json` under execution.
Use schema_version=1, matching task ID, full source_commit plus actual build/
dirty identity. `checks` maps EVERY required_proofs key to status=pass,
exit_code=0, actual executable argv in `command`, and nonempty `artifacts`
with path and SHA256. A fixture/schema check cannot claim a live proof.
Measurement_complete and goal_status are different fields: a completed slow
benchmark can pass measurement while goal_status remains miss.
Keep receipt/handoff compact; raw requests/logs remain under the task raw root.

Before the state completion transition run the exact command:

```bash
python3 .wiretail/execution/v10/validate.py --task TASK_ID --receipt .wiretail/execution/evidence/V10_TASK_ID.json
PROJECT_ROOT="$PWD" python3 /srv/wiretail/task_state.py validate
```

Replace both TASK_ID placeholders with the CURRENT task. Only after both pass:

```bash
PROJECT_ROOT="$PWD" python3 /srv/wiretail/task_state.py complete TASK_ID --summary "verified named proofs and receipt"
```

Replace TASK_ID with the current task here too; the CLI advances current_task.
Do not manually mark done first and fabricate a receipt on retry. A deliberate
planning correction preserves usage and immutable historical evidence. Use
`reconcile-current` for a stale derived pointer, then validate again.

When future work is necessary, schedule an actual repair+retest before its
consumer and move/defer the original incomplete measurement to that owner.
Never leave the current test waiting for a later task that cannot be reached.
Task status deferred is not a passed proof and never releases a consumer that
explicitly requires that proof; update dependencies to its replacement owner.
