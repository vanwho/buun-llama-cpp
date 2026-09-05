# Execution cookbook for the post-17 packets

This supplies command/fixture/evidence detail; it does not add new runtime
requirements or a second task runner. Read only the sections named by your
packet, once per cluster. The current task agent owns its experiments; never
execute these service/build commands concurrently with another task's run.

## C1. Locate, build and run the actual tests

From the project root, inspect instructions and the named source symbols with
`rg -n`, then read their implementations, not every old handoff. Preserve dirty
work. The existing development CUDA tree is `build-cuda`; a final immutable
bundle is a different artifact. Do not rebuild loaded DSOs underneath a live
measurement. Stop/quiesce its owned candidate or build a separate tree first.

Commands below are existing commands, not new interfaces:

```bash
git status --short
git branch --show-current
rg '^(GGML_CUDA|CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|LLAMA_BUILD_TESTS)' build-cuda/CMakeCache.txt
cmake --build build-cuda --target llama-server -j2
cmake --build build-cuda --target test-kv-pager test-kv-residency test-kv-policy -j2
ctest --test-dir build-cuda --output-on-failure -R '^test-kv-(pager|residency|policy)$'
python3 -m unittest discover -s tools/server/bench -p 'test*.py' -v
```

Select targets for the current change; do not run this whole list every turn.
Additional existing targets: `test-kv-attention-view`,
`test-kv-attention-execution`, `test-kv-attention-exact`,
`test-kv-attention-telemetry`, `test-kv-routing-summary`,
`test-kv-routing-retrieval`, `test-cuda-fattn-paged-turbo4`,
`test-llama-archs`, `test-server-mmproj-lifecycle`, `test-arg-parser`.
Their sources are in `tests/` with the corresponding `.cpp` or `.cu` basename.
The CUDA fixture is implemented in `tests/test-cuda-fattn-paged-turbo4.cu`;
the production paged kernels are in `ggml/src/ggml-cuda/fattn.cu`.

`ctest -N` lists registrations; it does **not** prove binaries were built.
At this audit, the CUDA paged fixture was registered but missing in build-cuda.
Build that named target before running it. A missing executable is not a test
failure in the pager algorithm. If the target is absent, inspect
`tests/CMakeLists.txt` and configure the selected build with CUDA/tests enabled,
using `native` or the actual chosen device architecture, not a production GPU
constant. A long healthy compile continues with progress, not a 240s timeout.
For modified GGML operators, extend `tests/test-backend-ops.cpp` and use the
binary's `--help` to select the relevant operation/backend; do not mislabel an
ADD-only probe as attention coverage. Expensive live tests get a separate run.

## C2. Site launch boundaries and CLI facts

Only site execution/configuration may use these paths. Generic library/tests
take caller-supplied paths. Do not print authentication values.

```bash
sudo -n true
systemctl show llama-server.service -p MainPID -p ActiveState -p SubState -p NRestarts
python3 tools/server/bench/run-pager-profile-benchmark.py --help
python3 tools/server/bench/run-quality-corpus.py --help
```

The site environment boundary is `BENCH_ENDPOINT` (the authenticated target's
chat-completions URL), `CANONICAL_BENCHMARK_RUNNER` (the existing
`/srv/ai/benchmarks/run-profile-benchmark.sh`), `LLAMA_API_KEY_FILE`
(`/srv/ai/config/llama/api-keys`), `LLAMA_ACTIVE_PROFILE`
(`/srv/ai/config/llama/active-profile`), `PAGER_CORPUS` (the validated current
corpus file), and `BENCH_SERVER_BIN` (the **absolute** coherent candidate path).
Use actual current `current.gguf` realpath/model hash; do not hunt for another
Qwen model because the implementation architecture says Qwen35.

Current launcher example, for an owned, purpose-appropriate small diagnostic:

```bash
# Set PAGER_TEST_CONTEXT and PAGER_RESULT_DIR for this experiment first.
python3 tools/server/bench/run-pager-profile-benchmark.py \
  fast short "$PAGER_RESULT_DIR" --mode selective \
  --context "$PAGER_TEST_CONTEXT" --diagnostic --mtp native
```

The current adapter supports `--diagnostic` for sub-corpus-ceiling contexts.
That label means not the full corpus, not a failed development test. Do not
omit it then call the resulting ceiling rejection a model-capacity failure.
Avoid long variants until 18-02 fixes their guessed padding.

`run-quality-corpus.py --mode selected-all` is **a record label**, not a server
mode switch! It targets an already launched server. Likewise an exact label
does not activate exact attention. Verify live mode/forced selection before
scoring, or use the explicit model parity driver described in C3. The existing
launcher knows only `off|observe|selective|exact` and `native|off` MTP. New
resume/case-selection/force-route options in recipes are implementation work,
not options to invoke before they have been implemented and tested.

Generic server controls already exist in `common/arg.cpp`:
`--kv-hot-pages N|auto`, `--kv-vram-budget SIZE|auto`,
`--kv-host-budget SIZE|auto`, `--kv-pin-recent TOKENS|auto`,
`--spec-draft-kv-device auto|gpu|cpu`, plus four cache-type options.
The site adapter does not automatically forward every server flag. 18-01 must
provide an allowlisted explicit override path for small forced-hot experiments;
then verify effective argv/metrics. Do not assume setting an unused env var
changed the hot capacity. Check CLI help and record the exact syntax selected.

Retain a healthy candidate; restore exact runtime identity only on declared
control/recovery. Model/service telemetry may be authenticated even when health
is public. Do not use a raw unauthenticated curl then conclude metrics are absent.
Never touch 8092, alter another task's branch or kill generic Python processes.

## C3. Minimal fixtures and the missing model-parity harness

These are **test-only** sizes; derive production geometry/budgets at runtime.
Use the smallest fixture exposing the bug, never the final six-point curve.

| Fixture | Input / assertion |
| --- | --- |
| F1: transform | One/two 128-value Turbo blocks, deterministic nonuniform Q/K/V, actual encode/get-rows/dense FA; same-domain logits and exactly-once inverse transform |
| F2: mapping | Full-attention model-layer IDs `[3,11,23]`, compact ordinals `[0,1,2]`; page mapping `[0,1,2,3] -> [2,0,3,1]`; last page has 17 valid rows; verify each layer/row/head byte address |
| F3: causal queries | Query counts 1/2/3 and a prefill tile; native position per query, page-boundary crossing, future rows masked, one all-masked partition |
| F4: physical pressure | A small model diagnostic around 2K–4K context if supported, at least seven occupied pages and a test hot cap of four; pin budget must leave an evictable slot; cold round-trip exact bytes and valid continuation |
| F5: rollback | Proposed block length three, accept 0/1/2/3, then continue; committed target/recurrent/MTP/host frontier agrees and rejected host rows are absent |
| F6: client resume | Three cases on a local fake HTTP server; interrupt after the second completed record, resume without duplicate completed rows; alter provenance and reject reuse |

F2 is a synthetic mapping test, not hardcoded Qwen layer geometry. F4 alone
is not sparse-quality proof: all-selected parity requires all logical rows
selected, or an exact cold-wave comparison if they cannot all be resident.

18-05 owns a real model-parity driver. Existing CUDA fixtures have `main()`
without model arguments and do not supply per-layer live parity by magic.
Preferred implementation: add one justified, optional model-backed test target
`tests/test-kv-pager-model.cpp`, built with the existing `llama_build` helper
and internal include patterns. Follow model/argument loading in
`tests/test-state-restore-fragmented.cpp`, but do not copy its multi-slot
semantics. Do not make normal CTest require a downloaded 27B model or server.

"Optional" means opt-in model-backed execution outside default CI, not an
optional task checkpoint: 18-05 must implement and run it (or extend a proven
equivalent already present). Start with MTP off and the existing reference/
single-query direct shapes. 18-06 adds native-MTP controls, 19-04 multiquery
direct, and 19-06 cold exact execution. Until those owners implement a mode,
report it unsupported; don't turn the driver's eventual interface into a
circular requirement to implement every later feature in 18-05.

Driver contract (new private test options, to implement, not current CLI):
load caller's GGUF; accept the same saved token-ID prefix; select
dense/reference/direct/exact and MTP policy explicitly; set forced page lists
through internal test seams; capture bounded named tensors and logits at chosen
positions; write machine-readable max-absolute/RMS/relative errors and first
divergence with shape/domain/mask/page metadata. Run contexts sequentially and
reuse model weights when safe so comparisons do not double GPU weight memory.
No arbitrary debug callback may override an existing user callback silently.

First validate one prefix using dense versus dense with identical settings.
Then freeze tolerances appropriate to that measured numeric baseline and
existing backend tests before measuring the candidate. Exact byte movement
uses byte equality. Existing constant CUDA fixtures use ~2e-6 output and 1e-5
mass tolerances; retain those fixtures' limits. Do not impose those constants
blindly on full-model logits or loosen them after a candidate fails. Report
argmax margin when a tiny numeric change flips a near-tie; large layer errors
are not explained away as greedy ties. Raw token IDs, not re-generated text,
define the identical prefix.

## C4. Small receipts, common provenance and validator ownership

18-03 implements shared receipt/case validation in the existing
`tools/server/bench/pager_benchmark_contract.py`, extending existing Python
tests. Do not create a new incompatible schema for each later task or overwrite
legacy manifests. Introduce a separately tagged schema, e.g. `pager-evidence-v5`.
17-16/18-01/02 may emit this simple envelope before the validator exists;
18-03 validates those receipts without changing their recorded results.

Minimum envelope (example structure only, never copy illustrative pass values):

```json
{
  "schema": "pager-evidence-v5",
  "schema_version": 1,
  "task_id": "<id>",
  "result": "pass|fail|not_run|incomplete",
  "kind": "implementation|benchmark|review",
  "procedure": "<generic-procedure-name>",
  "provenance": {
    "source_commit": "<sha>", "source_diff_sha256": null,
    "bundle_manifest_sha256": "<sha-or-null>",
    "model_sha256": "<sha-or-null>", "tokenizer_template_sha256": "<sha-or-null>",
    "corpus_sha256": "<sha-or-null>", "config_sha256": "<sha-or-null>"
  },
  "checks": [{"name": "<invariant>", "status": "pass|fail|not_measured", "raw_ids": []}],
  "raw_index": [{"id": "<id>", "path": "<durable-path>", "sha256": "<sha>"}],
  "measurements": {},
  "failure": null,
  "resume": {"command": ["<argv-elements-no-secrets>"], "next_case": null},
  "implementation_owner": null
}
```

`failure`, when present: class, fingerprint, minimized raw ID, changed variable,
observed result, next falsifiable hypothesis and owning task. Measurement rows
carry units, status and reason for null. Each process/config epoch is separate.
Use actual reported target/draft codecs, rows/backends, host/resident/selected
counts and route fractions; don't fill missing fields from requested flags.
Provenance-only receipts may have null model/bundle fields with reasons when
no live run is required. Append future phase-specific tables inside this common
envelope rather than changing its meaning. Review can include its documented
additional fields without forcing a build or live run.

Phase-21 summary tables and phase-22 review fields named in those packets are
additional top-level fields on this envelope; use the same schema/version and
validate them according to `kind`/`procedure` (for example parity, quality,
curve, controls, soak, summary or review), never hardcoded project task IDs
in generic benchmark code. `reviewed_phase` is a string such
as "21". A benchmark receipt's `result=pass` means its measurement procedure
is valid, not that quality, speed or the overall goal passed. Goal claims have
their own status and raw IDs. Store reasons for null provenance in a
`provenance_unavailable` map. Never put the receipt's own hash in its raw index;
the consumer/handoff can hash the completed receipt without self-reference.

Layer/kernel tasks' handoffs must export: new/changed type and symbol names,
ownership/lifetime, shapes/strides/domains, callback order, runtime switch and
tests. Later packets read that handoff rather than rediscovering those APIs.

## C5. Checkpoint and stop semantics

Each task's local checkpoints are small implementation steps, not new runner
tasks. A passing compile or fake test advances a checkpoint but does not mark
the task done when its live checkpoint is still required. Keep pending work
and the next exact command in a short handoff; don't append another history
essay on every retry. A failed attempt must change a causal variable or isolate
a smaller reproducer before repeating the expensive command.

The outer runner owns Git. State commands below operate only on the current
task and preserve other records; do not use them to bypass dependencies:

```bash
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp python3 /srv/wiretail/task_state.py validate
# Replace TASK_ID and SUMMARY; checkpoint only after updating the compact handoff.
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp python3 /srv/wiretail/task_state.py checkpoint TASK_ID --summary SUMMARY
```

When all task acceptance is actually met, create the handoff, then use the
helper's `complete TASK_ID --summary SUMMARY`; do not claim the placeholder
command ran. For a genuine exhausted blocker use `block TASK_ID --reason REASON`.
Do not use `defer` to clear an implementation gate. Benchmark tasks may complete
with negative measurements as their packets specify; that is not overall-goal
completion. Required unmet work must have a remediation owner, not disappear.

If an unexpected prerequisite is part of the current implementation scope,
fix it here first. If an assessment genuinely splits it out, add the new packet,
cluster and dependency while keeping the current task first unfinished under
the helper's invariant; do not naively insert an unfinished dependency before
an in-progress task. A substantial reordering requires the runner's assessment
to update statuses/current pointer consistently, not a blind JSON append.
