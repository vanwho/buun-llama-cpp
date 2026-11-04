### Server benchmark tools

Benchmark is using [k6](https://k6.io/).

##### Install k6 and sse extension

SSE is not supported by default in k6, you have to build k6 with the [xk6-sse](https://github.com/phymbert/xk6-sse) extension.

Example (assuming golang >= 1.21 is installed):
```shell
go install go.k6.io/xk6/cmd/xk6@latest
$GOPATH/bin/xk6 build master \
--with github.com/phymbert/xk6-sse
```

#### Download a dataset

This dataset was originally proposed in [vLLM benchmarks](https://github.com/vllm-project/vllm/blob/main/benchmarks/README.md).

```shell
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

#### Download a model
Example for PHI-2

```shell
../../../scripts/hf.sh --repo ggml-org/models --file phi-2/ggml-model-q4_0.gguf
```

#### Start the server
The server must answer OAI Chat completion requests on `http://localhost:8080/v1` or according to the environment variable `SERVER_BENCH_URL`.

Example:
```shell
llama-server --host localhost --port 8080 \
  --model ggml-model-q4_0.gguf \
  --cont-batching \
  --metrics \
  --parallel 8 \
  --batch-size 512 \
  --ctx-size 4096 \
  -ngl 33
```

#### Run the benchmark

For 500 chat completions request with 8 concurrent users during maximum 10 minutes, run:
```shell
./k6 run script.js --duration 10m --iterations 500 --vus 8
```

The benchmark values can be overridden with:
- `SERVER_BENCH_URL` server url prefix for chat completions, default `http://localhost:8080/v1`
- `SERVER_BENCH_N_PROMPTS` total prompts to randomly select in the benchmark, default `480`
- `SERVER_BENCH_MODEL_ALIAS` model alias to pass in the completion request, default `my-model`
- `SERVER_BENCH_MAX_TOKENS` max tokens to predict, default: `512`
- `SERVER_BENCH_DATASET` path to the benchmark dataset file
- `SERVER_BENCH_MAX_PROMPT_TOKENS` maximum prompt tokens to filter out in the dataset: default `1024`
- `SERVER_BENCH_MAX_CONTEXT` maximum context size of the completions request to filter out in the dataset: prompt + predicted tokens, default `2048`

Note: the local tokenizer is just a string space split, real number of tokens will differ.

Or with [k6 options](https://k6.io/docs/using-k6/k6-options/reference/):

```shell
SERVER_BENCH_N_PROMPTS=500 k6 run script.js --duration 10m --iterations 500 --vus 8
```

To [debug http request](https://k6.io/docs/using-k6/http-debugging/) use `--http-debug="full"`.

#### Metrics

Following metrics are available computed from the OAI chat completions response `usage`:
- `llamacpp_tokens_second` Trend of `usage.total_tokens / request duration`
- `llamacpp_prompt_tokens` Trend of `usage.prompt_tokens`
- `llamacpp_prompt_tokens_total_counter` Counter of `usage.prompt_tokens`
- `llamacpp_completion_tokens` Trend of `usage.completion_tokens`
- `llamacpp_completion_tokens_total_counter` Counter of `usage.completion_tokens`
- `llamacpp_completions_truncated_rate` Rate of completions truncated, i.e. if `finish_reason === 'length'`
- `llamacpp_completions_stop_rate` Rate of completions stopped by the model, i.e. if `finish_reason === 'stop'`

The script will fail if too many completions are truncated, see `llamacpp_completions_truncated_rate`.

K6 metrics might be compared against [server metrics](../README.md), with:

```shell
curl http://localhost:8080/metrics
```

### Using the CI python script
The `bench.py` script does several steps:
- start the server
- define good variable for k6
- run k6 script
- extract metrics from prometheus

It aims to be used in the CI, but you can run it manually:

```shell
LLAMA_SERVER_BIN_PATH=../../../cmake-build-release/bin/llama-server python bench.py \
              --runner-label local \
              --name local \
              --branch `git rev-parse --abbrev-ref HEAD` \
              --commit `git rev-parse HEAD` \
              --scenario script.js \
              --duration 5m \
              --hf-repo ggml-org/models	 \
              --hf-file phi-2/ggml-model-q4_0.gguf \
              --model-path-prefix models \
              --parallel 4 \
              -ngl 33 \
              --batch-size 2048 \
              --ubatch-size	256 \
              --ctx-size 4096 \
              --n-prompts 200 \
              --max-prompt-tokens 256 \
              --max-tokens 256
```

### Pager profile adapter

#### Frozen pager-corpus-v4 contract

The portable contract is implemented by
`pager_benchmark_contract.py`. Generate the versioned corpus artifact with:

```shell
python3 tools/server/bench/generate-pager-corpus.py \
  <corpus.json> \
  --model <model-gguf-sha256> --tokenizer <tokenizer-sha256>
```

The checked-in fixture is `fixtures/pager-corpus-v4.json`. It contains real
facts inside 256-token logical-page fixtures, including tails, odd page counts,
and cold/focus/churn distances. For measured Qwen lengths, pass
`--tokenizer-command` and `--tokenizer-model`; the generator records the
tokenizer identity and measured count. It produces deterministic calibration
and held-out cases with immutable case, prompt, and corpus hashes. Validate a
corpus, dry-run, or acceptance manifest with
`validate-pager-benchmark.py`; schema 1 is accepted only as
`legacy/non-acceptance`, while the prior v2 and v3 corpora remain historical and
unknown schemas or missing runtime evidence fail closed. Acceptance manifests require model/tokenizer hashes, Turbo4 MTP
placement, the reconciled pager ledger, timing, raw requests, and all required
telemetry. The frozen gate values and fields are summarized in
the fork-local execution package.

`run-pager-profile-benchmark.sh` preserves the canonical profile runner and
adds a joinable pager/corpus envelope to its existing artifacts. It supports
the historical short and large runs plus `stable-focus`, `cold-needles`,
`focus-shifts`, and `churn` variants:

```shell
tools/server/bench/run-pager-profile-benchmark.sh <profile> <variant> <results-dir>
tools/server/bench/run-pager-profile-benchmark.sh <profile> <variant> <results-dir> --dry-run
```

The launcher exposes the live configuration explicitly. `--mode selective`,
`--page-size 256`, `--context derived`, and `--mtp native` are the defaults;
`--device` accepts the server's device list. Native MTP requires the Qwen3.8
fast profile and pins both target and draft K/V to Turbo4/GPU. `derived` asks
the frozen V4 corpus to supply the context ceiling (currently 22,016 tokens),
so the target KV, native draft KV, and large prompt use one resolved value.
Use `--context TOKENS --diagnostic` only for an explicitly diagnostic
sub-ceiling run; its manifest is never acceptance evidence. A dry run prints
the resolved context and acceptance/diagnostic mode.

The quality runner has the same boundary:

```shell
tools/server/bench/run-quality-corpus.py fixtures/pager-corpus-v4.json <results-dir> \
  --endpoint <url> --model <model> --mode selective --context derived
```

Long quality and canonical campaigns support `--resume` plus repeatable
`--case-id ID` and `--case-index N` selection.  Resume requires the existing
case manifest to match bundle/model/tokenizer/template, corpus, configuration,
mode, cache condition, sampling, context, and timeout provenance; only
successful completed case keys are skipped.  The shared case ledger writes
append-only `case-state.jsonl` and atomically replaces `progress.json` after
raw response/metrics artifacts are flushed.  The separate connect, startup,
prefill-idle, decode-idle, and total limits are configurable with the matching
`--*-timeout` options.  A total expiry is recorded as `incomplete_timeout` and
can be resumed without fabricating a zero-speed or wrong-answer row.

Before a campaign, run the bounded harness gate with `--preflight` (or use
`--mode preflight` for a standalone check).  It sends one warm, one cold-needle,
and one selected-all request, and refuses the campaign if route, page,
residency, movement, timing, placement, hot-page-budget, or occupied-token
telemetry is absent.  Every request record includes the effective argv,
client/server timestamps, actual occupancy, and the complete runtime envelope.
Use `--max-cases`, `--stop-after-failures N`, and `--fail-fast-capability` to
bound a diagnostic run.  Fail-fast checkpoints write remaining cases as
`not_run`; they are not counted as failures and can be resumed from
`campaign-checkpoint.json`.  `report.json`/`report.md` separate setup failure,
capability refusal, quality mismatch, timeout, valid measurement, and not-run
records.

`run-pager-soak.py <results-dir> --dry-run` resolves the same corpus ceiling
without contacting a service. Its live synthetic marker prompts are sized from
that resolved context. Lower contexts require `--diagnostic` and are recorded
with their odd tail and diagnostic-only provenance.

Successful live runs keep the tested profile loaded and write
`lifecycle-state.json`, including whether the health-checked service is usable
for resume. Use `--restore-control` only for an intentional control/revert
run; failed or interrupted runs restore the previous healthy profile. If the
canonical runner returns success but adapter validation rejects telemetry,
health, records, or identity, the adapter invokes the configured activator and
verifies the prior profile again. Cleanup is safe to repeat.

`canonical_exit_code` and `adapter_validation` are recorded independently, so
a canonical benchmark failure cannot be mistaken for a harness validation
failure. Failed manifests also include `failure_class` (for example,
`canonical_runner_failure`, `missing_runtime_telemetry`,
`runtime_identity_mismatch`, `authentication_configuration`, or
`restoration_failure`) plus the complete `validation_errors` list. Raw
`records.jsonl`, `run-config.json`, and lifecycle state remain available for
diagnosis.

The lifecycle manifest records a requested candidate identity and the observed
managed-service identity: profile, systemd MainPID, executable, model,
context, pager mode/page size, and native-MTP placement/types. Identity is
scoped to `LLAMA_SERVICE_NAME` (and never selects an arbitrary `llama-server`
process), so an unrelated service such as one on another port is not touched.
The run fails closed before measuring when the active candidate does not match
the requested binary or configuration.

The pager profile adapter explicitly passes `B=1024` and `U=256` by default;
use `--batch` and `--ubatch` together to request another matched geometry.
Measured records retain prompt and cached token counts independently. Fresh
prefill throughput is computed only when both are known, so cached prompt rows
cannot increase the fresh-token rate. `stage_accounting` records the supported
request-local stages; unavailable stages remain `not_exposed` rather than zero.
Ordinary counter accounting does not add CUDA synchronization. The compact
checker in `.wiretail/execution/forward/check_canonical_result.py` validates
result identity, finite measured rows, MTP counts, logical/physical byte
geometry, frozen-decode PCIe movement, and raw artifact hashes.

Live runs require `BENCH_ENDPOINT`, `CANONICAL_BENCHMARK_RUNNER`,
`LLAMA_ACTIVE_PROFILE`, and `LLAMA_PROFILE_ACTIVATOR`. Set
`PAGER_CORPUS` for the frozen corpus, and optionally set
`LLAMA_API_KEY_FILE` for authentication. If the
metrics endpoint is protected, `LLAMA_API_KEY_FILE` (or `BENCH_API_KEY`) is
required: an HTTP 401/403 is a launcher-configuration error, not
`not_configured` pager telemetry. The adapter reads the key locally and never
places its value in output or manifests. These machine-specific paths are
deliberately not supplied as defaults; a dry run remains usable without them
and records the boundary as `not_configured`.

For a managed profile runner, use `BENCH_SERVER_BIN` to select the candidate
server binary. This keeps the source tree portable while ensuring a live retry
does not accidentally validate an older deployed executable.

The adapter reads pager telemetry directly from the server's `/metrics`
endpoint. A live run fails if required pager fields are absent; it never
substitutes `PAGER_*` environment values for runtime measurements. A dry run
remains explicitly `not_configured`. A live run delegates
profile activation, clean isolation, interruption summaries, and restoration
to `run-profile-benchmark.sh`, then records pre/post profile, PID, command,
health, and running-service snapshots. A failed restoration or post-run health
check is an error.

Pager samples use the names `llamacpp:kv_pager_<field>`; `mode`, `route`,
`mtp_backend`, target type, and realized native-MTP type are represented as
labels. The snapshot includes page/byte ledgers, epochs, attention and graph
counters, publish/queue and transfer timings, transfer bytes, waits, host
residency, and routing configuration. `attention_publish_time_us` is the
runtime source for the adapter's required `queue_us` field. Native-MTP labels
come from the draft context's measured allocation; `not_present` and
`unsupported` are preserved as explicit states and are never encoded as
numeric zero. Epoch fields are the generation boundary for consumers that need
to reconcile multiple scrapes. The adapter preserves the raw `mtp_rows`,
`mtp_bytes`, `mtp_type_k`, `mtp_type_v`, and `mtp_backend` fields in its pager
envelope in addition to its normalized evidence fields.

Calibration controls and historical short/large runs are not acceptance
results. Held-out acceptance requires the immutable corpus and all runtime
telemetry. Exact and selective results are reported separately, including
failed, fault, cancellation, and churn cases; missing telemetry fails closed.
## Promotion campaign setup (explicit geometry)

`run-pager-promotion.py` requires `--context` and `--hot-pages`; no hidden
8K geometry is assumed. They bind argv identity, allocator checks, prompt
sizing and reports. B/U remains 1024/256, page size 256, CUDA0, Turbo4 target
and GPU native Turbo4 MTP. `--selector-trace-page` is optional diagnostic
instrumentation, not a normal speed-test prerequisite. Use the repository
fixture root, not an assumed site path. Local fixture validation precedes
service access. Print the exact matching server argv without launching it:

`python3 tools/server/bench/run-pager-promotion.py --output /tmp/promotion-command-only --server-binary "$PWD/build-cuda/bin/llama-server" --model /srv/ai/models/text/current.gguf --context 16384 --hot-pages 16 --print-server-command`

Reuse a healthy matching binary/model/DSO/config, otherwise replace the ONE
managed model owner through the established lifecycle and validate again.
Do not start another model, silently adopt its geometry, or send a generation
"setup probe" merely to check counters; a short acknowledgement can correctly
have zero drafts. Setup is established by identity, health and allocator
evidence; MTP acceptance comes from completed measured responses.

## Active comparison completion — historical results count

The [numbered gap-only plan](../../../.wiretail/execution/benchmarks/COMPARISON_COMPLETION_20261104.md)
and [frozen machine configuration](../../../.wiretail/execution/benchmarks/COMPARISON_COMPLETION_20261104.json)
pin the server commands, binaries/libraries, 32K repository chunk map, and exact
missing prompt/mode selections. Complete historical cells are reused at their
recorded occupancies; different old checkpoint coordinates are not automatically
missing tests. An empty checkpoint selection means prefill-only, not omission of
that ingest point. Only a newly observed sparse-result anomaly permits one fresh
full-curve escalation. Older graphs with reasoning off do not supply unmeasured
low/medium/xhigh values.

MTP-on controls and selective ACO keep a resident prefix with paired target and
draft checkpoint state. A target-only slot-file restore drops the native MTP
prefix/carry, so it must not be used to benchmark normal MTP behavior. MTP-off
controls may restore the full target snapshot outside the timed request,
recording `base_restore_seconds`, without re-prefilling the repository. A
source-directed checkpoint-free MTP-off profile avoids optional checkpoint
capture while using full target files; it does not disable CUDA graphs or FA.
An HTTP-200 SSE stream
that closes without final usage and `[DONE]` is a runtime fault, not evidence of
a prefix-cache miss.

## Current completed ACO full-curve benchmark (2026-11-04)

The completed campaign has two fresh persistent server variants, MTP on and
off. Each variant ran the same target-context ladder with L=262,144,
H=65,536, B/U=1024/256, GPU Turbo4 target KV, and one GPU. MTP-on also used
GPU Turbo4 draft KV; MTP-off had no draft allocation. Three canonical
prompts were measured at each checkpoint in reasoning off, low, medium, and
xhigh; each prompt/mode cell had one 40-token warmup and three measured
requests capped at 400 tokens. Each variant kept one process across the fill
ladder; it did not refill/reset for individual prompts or reasoning modes.
The run completed 11 checkpoints per variant (22 rows), 264/264 cells, 792
measured requests, and 264 warmups. Configured targets were 6,144, 12,288,
28,672, 62,464, 78,848, 88,064, 100,352, 130,048, 196,608, 256,000, and
261,120. Actual occupancy is reported separately; the final target 261,120
reached C=261,121 and left 1,023 tokens of reserve.

Independent benchmark questions start from the same filled repository base.
Restoring that base discards the preceding question/answer tail; it does not
reload the model or re-prefill repository content. The benchmark-only shared-prefix
template places the current reasoning-effort instruction immediately before the
current question, so switching modes retains the base. This isolates benchmark
questions; ordinary growing-conversation tests use the native model template.

See the [full curves and protocol report](../../../.wiretail/execution/evidence/ACO_FULL_CURVES_20261104.md),
[whole-cell best results](../../../.wiretail/execution/evidence/ACO_BEST_RESULTS_20261104.md),
and [ACO implementation guide](../../../docs/aco.md). MTP-on process `VmSwap`
peaked at 530.4 MiB at target C=100,352, while MTP-off recorded zero across
958 samples. These samples do not identify swapped allocations or swap-in I/O;
MTP-on is not certified RAM-only.

## Archived staged performance matrix v2 (2026-11-03)

The following v2 manifest and sequencing notes are retained for provenance.
They are not current run instructions; the 2026-11-04 campaign above is
complete and reported in the linked full-curve report.

The staged v2 plan is `/srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/remaining-matrix.json`. Its fixed bundle is
`/srv/ai/paged-kv/builds/aco-packed-repair-1d74e9a31/bin/llama-server` (source/build `1d74e9a315acaf0379eb2af1245202b820cff1e1`,
SHA-256 `189473719da99f8c391593c61a4c928769c80f72df8ceb6e527e707f95f35794`).
The prefix-anchor proof passed. The MTP-on matrix completed all ten checkpoints,
all four modes and all three prompts (120 pairs and 480 requests). MTP-off has
not started. Final clean prefill retests are paused by the operator while the
mapped-ingest protocol replaces variable-sized per-file turns. Keep the valid
completed decode/MTP measurements; do not blend new ingestion curves into the
historical v2 output root.
The prior ca549 full-matrix attempt root is archived and excluded from current
comparisons; its hybrid checkpoint-retention failure invalidated it as a complete
full-matrix result. New output and per-variant slot-state paths use
`/srv/ai/paged-kv/results/forward/final-after-packed-repair-20261103-v2`. The other eight variant selections and the benchmark source,
model/template, sampling, geometry and context ladder are unchanged. The v2
command sheet is `.wiretail/execution/benchmarks/REMAINING_MATRIX.md`.

Validate/list without starting a server:

`python3 /srv/repos/vanwho/buun-llama-cpp/tools/server/bench/run-benchmark-gaps.py --manifest /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/remaining-matrix.json --validate`

Execution is held in `.wiretail/execution/benchmarks/GPU_POLICY.json`. The runner
checks this hold before loading a model or changing a service. Once the operator
resumes final retests, freeze a new manifest/output root that explicitly records
the source chunk map path and SHA-256. Historical manifests are readable but
cannot silently append results using the new ingestion protocol.

The staged v2 manifest's shared-prefix mechanics were carried forward into the
completed campaign: each configuration keeps one server and adds disjoint
repository chunks. All prompts for one reasoning mode run before the next mode
on the same filled base, restoring only the independent question/answer tail.
The benchmark-only template places low/xhigh effort instructions immediately
before the final query. The archived build, queue, and remaining-work status
above are historical; this prefix-reuse mechanism remains applicable.

The runner hash-checks binaries, adjacent libraries, model and benchmark template;
enforces 1024/256, explicit 16-thread CPU settings and exclusive GPU use; and
pauses/restores the profile gateway and managed server. It retains partial results,
rejects empty/two-token acknowledgments as speed measurements, and resumes only
missing prompt/mode pairs. It never patches a master control binary or calls a
capacity/runtime failure a successful benchmark. This harness and its site plan
are development benchmark tooling to exclude from the final runtime upstream PR.

## Fixed multi-file ingestion map

The mapping is `.wiretail/execution/benchmarks/repo-prefill-map.json`:
262,144 native content tokens from one frozen repository revision, grouped into
four 65,536-token payloads. Small files share a payload; large files may span
payloads. Each range records its source path, content hash and byte offsets.
Payload text stays outside Git under the manifest's `payload_root`.

`build-repo-prefill-map.py` uses a vocabulary-only native tokenizer, without
model weights, an inference context or a GPU allocation. Each fill consumes the
next disjoint source range. A checkpoint may split a group; its leftover bytes
are regrouped with the next group rather than becoming individual-file turns.
Reports distinguish full groups from checkpoint tails and keep backend prompt
tok/s, net-new prompt tok/s and net-new wall-time tok/s separate. Checkpoint
framing, generated output and the context reserve are additional to mapped
content tokens; never send the entire 262,144-token corpus into L=262,144.

Sizing checks with B1024/U256, L262144/H65536 and Turbo4 GPU MTP admitted
8K/16K/32K/64K multi-file requests. These are allocation checks from fresh
servers, not final ingest-speed results or proof of every occupied frontier.
See `.wiretail/execution/engineering-notes/repo-prefill-chunk-map-20261103.md`.

Jellyfin is protected. Never stop its container, transcodes or media-stack
service to isolate the GPU. Sizing may run with Jellyfin up. If Jellyfin uses
the GPU during a future exclusive benchmark, cleanup pauses and leaves it
running instead of killing it.
