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
the selected profile/model to supply the context rather than applying a
historical fixed hotset or context default.

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
