# ACO: Attention-Aware Context Offloading

ACO is an experimental CUDA path for keeping a bounded target-model KV working
set on the GPU while retaining sealed target pages in host memory. The default
`--kv-pager off` path is unchanged. ACO is not a general replacement for dense
attention, and `--kv-pager selective` is an approximation: pages outside the
selected set do not contribute to that attention result.

## Architecture

```mermaid
flowchart LR
    Q[Query and page-mass evidence] --> S[Query-boundary selector]
    S --> H[GPU-resident target pages]
    H --> R[CUDA attention route]
    C[Canonical sealed pages] -->|bounded promotion| H
    H -->|clean eviction| C
    M[Native MTP KV] -. separate allocation .-> G[GPU memory]
```

Target K/V is divided into logical pages. The pager keeps a bounded hot set on
the GPU, while canonical sealed pages and their metadata live in pageable host
memory. A pinned transfer ring stages bounded copies. Clean eviction can drop
device residency after host backing and the routing summary are current. The
pager does not own recurrent state or native MTP K/V; native MTP uses a separate
full-context allocation when enabled.

Attention evidence and retrieval determine which history pages are selected at
a query boundary. The selected history is frozen for that turn. A page being
resident is not proof that a particular attention operation read it; residency,
selection, and actual kernel execution are separate facts. For an eligible
CUDA shape, execution can use contiguous dense Flash Attention, a packed
selected-page view followed by mature Turbo4 Flash Attention, or the direct
paged Turbo4 kernel. `selected_reference` is an explicit correctness
diagnostic. It is not an automatic fallback and its timing is not performance
evidence.

## Supported shape and limits

The documented target geometry is Qwen3.8-family CUDA attention with 256-token
pages, 16 full-attention layers, 4 KV heads, K/V width 256, and Turbo4 K/V
(4.125 bits/value). A complete cross-layer target page is 4,325,376 encoded
bytes (4.125 MiB). The current documented execution boundary is causal,
batch-one CUDA attention. Do not infer support for Qwen3Next or other model
families from this geometry.

Selective routing can differ from full-context dense output and needs
model-and-prompt quality evaluation. It is not dense-equivalent. CUDA
correctness for a selected route does not establish output quality relative to
dense attention. Likewise, a page being GPU-resident does not establish that
it was attended in a measured request.

The hot-page count is a cap resolved against the target context and runtime
memory ledger. GPU memory must also fit model weights, graph and scratch space,
recurrent state, transfer staging, and any native MTP allocation. Host RAM must
fit the sealed-page backing and metadata. Pinned memory is separately bounded
by the transfer ring. `--fit off` disables automatic fitting; use it only when
you have checked the complete allocation on the target device. Unsupported
shapes, missing host backing, stale identities, unsafe eviction, or inadequate
capacity are refused rather than routed through CPU Turbo4 attention.

The available modes are `off`, `observe`, `selective`, and `exact`. `off` is
the normal dense path. `observe` collects pager evidence without selective
eviction. `selective` enables the supported selective route. `exact` requires
its exact-mode invariants and fails closed when they are not met. The optional
`--kv-router probe-rerank` selects query-boundary reranking; the default is
`legacy`.

## Build

Use the CUDA toolkit and CMake versions required by
[`docs/build.md`](build.md#cuda). Configure a Release build for the local CUDA
architecture and build the server:

```bash
cmake -B build-aco \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-aco --target llama-server -j16
```

The resulting executable is `build-aco/bin/llama-server`. Use a supported
CUDA device and a Qwen3.8-family model with the documented KV geometry. The
generic build command does not imply that every CUDA card can hold a particular
model, hot set, or MTP allocation.

## Launch

The following is a validation configuration for a CUDA device with enough
memory for the complete model, runtime, a 64K-token target hot set, and
full-context native MTP. It is not a universal preset. With 256-token pages,
`--kv-hot-pages 256` caps the target hot set at 65,536 tokens. A model path and
API-key file must be supplied by the operator:

```bash
MODEL=/path/to/qwen3.8-model.gguf
API_KEY_FILE=/path/to/llama-api-keys

build-aco/bin/llama-server \
  -m "$MODEL" \
  --host 127.0.0.1 \
  --api-key-file "$API_KEY_FILE" \
  --device CUDA0 \
  -ngl 999 \
  --fit off \
  -fa on \
  -c 262144 \
  -b 1024 \
  -ub 256 \
  -np 1 \
  -ctk turbo4 \
  -ctv turbo4 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-kv-device gpu \
  -ctkd turbo4 \
  -ctvd turbo4 \
  --kv-pager selective \
  --kv-router probe-rerank \
  --kv-page-size 256 \
  --kv-hot-pages 256 \
  --no-context-shift \
  --ctx-checkpoints 0 \
  --cache-ram 0 \
  --no-cache-idle-slots
```

For a smaller smoke configuration, use `-c 8192 --kv-hot-pages 16`, omit the
native-MTP options, and make sure the model itself fits the selected device.
This still requires the documented model geometry; it does not establish
quality or speed for a full-context run.

For API-key setup, choose a private file containing one key per line and pass
its path through `--api-key-file`. Keep the server bound to `127.0.0.1` unless
you have separately configured network access and authentication.

## Bulk ingestion and retrieval requests

Native `/completion` and OpenAI-compatible `/v1/chat/completions` accept an
optional top-level `aco_mode` string:

- `query` is the default. ACO ranks and promotes history at the user-query
  boundary, then replays the query if the selected history changed.
- `ingest` explicitly declares a bulk append. ACO evaluates its new tokens
  once using the bounded GPU-resident causal window. It retains normal page
  writes, asynchronous host sealing, clean eviction, output generation and
  MTP verification, but performs no cold-history retrieval or query replay.

For example, a source-loading chat request can include:

```json
{
  "aco_mode": "ingest",
  "messages": [{"role": "user", "content": "Updated src/example.py:\n<current file contents>"}],
  "cache_prompt": true,
  "max_tokens": 1
}
```

The output budget is the caller's normal choice, not an ingestion restriction.
Keep the actual conversation history for subsequent requests as usual; this
extension does not create a server-side file store or a new session protocol.
Ask the later question in a separate ordinary request, omitting `aco_mode` or
setting it to `query`. Only that final question needs query selection/replay;
the preceding cached source tokens remain the prefix.

Clients can also put new source in earlier messages and a short question in
the final user message of a single ordinary query request. Existing authenticated
message-span handling then limits replay to that final user span, when the
template preserves those boundaries. No filename lookup is involved. A source
block and question inside one user message remain one query span; the server
does not split arbitrary text or rely on magic delimiters.

Use `query` for a message mixing file contents with a question that may need
cold history, tool-result interpretation, or instructions referencing earlier
context. The server never infers ingestion from filenames, attachment-looking
text, message length, role, or phrases such as "read these files". Explicit
ingestion is a choice of attention policy, not a guarantee of dense-equivalent
output: a response during ingestion can use the hot window, not cold retrieval.

Repeated filenames are labels. New bytes, including updates of an existing
filename, remain ordinary new tokens. The server does not hash, silently
deduplicate, overwrite older file occurrences, or relocate their KV. Unchanged
prefixes use existing token caching; edits to earlier messages use the normal
prefix invalidation/recompute behavior. No per-file registry or GPU file map
is allocated.

The mode belongs to one request and is cleared on completion/cancellation.
Ingestion uses the same resident-window selection rule for prefill, decode
and MTP verification. Switching ingestion/query modes changes the graph's
topology identity so selector-free graphs cannot leak into retrieval requests.
The pager's normal ownership, admission and cleanup checks remain in force.
The hint is inactive with pager `off` or `observe`; unmarked requests retain
their existing behavior. Invalid values and types are rejected.

`/props` advertises `aco_request_modes: ["query", "ingest"]`. Clients requiring
single-pass ingestion should check this capability before relying on the hint:
older builds may ignore unknown request fields. This is a runtime API capability,
not an additional model, GPU allocation, quantization, or batch-size setting.

## Evidence and comparisons

Treat routing counters, residency snapshots, attention execution, quality, and
latency as distinct measurements. The current 107-02 natural A/B/A cohort does
not provide cold-promotion timing; 107-01 cold-promotion timing was invalidated.
Two correct near-256K promotions from the earlier 105-21 cohort remain valid
for that older cohort only. Do not present those results as current-cohort
timings or as a general 256K guarantee.

CPU-target F16 versus CUDA-target Turbo4 changes both placement and KV
representation, so it is a confounded comparison and cannot isolate the
offloading effect. Report the exact model, quantization, context, occupied
tokens, hot-page count, KV types, native-MTP placement, device, and executed
attention route with any comparison. Do not infer fully occupied context from
an allocation or configuration that only started successfully.

This implementation and documentation were developed with AI assistance and
require human review before contribution.
