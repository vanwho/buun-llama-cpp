# Focused natural cold-recall fixture

These five small Python/Bash source fixtures are copied verbatim from the
repository's `tools/server/bench/fixtures/pager-promotion` corpus. Their
SHA-256 values and the exact A1/B/A2 message sequence are frozen in
`manifest.json`; the repository MIT license is included as `LICENSE`.

The A1 turn reads two Python fixtures. B appends three Bash fixtures and a
brief acknowledgment request. A2 asks the same PY_MERGE_03 fact as the
previously validated natural cold-recall case. Use one fresh L8192/H4096 ACO server and
run this sequence once after the small placement matrix. The runner maps the
retrieval-comment span with the active server tokenizer, snapshots the native
page inventory before A2, then records the A2 page transition, selected page,
natural-proof witness, transfer counters, timings, and answer. It never pins,
forces, or names a page in a request. If the target comment's page is not
cold, report that outcome without a retry or promotion matrix.

The public output should report actual client wall and backend stage times
separately, plus exact native occupancy and the observed fresh/cached work.
Review the answer semantically; the expected response is not an exact-string
gate. Bash files are context only and must not be executed as watcher scripts.

Start the isolated server with the runner's exact ACO argv (substitute only
the local binary/model/key paths and alias), with diagnostic override
environment variables unset:

```sh
python3 tools/server/bench/release_evidence/run_natural_cold_recall.py \
  --server-bin /path/to/llama-server --model /path/to/model.gguf \
  --model-alias release-model --api-key-file /path/to/api.key \
  --source-root . --output /path/to/results/natural-cold-recall \
  --lock-file /path/to/ai-pager-benchmark.lock --preflight
```

This offline check validates fixture hashes and prints the exact service argv
and step hashes. It does not connect to the server. After service startup and
idle/GPU ownership verification, run the same command without `--preflight`
to execute the one A1/B/A2 sequence.

```sh
env -u LLAMA_KV_ATTENTION_ROUTE -u LLAMA_DENSE_TENSOR_DIAGNOSTIC \
  -u LLAMA_GRAPH_REUSE_DISABLE -u LLAMA_HOTPATH_PROFILE \
  -u LLAMA_KV_PAGER_DIAGNOSTIC_PAGE_MASS -u LLAMA_KV_PAGER_PROGRESS_TRACE \
  -u LLAMA_KV_PAGER_SELECTOR_TRACE -u LLAMA_MTP_FIRST_REJECT_TRACE \
  -u LLAMA_MTP_STATE_DIAGNOSTIC -u LLAMA_KV_ROUTER \
  /path/to/llama-server -m /path/to/model.gguf --alias release-model \
  -ngl 999 --device CUDA0 --fit off -fa on -c 8192 -np 1 \
  -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 \
  --host 0.0.0.0 --port 8080 --api-key-file /path/to/api.key --metrics \
  --reasoning-preserve --no-context-shift --ctx-checkpoints 0 --cache-ram 0 \
  --no-cache-idle-slots --kv-pager selective --kv-router probe-rerank \
  --kv-page-size 256 --kv-hot-pages 16 --kv-pin-recent 0 \
  --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 \
  --spec-draft-type-k turbo4 --spec-draft-type-v turbo4
```

After verifying that the service is idle and owns the managed port/GPU, run
the single sequence into a fresh external results directory:

```sh
python3 tools/server/bench/release_evidence/run_natural_cold_recall.py \
  --server-bin /path/to/llama-server --model /path/to/model.gguf \
  --model-alias release-model --api-key-file /path/to/api.key \
  --source-root . --output /path/to/results/natural-cold-recall \
  --lock-file /path/to/ai-pager-benchmark.lock
```
