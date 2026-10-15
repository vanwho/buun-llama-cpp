# Release benchmark evidence

This folder contains the portable protocol and exporter for the release
benchmark package. The run records and prompt transcripts stay in the
operator's external results directory. Publish only the JSON and CSV produced
by `export_public_results.py`; those files use an allowlist and omit local
paths, command lines, request bodies, credentials, and raw responses.

## Protocol

Use the same cleaned upstream-integrated release binary for every placement.
Record its commit and SHA-256, plus the model and tokenizer/template hashes,
outside the public export. Each run uses reasoning off, temperature 0, seed
42, native MTP draft on (`draft-mtp`, maximum draft length 2), Turbo4 for
target and draft KV when GPU placed, batch 1024, microbatch 256, one slot,
GPU layer offload 999, and flash attention enabled. CPU target KV may use F16;
keep the draft KV on GPU Turbo4. Record the actual startup argv and runtime
KV residency in private evidence.

For each canonical prompt, run one 40-token warmup followed by three measured
400-token requests. The prompts are:

1. `write a python function that merges two sorted lists into one sorted list, with docstring.`
2. `explain the difference between mmap and read for loading large files, one paragraph.`
3. `write a bash script that watches a directory and prints new files as they appear.`

At one shared occupancy, compare GPU-only KV, CPU target KV with GPU draft
KV, and ACO (Attention-Aware Context Offloading). This protocol fixes the
small workload at 8K logical context, 4K hot KV, and C=6144. The checked-in
source ranges and hashes are mandatory; if they no longer fit, stop and
revise/version the fixture instead of silently changing geometry. Prepare
and hash the server-tokenized prefix once, then give the same frozen prefix
file to all three placement runs. The system message ends with an explicit
repository-context boundary and says to answer only the following user
request. Before any generation, the runner uses the
server's native template/tokenizer on each exact system+canonical-user
message and verifies room for a 400-token answer plus a 64-token context
margin. Keep prefix occupancy, full rendered prompt tokens, backend-processed
tokens, cached tokens, fresh tokens, context allocation, and hot capacity
independently recorded. The normal prompt-cache policy is enabled in every
row; report observed cached and fresh tokens for every request. Repeated
canonical turns can reuse their exact tokenized prompt, so their prompt-stage
time is not a bulk-fresh-prefill measurement. Prefix preparation sits outside
each generation's HTTP wall time; prompt-cache reuse and fresh work are
reported as observed rather than presumed.

Then run one 256K logical allocation with 64K hot KV. Fill toward the
approximately 256K target while preserving the append-by-append occupancy
and prefill curve, targeting C=258,048 tokens or 260,000 only when the frozen
source fit leaves the full 1,536-token context reserve. Capture all three
canonical prompts at that occupancy. The shared small GPU-only row is the
dense output-sanity control; do not add a second long full-prefill matrix.
Finally run one known successful natural cold-recall case; this is a focused
recall check, not a promotion matrix.

The historical master106 cohort is a patched-master baseline: it includes a
narrow draft-placement patch and compares CPU F16 target KV against GPU
Turbo4 target KV with the Turbo4 GPU draft held fixed. Label it as that
cohort, never as stock upstream. The release candidate must be identified by
the cleaned upstream-integrated source and binary hashes.

Report client wall time separately from backend prompt and decode stages.
Where exposed, keep prompt processed tokens, fresh prompt tokens, cached
tokens, prompt stage time, decode stage time, and response-local MTP drafted
and accepted counts as distinct fields. Do not present transfer-stage time as
total turn overhead. Historical 10521 evidence has 33 MiB transfer and 5132
and 5117 microsecond transfer stages; those are stage timings only. The
successful 10702 8K cold-recall observation is 2447 ms wall and 33 MiB
transferred, with mixed-control repeats. The entire 10701 cohort is excluded
from numerical comparison, including rows that individually completed.

Before model requests, use the host's established lifecycle lock and verify
the exact managed service and GPU owner. Preserve the unrelated 8091 and
8092 listeners. Capture per-process RSS and swap at each run boundary; host
wide swap changes are outside this protocol.

## Export format

The exporter accepts a manifest whose `runs` entries point to private JSONL
records. Each entry needs a safe `name`, a `placement` (`gpu-only`,
`cpu-target-gpu-draft`, or `aco`), logical context, occupied tokens, hot KV
tokens, and a `records_jsonl` path. The exporter copies only explicitly
approved scalar fields into `results.json` and `results.csv`; private path
values and unrecognized fields are never copied.

Run it with:

```sh
python3 tools/server/bench/release_evidence/export_public_results.py \
  --manifest /path/to/private-manifest.json \
  --output-dir /path/to/public-results
```

`run_release_matrix.py` uses only the Python standard library and its frozen
source manifest. `--preflight` checks source hashes, geometry, model/binary
identity, and prints exact service argv for all placements without contacting
a server. `--prepare-prefix` requires the already-started server to match its
generated argv, then saves a single frozen JSON prefix (including its exact
text, hashes, selected repo ranges, and native prompt-budget checks). Run it
once, before changing placements. `--run` requires that same prefix file,
re-tokenizes it to verify exact occupancy, validates each exact canonical
message's 400+64 budget, verifies the running candidate/argv, then executes
one placement. Repeat with the three placement argv from preflight, always
passing the same prefix file. The runner must never be used to change service
state. Unset diagnostic route overrides (including
`LLAMA_KV_ATTENTION_ROUTE`) for the normal automatic route; explicit route
overrides are diagnostic-only and must be recorded as such.

Example (replace the placeholders with local paths; the server must already
be running with the exact argv printed by preflight):

```sh
PKG=tools/server/bench/release_evidence
COMMON="--server-bin /path/to/llama-server --model /path/to/model.gguf --model-alias release-model --api-key-file /path/to/api.key --slot-state /path/to/slot-state --source-root . --output /path/to/run --lock-file /path/to/ai-pager-benchmark.lock --context-tokens 8192 --occupied-tokens 6144 --hot-tokens 4096"
python3 "$PKG/run_release_matrix.py" --mode gpu-only $COMMON --preflight
python3 "$PKG/run_release_matrix.py" --mode gpu-only $COMMON --output /path/to/frozen-prefix.json --prepare-prefix
python3 "$PKG/run_release_matrix.py" --mode gpu-only $COMMON --prefix-file /path/to/frozen-prefix.json --output /path/to/gpu-only --run
python3 "$PKG/run_release_matrix.py" --mode cpu-target-gpu-draft $COMMON --prefix-file /path/to/frozen-prefix.json --output /path/to/cpu-target --run
python3 "$PKG/run_release_matrix.py" --mode aco $COMMON --prefix-file /path/to/frozen-prefix.json --output /path/to/aco --run
```

The long ACO row uses `--context-tokens 262144 --occupied-tokens 258048
--hot-tokens 65536`; it builds a deterministic Git source schedule and
measures incremental fill requests at 16K-token frontiers before the three
canonical prompts. Use a separately prepared frozen long-prefix file. The
caller owns service stop/start and restoration so the runner cannot silently
replace a managed service. The focused natural cold-recall check uses the
frozen known-successful 8K case after the victim/page mapping fix; record the
actual mapped span and do not expand it into a promotion matrix.
