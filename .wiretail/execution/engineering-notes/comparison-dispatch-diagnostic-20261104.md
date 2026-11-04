# Post-comparison CUDA dispatch diagnostic

This is an explicitly separate diagnostic after all primary matrix and fallback
runs. It never replaces primary timings or modifies inference code/configuration.
Canonical tracked launcher: `.wiretail/execution/benchmarks/profile-comparison-dispatch.py`.

# R6 CUDA dispatch diagnostic

This is a post-matrix diagnostic only. It compares the frozen R6 master and
fork GPU/MTP-on server builds while ingesting the same mapped repository
prefix. It does not measure benchmark throughput and must not replace any
primary curve values.

## When to run

Wait until the R6 primary work and the already-planned L=122880 fallback
series are complete and the shared benchmark lock is free. The launcher
defaults to dry-run. The runtime owner can start the diagnostic with:

```bash
python3 /srv/ai/paged-kv/results/comparison-dispatch-diagnostic-20261104//srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/profile-comparison-dispatch.py
python3 /srv/ai/paged-kv/results/comparison-dispatch-diagnostic-20261104//srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/profile-comparison-dispatch.py --execute
```

The first command validates the frozen binary hashes and exact server argv,
checks installed Nsight Systems options, and prints the launch plan without
starting a server. The second command runs master and fork sequentially, each
in a fresh server process, at L=65536 and C=6144, 12288, 28672. It reuses the
existing repository map and resident-prefix ingestion driver, with prefill
only requests and no canonical decode cases.

The launcher takes `/tmp/ai-pager-benchmark.lock`, saves the prior state of
`ai-profile-gateway.service` and `llama-server.service`, stops only those
services that were active, and restores their prior active state in `finally`.
It records and checks Jellyfin PID and listeners on ports 8091/8092. It tracks
the actual `llama-server` PID behind Nsight and only signals that PID after
verifying `/proc/PID/exe`; it does not kill processes by name. It uses
`sudo -n` and fails if privileged actions are unavailable.

## Profile options and limits

The checked Nsight Systems build is 2023.4.4.54. The profile command traces
CUDA, NVTX, and OS runtime events, explicitly uses CUDA graph node granularity,
disables CPU instruction-pointer sampling and context-switch tracing, disables
GPU memory-usage tracing, and sets GPU metrics to `none`. It does not collect
hardware counter metrics. Node tracing can add substantial runtime and memory
overhead; these captures are for dispatch diagnosis only.

Nsight summary exports are limited to CUDA GPU kernel, CUDA API, and CUDA GPU
memory-operation summaries. `summary.json` reports attention-named kernel
variants and call counts plus separate CUDA copy/synchronization API and GPU
memory-operation totals. These views overlap and must not be summed. The
profiles include model startup as well as ingest; kernel dispatch/counts are
the primary evidence, while total API times are not request-exclusive latency.
The summary fails closed: an absent/empty CUDA kernel table is marked
`unmeasured_or_unsupported_cuda_trace`, and a populated table with no matching
attention names is marked `kernel_trace_present_but_attention_names_unmatched`.
Neither state is reported as zero attention dispatch or successful evidence;
check Nsight/CUDA compatibility and symbol naming before interpreting it.
Keep raw `.nsys-rep` files and append SSE/JSON artifacts in the external result
directory; send only compact summaries upstream.

## Output

Each invocation writes to a new UTC-stamped directory under
`/srv/ai/paged-kv/results/comparison-dispatch-diagnostic-20261104/runs/`.
`identity.json` captures the exact binary hash, server argv/environment, and
Nsight command. Per-build summaries and stage logs are kept beside the raw
profile. No runtime source, build, or benchmark configuration is changed.
