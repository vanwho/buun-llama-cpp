# Cluster 14b — final speed acceptance and operational soak

Tasks: `14-04`, `14-05`.

Purpose: decide whether the requested speed goal and long-running operational behavior are actually met.

Carry forward:

- use the exact release SHA/config/corpus hashes accepted by 14-01–14-03;
- run at least ten measured trials for same-context ordinary CPU target K/V plus GPU MTP, selective warm
  focus, cold needles, focus shifts, churn, and feature-off; discover the safe all-GPU comparison context
  anew for this SHA;
- report pp/tg, TTFT, inter-token p50/p95/p99, CPU/RSS/pinned RAM, VRAM by category, MTP acceptance,
  selected rows, faults, transfers, overlap, waits, graph changes, and errors;
- hard warm gate is >=3x CPU-KV median decode; report 5x target and >=70% comparable all-GPU ratio;
  observe overhead <=5%; no-fault p95 <=1.5x median. Fault/churn segments remain separate;
- soak repeated long conversations, focus shifts, cache/checkpoint restore, speculative rejection,
  cancellation, slot reuse, clear, shutdown/restart, and memory recovery. No monotonically growing host,
  pinned, device, event, or page-table resource is allowed;
- always restore the starting profile and both required health endpoints; never touch port 8092.

Read: plan section 24; handoffs 08-05, 12-02, 13-05, 14-01–14-03; benchmark scripts and raw manifests.

Exit artifacts: final machine-readable performance comparison, confidence/tail summaries, raw soak logs,
resource time series, fault examples, restored-service evidence, and a single pass/block acceptance. These
tasks cannot be deferred or passed from historical controls.
