# Cluster 13a — measured kernel and transfer optimization

Tasks: `13-01`, `13-02`, `13-03`.

Purpose: establish stable profiles, then remove the dominant decode and page-movement costs while keeping
Turbo4 direct and every correctness gate intact.

Carry forward:

- task 13-01 freezes microbenchmark shapes and budgets from live traces: selected rows/pages, logical
  gaps, tails, page-table sizes, query counts, fault batches, coalesced runs, and warm/cold cases;
- profile end-to-end and kernel/device timelines. Attribute table upload, graph rebuild, Turbo4 decode,
  page-mass reduction, D2H/H2D, event waits, host copies, policy, and server overhead separately;
- kernel candidates include table metadata layout/cache, run coalescing, vector loads, occupancy,
  registers/shared memory, fused mass reduction, and specializations only for runtime-validated geometry;
- transfer candidates include pageable-versus-pinned backing, bounded pinned ring size, batching,
  contiguous layer/side runs, asynchronous copies, stream priority, double/triple buffering, VMM mapping,
  and prefetch distance;
- benchmark one change at a time, preserve before/after raw data, repeat enough to reject noise, and run
  output/sanitizer checks after each retained optimization;
- never win by reducing selected capacity, skipping required pages, changing output length, disabling
  MTP, weakening quality, or hiding fault intervals.

Read: plan sections 22–24; handoffs 03-03, 06-02, 09-05, 10-03, 11-03–11-05; CUDA kernel/resource logs,
transfer counters, and the frozen benchmark contract.

Exit artifacts: reproducible microbench driver, profiles, ranked bottleneck table, retained optimization
commits, rejected-hypothesis log, parity/sanitizer evidence, and an integrated checkpoint showing how much
of the CPU-offload gap has closed.
