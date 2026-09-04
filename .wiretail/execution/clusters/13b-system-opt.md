# Cluster 13b — prefill/graph optimization and policy Pareto calibration

Tasks: `13-04`, `13-05`.

Purpose: optimize costs above the decode kernel and choose quality-safe automatic policy defaults using
only the frozen calibration split.

Carry forward:

- prefill/graph work targets TTFT, bounded scratch, graph reuse, allocation churn, table staging, chunk
  sizing, and the transition to first decode; it cannot allocate target buffers proportional to `C` on GPU;
- policy calibration sweeps capacity-relative recent/history/transient shares, summary shape/top-K,
  exploration, EMA/peak decay, hysteresis, sample cadence, and prefetch depth within the live admitted `H`;
- measure warm throughput, cold recall, captured exact attention mass, needle/task score, MTP acceptance,
  fault rate, late waits, transfer bytes, and p95/p99 latency as a Pareto frontier;
- lock defaults from the calibration split before held-out phase 14. Defaults are ratios/minima or `auto`
  functions, never a hot page count;
- if the integrated 3x floor is missed, test at least three profiler-driven hypotheses across kernel,
  transfer, graph, and policy ownership before blocking. Preserve failed results.

Read: plan sections 23–24; handoffs 08-04, 10-04–12-03, 13-01–13-03; frozen corpus manifest, live metrics,
and raw profile outputs.

Exit artifacts: prefill/graph before-after evidence, bounded-memory proof, calibration sweep tables, chosen
default rationale, config hash, quality floor verification, and a release-candidate parameter set that
phase 14 must use unchanged.
