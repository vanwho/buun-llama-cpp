# Cluster 06c — exact all-pages reference

Task: `06-05` only. Use a fresh session because online-softmax partitioning has different numerical and
staging concerns from selective benchmark tuning.

Purpose: validate storage/page-table correctness and measure selective quality delta without applying a
selective throughput gate.

Carry forward:

- every valid logical position participates exactly once;
- resident and cold page waves produce `(m,l,o)` partial states merged by the stable formula in plan
  section 8.10, normalizing only after all partitions;
- first backend streams opaque Turbo4 pages through bounded GPU staging; no full-context GPU allocation,
  F16 gather, CPU Q8_0 fallback, or assumed CPU Turbo4 support;
- serial deterministic execution precedes optional double buffering;
- native positions, causal masks, GQA, tails, and page generations remain authoritative;
- reuse the frozen 06b corpus and compare selective output without retuning thresholds.

Read: plan sections 2, 8.10, 9.4–9.5, 12, 16–17; task 06-05; Phase 03, host residency, correctness,
and selective acceptance handoffs.

Exit artifacts: merge-formula unit evidence, page coverage ledger, dense bounded comparisons, 256K exact
evidence where feasible, and exact-vs-selective quality delta.
