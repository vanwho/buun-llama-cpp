# Cluster 21b — bounded harness and live correctness

Tasks: `21-04`, `21-05`, `21-06`.
Model policy: Luna High; do not change tool-wide defaults.

## Purpose

Make benchmark execution fail-fast and observable, then prove the repaired
runtime on small, representative workloads before spending long-context time.
The cluster separates harness failures, model-quality failures, and runtime
movement failures.

## Shared context

Read `BENCHMARK_PROTOCOL_V5.md` A–E, `EXECUTION_COOKBOOK.md` C2–C5,
`IMPLEMENTATION_CONTRACTS.md` I2/I5/I6, and the receipts from 21-01 through
21-03. Reuse the frozen release only if its source, DSOs, model, corpus,
template, and policy hashes match.

## Invariants

Every live record reports route (`selected_direct`, `selected_reference`, or
fallback), selected/physical/logical pages, host-valid rows, H2D/D2H/fault and
eviction counters, queue/copy/wait time, target/draft placement, and actual
occupied tokens. Missing fields are a capability failure, never zero. A mode
stops after a confirmed capability refusal or two same-prefix quality
failures. The unrelated 8092 service is never touched.

## Handoff

Only after 21-06 proves bounded correctness and real movement may the final
speed curve and controls run. Keep successful tested profiles loaded for the
next dependent task; restore only for an explicitly declared control or
failed-start recovery.
