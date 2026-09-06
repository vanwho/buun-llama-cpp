# Cluster 21a — capability gate and core runtime repair

Tasks: `21-01`, `21-02`, `21-03`.
Model policy: Luna High; do not change tool-wide defaults.

## Purpose

Turn the stopped 21-01 campaign into a bounded capability gate, then repair
the two causal runtime problems it exposed: three-query serialized prefill and
selective attention that does not reliably promote/calculate cold pages. No
long corpus or speed campaign may run in this cluster.

## Shared context

Read `POST17_IMPLEMENTATION_STRATEGY.md`, the relevant paged Turbo4 and
residency sections of `TECHNICAL_CHANGE_SPEC.md`, `BENCHMARK_PROTOCOL_V5.md`
A–C, the 20-07 release receipt, and the stopped 21-01 raw receipt. Use only
the compact handoff and the smallest raw records needed to reproduce one
failure; never load the entire historical ledger.

## Invariants

Target K/V, draft K/V, and every benchmark control remain Turbo4. Native MTP
draft rows are GPU-resident and sized from the resolved context. Host RAM is
the canonical target-K/V backing store. Hot capacity is derived from an
explicit VRAM budget after model and MTP reservations; a fixed four-page cap is
diagnostic-only. Logical positions, page generations, host-valid state,
transaction fences, and exact/selective semantics must remain intact.

## Exit rule

Each task leaves a compact receipt with a falsifiable hypothesis, changed
variable, command, and raw pointer. A failed capability probe is useful only
when it identifies an implementation owner and the next minimized test. Do not
start a 24/48-case campaign while route, movement, or batching telemetry is
missing.
