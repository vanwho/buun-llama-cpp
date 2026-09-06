# Cluster 23a — measured runtime repair

Tasks: `23-01`, `23-02`, `23-03`.
Model policy: Luna High; do not change tool-wide defaults.

## Purpose

Repair the three causal runtime failures measured in phase 21: automatic hot-set
admission consumes space needed by late CUDA allocations, compact paged storage
is serialized with dense logical offsets during checkpoint save, and the live
Qwen prefill remains on a slow selected-reference route despite tiled-kernel
work. These are new minimized hypotheses, not repetitions of the phase-21
campaign.

## Shared context

Read `POST17_IMPLEMENTATION_STRATEGY.md`, `TECHNICAL_CHANGE_SPEC.md` T2/T3/T5/T9,
`IMPLEMENTATION_CONTRACTS.md` I1/I2/I3/I6, `EXECUTION_COOKBOOK.md` C1–C5,
`BENCHMARK_PROTOCOL_V5.md` A–C, the 22-01 handoff, and only the raw IDs named by
the current packet. Use the phase-21 summary to avoid retrying an unchanged
fixed-four-page or three-token workaround.

## Invariants and exit

Logical context and GPU-native MTP rows remain equal; all four K/V codecs remain
Turbo4. Automatic admission is budget-derived, checkpoint bytes follow physical
slot identity, and route changes preserve selected-all numerical parity. Each
task records its attempt fingerprint, changed variable, minimized result and
source owner. Leave a successful candidate loaded for the next dependent task.
