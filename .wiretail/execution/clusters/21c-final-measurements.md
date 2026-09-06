# Cluster 21c — final measurements

Tasks: `21-07`, `21-08`, `21-09`.
Model policy: Luna High; do not change tool-wide defaults.

## Purpose

Run the requested three-prompt Turbo4 speed curve, matched controls, and
pressure/soak only after the runtime and harness gates are real. Context
coordinates are campaign inputs, not engine defaults; the curve is a finding,
not a historical numeric acceptance gate.

## Shared context and invariants

Read the compact 21-06 receipt and `BENCHMARK_PROTOCOL_V5.md` D–F. Use one
immutable release/model/corpus/config identity. Keep target and MTP K/V Turbo4,
keep native MTP on GPU, derive the hot set from safe VRAM, and report actual
occupied/hot/host rows and movement. Never replace a missing point with a
shorter context or a successful selected-all run.
