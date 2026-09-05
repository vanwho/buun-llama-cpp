# Cluster 21a-quality-curve

Tasks: `21-01`, `21-02`.
Model policy: Luna High; do not change tool-wide defaults.

## Shared context

Read `POST17_IMPLEMENTATION_STRATEGY.md` once, then the current task packet
and its named `TECHNICAL_CHANGE_SPEC.md` sections. Live work also reads
`BENCHMARK_PROTOCOL_V5.md`. Reuse these within this cluster; do not load
historical phase-14/15/16 audit transcripts, all previous handoffs, or entire raw
JSONL files. The immediate dependency handoff is the entry receipt.

- 21-01: Measure release quality and renewed full-256K functionality; exit receipt `evidence/PHASE21_QUALITY.json/.md`.
- 21-02: Run the original three prompts across the Turbo4 context speed curve; exit receipt `evidence/PHASE21_CURVE.json/.md`.

## Invariants

Target/MTP K and V stay Turbo4; native MTP rows equal resolved target context
and stay GPU resident. Target hot capacity is budget-derived; CPU backing is
canonical. Preserve exact recurrent state, logical positions, representation
identity, transaction generations and actual telemetry. Single-sequence support
is the base scope. Speeds are findings, not the historical 3x gate.
Implementation is generic; site service/config/results and execution state do
not belong in upstream code slices. Outer runner owns auto Git operations.

The current packet also names C-sections in `EXECUTION_COOKBOOK.md` and
I-sections in `IMPLEMENTATION_CONTRACTS.md`. Read only those sections once;
they contain actual commands, minimal fixtures, interface and receipt contracts.

## Handoff boundary

Finish each packet's focused live/portable checks and compact receipt before the
next task. Preserve a successful runtime for reuse. Cluster exit is the final
task's receipt, not a request to reread every earlier phase.
