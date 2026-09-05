# Post-17 plan revision — 2026-09-05

The active 17-15 packet and server were not changed or interrupted. Thirty
future tasks in fifteen clusters now follow it; state validates with 122 tasks.
All completed task records and reported token accounting were preserved.

17-16 captures the completed campaign compactly. Phases 18–20 repair specific
runtime, representation, storage, CUDA, routing and performance gaps. Phase 21
measures quality, full-context functionality, the original-three-prompt speed
curve, matched controls and real physical-pressure soak; YaRN is conditional.
22-01 is the Sol High benchmark-only reviewer and can create the next measured
remediation chain. Other tasks remain Luna High; tool defaults are unchanged.

The six 20K/40K/60K/100K/175K/256K coordinates are exclusively the final results
curve after functionality works. Development tests choose purpose-appropriate
context and hot budget; no routine task or phase repeats the six-point ladder.

Technical decisions, source symbols and falsifiable repair hypotheses live in
[TECHNICAL_CHANGE_SPEC.md](../TECHNICAL_CHANGE_SPEC.md). Overall reasoning and
current findings are in [the strategy](../POST17_IMPLEMENTATION_STRATEGY.md).
The [benchmark protocol](../BENCHMARK_PROTOCOL_V5.md) replaces guessed sizing,
blanket timeouts and misleading control/metric comparisons.

Checks: generic state validator, all future dependency/packet/cluster/model
links, maximum three tasks per cluster, unchanged active packet and historical
records, preserved token accounting, and git diff --check all passed.
This is plan validation, not a new runtime benchmark or a speedup claim.

Changes are persisted on the active worktree for the outer auto runner's
execution-metadata checkpoint and integration. No competing branch switch,
commit, push, upstream post or production-code modification was performed by
this revision. Old unstarted packets were archived, not erased from history.
The machine-readable receipt is [POST17_PLAN_REVISION.json](POST17_PLAN_REVISION.json).
