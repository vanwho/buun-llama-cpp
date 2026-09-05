# Post-17 task detail audit

Date: 2026-09-05. Scope: instructions and task context, not implementation or
new runtime acceptance. All 30 scheduled packets from 17-16 through 22-01 now
have concrete execution recipes across 15 clusters (at most three tasks each).

The earlier packets named architecture and tests, but still required agents
to invent command details, test-driver interfaces, tensor layouts, evidence
schemas and several dependency boundaries. The additions are:

- [Execution cookbook](../EXECUTION_COOKBOOK.md): verified existing CLI/build
  entrypoints, six small fixtures, one reusable live model-parity driver,
  common receipt/resume contract and named implementation owners.
- [Implementation contracts](../IMPLEMENTATION_CONTRACTS.md): allocation-plan
  inputs/outputs; physical-write tickets and host commit order; multiquery and
  partial-state strides; GPU-owned exact-wave execution; complete cold inventory
  and fresh-query execution boundary; staging versus production-proof rules.
- Each packet names the necessary C/I sections and checkpoint sequence, with
  exact outputs needed by its successor. Packets are about 743–1,021 words;
  shared context is reused within a cluster rather than copied into every task.

Important findings that changed the instructions:

1. The quality client's mode option labels a result; it does not switch the
   server. Require observed dispatch or explicit internal test selection.
2. The CUDA fixture has no model arguments. Registering it with CTest does not
   create live per-layer/logit parity; 18-05 must provide that reusable driver.
3. The current exact callback returns a CPU vector. 19-06 must build device-owned
   partial accumulation, not transfer every weighted-V partial through CPU.
4. Direct single-query numerical proof, production slab binding and direct
   MTP verification have different owners. I7 prevents premature dependencies
   while explicitly preventing a diagnostic preload from proving live paging.
5. Generic harness schemas use procedure types, not hardcoded project task IDs.
   The six-point speed curve remains final results only after functionality.

Validation: state helper passed with 122 tasks; all 30 recipe/C/I links, models,
packet paths and 15 cluster contexts checked; `git diff --check` passed.
Completed records before 17-15 and the 17-15 packet were unchanged. The outer
runner independently completed 17-15 and advanced current_task to 17-16 during
the audit; that legitimate transition and reported usage were preserved.
No service was changed and no GPU benchmark was run for this documentation audit.

The recipes reduce missing-context/design reconstruction. They cannot guarantee
that a simple model will implement a difficult CUDA/scheduler change correctly.
Luna High remains the implementation/benchmark recommendation and Sol High the
final reviewer; no Wiretail default or model setting was changed by this audit.
Numerical/live checkpoints and bounded recovery assessments remain necessary.
