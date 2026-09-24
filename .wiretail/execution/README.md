# Current hot-KV execution package

Read the [long-horizon context and usage policy](CONTEXT_POLICY.md) first. For
execution, trust the current task in `WORK_STATE.json`, its packet/cluster, and
its explicit `context_files`; those identify the active design slice. Do not
assume an older V10/V9 plan or phase summary is current merely because it is
linked from historical evidence. Wiretail injects `CONTEXT_POLICY.md` into
each task and assessment prompt, while task-specific context remains bounded.

```bash
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
PROJECT_BRANCH=plan/attention-aware-kv-paging /srv/wiretail/wiretail.sh
```

Each bounded task explicitly lists `context_files`. Read that list, the current
packet/cluster, repository instructions, and the current task handoff if
present. Do not recursively read old plans, the full state/log, old acceptance
gates, or every dependency handoff. Every task has its own named-proof
completion contract; follow the contract linked by that task.

V9 and older plans/packets/handoffs/results remain historical, not active.
Do not PR any execution metadata/history upstream. The historical cleanup
inventory remains in `v9/OPERATIONS.md` for a future publication-only task;
do not load it in current implementation sessions. The V10 repair/testing
documents are historical design references unless a current task names them.
Portable code commits remain separate.

Task and phase token totals are Codex-reported `turn.completed` usage
aggregates, not the size of one prompt or active context window. See
[`CONTEXT_POLICY.md`](CONTEXT_POLICY.md) for field semantics, compaction, and
the handoff format.

Automatic selective paging is fast-route-only. `selected_reference` is a
temporary explicit correctness oracle, never a production fallback or valid
speed result. Unsupported automatic shapes must be refused with a reason until
their GPU-native dense, direct Turbo4, or bounded packed route is implemented;
see task [92-01](tasks/92-01.md).

The final review reads only the new benchmark summary. If the goal is unmet,
it must append actual remediation/benchmark/summary/review task entries before
completing, not merely recommend another action in prose.
