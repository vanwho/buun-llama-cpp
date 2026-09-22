# Current hot-KV execution package

For unfinished tasks85-09 onward start with the
[corrective plan](v10/REPAIR85_PLAN.md),
[source audit](v10/REPAIR85_ASSESSMENT.md) and
[short-test protocol](v10/REPAIR85_TESTING.md), dated2026-09-21.
They supersede older packed-only/reference-fallback directions. Tasks85-09–19
repair CUDA numerics, lifetime, MTP and maintenance, then measured successors
advance promotion, speed, scale, and route policy. Explicit Luna Medium/High and first retry High;
Wiretail shared defaults unchanged. Scheduling/usage state is `WORK_STATE.json`.

```bash
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
PROJECT_BRANCH=plan/attention-aware-kv-paging /srv/wiretail/wiretail.sh
```

Each task explicitly lists `context_files`. Read that list, the current packet,
its new `repair85-*` cluster and repository instructions; don't recursively read old plans,
the full state/log, old acceptance gates or all dependency handoffs. V10 folds
in the useful findings and corrects stale-bundle/wrong-workload claims. New
cluster IDs prevent old-session reuse. Every task has a named-proof completion
check; see [receipt contract](v10/RECEIPTS.md).

V9 and older plans/packets/handoffs/results remain historical, not active.
Do not PR any execution metadata/history upstream. The historical cleanup
inventory remains in `v9/OPERATIONS.md` for a future publication-only task;
do not load it in current implementation sessions. Current site/testing rules
are in [current testing](v10/REPAIR85_TESTING.md). Portable code commits remain separate.

Automatic selective paging is fast-route-only. `selected_reference` is a
temporary explicit correctness oracle, never a production fallback or valid
speed result. Unsupported automatic shapes must be refused with a reason until
their GPU-native dense, direct Turbo4, or bounded packed route is implemented;
see task [92-01](tasks/92-01.md).

The final review reads only the new benchmark summary. If the goal is unmet,
it must append actual remediation/benchmark/summary/review task entries before
completing, not merely recommend another action in prose.
