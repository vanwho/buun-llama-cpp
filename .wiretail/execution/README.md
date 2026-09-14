# Current hot-KV execution package

Start with [V10 overview](v10/OVERVIEW.md) and the
[2026-09-14 audit](v10/ASSESSMENT.md). Phases49–52 contain19 concrete tasks,
with explicit risk-based Luna Medium/High and High first retry. Task48-01's
user-requested audit is complete; next implementation task is49-01. Wiretail
shared defaults are unchanged. Scheduling/usage state is `WORK_STATE.json`.

```bash
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
PROJECT_BRANCH=plan/attention-aware-kv-paging /srv/wiretail/wiretail.sh
```

Each task explicitly lists `context_files`. Read that list, the current packet,
its new cluster and repository instructions; do not recursively read old plans,
the full state/log, old acceptance gates or all dependency handoffs. V10 folds
in the useful findings and corrects stale-bundle/wrong-workload claims. New
cluster IDs prevent old-session reuse. Every task has a named-proof completion
check; see [receipt contract](v10/RECEIPTS.md).

V9 and older plans/packets/handoffs/results remain historical, not active.
Do not PR any execution metadata/history upstream. The historical cleanup
inventory remains in `v9/OPERATIONS.md` for a future publication-only task;
do not load it in current implementation sessions. Current site/testing rules
are in [testing](v10/TESTING.md). Portable code commits remain separate.

The final review reads only the new benchmark summary. If the goal is unmet,
it must append actual remediation/benchmark/summary/review task entries before
completing, not merely recommend another action in prose.
