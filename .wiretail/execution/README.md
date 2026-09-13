# Current hot-KV execution package

Start with [V9 overview](v9/OVERVIEW.md). Phases33–38 contain 23 concrete tasks,
all Luna Medium with an opt-in Luna High first retry. Wiretail shared defaults
are unchanged. Current scheduling/usage state is `WORK_STATE.json`.

```bash
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
PROJECT_BRANCH=plan/attention-aware-kv-paging /srv/wiretail/wiretail.sh
```

Each task explicitly lists `context_files`. Read that list, the current packet,
its new cluster and repository instructions; do not recursively read old plans,
the full state/log, old acceptance gates or all dependency handoffs. V9 folds
in the useful historical findings. New cluster IDs prevent old-session reuse.

Old plans and README contracts are under `archive/before-v9-20260913/`; old
packets/handoffs/results remain historical, not active. Do not PR any execution
metadata/history upstream. The cleanup inventory and site commands are in
[operations](v9/OPERATIONS.md). Portable implementation commits remain separate.

The final review reads only the new benchmark summary. If the goal is unmet,
it must append actual remediation/benchmark/summary/review task entries before
completing, not merely recommend another action in prose.
