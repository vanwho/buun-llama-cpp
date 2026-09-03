# Attention-aware KV paging execution package

Start with `ATTENTION_AWARE_KV_PAGING_PLAN.md`, then use `WORK_STATE.json` as the sole task-order and
status authority. Each task packet under `tasks/` is written to be executable without loading the full
plan. Each `WORK_STATE.json` cluster has a matching shared context under `clusters/`, allowing two or
three tightly coupled tasks to reuse one session without carrying unrelated phase history. Handoffs
belong under `handoffs/` and must record commands, results, changed files, unresolved risks, and deferred
hardware or human actions.

For this upstream-bound C++ project, run the shared clustered runner only in local Git mode:

```bash
CODEX_PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
CODEX_GIT_MODE=local \
/srv/codex/run_until_complete_clustered.sh --status
```

The checked-in wrapper sets and verifies those safe defaults:

```bash
tool/codex/run_clustered.sh --status
tool/codex/run_clustered.sh --show-clusters
```

When the user explicitly authorizes autonomous fork commits and pushes, run the shared runner directly
in managed mode. Use the current plan branch as the integration branch because it contains the execution
package; the wrapper intentionally forces local mode and therefore must not be used for this variant:

```bash
cd /srv/repos/vanwho/buun-llama-cpp
CODEX_PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
CODEX_PROJECT_REMOTE=origin \
CODEX_PROJECT_BRANCH=plan/attention-aware-kv-paging \
CODEX_GIT_MODE=managed \
CODEX_SESSION_MAX_TURNS=4 \
CODEX_SESSION_MAX_INPUT_TOKENS=90000 \
/srv/codex/run_until_complete_clustered.sh
```

Leave `MAX_TASKS_PER_RUN` unset to run through all remaining tasks. Managed mode creates and pushes
temporary `codex/task-<id>` branches, merges them into the plan branch, and removes those temporary
branches. Each task's implementation commit excludes `docs/execution/**`; a following completion commit
contains the state, work log, handoff, and other execution metadata. This keeps code commits clean for
upstream range-diffs while retaining resumable controller history in the fork. Upstream-facing issues,
pull requests, and merges remain human-owned.

The wrapper defaults each cluster thread to four turns or 90,000 reported input tokens before rotation.
These are context-efficiency guardrails and can be lowered with `CODEX_SESSION_MAX_TURNS` or
`CODEX_SESSION_MAX_INPUT_TOKENS`. The state validator also requires every task's cluster context file.

The runner must not author commits, push, merge, create issues, or create pull requests. A user-authorized
outer agent may commit/push `vanwho/*`; a human owns upstream-facing prose, PR creation/replies, and merges.
