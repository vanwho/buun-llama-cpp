# Attention-aware KV paging execution package

Start with `ATTENTION_AWARE_KV_PAGING_PLAN.md`, then use `WORK_STATE.json` as the sole task-order and
status authority. Each task packet under `tasks/` is written to be executable without loading the full
plan. Each `WORK_STATE.json` cluster has a matching shared context under `clusters/`, allowing two or
three tightly coupled tasks to reuse one session without carrying unrelated phase history. Handoffs
belong under `handoffs/` and must record commands, results, changed files, unresolved risks, and deferred
hardware or human actions.

Foundation phases 00–07 are historical completed work. Production-completion phases 08–14 contain the
original implementation, phase 15 is the corrective live-product repair/re-acceptance series, and phase
16 is the shifted handoff/reproducibility series. The current state is intentionally `ready`, not
`complete`. Native MTP capacity follows the resolved target context and target hot capacity is derived
from the runtime memory ledger; no prior Fast-profile hot count is a default.

The wrapper defaults to auto Git mode. For supervised upstream-bound work, explicitly select manual mode:

```bash
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
GIT_MODE=manual \
/srv/codex/run_until_complete_clustered.sh --status
```

The shared runner and its adjacent `/srv/codex/task_state.py` executable are the only execution
programs. `task_state.py` receives `PROJECT_ROOT` from the runner; all state, logs, packets, and
handoffs remain under that project root.

The checked-in wrapper sets and verifies those safe defaults:

```bash
tool/codex/run_clustered.sh --status
tool/codex/run_clustered.sh --show-clusters
```

When the user explicitly authorizes autonomous fork commits and pushes, run the shared runner directly
in auto mode. Use the current plan branch as the integration branch because it contains the execution
package; set the integration branch explicitly as shown below:

```bash
cd /srv/repos/vanwho/buun-llama-cpp
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
PROJECT_REMOTE=origin \
PROJECT_BRANCH=plan/attention-aware-kv-paging \
GIT_MODE=auto \
CODEX_SESSION_MAX_TURNS=0 \
CODEX_SESSION_MAX_INPUT_TOKENS=0 \
/srv/codex/run_until_complete_clustered.sh
```

Leave `MAX_TASKS_PER_RUN` unset to run through all remaining tasks. Auto mode creates and pushes
temporary `codex/task-<id>` branches, merges them into the plan branch, and removes those temporary
branches. Each task's implementation commit excludes `docs/execution/**`; a following completion commit
contains the state, work log, handoff, and other execution metadata. This keeps code commits clean for
upstream range-diffs while retaining resumable controller history in the fork. Upstream-facing issues,
pull requests, and merges remain human-owned.

The shared runner and wrapper default both proactive session-rotation guardrails to `0` (disabled). Set a
positive value only when you deliberately want conversation rotation for context hygiene; these settings
do not truncate prompts. The Codex service's own hard context/usage limits still apply. The state validator
also requires every task's cluster context file.

In manual mode, the runner must not author commits, push, merge, create issues, or create pull requests. In
auto mode it may commit/push/merge only on the configured `vanwho/*` fork; a human owns upstream-facing
prose, PR creation/replies, and merges.
