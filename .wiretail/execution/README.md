# Attention-aware KV paging execution package

For work after the running 17-15, start with `POST17_IMPLEMENTATION_STRATEGY.md`,
then the current packet's `TECHNICAL_CHANGE_SPEC.md` sections and
`BENCHMARK_PROTOCOL_V5.md`. The large `ATTENTION_AWARE_KV_PAGING_PLAN.md` now
points to this amendment; old acceptance ledgers are historical context only.
Each future packet also names sections of `EXECUTION_COOKBOOK.md` for verified
commands, minimal fixtures and common evidence, and `IMPLEMENTATION_CONTRACTS.md`
for internal APIs, memory ownership, tensor layouts and checkpoint boundaries.
Read those named sections once per cluster, not the entire history. A command
or interface marked proposed must be implemented/tested before it is invoked.
Use `WORK_STATE.json` as the sole task-order and
status authority. Each task packet under `tasks/` is written to be executable without loading the full
plan. Each `WORK_STATE.json` cluster has a matching shared context under `clusters/`, allowing two or
three tightly coupled tasks to reuse one session without carrying unrelated phase history. Handoffs
belong under `handoffs/` and must record commands, results, changed files, unresolved risks, and deferred
hardware or human actions.

Foundation phases 00–07 and follow-on phases through the active 17-15 remain
historical/current evidence, not a proof that all live capabilities work.
17-16 now bridges that campaign to 18–20 implementation repairs/advancements.
Phase 21 produces new quality, context-speed, paired-control and physical-soak
evidence; **22-01 is the Sol High benchmark-only reviewer**. It creates a new
measured remediation chain if needed, not another historical closure audit.
All other task recommendations stay Luna High; Wiretail defaults are unchanged.
The project remains `in_progress`, not complete.
Native MTP capacity follows the resolved target context and target hot capacity is derived from the
runtime memory ledger; no prior Fast-profile hot count is a default.

The shared runner defaults to auto Git mode. For supervised upstream-bound work, explicitly select manual mode:

```bash
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
GIT_MODE=manual \
/srv/wiretail/wiretail.sh --status
```

The shared runner and its adjacent `/srv/wiretail/task_state.py` executable are the only execution
programs. `task_state.py` receives `PROJECT_ROOT` from the runner; all state, logs, packets, and
handoffs remain under that project root.

The shared runner provides status and cluster inspection:

```bash
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp /srv/wiretail/wiretail.sh --status
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp /srv/wiretail/wiretail.sh --show-clusters
```

To give the next task's initial agent prompt a one-shot operator directive, set
`PROMPT_PREFIX`. It is consumed when that task starts, is injected through
`build_prompt()`, and is not carried into substantive retries, recovery
assessments, or later tasks:

```bash
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
PROMPT_PREFIX='First inspect the current blocker and choose a distinct recovery path.' \
/srv/wiretail/wiretail.sh
```

When the user explicitly authorizes autonomous fork commits and pushes, run the shared runner directly
in auto mode. Use the current plan branch as the integration branch because it contains the execution
package; set the integration branch explicitly as shown below:

```bash
cd /srv/repos/vanwho/buun-llama-cpp
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
PROJECT_BRANCH=plan/attention-aware-kv-paging \
/srv/wiretail/wiretail.sh
```

Leave `MAX_TASKS_PER_RUN` unset to run through all remaining tasks. Auto mode creates and pushes
temporary `codex/task-<id>` branches, merges them into the plan branch, and removes those temporary
branches. Each task's implementation commit excludes `.wiretail/execution/**`; a following completion commit
contains the state, work log, handoff, and other execution metadata. This keeps code commits clean for
upstream range-diffs while retaining resumable controller history in the fork. Upstream-facing issues,
pull requests, and merges remain human-owned.

The shared runner defaults both proactive session-rotation guardrails to `0` (disabled). Set a
positive value only when you deliberately want conversation rotation for context hygiene; these settings
do not truncate prompts. The Codex service's own hard context/usage limits still apply. The state validator
also requires every task's cluster context file.

In manual mode, the runner must not author commits, push, merge, create issues, or create pull requests. In
auto mode it may commit/push/merge only on the configured `vanwho/*` fork; a human owns upstream-facing
prose, PR creation/replies, and merges.

When Codex reports usage on a `turn.completed` event, Wiretail persists the
turn under that task and aggregates it by phase and project in `WORK_STATE.json`.
Recorded fields are total, input, cached-input, output, and thinking/reasoning
tokens. If Codex omits total, Wiretail derives it as input plus output and
marks the turn with `total_tokens_derived`.
