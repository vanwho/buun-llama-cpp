# Attention-aware KV paging execution package

Current authority (20 26-09-11): [PHASE25_SPEED_FIRST_STRATEGY.md](PHASE25_SPEED_FIRST_STRATEGY.md)
and [BENCHMARK_PROTOCOL_V6.md](BENCHMARK_PROTOCOL_V6.md), then the current task
and cluster in WORK_STATE.json. Do not read the large historical plan or old
acceptance chains for routine execution.

Resume command:

```bash
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp PROJECT_BRANCH=plan/attention-aware-kv-paging /srv/wiretail/wiretail.sh
```

Phase 25 now fixes short-run measurement, incremental host/routing work, GPU
batching/attention kernels, real movement, graph reuse, MTP and whole-model
speed. Phase 26 proves populated 262144 context and records the final original-
question curve plus affordable controls. Sol High 27-01 reviews that compact
summary only. Broad quality/exact/soak/YaRN/upstream gates are not on this
speed-prototype critical path. Minimal byte/causal/lifetime checks remain.

The previous 25-02 blocker was re-scoped, not declared solved. 140 done records
and all reported token usage remain; old packets/handoff are archived in
archive/phase25-before-speed-first-20260911/. The first new task is 25-02 in a
fresh cluster. All implementation/measurement recommendations remain Luna
High; Wiretail's own defaults are unchanged.

Use WORK_STATE.json for order/status and the current packet for source seams,
fixtures and commands. Clusters contain at most3 consecutive tasks. Each
handoff is compact and points to durable raw evidence; don't read full runner
JSONL or historical ledgers unless a specific unresolved issue requires it.
Target/draft KV are always Turbo4; native GPU MTP follows resolved context,
not hot capacity. Source stays generic; machine config/data in /srv/ai or
execution metadata. Only the runner owns automated fork Git operations.

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
