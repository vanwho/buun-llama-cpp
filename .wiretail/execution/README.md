# Attention-aware KV paging execution package

Current authority: [GPU_HOT_PATH_REDESIGN_V8.md](GPU_HOT_PATH_REDESIGN_V8.md),
[BENCHMARK_PROTOCOL_V8.md](BENCHMARK_PROTOCOL_V8.md), current packet and cluster
in WORK_STATE.json. Read current context, not historical acceptance diaries.

Resume (runner deliberately stopped for the user-requested replan):

```bash
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp PROJECT_BRANCH=plan/attention-aware-kv-paging /srv/wiretail/wiretail.sh
```

Next:27-02, kernel memory repair and small CUDA/live attribution.27-03 restores
mature FA. Phase28 implements bounded GPU per-layer selection, reusable inputs,
optimized paged decode/prefill and asynchronous layer transfers. Phase29
produces speed/capacity evidence;30-01 assesses only HOTPATH29_SUMMARY.

13 unfinished tasks in11 fresh clusters replace the3 old remaining packets.
163 completed tasks and every reported usage record are preserved. The former
27-02 scaling campaign is interrupted negative evidence, not completion.
Its raw data and runtime fingerprints: evidence/GPU_HOT_PATH_AUDIT_20260912.md.
Superseded packets: archive/before-gpu-hot-path-v8-20260912/.
Implementation/benchmark tasks use Luna High; the previous higher-model reviewer
moves from28-01 to30-01 unchanged. Wiretail defaults are unchanged.

Target and native-MTP K/V remain Turbo4. MTP capacity equals full resolved L;
H is resident target capacity and A is per-layer attended work, not synonyms.
Primary8K/4K real cold data first, then32K/16K and useful bounded128K tests.
No fixed hot production size or256K/six-coordinate campaign in this repair.
Keep successful Qwen candidate loaded; use sudo -n for authorized operations;
unrelated8092 untouched. Source generic, site profiles/data in /srv/ai,
execution metadata commits separate. No runtime speedup is claimed by the plan.

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
