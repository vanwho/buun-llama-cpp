# Attention-aware KV paging execution package

Current authority (2026-09-12): [PHASE26_INTERACTIVE_KV_STRATEGY.md](PHASE26_INTERACTIVE_KV_STRATEGY.md)
and [BENCHMARK_PROTOCOL_V7.md](BENCHMARK_PROTOCOL_V7.md), then the current packet
and cluster in WORK_STATE.json. Read the compact current context, not historical
acceptance chains or the old full-context retry diary.

Resume command:

```bash
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp PROJECT_BRANCH=plan/attention-aware-kv-paging /srv/wiretail/wiretail.sh
```

Phase26 now fixes typed units, actual B/U launch, role/route scratch,
prefill-to-decode frontiers and natural recall, then measures kernel dispatch,
maintenance and memory trade-offs on the primary8192 logical/4096 hot fixture.
Phase27 tests32768/16384 then up to131072 with safe maximum H and incremental
inputs. Phase28 reviews only INTERACTIVE27_SUMMARY and creates targeted further
remediation if needed. No active256K-first or six-coordinate curve requirement.

154 completed task records, all reported token usage and dirty implementation
are preserved. Five old unfinished packets are replaced by12 focused tasks in
8 fresh clusters, beginning26-01 (todo/ready). Old packet/handoff/state context
is in archive/phase26-before-interactive-20260912/. Historical negative evidence
is preserved, not promoted to success. New source findings are in
evidence/PHASE26_REPLAN_FINDINGS.md.

Use WORK_STATE.json for exact task recommendations: Luna Medium for bounded
driver/measurement work, Luna High for critical source repairs; reviewer28-01
retains the prior reviewer's configured model. Wiretail tool defaults unchanged.
Target and native-MTP K/V remain Turbo4; full resolved-context GPU MTP, not H.
B128/U64 is a starting experiment, not a universal default or input length cap.
Use sudo -n for authorized protected files/Qwen lifecycle. Keep successful
candidate loaded; protect unrelated8092. Source stays generic and site data
stays in /srv/ai. No new speed benchmark was run during this plan revision.

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
