# Long-horizon context, accounting, and handoff policy

This compact policy is injected into each Wiretail task and recovery-assessment
prompt. It supplements the current task packet; it does not replace its scope or
acceptance criteria.

## Build parallelism

For every build in this project, use CMake's `--parallel N` rather than
defaulting to `-j2`. On this 16-core/32-thread Ryzen 9 7950X3D host, every
project build—CPU or CUDA, clean or incremental—must use exactly
`--parallel 16`. Do not silently reduce or increase the job count for a slow
build; diagnose the actual bottleneck first. If parallel-16 compilation has a
verified resource failure, record it and ask the operator before using a
different count. This is local execution guidance, not a value for portable
product code. Use `--clean-first` only when the task requires a clean candidate;
it is not a general speed setting. If an active packet still specifies another
job count, correct its command to `--parallel 16` before building.

## Current phase-103/104 override

103-02a completed transport/owner checks; its compact handoff records ranking
misses and separate queued-but-unpublished transfers. 103-03 is deferred:
C=254393 was committed, but exact C=L is unproven after an attention-ubatch
planning failure. Do not resume its campaign, inherit its exact-fill helper
archive, or claim that ranking changes solve that capacity boundary.
Next104-01–08 run an isolated ranking experiment; then
103-04/05 consume its verdict for main benchmark/review. RANKING104.md and
the current packet's named sections supersede legacy Mean-Q/Mean-K algorithm
directions only inside this experiment. Never load old transcripts, full
request bodies, retired103-02 or unregistered103-02b as active context.

The experiment uses /srv/repos/vanwho/buun-llama-cpp-ranking-v1 and branch
experiment/attention-aware-ranking-v1 created from the then-current local
plan branch. Explicit experiment Git milestone commits are task-authorized;
the main project's branch/commits remain owned by Wiretail. Do not switch
the active main worktree, run nested Wiretail, modify the experiment's copied
state ledger, merge it into plan or change main default routing implicitly.
Keep GPU Turbo4 full-L MTP, B1024/U256 and frozen history during generation.
Experiment misses and speed losses become recorded decisions, not endless
retries. Only the current packet/design slice and immediate compact handoff
are startup context; source paths identify symbol regions to inspect.

Forward OVERVIEW/TESTING remain the main architecture contract. Old93/100
repair plans, SOURCE_FORWARD_PLAN, retired packets, acceptance diaries and
whole predecessor gate sets are historical references, not startup context.
Resume the first unfinished task in WORK_STATE rather than a historical task
number. Source entries are symbol pointers: use rg and selected regions,
never ingest all of server-context.cpp, llama-context.cpp or llama-kv-cache.cpp.
Verify the current packet's receipt before completion; metadata correction
preserves prior implementation commits and every usage event.

## Token accounting: what the numbers mean

Wiretail records one detail entry per Codex JSONL `turn.completed` usage object.
That is the accounting unit exposed by the stream, not a per-generation record
and not the active prompt/context-window size. A long task can perform many
model/tool cycles, so task and phase totals can be very large without any one
model request having that many context tokens. The current stream does not
provide a reliable per-generation or active-context measurement; do not infer
either from the accumulated input total.

- `input_tokens` is the reported input total for the completed-turn usage
  object. `cached_input_tokens` and `cache_write_input_tokens` are retained as
  separate reported fields; do not add them on top of input or total.
- `output_tokens` includes `thinking_tokens`/`reasoning_output_tokens` as a
  subset. Do not add thinking again when computing total usage.
- `total_tokens` is the Codex-reported total when present; otherwise Wiretail
  derives it as `input_tokens + output_tokens` and marks the source. It does not
  add cached, cache-write, or thinking subsets again.
- `input_tokens_minus_cached_tokens` is explicitly just input minus the
  reported cached subset; it is not labeled as all uncached work.
- A missing field is `null`/unknown, not a measured zero. Known-field counts
  make partial aggregates visible.
- Task state owns detailed usage events. Phase and project state contain
  aggregate totals only, not duplicate copies of every event.

The `CODEX_SESSION_MAX_INPUT_TOKENS` option is an opt-in task-boundary session
rotation guardrail, disabled by default. It compares the latest completed
invocation's sum of known `input_tokens` across its `turn.completed` events. If
no input metric was reported, it remains unknown and cannot trigger rotation;
it is not a prompt/context limit and never stops or truncates a task. Context
compaction is a separate Codex feature.

## File-backed long-horizon work

The working session is useful continuity, not the source of truth. Trust the
current worktree, current task entry, packet/cluster, named evidence, and
handoff. With `context_files`, read only the current task/cluster, current task
handoff if present, repository instructions, and those selected files. A
dependency establishes order; it does not authorize loading all predecessor
plans, diaries, or raw transcripts. Do not dump the whole work state or work
log into a prompt. Use Wiretail status or a narrow current-task projection for
state; inspect legacy long handoffs by headings and current outcome/state first,
then retrieve only task-relevant sections or excerpts.

Wiretail enables Codex automatic history compaction for every task and
assessment. By default, its threshold is 65% of the selected model's context
window from the local bundled model catalog, with `total` scope; if that model
cannot be resolved, Wiretail leaves the model's own threshold in force. Tool
outputs retained in the Codex history are capped at 12,000 tokens by default.
The operator can override these with `CODEX_AUTO_COMPACT_TOKEN_LIMIT`,
`CODEX_AUTO_COMPACT_TOKEN_LIMIT_SCOPE`, and `CODEX_TOOL_OUTPUT_TOKEN_LIMIT`.
Native Codex compaction preserves session continuity; it does not replace
checkpoint files or justify carrying irrelevant project history.

For long tasks, update the current task handoff in place at meaningful verified
milestones and before a long-running test. Use it as a current-state snapshot,
not an append-only diary. Prefer these sections, following the packet's exact
required headings where specified:

1. Result and current state
2. Decisions/invariants that the next step must preserve
3. Changed files and relevant symbols
4. Exact validation commands/results and raw artifact paths
5. What remains, with the next concrete action
6. Loaded candidate/profile identity and safe resume state, when relevant

Aim for at most 120 lines. Put full command logs, benchmark records, request
bodies, and attempt-by-attempt details in the raw result root; reference them by
path and hash. Replace stale handoff narratives with the latest diagnosis and
archive detailed history only when the task's evidence contract requires it.
For verbose commands, redirect complete output to a file and show only concise
summaries or targeted excerpts to Codex. Never weaken verification to save
context.

Reuse a session across adjacent tasks when the cluster shares source area,
design decisions, and test setup. Let native compaction manage a long cohesive
session. Start a fresh session at an actual cluster/risk/context boundary, after
stale assumptions, or when recovery would inherit noisy irrelevant history;
seed it from the compact handoff and selected files. Do not force a fresh
session for every task or preserve a thread merely for its own sake.
