# Long-horizon context, accounting, and handoff policy

This compact policy is injected into each Wiretail task and recovery-assessment
prompt. It supplements the current task packet; it does not replace its scope or
acceptance criteria.

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
