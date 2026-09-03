# Attention-aware KV paging execution package

Start with `ATTENTION_AWARE_KV_PAGING_PLAN.md`, then use `WORK_STATE.json` as the sole task-order and
status authority. Each task packet under `tasks/` is written to be executable without loading the full
plan. Handoffs belong under `handoffs/` and must record commands, results, changed files, unresolved
risks, and deferred hardware or human actions.

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

The runner must not author commits, push, merge, create issues, or create pull requests. A human owns
those actions and all GitHub-facing prose.
