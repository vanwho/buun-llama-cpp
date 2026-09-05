# Superseded unstarted execution packets

These five original packets were unstarted when the post-17 revision was made
on 2026-09-05. They are retained for history, not referenced by active state.

| Old task | Replacement |
| --- | --- |
| 17-16 performance | 17-16 is now the bridge; actual measurements 21-02/03 |
| 17-17 pressure soak | 21-04 after live implementation |
| 17-18 full256K/curve | 19-08 implementation proof, then 21-01/02 release measurement |
| 17-19 consolidation | 21-06 |
| 18-01 review | 22-01 Sol High; 18-01 now runtime repair |

No active/completed task data or raw benchmark outputs were removed. Historical
cluster files may still mention these old IDs; only WORK_STATE.json assigns
executable clusters. All execution folders (old docs/execution, tool/codex,
build/codex-autonomous, .codex-runner/.codex_runner and current .wiretail) and
their commits are excluded from upstream code slices. Keep cleanup/rebase work
separate from a running task; do not delete ordinary repository docs.
