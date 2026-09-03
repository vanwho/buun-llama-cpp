# Cluster 15a — final rebase, portable handoff, and clean reproduction

Tasks: `15-01`, `15-02`, `15-03`.

Purpose: leave a clean, understandable fork series and reproducible operator/evidence package after all
live gates pass.

Carry forward:

- fetch upstream and re-read `CONTRIBUTING.md`/`AGENTS.md`; do not overwrite unrelated work or silently
  resolve semantic conflicts during rebase;
- partition generic draft placement, dynamic sizing/accounting, page core, host/CUDA residency, Turbo4
  attention, routing/telemetry/policy, Qwen/server lifecycle, exact mode, tests, and docs into dependency-
  ordered review slices;
- implementation commits exclude `docs/execution/**` and machine paths. Execution metadata stays in
  separate fork-only commits and will not be proposed upstream;
- run portability, secret, generated-file, large-artifact, and server-specific-path scans; retain Qwen
  numeric geometry only in capability tests or measured evidence;
- regenerate help/docs, build Release and test configurations from a clean worktree, rerun a bounded
  smoke and verify raw artifact hashes/pointers;
- follow the AI policy: agents may prepare code, slice maps, checklists, and factual evidence, but never
  write/post GitHub issue or PR prose, replies, or review messages. A human owns those actions.

Read: plan sections 13, 16–18, 25–26; `CONTRIBUTING.md`; old and new slice maps; all phase 14 handoffs.

Exit gate: upstream relation is recorded, branch/slices are clean and pushed to the user's fork by the
outer auto runner, all tests and bounded reproduction pass, final evidence links resolve, and
`WORK_STATE.json` becomes complete only when every phase 08–15 task is done. No acceptance item may be
converted to deferred during packaging.
