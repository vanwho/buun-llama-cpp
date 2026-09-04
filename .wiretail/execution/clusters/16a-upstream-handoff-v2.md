# Cluster 16a — rebase and portable handoff preparation

Tasks: `16-01`, `16-02`.

Purpose: leave a clean, understandable fork series and portable operator/evidence package before phase 17.

Carry forward:

- fetch upstream and re-read `CONTRIBUTING.md`/`AGENTS.md`; task agents do not perform Git mutations. If
  the integration base moved, an authorized outer Git owner must supply a clean updated branch before
  source-level reconciliation continues;
- partition generic draft placement, dynamic sizing/accounting, page core, host/CUDA residency, Turbo4
  attention, routing/telemetry/policy, Qwen/server lifecycle, exact mode, tests, and docs into dependency-
  ordered review slices;
- implementation commits exclude `.wiretail/execution/**` and machine paths. Execution metadata stays in
  separate fork-only commits and will not be proposed upstream;
- run portability, secret, generated-file, large-artifact, and server-specific-path scans; retain Qwen
  numeric geometry only in capability tests or measured evidence;
- regenerate help/docs, build Release and test configurations from a clean worktree, rerun a bounded
  smoke and verify raw artifact hashes/pointers;
- follow the AI policy: agents may prepare code, slice maps, checklists, and factual evidence, but never
  write/post GitHub issue or PR prose, replies, or review messages. A human owns those actions.

Read: plan sections 13, 16–18, 25–26; `CONTRIBUTING.md`; old and new slice maps; all phase 14 handoffs.

Recovery note: a `codex_core::tools::router` `apply_patch verification failed` diagnostic is a stale
patch-anchor failure, not a runtime acceptance blocker. Re-read the current file and use a minimal exact
patch; never repeat the same failed patch or append duplicate evidence. The runner rotates the session
for the next substantive attempt and supplies this signal to the retry assessor. Its transient recovery
counter resets on a new runner invocation, so an externally repaired blocker remains resumable.

Exit gate: upstream relation is recorded, branch/slices are clean and pushed to the user's fork by the
outer auto runner, and portable documentation/evidence links resolve. Runtime completion is intentionally
owned by phase 17 and the benchmark-only phase 18 review; no acceptance item may be converted to deferred
during packaging.
