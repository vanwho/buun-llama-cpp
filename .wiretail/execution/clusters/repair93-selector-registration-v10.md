# Cluster repair93-selector-registration-v10

Revision: `hotpath-v10-20260914`; repair: `repair93-selector-promotion-20260925`.

This is a fresh, single-task diagnostic cluster for 93-11o. The previous
93-11n live task accumulated roughly 90.96M reported tokens across three
attempts and ended with no selector query row captured. Do not resume or import
that Codex thread. Use only this cluster, the 93-11o packet, and the compact
93-11n handoff observation.

## Scope

Diagnose and repair the exact graph-registration/capture gate on the existing
Qwen path. Start with bounded source-level reason codes and a deterministic
regression. Only after those pass, reload/verify the candidate through the
managed single-service lifecycle and issue one short request to prove that a
real causal query row and position reach selector capture. The task does not
prove eviction, nomination, transfer, promotion, speed, or answer quality;
93-11n owns the later full campaign.

## Non-negotiable runtime constraints

- One managed Qwen process only; preserve the service after successful checks.
- Candidate identity must include server SHA, loaded implementation DSO SHA,
  model SHA, PID/start time, and effective argv before the request.
- Keep target and native-MTP K/V Turbo4/Turbo4, MTP on GPU, one slot, no forced
  page IDs, no reference-route substitution, and no port 8092 access.
- Diagnostic state is opt-in, fixed-size, request-correlated, and captured at
  graph-build/refresh boundaries. No tensor-content copies, per-token logs,
  full page-table scans, or new CUDA synchronization.
- Keep raw live output under `/srv/ai/paged-kv/results/v10/93-11o/`; task
  context receives only compact summaries/hashes.

## Fresh-session context

Do not read WORK_LOG, raw JSONL, the full WORK_STATE, previous session output,
or unrelated phase evidence. If an exact diagnostic field is needed, extract
only that field from the named raw attempt. On completion, leave a concise
handoff naming the diagnosed gate, source/test locations, candidate identity,
and the next action for 93-11n.
