# Admission defect carried from deferred phase 93

Disposition, 2026-09-27: the user authorized deferring 93-11n and implementing
the forward architecture first. The old campaign did not pass. Its erroneous
`done` state is corrected to `deferred`; its failed executable proof remains
failed. Completed selector repairs and current dirty code must be preserved.

Attempt 13: page 20, page generation 57, content version 355 was cold and
host-backed before request 3. Query generation 211, position 10636, returned
`[20,21]`; the mailbox published and authenticated page 20. The trace records
`policy_admitted=false`, zero H2D, no mapping publication or later target use.
Its observed outcome is `selected_pending`: this alone does not distinguish
policy never evaluated from an evaluated target omission. Generation 212 later
returned `[-1,-1]`; it must not erase the earlier authenticated nomination.
GPU Turbo4 MTP placement passed; answer correctness failed independently.

Relevant current code is `llama_kv_cache::apply_pager_live_policy()`:
mailbox take/readback -> identity authentication -> resident/cold aggregation ->
`boundary.retrieval.selected` -> `llama_kv_live_policy_build_trace()` ->
`llama_kv_policy_decide()` -> transfer plan -> `pager_->apply_live_policy()`.
There are two target-decision passes, old recent/structural quotas, per-layer
attention limits, a two-page transfer cap, and query-generation guards. These
are inspection points, not a proven diagnosis of the particular rejection.

94-01a implements authoritative, validated query-boundary historical admission
instead of more legacy per-token ranking patches. 95-03 proves the actual
query/cold-nomination/admission/H2D/publication/replay/target-use chain. No
forced selector IDs, page-20 special cases, filename-format gates or old A/B/A
prerequisite. A focused synthetic fixture tests internal admission mechanics;
it is never labeled natural model retrieval.

Raw root: `/srv/ai/paged-kv/results/v10/93-11n/attempt-13/final-campaign/`.
Direct trace: `cases/PY_MERGE_03/request-03/selector-trace-poll-snapshots.json`.
Candidate/model identity is in `candidate-identity.json`; full logs stay there,
not in task startup context. Do not reload raw Codex JSONL or the attempt diary.
