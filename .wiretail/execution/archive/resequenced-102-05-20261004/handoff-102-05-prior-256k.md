# 102-05 handoff

## Result and current state
- Task state is `blocked`; both named 256K proofs are truthfully recorded as failed. The state CLI preserved `current_task=102-05` and `current_phase=102`; no completion status is claimed.
- The managed CUDA0 Qwen candidate is loaded at L=262144 with full-L GPU Turbo4 MTP: 276,955,136 bytes for 262,144 rows. Candidate binary/model/DSO/PID/start-time/argv identity is recorded in each attempt's `repo-content-preflight.json`.
- Requested H=59,904 admitted only 62 pages under current measured VRAM headroom. Restarted with measured H=15,872 (62 pages), keeping L/B/U and all other settings fixed.
- First zero-generation preflight found the runner filled to the reserve boundary (projected A2 prompt 259,179, only 2,965 tokens below L) instead of honoring the requested C=250,000 target. It sent zero generations. Preserve its raw record at `/srv/ai/paged-kv/results/forward/102-05/attempt-01/preflight.log`; the saved candidate-bound preflight is in `occupancy/repo-content-preflight.json`.
- Corrected the schedule fill cap to the requested occupied target (never beyond measured reserve capacity) and enabled the 262144 validator geometry. The corrected preflight will use a fresh output directory because the source identity changed.
- Corrected preflight selected 79 ranges/18 turns with A2 at 252,126 tokens, leaving 10,018 tokens below L. The resume-preflight path had a control-flow bug and sent A1 then B when asked to validate the frozen plan. Preserve this run at `occupancy-target250k`; A1 succeeded with 400 output tokens and MTP drafted/accepted 365/48, then B failed with a captured CUDA OOM while reserving a 160.94 MiB graph buffer (only 212 MiB free). No occupancy beyond 4,289 tokens committed; do not retry this unchanged H=15,872 row.
- Fixed `--preflight-only --resume-state` to return after validating the candidate/source/slot-bound checkpoint, before any generation. H=4,096 (16 pages) could not freeze the unchanged B fixture because its page-safe fresh-query budget left no repository range; no request was sent. The H=15,360 retry is two pages below the OOM row and keeps L/B/U and model/codecs/MTP unchanged.
- Final H=15,360 preflight caps the last repository chunk at the requested frontier: A2 prompt is 250,121 tokens, leaving 12,023 tokens below L; 78 ranges/19 turns, zero generations. Resumed `--preflight-only` returned the same plan in 6.3s with `requests_sent=0`.
- Retrying at H=15,360 reproduced the identical B-turn CUDA OOM (160.94 MiB graph allocation failed with 220 MiB reported free). The report shows 121,111,000 bytes target pool, 259,523,000 bytes scratch high-water, and 28 physically accounted pages. This lower-H row therefore did not progress beyond the 4,289-token A1 prefix.
- H=8,192 was not a viable preflight geometry for the fixed A/B workload (no repository range fits its H-derived query budget). H=15,104 (59 pages) was the smallest page-aligned value expected to admit the measured 14,301-token B request. The managed server is now restarted at H=15,104, with full-L MTP reservation confirmed and 1,268 MiB free immediately after startup.
- Fresh H=15,104 preflight passed with 78 ranges/19 turns, projected A2 prompt=250,123, gap to L=12,021, and zero generations. A second `--resume-state --preflight-only` returned the same frozen schedule with `requests_sent=0`. During the live row A1 passed (3,890 prompt/400 output; MTP drafted/accepted 286/206), but B ended at 18,595 preflight prompt tokens with `packed selected attention allocation failed`; the stream generated no output and the frontier remained 4,289 tokens. The raw request, SSE, and journal are in `occupancy-h15104/`; CUDA reported repeated 7.22 MiB allocation failures and 118 MiB was free at failure.
- The task requires retrying OOM/scratch failures by lowering only H in page-aligned steps. H=14,848 (58 pages) is now active; L/B/U/model/MTP and all other service settings remain fixed. Fresh and resumed preflight both pass with 78 ranges/19 turns, projected A2=250,126, and zero requests in preflight. The H=15,104 row is incomplete and is not a proof.
- H=14,848 A1 again passed at 3,890 prompt/400 output with MTP drafted/accepted 286/206. B again failed before generation with `Compute error` while allocating a 160.94 MiB CUDA graph buffer; 228 MiB was free. The resulting page capacity/high-water remained 28 pages and 121,111,000 bytes, the same as H=15,104; frontier stayed at 4,289 and validator correctly rejected the row. Raw campaign, SSE, journal, candidate-bound frontier, and validation failure are under `/srv/ai/paged-kv/results/forward/102-05/attempt-01/occupancy-h14848/`.
- H=14,848 is the lowest query-capable page geometry for this frozen B request: the successful preflight admitted a 14,069-token fresh B delta with effective cap 14,081 (58 hot pages, two generation-write pages). The next 256-token reduction to H=14,592 lowers the page-safe cap to 13,825, below the unchanged B delta, so it cannot produce a candidate-compatible schedule. Further H reduction would make the fixed A/B workload unqueryable before it could test the OOM allocation. Keep the server on the last viable H=14,848 row.
- Overall result remains incomplete: only 4,289 tokens committed versus the requested 250,000; no near-full promotion/replay/ring proof exists. Full-L GPU MTP reservation is independently observed at 262,144 rows/276,955,136 bytes. Do not claim the named occupancy or integrated promotion proofs.
- `.wiretail/execution/evidence/V10_102-05.json` records both named proofs as `fail`, with candidate identity, exact campaign argv, and hashed raw artifacts. The required completion check was run and correctly rejected the receipt because neither named test passed; raw output is `/srv/ai/paged-kv/results/forward/102-05/attempt-01/completion-check.log`.

## Decisions and invariants
- Update occupancy schedule to stop corpus growth at the requested committed frontier while retaining the measured final-query/output/MTP/replay reserve; do not run the saved boundary-filling selection.
- Keep L=262144, B/U=1024/256, Turbo4 target/draft KV, GPU model/MTP, one slot, no context shifting. H=14,848 is the last frozen geometry that admits the required B request under the <=16K fresh limit.
- The A1/B/repository-prefix/A2 workload must be preflighted in a fresh candidate/source-bound output directory before generation. Max fresh chunk remains <=16K; every successful request gets a durable journal/checkpoint.

## Changed files and symbols
- `tools/server/bench/run-occupancy-frontier.py`: `build_repo_schedule` now receives the requested target and stops corpus growth there instead of filling all available context up to the final reserve ceiling.
- `tools/server/bench/run-occupancy-frontier.py`: resumed preflight now validates and reuses the checkpoint without sending scheduled requests.
- `tools/server/bench/run-occupancy-frontier.py`: final corpus request is capped by remaining target distance while preserving A2 reserve.
- `tools/server/bench/validate-occupied-frontier.py`: accepts L=262144.
- `tools/server/bench/test_occupancy_frontier.py`: asserts 262144 validator support and target-bounded repo schedule metadata.

## Validation and raw evidence
- `PYTHONPATH=tools/server/bench python3 -m unittest test_occupancy_frontier` — passed, 14 tests after the harness change; raw log `/srv/ai/paged-kv/results/forward/102-05/attempt-01/occupancy-runner-tests.log`.
- `PYTHONPATH=tools/server/bench python3 -m unittest test_pager_promotion` — passed, 15 tests.
- `python3 tools/server/bench/validate-occupied-frontier.py --frontier /srv/ai/paged-kv/results/forward/102-05/attempt-01/occupancy-h14848/occupied-frontier.json --output /srv/ai/paged-kv/results/forward/102-05/attempt-01/validation-occupancy-h14848.json` — correctly failed: only 4,289 committed tokens, below H+8 pages and the requested frontier; target valid rows/bytes were not measured.
- `python3 .wiretail/execution/v10/validate.py --task 102-05 --receipt .wiretail/execution/evidence/V10_102-05.json` — correctly failed both required named proofs (`named test must actually pass`).
- Intermediate preflight: 79 selected ranges, 18 turns, A2=252,126; superseded by final target cap.
- Final H=15,360 plan: 78 ranges, 19 turns, A2=250,121, reserve gap=12,023; `/srv/ai/paged-kv/results/forward/102-05/attempt-01/preflight-h15360-final.log` and `/srv/ai/paged-kv/results/forward/102-05/attempt-01/preflight-resume-h15360-final.log`.
- OOM raw request/SSE, checkpoint and journal remain under `/srv/ai/paged-kv/results/forward/102-05/attempt-01/occupancy-target250k/`; diagnostic `cudaMalloc` excerpts are in the task system journal and copied into the final task raw log before completion.
- H=15,104 fresh and resumed preflight logs: `/srv/ai/paged-kv/results/forward/102-05/attempt-01/preflight-h15104.log`, `preflight-resume-h15104.log`; frozen candidate-bound plan is under `occupancy-h15104/`.
- H=14,848 fresh and resumed preflight logs: `/srv/ai/paged-kv/results/forward/102-05/attempt-01/preflight-h14848.log`, `preflight-resume-h14848.log`; frozen candidate-bound plan is under `occupancy-h14848/`.
- Live H=14,848 campaign: `/srv/ai/paged-kv/results/forward/102-05/attempt-01/campaign-h14848.log`; failed occupancy validation: `/srv/ai/paged-kv/results/forward/102-05/attempt-01/validation-occupancy-h14848.json`.
- Full raw attempt root: `/srv/ai/paged-kv/results/forward/102-05/attempt-01/`.

## Deferred verification and next action
- The managed CUDA host is available, but its measured VRAM headroom cannot admit the required selected-attention graph at the last H that still admits the fixed B query. Full-L allocation succeeded; near-full committed occupancy and integrated promotion/replay/ring/memory did not. No hardware success is inferred from the failed rows.
- No smaller page-aligned H can preserve the fixed A/B query schedule; resolving this requires a new source/runtime or capacity-directed follow-up, not weakening L, B/U, model, codecs, draft, or the task's fresh-token bound.
- Resume only after a source/runtime or capacity-directed repair makes the selected-attention graph fit while preserving L=262144, B/U=1024/256, full-L GPU Turbo4 MTP, the fixed A/B workload, and the <=16K fresh-request contract. Start from this handoff and the H=14,848 raw row; do not call `task_state.py complete` against this receipt.

## Wiretail status
<!-- wiretail:runner-status:start -->
- State: `blocked`
- Source of truth: `.wiretail/execution/WORK_STATE.json`
- Blocker: Full-L GPU MTP allocation succeeds, but selected-attention graph allocation repeatedly fails at B: 160.94 MiB CUDA allocation with 228 MiB free at the last query-capable H=14848 row. H=14592 would cap fresh B at 13825 tokens below the measured 14069-token B delta, so no lower page-aligned H preserves the frozen A/B schedule. Only 4289 tokens committed; both required 256K proofs fail. See .wiretail/execution/handoffs/102-05.md and /srv/ai/paged-kv/results/forward/102-05/attempt-01/
<!-- wiretail:runner-status:end -->
