# Historical handoff — pre-amendment 102-01 (32K occupancy attempt)

This is preserved provenance for the earlier 32K/16K live-occupancy task.
After amendment `repo-context-scale-20261002`, task 102-01 is a driver-limit
regression-only task and task 102-02 owns the new 8K/4K repo-content baseline.
Do not load this historical handoff as task context or treat its missing live
proof as a blocker for the amended task sequence.

## Result and current state

102-01 remains incomplete and is blocked pending its scheduled prerequisite. The occupancy driver, validator, and deterministic resume regression are implemented, but no candidate was loaded and no 32K live occupancy, semantic, CUDA, or speed run was started. `GPU101_RELEASE.json` still reports `goal_miss`; its passing validator confirms receipt consistency only, not release eligibility.

## Decisions and invariants

- Do not load/reload the candidate or run occupancy until the exact-candidate `GPU101_RELEASE.json` reports `goal_status=pass` after scheduled 101-12n review.
- Preserve L32768/H16384/B1024/U256, GPU Turbo4 target/draft placement, and the single-slot committed-frontier invariant. Resume requires matching executable/DSO/model/process/geometry/frontier identity.
- Keep the release `goal_miss`; do not synthesize live proof or a V10 receipt. 101-12n remains the scheduled next task; this attempt did not start it.

## Changed files and relevant symbols

- `tools/server/bench/run-occupancy-frontier.py`: configurable geometry, runtime identity hashing, effective-geometry checks, atomic checkpoint and request journal, resumable per-request artifacts, slot-generation/frontier confirmation, and explicit incomplete status.
- `tools/server/bench/validate-occupied-frontier.py`: validates fixture geometry, identity-bound committed request chain, DSO and model hashes, frontier threshold, per-request timings, and allocation values against the final runtime snapshot without legacy fixed-allocation assumptions.
- `tools/server/bench/test_occupancy_frontier.py`: deterministic committed-interruption/resume, unique numbering, incomplete-frontier, identity-rejection, and geometry-rejection regression.

## Validation and evidence

Raw command logs are in `.wiretail/build/102-01-attempt-3/`:

- `python3 -m py_compile tools/server/bench/run-occupancy-frontier.py tools/server/bench/validate-occupied-frontier.py tools/server/bench/test_occupancy_frontier.py` — pass; `py_compile.log` SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `python3 tools/server/bench/test_occupancy_frontier.py` — pass, 1 test; `occupancy-regression.log` SHA-256 `089d0839d954b9a44fdc2dd1b8565fcc1fa1a84d3c547561f4019d04b6674324`.
- `git diff --check` — pass; `diff-check.log` SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `PROJECT_ROOT="$PWD" python3 /srv/wiretail/task_state.py validate` — pass, 489 tasks; `state-validation.log` SHA-256 `ff64e488f3ad342ae205594ce75d288be0557784819d27da25a0e42fc003fdc5`.
- `python3 tools/server/bench/validate-gpu101-release.py --receipt .wiretail/execution/evidence/GPU101_RELEASE.json` — validator passes with `goal_miss`; `gpu101-release-validation.log` SHA-256 `2b8ffa97ff92ceff0f3dda80ee8d054c78e63dcf15746eb90d0dd1ae61237a1e`.
- Packet completion check `python3 .wiretail/execution/v10/validate.py --task 102-01 --receipt .wiretail/execution/evidence/V10_102-01.json` — unmet: receipt is absent; `v10-completion-check.log` SHA-256 `7e8e6e4720b2494340e985b9561b74717dcd7004890ed00e32abdf4a3c0f22cb`.

## Deferred verification

The required 32K/16K hardware proof, natural retrieval/replay/MTP proof, allocation/headroom ledger, and canonical speed suite remain unrun pending 101-12n review. Under the owner-approved policy, an MTP-only result below the 40% median floor remains a reported goal miss but does not block these capacity measurements after review. Candidate identity, Turbo4 GPU draft placement, state correctness, and setup failures remain hard requirements. The 102-01 V10 receipt must be generated only after the live proofs run.

## Next concrete action

Complete scheduled 101-12n's exact-candidate release review. Then proceed with 102-01 using the reviewed candidate even if the preserved release decision is `goal_miss` solely because of MTP acceptance; keep that miss explicit while measuring occupied-context MTP and capacity. Run the required live proofs and generate the V10 receipt only from actual evidence.

## Wiretail status
<!-- wiretail:runner-status:start -->
- State: `in_progress`
- Source of truth: `.wiretail/execution/WORK_STATE.json`
<!-- wiretail:runner-status:end -->
