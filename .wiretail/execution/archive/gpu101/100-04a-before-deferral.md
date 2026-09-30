# 100-04a handoff

## Result and current state

Task is currently `blocked` in WORK_STATE.json. The production loop routes final-user pager commit,
checkpoint restore, and replay choice through `server_query_replay_transition`,
but the fixture still exercises only that helper's callback branches. It does
not execute `server_query_checkpoint` capture/restore with real target/draft
contexts or compare decode/recurrent/MTP state. No V10 receipt was created.

Attempt 04 rebuilt and ran both available CUDA focused executables. The server
prompt-cache and selected CUDA consumer checks pass; they are supplemental and
do not establish `final_query_replay_one_pass_parity`.

## Decisions and invariants

- Preserve the server-owned checkpoint and one-replay policy in
  `server-context.cpp`; do not copy its decision logic into tests.
- Real parity must compare the same deterministic final-user input and page
  map through Path A checkpoint/commit/restore/replay and Path B commit/decode,
  including real target/draft decode, recurrent state, speculative carry,
  suffix handling, and generation-boundary bookkeeping.
- CUDA hardware is available, but the managed server currently uses about
  14.3 GiB of 16 GiB VRAM. The final hybrid candidate is a 14 GiB model at
  `/srv/ai/models/text/current.gguf`. Do not load another candidate or alter
  the managed service lifecycle to force the test.

## Changed files and relevant symbols

- `tools/server/server-context.h`: declares transition result/status and the
  shared `server_query_replay_transition` seam.
- `tools/server/server-context.cpp`: defines that transition and invokes it at
  the actual prompt loop boundary. `server_query_checkpoint::capture` and
  `restore` remain the production state owners.
- `tests/test-server-prompt-cache.cpp`: callback-branch tests only; not the
  integrated decode fixture.
- `tests/test-cuda-kv-promotion.cpp`: selected final-map CUDA attention proof;
  not server replay parity.

## Validation and raw artifacts

- `cmake --build build-cuda --target test-server-prompt-cache test-cuda-kv-promotion -j8`
  — PASS. Log:
  `/srv/ai/paged-kv/results/forward/100-04a/attempt-04/focused-build.log`
  SHA256 `f58f0858a3fb5851391659a75d18a4fa16d40c8967c1431089f6b825e5eb88c7`.
- `build-cuda/bin/test-server-prompt-cache` — PASS. Log:
  `/srv/ai/paged-kv/results/forward/100-04a/attempt-04/server-prompt-cache.log`
  SHA256 `c9beb03e7a9c2146f763cd3cdb94049d39d9e3204cf2a8925ac4a3610affbeb2`.
- `LD_LIBRARY_PATH="$PWD/build-cuda/bin" build-cuda/bin/test-cuda-kv-promotion`
  — PASS, including `selected_final_map_cuda_attention_parity`. Log:
  `/srv/ai/paged-kv/results/forward/100-04a/attempt-04/cuda-kv-promotion.log`
  SHA256 `611afcd079b9d4f938544032dbabadd4b61da9bfa8cc3345884f40c255d1a241`.
- V10 validation — FAIL (expected until integrated proof and receipt exist):
  `python3 .wiretail/execution/v10/validate.py --task 100-04a --receipt .wiretail/execution/evidence/V10_100-04a.json`.
  Output: `/srv/ai/paged-kv/results/forward/100-04a/attempt-04/v10-validate.log`
  SHA256 `9bd3a20c4c40f244abb056437a2480cf0cd485d23c55f56316adbef310e077f2`.
- `PROJECT_ROOT="$PWD" python3 /srv/wiretail/task_state.py validate` — PASS
  (`Valid state: 456 tasks`). Log:
  `/srv/ai/paged-kv/results/forward/100-04a/attempt-04/task-state-validate.log`
  SHA256 `2d0f4c99d7a3dfd0669136f8f79989bdfc56d18a761cd12de2481055a1cd643b`.
- `git diff --check` — run after handoff update.

## Deferred verification

Integrated real target/draft replay parity is unavailable in this attempt. The
existing build's CUDA consumer proof and helper-branch test do not substitute
for it. Do not claim or receipt the required proof. A follow-up must add a
model-backed fixture seam that invokes the actual checkpoint capture/restore
and production transition with real decode, or establish a safe isolated
candidate runtime with sufficient VRAM; then run both parity fixtures and
record their hashed outputs.

## What remains

Task is currently `blocked`. WORK_STATE.json controls progression to 101-01. The missing work is the
model-backed server fixture and its integrated output comparisons, followed by
the truthful V10 receipt, V10 validation, and task-state validation. No
hardware or service success is inferred from attempt 04.

## Wiretail status
<!-- wiretail:runner-status:start -->
- State: `blocked`
- Source of truth: `.wiretail/execution/WORK_STATE.json`
- Blocker: Automatic substantive retry budget exhausted after 4 total attempts; see the latest handoff and recovery-assessment artifacts. Latest agent output: /srv/repos/vanwho/buun-llama-cpp/.wiretail/build/100-04a-attempt-4-20260930T090333482253493-final.md.
<!-- wiretail:runner-status:end -->
