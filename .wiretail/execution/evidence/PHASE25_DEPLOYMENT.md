# Phase 25 deployment and telemetry preflight

## Result

Task `25-01` passes its owned boundary: the managed 8080 service was
transitioned to an immutable task-owned executable-plus-DSO bundle, the live
executable and all project DSO maps resolve inside that bundle, and the
candidate remains loaded. Port 8092 and its unrelated process were not
touched.

The bounded preflight completed all three live requests (`warm`,
`cold_needle`, `selected_all`). Every record is `valid_measurement` with HTTP
200, exact local/server prompt-token agreement, zero telemetry validation
errors, stable snapshot/config generations, CUDA target placement, GPU Turbo4
MTP, 262144 MTP rows, budget ledger, route, page domains, movement counters,
and timings. The server route label `selected direct` is preserved verbatim
and normalized by the repaired adapter to `selected_direct`.

The immutable bundle, model, tokenizer, runtime chat-template, profile and
effective launch configuration hashes are recorded in
`PHASE25_DEPLOYMENT.json`. Raw receipts are under
`/srv/ai/paged-kv/results/25-01-runtime-smoke-20260906T090900Z` and
`/srv/ai/paged-kv/results/25-01-telemetry-preflight-20260906T091100Z`.

## Changes

- `run-quality-corpus.py` now normalizes the server’s space-separated route
  labels and has an explicit `--telemetry-only` preflight mode. This mode
  continues after an answer mismatch only when the complete telemetry envelope
  is valid, preserving the quality result separately.
- Added regression coverage for route aliases and telemetry-only preflight
  continuation.
- Built and staged the CUDA `llama-server` plus every project DSO in the
  read-only bundle recorded by the evidence receipt.

## Verification

- CUDA server build: pass, `cmake --build build-cuda --target llama-server -j2`.
- Benchmark tests: pass, 50 tests via
  `python3 -m unittest discover -s tools/server/bench -p 'test*.py' -q`.
- Managed lifecycle smoke: pass, canonical exit 0 and adapter validation
  passed; successful candidate kept loaded.
- Three-request telemetry preflight: pass, decision `allow_campaign`, 3/3
  records valid, zero telemetry errors.
- Evidence envelope validation: run after writing this receipt.
- Repository state validation and `git diff --check`: run before task state
  completion.

## Deferred verification

Corpus answer scoring is explicitly separated from this telemetry receipt:
the model returned answer mismatches under the preflight sentinel, so quality
is deferred to the quality/parity tasks rather than hidden by this deployment
pass. Full occupied-context population, cold identity/parity and lifecycle
soak remain later phase-25 tasks. No hardware, credential or human upstream
action was unavailable for this task.
