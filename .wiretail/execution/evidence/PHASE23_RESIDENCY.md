# Phase 23 residency acceptance

Task 23-04 is locally complete with the production prompt probe deferred.

The live-policy production boundary now collects all selected cold pages into
one authenticated H2D transfer plan. `page_index` is assigned by the plan
builder across that complete page vector, matching the host-read callback's
plan-local lookup and preventing later promotions from reading host page zero.
The policy and transaction layers still own full page identity, destination
slot assignment, event completion, stale recheck, rollback, and immutable table
publication.

The deterministic live-policy fixture promotes two retrieval-selected cold
pages, records distinct page-indexed payloads, and verifies two loaded pages and
16 useful H2D bytes. The residency fixture separately verifies the generic
multi-page transfer contract. Existing pager tests retain partial-tail,
clean/dirty eviction, stale-generation, cancellation, slot-reuse, and
layer-major geometry coverage.

## Verification

- `cmake --build build-cuda --target test-kv-residency -j2` — passed.
- `cmake --build build-cuda --target test-kv-policy -j2` — passed.
- Focused pager/policy/residency/retrieval CTest set — 4/4 passed.
- Focused attention/telemetry/summary CTest set — 3/3 passed.
- `PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp python3 /srv/wiretail/task_state.py validate` — passed.
- `git diff --check` — passed.

## Deferred verification

The required live 6K–8K pressure sequence with an explicit four-page cap was
not run. A historical selected-reference candidate remains on port 8080 and a
separate `llama-server` owns port 8092; neither was stopped or changed. The
production receipt must still correlate query generation, logical identity,
host checksum, useful/aligned H2D bytes, CUDA completion event/fence, physical
slot, selected route, answer continuation, and fault/eviction counters after an
authorized lifecycle transition.
