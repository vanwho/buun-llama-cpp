# Phase 21 full 256K functionality checkpoint

Task 21-06 is complete as an explicit negative/incomplete checkpoint. The exact
262,144-token runtime allocation and GPU Turbo4 MTP setup worked, but the
near-full population did not finish within the bounded 1,200-second request.
The base full-context goal is therefore **not demonstrated**.

## Frozen runtime

The test used the immutable `20-07-runtime-bundle-20260905T220000Z` binary,
the pinned Qwen3.8 model (`40fac405...e6199`), selective paging with 256-token
pages, and logical context `262144`. The unrelated service on port 8092 was
not touched. The successful exact profile was left loaded on port 8080 after
the run and the timed-out slot was erased through the existing `/slots/0`
control, returning the service to idle.

## Results

| Area | Result | Evidence |
|---|---|---|
| Allocation | Pass with fixed-four-page recovery | `262144` logical tokens, `1024` logical pages, target allocation `17,301,500` bytes, CUDA target, CPU Turbo4 target types |
| Budget-derived hot set | Fail | `--kv-hot-pages auto` hit CUDA OOM during compute-buffer reservation; fixed `4` hot pages then allocated successfully |
| Near-full population | Incomplete/fail | Locally rendered prompt was `262136` tokens with an eight-token reserve; server processed `15360` tokens by `1200.0778666429687` seconds and returned no response |
| MTP | Pass for allocation only | `262144` GPU-resident MTP rows, `276955000` bytes, Turbo4 K/V; population and generation were not completed |
| Movement | Not demonstrated | Partial snapshot showed no page IDs/checksum/promotion and zero H2D/D2H useful bytes, faults, or evictions |
| Retrieval | Not measured | Early, middle, late, and recent fact probes were not attempted after population timeout |
| Continuation | Not measured | Focus shift and short continuation were not attempted |
| Base goal | No | No shorter context was substituted |

The partial post-timeout metrics reported route `selected_reference`, four
selected pages, 1024 target-valid rows, and 1024 host-valid rows. Those values
describe partial work and are not treated as full-context population or real
movement proof.

## Failure classification

The budget-derived allocation failure is a runtime configuration/headroom
failure: the automatic hot-set choice could not reserve the graph, while the
fixed-four-page recovery preserved the full logical capacity. The population
failure is a bounded prefill/attention execution timeout: the server was still
processing the request, had reached 15,360 processed prompt tokens, and had not
produced a completion. This receipt makes no model-quality claim and does not
claim that pager promotion is broken.

The benchmark adapter also emitted a profile identity/restore warning because
its profile parser returned null before and after the run. Canonical execution
returned success, the exact process remained loaded, and direct service state
was checked after cancelling slot 0. This is recorded as a harness warning,
not silently converted into a runtime success.

## Verification

- `ctest --test-dir build-cuda --output-on-failure -R '^test-(kv-pager|kv-attention-view|kv-attention-telemetry|kv-attention-execution|kv-attention-exact|cuda-fattn-paged-turbo4)$'` — 6/6 passed.
- `python3 -m unittest discover -s tools/server/bench -p 'test_*.py'` — 48/48 passed.
- Evidence-envelope validation and repository state validation are recorded in the task handoff.
- `git diff --check` is required after the receipt/state update.

## Deferred verification

The following remain deferred because the bounded full-context population did
not complete: full near-262K population with generation headroom;
early/middle/late/recent retrieval; focus shift and continuation; correlated
logical-page IDs, host checksum, promotion, and transfer route; and automatic
budget-derived hot-set allocation without the fixed-four-page recovery. The
final speed curve is not authorized by this checkpoint.

The machine-local raw artifacts and hashes are indexed in
`PHASE21_FULL256K.json`.
