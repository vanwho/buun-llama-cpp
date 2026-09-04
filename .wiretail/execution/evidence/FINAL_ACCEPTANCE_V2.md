# Final acceptance V2

## Decision

**BLOCKED — not a release acceptance.** The clean local reproduction and
deterministic checks pass, but the required phase-14 model-backed gates do not:
quality, selective/exact performance, and pager soak are all recorded as
`block`. No accepted quality score, speed ratio, or soak claim is available.

## Reproduction audit

- Final repository SHA: `1d059e345b1f0bedb87f93f3c16ea0d5168bf42e`
- Recorded upstream base: `cb703be37e3628dadb71912f3b3b25b82090555b`
- Current upstream tip: `3823c9eb6541725bffa70fe3f8508c55b3b5ca7b`
- Relation: upstream is one commit ahead; the integration branch is 204 commits
  ahead. The fork's `plan/attention-aware-kv-paging` branch now points at the
  recorded repository SHA; upstream re-sync remains an outer Git-owner action.
- Fresh build: Release CUDA server and docs generator built in a new `/tmp`
  build directory. Enabling repository CUDA tests is currently prevented by
  CMake's empty `CUDA_ARCHITECTURES` on `test-cuda-fattn-paged-turbo4`; this
  audit used the successful server-only CUDA configuration instead.

| check | result |
| --- | --- |
| server `--help` | pass |
| Release CPU main CTest | pass, 105 tests |
| focused CUDA ADD probe | pass, 99/99 CUDA cases |
| generated docs target | pass |
| state validation | pass, 69 tasks |
| `git diff --check` | pass |
| active profile service | active |
| health 8080 / 8091 | HTTP 200 / HTTP 200 |
| 8092 | not touched |

The machine-readable record is
[`FINAL_ACCEPTANCE_V2.json`](FINAL_ACCEPTANCE_V2.json).

## Gate evidence

- [Quality](QUALITY_ACCEPTANCE_V2.md) is **BLOCKED**: frozen held-out quality
  rows are not accepted and exact/telemetry requirements were unavailable.
- [Performance](PERFORMANCE_ACCEPTANCE_V2.md) is **BLOCKED**: no valid warm
  selective+native-MTP segment exists, so no 3x floor, 5x target, or 70%
  comparable-GPU ratio is claimed.
- [Soak](SOAK_ACCEPTANCE_V2.md) is **BLOCKED** because its accepted pager
  prerequisite is absent; control cancellation, restart, and restoration
  evidence remains recorded.
- [V2 evidence index](INDEX_V2.md) preserves exact hashes, raw roots, faults,
  rollback, and churn pointers.

These are substantive runtime results, not documentation-only completion. The
next required work belongs to the owning phase-14 acceptance path.

## Deferred verification

No new hardware or service verification is deferred: the available service was
healthy and the deterministic CUDA probe passed. Upstream re-sync is deferred
to the authorized outer Git owner. Model-backed quality, selective speed, and
pager soak are blocked, not deferred, and must not be promoted during closure.

## Attempt 25 — current clean reproduction

The final local checks were rerun against the current repository state. State
and all V2 acceptance JSON documents parsed; the contract API validated all 24
`pager-corpus-v2` cases; 271/271 repository-relative documentation links
resolved; and portability scans found no production fixed-hot-count,
non-test credential, or tracked-large-artifact violation. CPU main CTest
passed, focused CUDA contract CTest passed 4/4, `llama-gen-docs` built, and
server help passed. Both required services were active and 8080/8091 returned
HTTP 200; 8092 was untouched. The fork branch still matches the repository
tip, while upstream remains one commit ahead of the recorded base.

These checks do not change the authoritative phase-14 decisions: quality,
selective/native-MTP performance, and pager soak remain `block`, with no
accepted model-backed result or speed ratio. The final decision remains
blocked and 15-03 remains `in_progress`.

## Recovery attempt 4

The focused CUDA lifecycle selection reproduced the known fixture/configuration
boundary: 19/20 tests passed, and `test-state-restore-fragmented` aborted only
because its default Turbo4 setup uses `n_embd_head_k=48`, which is not divisible
by the Turbo4 block size 128. The contract-compatible recovery passed:

`build-14-01-cuda/bin/test-state-restore-fragmented -m build-14-01-cuda/tinyllamas/stories15M-q4_0.gguf --cache-type f16 --fit off`

It saved, cleared, restored, and decoded a fragmented sequence successfully.
The direct `test-cuda-fattn-paged-turbo4` executable also passed. These
deterministic results do not replace the unavailable accepted phase-14
model-backed quality, selective/native-MTP performance, and pager-soak runs.

The pinned Qwen3.8 model was also located and its SHA256 matched the recorded
provenance. No readable executable phase-14 acceptance harness was available.
The currently mounted corpus has the valid `pager-corpus-v2` schema and the
immutable canonical corpus hash `e8cd002b...0e35`; `513eb790...c5a4` is only
the raw JSON file digest. Its held-out descriptors still contain static
`Context contains sealed pages` prompts with no live retrieval, so this does
not create a new accepted runtime quality result.

## Attempt 6 adapter audit

The canonical corpus validator passed and recomputed the immutable corpus hash
for all 24 cases. The pager adapter dry-run also passed, producing its explicit
`not_configured` envelope without contacting a service. It contains no runtime
telemetry, MTP placement, timing, or accepted quality/performance/soak result,
so the final decision remains blocked.

## Deferred verification

No new hardware or service verification is deferred. Upstream re-sync remains
an outer Git-owner action. Phase-14 model-backed acceptance remains blocked and
must return to its owning acceptance path before this task can be marked done.
