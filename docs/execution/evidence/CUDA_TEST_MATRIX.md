# CUDA correctness and fault matrix

Task 06-02 records CUDA evidence for the currently implemented Turbo4 direct
page-wave boundary and its available backend/lifecycle fixtures. Results below
are tied to raw logs in this directory. A row marked `CPU fake` is not a CUDA
pass; it records the already-closed deterministic fallback for a boundary that
does not yet have a live CUDA/model harness.

## Build and platform provenance

| Field | Recorded value |
|---|---|
| Repository SHA | `bfc536912fb81252b3ef7eb47f861fae158d5856` |
| Build directory | `build-cuda-test` |
| Build type | `Release` |
| CUDA | `GGML_CUDA=ON`, `GGML_CUDA_GRAPHS=ON` |
| CUDA architecture | `CMAKE_CUDA_ARCHITECTURES=89-real` |
| Tests | `LLAMA_BUILD_TESTS=ON`, `LLAMA_BUILD_SERVER=ON` |
| GPU | NVIDIA GeForce RTX 4080, compute capability 8.9, 16376 MiB reported by `nvidia-smi` |
| Driver | 595.84 |
| CUDA toolkit | 12.4.131 |

Raw provenance is in [`cuda-fixture.log`](cuda-fixture.log) and
[`cuda-toolkit-version.log`](cuda-toolkit-version.log). The build configuration
was read with:

```sh
git rev-parse HEAD
rg '^(CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|GGML_CUDA|GGML_CUDA_GRAPHS|LLAMA_BUILD_TESTS):' build-cuda-test/CMakeCache.txt
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
nvcc --version
```

## Executed commands and outputs

The registered direct-kernel fixture was rebuilt and run with:

```sh
cmake --build build-cuda-test --target test-cuda-fattn-paged-turbo4 -j2
ctest --test-dir build-cuda-test -R '^test-cuda-fattn-paged-turbo4$' --output-on-failure
build-cuda-test/bin/test-cuda-fattn-paged-turbo4
```

CTest result: `100% tests passed, 0 tests failed out of 1`.
The direct run initialized the RTX 4080 and reported
`paged Turbo4 decode: 20.112 ms (four Q heads, 529 selected rows)`.
This timing is fixture telemetry only, not a performance claim.

Sanitizer commands:

```sh
compute-sanitizer --tool memcheck --error-exitcode=99 build-cuda-test/bin/test-cuda-fattn-paged-turbo4
compute-sanitizer --tool racecheck --error-exitcode=99 build-cuda-test/bin/test-cuda-fattn-paged-turbo4
compute-sanitizer --tool memcheck --error-exitcode=99 build-cuda-test/bin/test-vbr-vmm
```

Results: paged memcheck `ERROR SUMMARY: 0 errors`; racecheck `0 hazards
displayed (0 errors, 0 warnings)`; VMM memcheck `ERROR SUMMARY: 0 errors`.
Raw logs are [`cuda-sanitizer.log`](cuda-sanitizer.log),
[`cuda-racecheck.log`](cuda-racecheck.log), and
[`cuda-vmm-sanitizer.log`](cuda-vmm-sanitizer.log).

## Requirement matrix

| Requirement | Fixture and evidence | Disposition |
|---|---|---|
| Turbo4 K/V, head width 256, GQA=4 | [`test-cuda-fattn-paged-turbo4.cu`](../../tests/test-cuda-fattn-paged-turbo4.cu) sets T4 K/V, `head_dim_k=head_dim_v=256`, `n_head_q=4`, `n_head_kv=1` | Passed on RTX 4080; unsupported type/shape refusals also pass |
| One page and partial tail | The fixture reruns with one selected page and 17 rows and checks every output element against `0.011353` | Passed; absolute tolerance `< 2e-6` |
| Permuted physical slots and logical gap | Three selected logical pages are ordered `3,0,1`, mapped to physical slots `5,1,7`; logical page 2 is absent | Passed; direct output and page-mass bins agree with logical order |
| Mixed contiguous selected runs and 304-page geometry | CPU page/view arithmetic covers the 304-page partition; the CUDA direct fixture covers a bounded 529-row, three-page mixed run | 304-page CUDA allocation is not run in this bounded fixture; mapped to the explicit later Qwen3.8 acceptance run, not claimed here |
| Native positions, causal filtering, and compact mask | The last native position is set above the query position and one compact row is masked | Passed; causal/mask rows are excluded and output remains finite |
| Compact gather reference | Expected output is independently computed from the selected page values and valid-row count: `(17*c8 + 255*c9 + 255*c10)/527` | Passed; absolute output tolerance `< 2e-6` |
| Dense Turbo4 reference | No standalone dense-Turbo4 test entry exists for this internal backend callback; the existing dense graph route is covered by the CPU route contract | Not available as a CUDA comparison in this repository; no dense equivalence claim made |
| Score/page-mass reduction on versus off | Fixture runs once without reduction and once with four logical bins | Passed; output vectors compare exactly; bins use `< 1e-5` for nonzero mass and `< 1e-6` for zero mass |
| Invalid type and unsupported shape fail closed | Fixture requests F16 K and then value width 128 | Passed; exact `unsupported_type` and `unsupported_shape` statuses |
| Page-table and argument validation | Host page-table validation is exercised before launch; null/zero validation is checked; physical bounds and mass-bin bounds are validated by the launch boundary | Passed for the assertions present in the fixture; exhaustive malformed-pointer/device-memory injection is not a safe standalone test and is not claimed |
| CUDA graph capture/reuse across unchanged/changed table epochs | [`test-cuda-graph-key.cpp`](../../tests/test-cuda-graph-key.cpp) checks route-affecting key reuse; [`test-kv-attention-execution.cpp`](../../tests/test-kv-attention-execution.cpp) checks old/new graph epochs and fences | Contract passed; direct CUDA graph capture/replay with live page-table updates is not registered and remains a later integration check |
| VMM mapping and page granularity | [`test-vbr-vmm.cpp`](../../tests/test-vbr-vmm.cpp) runs through CUDA VMM; raw output reports `PASS: device 0 VMM range accounting (2048 KiB pages)` | Passed; direct and memcheck logs in [`cuda-vmm.log`](cuda-vmm.log) and [`cuda-vmm-sanitizer.log`](cuda-vmm-sanitizer.log) |
| No hidden F16 full-cache buffer | Direct fixture allocates raw Turbo4 page storage and launches the direct decoder; it does not allocate an F16 selected/full-cache tensor | Passed by fixture allocation/code-path audit; `cuobjdump` reports the direct kernel with `REG:40 STACK:0 SHARED:0 LOCAL:0` |
| Compiler/kernel resource bounds | [`cuda-resource-usage.log`](cuda-resource-usage.log) is the complete `cuobjdump --dump-resource-usage` output; [`cuda-kernel-resource-summary.log`](cuda-kernel-resource-summary.log) isolates the direct decoder | Passed: direct kernel `REG:40`, `STACK:0`, static `SHARED:0`, `LOCAL:0`; dynamic launch sizing is bounded by source validation to the supported page limit |
| D2H/H2D transfer reordering, cancellation, stale completion, host corruption, ring exhaustion, all-pinned, and clean zero-D2H eviction | [`CPU_TEST_MATRIX.md`](CPU_TEST_MATRIX.md) and its fake transfer/prefetch fixtures exercise these callback boundaries and rollback assertions | CPU fake passed in 06-01; no live CUDA transfer adapter is registered for this page boundary, so this is not reported as CUDA-passed |
| VRAM map denial and transfer-phase rollback | Fake backend failure injection covers map, issue, completion, catalog, stale, and every transaction phase in [`test-kv-residency.cpp`](../../tests/test-kv-residency.cpp) | CPU fake passed; CUDA fault injection remains an integration check because the production page transaction is callback-owned |
| Clear/teardown and recurrent operations | Deterministic server fault fixtures pass without a model; model-backed hybrid/recurrent state remains outside the direct CUDA fixture | Server clone/accounting/checkpoint faults passed; Qwen3.8 GPU teardown is not claimed |
| MTP accept/reject/checkpoint and target pressure | [`test-mtp-vocab-trim.cpp`](../../tests/test-mtp-vocab-trim.cpp) and server fault fixtures cover deterministic validation/rollback seams | CPU/deterministic checks passed; no configured model-backed Qwen3.8 MTP CUDA run is available in this worktree |

## Failure dispositions

No executed CUDA command failed. The non-pass dispositions above are scope
boundaries, not tolerated failures:

- The direct CUDA API intentionally supports only causal batch-one, one-query,
  T4/T4, D=256, GQA=4 decode. Prefill and other batch shapes are expected to
  use the selected reference route; they are not silently accepted by the
  direct kernel.
- Dense-Turbo4 parity and live graph replay require the graph integration seam,
  not just the direct callback. Their CPU contracts pass, while their live
  CUDA checks belong to the Qwen3.8 integration fixture.
- CUDA transfer faults require a production CUDA transfer adapter. The fake
  backend is the deterministic test double and its complete fault matrix is
  linked above; no fake result is promoted to a hardware result.
- The repository contains no Qwen3.8 model checkpoint or registered
  model-backed selective-pager runner. The pinned external model location is
  recorded in `docs/execution/baseline/reference.json`; use the later profile
  and acceptance harness only after that external artifact is provisioned.

## Deferred verification

The RTX 4080 and CUDA toolkit were available, so the bounded kernel and VMM
checks were run rather than deferred. The following require the external
Qwen3.8 checkpoint and the later integration/acceptance harness:

- 304-page live target allocation, dense/gather/paged output comparison through
  the full model graph, prefill-to-decode transition, and supported multi-
  sequence batches.
- Live CUDA D2H/H2D cancellation/reordering, stale event publication, host
  corruption, pinned-ring exhaustion, all-pinned pressure, clear, and teardown.
- Recurrent companion operations, native MTP acceptance/rejection/checkpoint,
  target pressure while MTP remains resident, and model-output equivalence.
- CUDA graph capture/reuse across actual unchanged and changed table epochs.

These are explicitly deferred to the model-backed integration/acceptance
tasks; they are not marked passed or hidden behind disabled assertions.
