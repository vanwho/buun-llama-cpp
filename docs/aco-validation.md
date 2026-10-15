# ACO preview validation

This note records targeted compile and correctness checks for the ACO preview.
It is not a full CI result, a human-review certification, or a performance
report. No benchmark numbers are claimed here.

## Build provenance

- Upstream base: `d2407c33f9595d1badccfd9a82a33c4c5ceda490`.
- Production source commit used for the CUDA server build:
  `bc294606cdcadc50e8806fd7a1f3e395f9c64ff0`.
- Source tree at that commit: `e3be4b9cea68d31ff0de05ea1d309985507469ee`.
- Release server SHA-256:
  `805f2ab1810288c0c639f24e61b6abcf46779fcf560512b3fb15718d4a121514`.
- Later changes through this note are documentation/evidence-tooling only; the
  production source delta from the build commit is empty.

The CUDA build used Release mode, `GGML_CUDA=ON`, CUDA architecture 89,
`LLAMA_BUILD_TESTS=ON`, ccache launchers, and 16-way parallelism. The host was
Ubuntu 26.04.1 x86-64, CMake 4.2.3, GNU C/C++ 15.2.0, CUDA Toolkit/NVCC
12.4.131 with GNU 13.4.0 as NVCC's host compiler, and ccache 4.12.3. The test
GPU was an NVIDIA GeForce RTX 4080 (compute capability 8.9, driver 595.99.02).

## Focused checks

In a separate CPU-only Release configuration (`GGML_CUDA=OFF`, tests enabled),
the following eight CTest targets passed (8/8):

- `test-kv-pager`
- `test-kv-prefetch`
- `test-kv-page-select`
- `test-kv-router-job`
- `test-server-mmproj-lifecycle`
- `test-kv-attention-execution`
- `test-kv-attention-view`
- `test-kv-attention-exact`

The pager test includes the regression for oldest-position victim ordering in
non-generation writes. The synthetic pager-model contract probes passed, and
`test-speculative-mtp-state` passed (1/1).

In the CUDA-enabled Release configuration, these additional host-run targets
passed: `test-kv-pager`, `test-kv-attention-exact`,
`test-kv-attention-view`, `test-cache-budget`, and
`test-server-prompt-cache`.

Two CUDA fixtures passed on the RTX 4080:

- `test-cuda-fattn-paged-turbo4`, including its
  `--prefill-long-context-regression` case. Coverage included fused/materialized
  parity, paged noncontiguous rows, graph capture/replay, pager layer-slot
  strides, and the long-context prefill shape.
- `test-cuda-kv-page-summary`, covering CUDA summary updates, reranking, mixed
  identity, softcap, and exact-pool mass parity.

These checks exercise focused code paths only; they do not substitute for the
full project test matrix or model-level quality and performance evaluation.

## Retained log fingerprints

Raw logs are retained outside the source tree. The artifact identifiers below
are stable labels; hashes fingerprint the retained files without publishing
machine-local paths.

| Artifact | SHA-256 |
| --- | --- |
| `ACO-CUDA-BUILD-01` | `2187ba3aa96217c93cf3c948264a54e2186ca910ef0b818e8b4a0e7caf472f0b` |
| `ACO-CUDA-HOST-TESTS-01` | `17aef37fbc40a22b4964410a779a5eadbd24960379bb17e25e74382d70297393` |
| `ACO-CUDA-ATTN-FIXTURES-01` | `62904ce3845714e241275940581dcb7016f495b455acf75837d23aa8742adb59` |
| `ACO-CUDA-PAGE-SUMMARY-01` | `666890ac93ea28b2b98005afa0c99a122058e1a46ef56009b85d9bc9db93be27` |
| `ACO-CPU-BUILD-01` | `c537a7951db574a5deade69d682f855b36cd897c033f7274f4624acca5452470` |
| `ACO-CPU-CTEST-01` | `acaa50be689c5536b6b3b60cd256828a8703e6c9ccb51ae93811eba2e35477ee` |
| `ACO-CPU-CONTRACT-BUILD-01` | `567e9305a04355e2ee529fade4f806ecff0e7f9e4e9d8ad07f8e47cac5a5be17` |
| `ACO-CPU-PAGER-CONTRACT-01` | `d9005c8b37e1f33e2af656fe51c6e79681d75287a746fe719c2c43d26f901cbb` |
| `ACO-CPU-MTP-CONTRACT-01` | `29484947e4cf608b683ef185ee789d57cd1bf9f459ada1dc91b90d21ff61dee7` |
