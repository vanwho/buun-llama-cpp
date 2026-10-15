# ACO preview validation

This note records targeted compile and correctness checks for the ACO preview.
It is not a full CI result, a human-review certification, or a performance
report. No benchmark numbers are claimed here.

## Build provenance

- Upstream base: `d2407c33f9595d1badccfd9a82a33c4c5ceda490`.
- Production source commit used for the final CUDA server build:
  `794edc21151d2841add685da0b41e3534cf88e94`.
- Source tree at that commit: `a59a52ac1474fde7836757a2a52fb23894655c42`.
- Release server SHA-256:
  `c9041ca96611ed4c9aa06be7b9a23443aa11d4b3bb30983d7dd2558c582659ec`.
- The executable reports build commit `794edc211`. Changes after this build
  are limited to documentation and evidence tooling; the production source
  delta from the build commit is empty.

The CUDA build used Release mode, `GGML_CUDA=ON`, CUDA architecture 89,
`LLAMA_BUILD_TESTS=ON`, ccache launchers, and `--parallel 16`. The host was
Ubuntu 26.04.1 x86-64, CMake 4.2.3, GNU C/C++ 15.2.0, CUDA Toolkit/NVCC
12.4.131 with GNU 13.4.0 as NVCC's host compiler, and ccache 4.12.3. The test
GPU was an NVIDIA GeForce RTX 4080 (compute capability 8.9, driver 595.99.02).

## Focused checks

In the earlier CPU-only Release configuration (`GGML_CUDA=OFF`, tests enabled)
at source commit `bc294606c`, the following eight CTest targets passed (8/8):

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

At final source commit `794edc211`, the focused
`test-kv-policy --query-commit-authoritative-admission` fixture passed in both
the CPU-only and CUDA-enabled Release builds. It covers retaining a dirty
physical owner outside the attention budgets, replacing a clean omitted owner,
and refusing a full dirty pool before transaction publication.

In the CUDA-enabled Release configuration at `794edc211`, these host-run CTest
targets passed (5/5): `test-kv-pager`, `test-kv-attention-exact`,
`test-kv-attention-view`, `test-cache-budget`, and
`test-server-prompt-cache`.

Two CUDA fixtures passed on the RTX 4080 at `794edc211`:

- `test-cuda-fattn-paged-turbo4`, including its
  `--prefill-long-context-regression` case. Coverage included fused/materialized
  parity, paged noncontiguous rows, graph capture/replay, pager layer-slot
  strides, and the long-context prefill shape.
- `test-cuda-kv-page-summary`, covering CUDA summary updates, reranking, mixed
  identity, softcap, and exact-pool mass parity.

These checks exercise focused code paths only; they do not substitute for the
full project test matrix or model-level quality and performance evaluation.

## Long-run exclusion

A pre-fix 256K-context / 64K-hot long run failed at actual context `C=234,592`
with `dirty_victim` during selected-history publication. Its measurements are
diagnostic only and are excluded from the fixed-source cohort; they are not
evidence of a complete 256K run or a performance result. The fix preserves
non-droppable physical owners while selecting only genuinely free or clean-
droppable slots. Any later long-run claim must reference the fixed binary above
and its separately published results artifact.

## Observed long-context replay limitation

The fixed candidate completed all 16 long-fill frontiers through actual
`C=258,048` without a runtime fault. The identical-question long canonical
warmup and first measured p0 replay both incurred substantial backend work:
the warmup took 135.7 seconds wall / 134.4 seconds in the prompt stage, and
the measured replay took 143.6 seconds wall / 134.5 seconds in the prompt
stage. Both reported `cache_n=149,668` and `prompt_n=108,424` for a full
rendered prompt of 258,071 tokens. Thus a warm cache did not eliminate
substantial prompt reprocessing at this frontier. These are two rows, not a
complete performance characterization; remaining canonical requests were
still in progress when this note was prepared. The source LCP advanced to
220,585 at `C=234,592`, while restore used the 149,668-token checkpoint and
attention-only trim. Checkpoint selection/retention remains a follow-up
optimization area; the observations do not establish cache-RAM pressure as the
cause.

## Retained log fingerprints

Raw logs are retained outside the source tree. The artifact identifiers below
are stable labels; hashes fingerprint the retained files without publishing
machine-local paths.

| Artifact | SHA-256 |
| --- | --- |
| `ACO-CUDA-BUILD-PRE-FIX-01` | `2187ba3aa96217c93cf3c948264a54e2186ca910ef0b818e8b4a0e7caf472f0b` |
| `ACO-CUDA-FIX-BUILD-01` | `fa3bfc11e96a36ff89e483b4db5038fca02fd3810d6ddf0ef602dd5d0101c7ea` |
| `ACO-CUDA-FIX-FOCUSED-BUILD-01` | `46e4233f80e98997329312972e42ace86a3842f979b4b2f286227c7ae01ea1b4` |
| `ACO-CUDA-FIX-POLICY-TEST-01` | `d0f57678e35850881d1ab655074735e31c28a796eb0b48cc09f004aec5f89cca` |
| `ACO-CUDA-FIX-HOST-TESTS-01` | `b3b6133c522bf4ae18cc08fe73a021f6ad493bc6000212c04bfb5060edb9d228` |
| `ACO-CUDA-FIX-ATTN-FIXTURE-01` | `c3251cb1d22f7685d85d1da5545b757d12cef645de574979435ec533ca40214e` |
| `ACO-CUDA-FIX-LONG-ATTN-01` | `3450be8f8df68fd2ebd48011b55239a406ca10d72e386c3d60e2ee830b678a72` |
| `ACO-CUDA-FIX-PAGE-SUMMARY-01` | `43d4696ff706eca3edf9ffd4aaaf4885af444f51fcf51c83380cec22f572e7ad` |
| `ACO-CPU-BUILD-01` | `c537a7951db574a5deade69d682f855b36cd897c033f7274f4624acca5452470` |
| `ACO-CPU-CTEST-01` | `acaa50be689c5536b6b3b60cd256828a8703e6c9ccb51ae93811eba2e35477ee` |
| `ACO-CPU-CONTRACT-BUILD-01` | `567e9305a04355e2ee529fade4f806ecff0e7f9e4e9d8ad07f8e47cac5a5be17` |
| `ACO-CPU-PAGER-CONTRACT-01` | `d9005c8b37e1f33e2af656fe51c6e79681d75287a746fe719c2c43d26f901cbb` |
| `ACO-CPU-MTP-CONTRACT-01` | `29484947e4cf608b683ef185ee789d57cd1bf9f459ada1dc91b90d21ff61dee7` |
| `ACO-CPU-FIX-BUILD-01` | `752bd0df6e13c9d99c172d8520a1c604bf2009822ebc7728706ede9fb1e8e9d6` |
| `ACO-CPU-FIX-POLICY-TEST-01` | `d0f57678e35850881d1ab655074735e31c28a796eb0b48cc09f004aec5f89cca` |
