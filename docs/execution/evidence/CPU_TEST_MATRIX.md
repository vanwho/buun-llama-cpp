# CPU and fake-backend correctness matrix

Task 06-01 closes the deterministic part of the Phase 06 matrix. Every row below
links the requirement to an existing test source/target and gives the raw
command used for verification. `Passed` means the command was executed in the
current worktree; GPU and model-backed rows are deliberately deferred.

## Commands

The feature-off build was configured with `GGML_CUDA=OFF` and
`LLAMA_BUILD_TESTS=ON` in `build`. The focused CPU run was:

```sh
cmake --build build --target test-vbr-policy test-vbr-downward test-vbr-physical \
  test-vbr-transaction test-kv-residency test-kv-attention-view \
  test-kv-routing-summary test-kv-attention-telemetry test-kv-attention-execution \
  test-cache-accounting test-cache-budget test-cache-authority test-mtp-vocab-trim \
  test-arg-parser test-kv-cell-position-index test-vbr-artifact-capture -j2
ctest --test-dir build -R '^(test-(vbr-(policy|downward|physical|transaction)|kv-(residency|attention-view|routing-summary|attention-telemetry|attention-execution|cell-position-index)|cache-(accounting|budget|authority)|mtp-vocab-trim|arg-parser|vbr-artifact-capture))$' --output-on-failure
```

Result: `100% tests passed, 0 tests failed out of 16`.

The VBR identity and generation fixture was also run explicitly:

```sh
cmake --build build --target test-vbr-representation-epoch -j2
build/bin/test-vbr-representation-epoch --identity-cpu
build/bin/test-vbr-representation-epoch --generation-cpu
build/bin/test-vbr-representation-epoch --operation-cpu
```

Result: all three commands reported `PASS`.

The server lifecycle fixture requires a server-enabled configuration. It was
run from `build-cuda-test` (the test itself is deterministic and does not load
a model):

```sh
cmake --build build-cuda-test --target test-server-prompt-cache -j2
ctest --test-dir build-cuda-test -R '^test-server-prompt-cache$' --output-on-failure
```

Result: `100% tests passed, 0 tests failed out of 1`.

## Requirement matrix

| Requirement | Existing coverage | Status / raw command |
|---|---|---|
| Page number, tail, and reference 304/1,024-page arithmetic | [`test-kv-residency.cpp`](../../tests/test-kv-residency.cpp), [`test-kv-cell-position-index.cpp`](../../tests/test-kv-cell-position-index.cpp) | Passed; focused CPU `cmake`/`ctest` command above |
| Actual row-byte, payload, fixed-window, and budget arithmetic | [`test-kv-residency.cpp`](../../tests/test-kv-residency.cpp), [`test-cache-budget.cpp`](../../tests/test-cache-budget.cpp), [`test-vbr-physical.cpp`](../../tests/test-vbr-physical.cpp) | Passed; focused CPU command above |
| Logical/physical bijection, gaps, duplicate rejection, and deterministic ties | [`test-kv-residency.cpp`](../../tests/test-kv-residency.cpp), [`test-vbr-policy.cpp`](../../tests/test-vbr-policy.cpp), [`test-vbr-physical.cpp`](../../tests/test-vbr-physical.cpp) | Passed; focused CPU command above |
| Sequence generation ABA, runtime identity, and stale completion rejection | [`test-vbr-representation-epoch.cpp`](../../tests/test-vbr-representation-epoch.cpp), [`test-kv-residency.cpp`](../../tests/test-kv-residency.cpp) | Passed; three explicit `test-vbr-representation-epoch` commands plus focused CPU command |
| Position runs, repeated positions, shift/copy/keep, and restore indexing | [`test-kv-cell-position-index.cpp`](../../tests/test-kv-cell-position-index.cpp) | Passed; focused CPU command above |
| Host catalog/capture, codec/topology identity, corruption/short input, and pins | [`test-vbr-artifact-capture.cpp`](../../tests/test-vbr-artifact-capture.cpp), [`test-cache-authority.cpp`](../../tests/test-cache-authority.cpp) | Passed by deterministic fake-provider/authority assertions; focused CPU command above |
| Host and VRAM budgets, pinned exhaustion, and admission rollback | [`test-cache-budget.cpp`](../../tests/test-cache-budget.cpp), [`test-cache-authority.cpp`](../../tests/test-cache-authority.cpp), [`test-vbr-downward.cpp`](../../tests/test-vbr-downward.cpp) | Passed; focused CPU command above |
| H2D/D2H transfer reordering, clean zero-D2H eviction, dirty reseal, and cancellation | [`test-kv-residency.cpp`](../../tests/test-kv-residency.cpp), [`test-kv-policy.cpp`](../../tests/test-kv-policy.cpp) | Passed with fake transfer/prefetch backends; focused CPU command above |
| Failure at every residency transaction phase and full rollback | [`test-kv-residency.cpp`](../../tests/test-kv-residency.cpp) | Passed; the fixture iterates snapshot through retire failure phases; focused CPU command above |
| Table epoch, immutable snapshots, graph reuse, and stale publication | [`test-kv-residency.cpp`](../../tests/test-kv-residency.cpp), [`test-kv-attention-execution.cpp`](../../tests/test-kv-attention-execution.cpp), [`test-kv-attention-view.cpp`](../../tests/test-kv-attention-view.cpp) | Passed; focused CPU command above |
| Routing top-K, structural union, exploration, retention, EMA, hysteresis, and unavailable evidence | [`test-kv-routing-summary.cpp`](../../tests/test-kv-routing-summary.cpp), [`test-kv-policy.cpp`](../../tests/test-kv-policy.cpp), [`test-kv-attention-telemetry.cpp`](../../tests/test-kv-attention-telemetry.cpp) | Passed; focused CPU command above |
| Prefill/decode boundary, fallback routes, and feature-off behavior | [`test-kv-attention-execution.cpp`](../../tests/test-kv-attention-execution.cpp), [`test-kv-attention-view.cpp`](../../tests/test-kv-attention-view.cpp), [`test-arg-parser.cpp`](../../tests/test-arg-parser.cpp) | Passed; explicit `off` route/status assertions; focused CPU command above |
| Hybrid/recurrent mutation and companion rollback seams | [`test-recurrent-state-rollback.cpp`](../../tests/test-recurrent-state-rollback.cpp), [`test-server-prompt-cache.cpp`](../../tests/test-server-prompt-cache.cpp) | Deterministic server lifecycle assertions passed; model-backed recurrent execution is deferred below |
| Server slot reuse, cancellation, stale completion, checkpoint, and speculative carry rollback | [`test-server-prompt-cache.cpp`](../../tests/test-server-prompt-cache.cpp), [`test-mtp-vocab-trim.cpp`](../../tests/test-mtp-vocab-trim.cpp) | Passed for deterministic fixtures; server command above |
| Feature-disabled ordinary behavior remains unchanged | [`test-kv-attention-execution.cpp`](../../tests/test-kv-attention-execution.cpp), [`test-kv-policy.cpp`](../../tests/test-kv-policy.cpp), [`test-arg-parser.cpp`](../../tests/test-arg-parser.cpp) | Passed in `build` with `GGML_CUDA=OFF`; focused CPU command above |

## Deferred rows

The following requirements are not CPU claims and are mapped to 06-02:

- CUDA T4/T4 output equivalence against dense, gather, paged, and F16 references.
- CUDA one/tail/304-page permutations, gaps, mixed runs, prefill/decode batches,
  score reduction, graph epochs, and compute-sanitizer fixtures.
- Real device map denial, pinned-ring exhaustion, allocator/VMM behavior, and
  model-backed Qwen3.8/MTP page pressure.

The required Qwen3.8 checkpoint is not configured locally, so model-backed
hybrid/recurrent and full server transfer scenarios are also deferred. They are
not represented as passed here. Existing bounded CUDA kernel evidence remains
in [handoff 03-03](../handoffs/03-03.md); the complete CUDA matrix belongs to
[task 06-02](../tasks/06-02.md).
