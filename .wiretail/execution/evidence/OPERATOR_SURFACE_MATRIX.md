# Operator-surface lifecycle matrix

This matrix closes the locally executable part of task 12-03. `Passed` means a
deterministic test or parser/refusal command ran in the current worktree;
`Deferred` means the check requires the configured Qwen3.8/Turbo4 model and
CUDA service. Deferred rows are not acceptance claims.

## Mode and lifecycle matrix

| Mode or boundary | Local coverage | Resource/lifecycle assertion | Status |
|---|---|---|---|
| `off` | [`test-kv-pager.cpp`](../../../tests/test-kv-pager.cpp), [`test-kv-attention-execution.cpp`](../../../tests/test-kv-attention-execution.cpp) | plan is disabled, uninitialized, has no physical window, and requests no backend allocation; dense route remains ordinary | Passed |
| `observe` | [`test-kv-pager.cpp`](../../../tests/test-kv-pager.cpp) | logical inventory is present but physical page count and residency slot capacity are zero; no pager allocation | Passed |
| `selective` | [`test-kv-pager.cpp`](../../../tests/test-kv-pager.cpp), [`test-kv-policy.cpp`](../../../tests/test-kv-policy.cpp) | five admitted pages in the fake geometry; allocation and release counts reconcile; live policy/prefetch generation tests pass | Passed locally; CUDA/model request deferred |
| `exact` | [`test-kv-pager.cpp`](../../../tests/test-kv-pager.cpp), [`test-kv-attention-exact.cpp`](../../../tests/test-kv-attention-exact.cpp) | bounded physical window is created and released; exact wave inventory/coverage is tested independently | Passed locally; production per-layer CUDA callback deferred |
| unsupported capability combinations | [`llama-kv-pager-config.cpp`](../../../src/llama-kv-pager-config.cpp), [`test-kv-pager.cpp`](../../../tests/test-kv-pager.cpp) | backend, architecture, causality, type, geometry, topology, sequence, host budget, MTP, and conflicting-authority reasons are typed and fail closed before ownership | Passed |
| cancellation, clear, checkpoint, slot reuse, shutdown | [`test-server-prompt-cache.cpp`](../../../tests/test-server-prompt-cache.cpp), [`test-kv-policy.cpp`](../../../tests/test-kv-policy.cpp) | generation rotation cancels tracked work; stale/cancelled completions cannot publish; shutdown is idempotent | Passed with deterministic fakes |

## Commands and results

```text
cmake --build build --target test-kv-pager test-kv-attention-execution \
  test-kv-attention-exact test-kv-attention-telemetry test-kv-policy \
  test-server-prompt-cache -j2
  PASS

ctest --test-dir build -R '^(test-kv-pager|test-kv-attention-(execution|exact|telemetry)|test-kv-policy|test-server-prompt-cache)$' --output-on-failure
  PASS: 6/6

cmake --build build --target test-arg-parser llama-server -j2
  PASS

ctest --test-dir build -R '^test-arg-parser$' --output-on-failure
  PASS: 1/1

python3 -m py_compile tools/server/bench/run-pager-profile-benchmark.py \
  tools/server/bench/pager_benchmark_contract.py
  PASS

git diff --check
  PASS
```

The pager mode fixture uses a 1,025-token synthetic context and 128-byte page
geometry. `observe` performs zero device allocations; `selective` and `exact`
each perform one allocation and one release. The test has no model output or
throughput claim.

## Refusal and live-service disposition

The parser refuses invalid mode and page geometry before model setup, including
the previously recorded commands:

```text
build/bin/llama-server --kv-pager invalid  -> exit 1, invalid --kv-pager mode
build/bin/llama-server --kv-page-size 128 -> exit 1, page-geometry diagnostic
```

The Qwen3.8 target GGUF and RTX 4080 are present, but the only GPU is fully
occupied by the pre-existing 8080 service (`nvidia-smi` reported 15,481 MiB
used of 16,376 MiB). Its external binary does not publish pager metrics, and
starting a second model-backed process would contend with that service. No
service was stopped or reconfigured. Therefore the following remain deferred:

- one real request in each of `off`, `observe`, `selective`, and `exact`;
- selected cold-page promotion, exact all-page waves, and output parity;
- native Turbo4 MTP context-sized GPU residency and transfer counters;
- repeated server starts, clear/cancel/shutdown, memory deltas, and sanitizer
  checks against the real service;
- restoration evidence for ports 8080/8091 and confirmation that 8092 is
  untouched during the external service run.

No service was started or modified by this task, so there are no external raw
result paths or SHA-256 hashes to claim. The required later run should write
raw artifacts under `/srv/ai/paged-kv/results/12-03/`, then add their paths and
hashes here and to the task handoff.
