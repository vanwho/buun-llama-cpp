# 18-01 runtime identity, crash, and lifecycle evidence

## Result

PASS. One coherent immutable CUDA candidate bundle started at the required 22,016-token context, survived idle authenticated health/props/metrics checks, a short generation, and five repeated authenticated metrics polls during generation. The candidate remained loaded after success. An intentional canonical-runner failure restored the exact prior runtime and allowlisted transient overrides.

## Candidate identity

- Bundle: `/srv/ai/paged-kv/results/18-01-runtime-bundle-20260905-v3/bundle`; 23 regular files, all non-writable.
- Executable: `bin/llama-server`; all nine observed project DSOs were mapped from the same bundle, verified from `/proc/<MainPID>/maps`.
- Model: `Qwen3.8-27B-UD-IQ4_XS.gguf`, SHA-256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
- Effective command: context 22016, selective pager, page size 256, GPU draft placement, Turbo4 target and MTP K/V, MTP `n_max=2`.
- Final service: `llama-server.service` active/running, MainPID 3082150 after recovery, `NRestarts=0`.

## Crash diagnosis and fix

The prior `/metrics` failure was reproduced as a deterministic server error, not a new SIGSEGV: the pager JSON telemetry loop called `get<double>()` on the boolean `admission_accepted`, producing nlohmann `type_error.302`. The loop now accepts only numeric JSON values, and `test-server-mmproj-lifecycle` covers the boolean regression field. The fixed candidate had no exception, SIGSEGV, or type-mismatch journal entry and stayed running through the live checks. A matching-symbol debugger/core backtrace remains unavailable because no usable coredump/debugger artifact exists.

## Verification

- CUDA build: `cmake --build build-cuda --target llama-server test-server-mmproj-lifecycle -j2` — pass.
- Regression: `build-cuda/bin/test-server-mmproj-lifecycle` and focused `ctest` — pass.
- Portable lifecycle/soak tests: `python3 -m unittest tools.server.bench.test_pager_benchmark_adapter tools.server.bench.test_pager_soak -v` — 24 passed.
- Syntax/format checks: Python `py_compile`, `bash -n` for the three site scripts, and `git diff --check` — pass.
- Live receipts: candidate lifecycle `adapter_validation=passed`; health/props/metrics/generation all HTTP 200; five metrics snapshots contained KV-pager metrics.
- Recovery receipt: `/bin/false` produced `canonical_runner_failure`; restoration state was `restored`, with exact context/pager/page size, bundle DSOs, and transient values.
- Isolation: no final candidate or recovery receipt references port 8092.

## Raw evidence

The machine-readable receipt is [18-01_RUNTIME.json](18-01_RUNTIME.json). Bulk artifacts are under `/srv/ai/paged-kv/results/18-01-runtime-20260905T0820Z/` and `/srv/ai/paged-kv/results/18-01-runtime-recovery-20260905T0828Z/`; the bundle manifest is `bundle-manifest.json` in the candidate result directory.

## Deferred verification

No human or upstream action is required. Full debugger/core replay remains deferred because the required artifact was unavailable. Long-corpus performance curves and physical transfer/residency calibration remain outside this packet and are owned by later tasks.
