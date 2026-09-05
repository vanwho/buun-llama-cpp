# 18-04 Telemetry Evidence

The live telemetry boundary now reports one coherent snapshot containing
request/slot/config/reset identity, monotonic time, target and native-MTP
codec/backend, admission and realized allocation, physical residency, host
validity, transfer bytes, route counts, and token acceptance denominators.
Admission estimates no longer overwrite measured native-MTP rows or bytes.

## Verification

- `cmake --build build-cuda --target test-kv-policy test-kv-residency test-kv-pager test-kv-attention-view test-kv-routing-summary test-kv-routing-retrieval test-kv-attention-telemetry test-kv-attention-execution test-kv-attention-exact test-server-mmproj-lifecycle -j2`: passed.
- `ctest --test-dir build-cuda -R '^test-(kv-(pager|residency|attention-(execution|telemetry|view|exact)|routing-(summary|retrieval)|policy)|server-mmproj-lifecycle)$' --output-on-failure`: 10/10 passed after rebuilding header-dependent tests.
- `python3 -m unittest discover -s tools/server/bench -p 'test*.py'`: 46 passed.
- `python3 -m py_compile tools/server/bench/pager_benchmark_contract.py tools/server/bench/run-pager-profile-benchmark.py`: passed.
- `git diff --check`: passed.
- `PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp python3 /srv/wiretail/task_state.py validate`: passed before completion.

## Deferred verification

The authenticated live `/metrics` checkpoint was not measured: an unauthenticated
probe returned HTTP 401 and no benchmark credential is configured in this
workspace. Later setup is to provide `BENCH_API_KEY` or `LLAMA_API_KEY_FILE`,
run the rebuilt CUDA server with the intended pager/native-MTP profile, scrape
before/after request snapshots, and verify nonzero transfer/residency behavior
against raw server output. No live performance or hardware correctness claim is
made here.

## Next action

Proceed to task 18-05. Preserve the snapshot identity and provenance rules when
adding parity comparisons.
