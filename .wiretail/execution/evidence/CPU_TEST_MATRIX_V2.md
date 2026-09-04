# Release CPU correctness matrix — task 14-01

This is the versioned release matrix for the dirty worktree rooted at
`1988b072cc0e3466c739a1762c4e5f44b00a59a4`. The frozen phase-13 release
configuration hash remains `2497f112026c71f904ac115cded4dd290a42bfd52a18ba202dc7337e5f88276b`.
The command/result transcript is [release-cpu-14-01.log](release-cpu-14-01.log).

| Configuration | Exact command | Result |
|---|---|---|
| Release CPU-only | `cmake -S . -B build-14-01-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_WASM=OFF` | Configure and build pass |
| Complete relevant release CTest | `ctest --test-dir build-14-01-cpu -L main --output-on-failure` | 105/105 pass, 27.35 s |
| Backend ops | `build-14-01-cpu/bin/test-backend-ops` | 1/1 backend pass |
| Debug ASAN+UBSAN | See [release-sanitizer-14-01.log](release-sanitizer-14-01.log) | 16/16 focused pass, no sanitizer diagnostics |

The release suite covers tokenizers/vocab, parser and argument surfaces,
budget/accounting/authority, graph/exact/attention, pager/VBR, speculative and
recurrent state, prompt-cache/server lifecycle, feature-off routes, and all
registered deterministic fault fixtures. The Debug matrix additionally keeps
assertion checking enabled and exercises the same pager/attention/recurrent
contracts without CUDA.

The test contract correction in `test-llama-archs.cpp` preserves all child
destinations when recurrent-copy capacity preflight rejects a composite copy.
The policy fixture corrections make setup calls execute in both Debug and
Release; `llama-kv-policy.cpp` now counts unavailable evidence across the
complete trace even after the target reaches capacity.
