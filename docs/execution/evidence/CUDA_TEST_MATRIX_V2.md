# Release CUDA correctness matrix — task 14-01

Release provenance: worktree rooted at
`1988b072cc0e3466c739a1762c4e5f44b00a59a4`, CUDA 12.4.131, NVIDIA GeForce RTX
4080 (compute 8.9, 16376 MiB), driver 595.84, architecture `89-real`. The
frozen phase-13 configuration hash is
`2497f112026c71f904ac115cded4dd290a42bfd52a18ba202dc7337e5f88276b`.
Raw command results are in [release-cuda-14-01.log](release-cuda-14-01.log) and
[release-faults-14-01.log](release-faults-14-01.log).

| Coverage | Exact command/result |
|---|---|
| Clean CUDA Release | `cmake -S . -B build-14-01-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89-real -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_WASM=OFF`; all registered test targets built |
| CUDA main, resource-independent | `ctest --test-dir build-14-01-cuda -L main -E '^(test-backend-ops|test-llama-archs|test-thread-safety|test-moe-cache)$' --output-on-failure`; 109/109 pass, 56.93 s |
| Turbo4/VBR/CUDA focused | 12/12 pass: paged Turbo4, VMM, graph key, Q8 selector, RDNA2 policy, HC combine/stream/grouped RMS, MMVQ post-SILU, and MoE cache selector |
| CUDA backend-ops narrow | `build-14-01-cuda/bin/test-backend-ops -b CUDA0 -o ADD`; 99/99 pass |
| CUDA fault and feature-off | Registered deterministic fault and feature-off fixtures in the 109/109 rerun; all pass |
| CUDA memory tools | Turbo4 memcheck 0 errors; racecheck 0 hazards/0 errors/0 warnings; initcheck 0 errors; VMM memcheck 0 errors |

The unfiltered `-L main` run was executed and reported four resource-bound
failures while `/srv/ai/paged-kv/build/buun/bin/llama-server` held 15326 MiB
of the single GPU, leaving 462 MiB free (218 MiB observed during the tests):

- `test-backend-ops`: CUDA allocation failures in the full operator sweep.
- `test-llama-archs`: dynamic-VBR budget exhaustion in the model stress path.
- `test-thread-safety`: cublas CUDA allocation failure under four contexts.
- `test-moe-cache`: `cache-expert-profile` failed under the same occupied-GPU
  condition.

The 109-test rerun and 99-case CUDA ADD rerun passed. A clean-GPU rerun of the
four excluded model/full-backend cases is explicitly deferred to the external
service owner; no model-backed quality result is claimed by this task.
