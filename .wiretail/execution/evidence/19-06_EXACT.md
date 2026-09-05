# 19-06 exact GPU page-wave evidence

Classification: local implementation and deterministic CPU/CUDA verification
passed. The model-backed cold-prefill/native-MTP checkpoint is deferred because
the shared RTX 4080 workload was retained.

The exact route now publishes an immutable coverage plan to graph construction.
Each attention layer emits one paged Turbo4 node per bounded wave. Resident
waves address the pager slab directly; cold waves copy canonical host page
payloads into a separately sized, compact layer-major staging slab. Intermediate
nodes carry device-resident unnormalized `[m,l,o]` state through stable
online-softmax rescaling, and only the final node normalizes before the existing
Turbo output transform/projection. The CPU `compute_wave` callback remains an
oracle seam and is not used by the GPU graph.

## Verification

| Command | Result |
|---|---|
| `cmake --build build-cuda --target llama test-kv-attention-exact test-cuda-fattn-paged-turbo4 -j8` | pass |
| `ctest --test-dir build-cuda -R 'test-(kv-attention-execution\|kv-attention-exact\|cuda-fattn-paged-turbo4)$' --output-on-failure` | 3/3 pass |
| `build-cuda/bin/test-cuda-fattn-paged-turbo4` | pass; 1/2/3-query tiles, tails, reordered pages, native causal positions, page mass, host upload, and split-wave `[m,l,o]` merge |
| `build-cuda/bin/test-kv-attention-exact` | pass; dense merge, empty partition, exact coverage, and CPU oracle |
| `./build-cuda/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 -j 2` | 2948/2948 pass |
| `git diff --check` | pass |

The deterministic CUDA fixture measured 19.077 ms for its 529-row, four-head
query case. This timing is a finding, not a throughput gate. Coverage telemetry
is populated from the validated immutable plan: wave/page counts, valid rows,
cold useful bytes, staging peak, and exact pages visited.

## Deferred verification

At the final probe, `nvidia-smi` reported an RTX 4080 with 14,815 MiB used of
16,376 MiB and 1,128 MiB free by an existing workload. No model-backed process
was started, killed, or disturbed. After GPU isolation, run the runtime-
supported Qwen exact cold-prefill checkpoint at two occupied lengths with
native MTP off/on; capture progress, bounded staging allocation, H2D/event
timings, exact coverage counters, and dense-vs-exact logits/checksums.

## Changed files

`ggml/include/ggml.h`, `ggml/src/ggml.c`,
`ggml/src/ggml-cuda/fattn-paged-turbo4.cuh`, `ggml/src/ggml-cuda/fattn.cu`,
`src/llama-context.cpp`, `src/llama-graph.cpp`, `src/llama-graph.h`,
`src/llama-kv-attention-exact.cpp`, `src/llama-kv-attention-exact.h`,
`src/llama-kv-attention-execution.cpp`,
`src/llama-kv-attention-execution.h`, and
`tests/test-cuda-fattn-paged-turbo4.cu`.
