# 19-04 direct Turbo4 native-MTP verification evidence

Classification: implementation and deterministic CUDA verification passed
locally. The isolated model-backed Qwen MTP checkpoint is deferred because the
shared RTX 4080 runtime workload was retained.

## Implementation

The public paged-attention descriptor now carries KV-head count and preserves
query-token shape. The CUDA primitive launches one CTA per query/head, uses
explicit Q/output/page-mass/partial-state query strides, derives the KV head
from the runtime GQA ratio, and validates the supported boundary of one to
three query tokens, batch one, head width 256, causal mode, and ratio four.

The graph and context paths retain the native `[head_dim, query_head,
query_token, batch]` layout, pass every query position, allocate three-dimensional
page-mass telemetry, and allow the direct path for both decode and
`mtp_verify`. Prefill remains on the reference path for the next task.

## Verification

| Command | Result |
|---|---|
| `cmake --build build-cuda --target test-cuda-fattn-paged-turbo4 llama -j2` | pass |
| `cmake --build build-cuda --target test-kv-attention-execution -j2` | pass |
| `ctest --test-dir build-cuda --output-on-failure -R '^test-kv-attention-execution$'` | 1/1 pass |
| `ctest --test-dir build-cuda --output-on-failure -R '^(test-kv-attention-view\|test-kv-attention-telemetry\|test-kv-attention-exact\|test-cuda-fattn-paged-turbo4)$'` | 4/4 pass |
| `build-cuda/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 -j 2` | 2948/2948 pass |
| `git diff --check` | pass |

The focused CUDA fixture exercised query counts 1, 2, and 3, rejected 4,
verified query/head output isolation, causal page mass, partial `[m,l,o]`
state, page permutation, a missing logical page, and a 17-row tail.
Machine-readable details and hashes are in `19-04_VERIFY_KERNEL.json`.

## Deferred verification

No model-backed Qwen MTP invocation was started or disturbed. After GPU
isolation, run the runtime-supported Qwen command with a deliberately small hot
budget and capture the original compressed-page checksum, canonical host
checksum, direct-attention checksum, CUDA event/H2D transfer accounting, and a
warm repetition proving no redundant upload. Model and corpus hashes remain
unmeasured for this task.

## Exact next action

Use the isolated live checkpoint above when the GPU is available; leave prefill
direct routing unchanged until task 19-05.
