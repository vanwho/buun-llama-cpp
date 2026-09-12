# SPEED25_05_PREFILL_KERNELS

Result: `pass_with_deferred_end_to_end_comparison_and_ncu`.

The direct paged Turbo4 kernel now treats a query tile as the scheduling unit.
The dispatcher accepts qualified B beyond 64, launches `ceil(B/tile_q)` in
`grid.z`, and uses query-base/query-count indexing while retaining explicit
strides, bounded shared memory, online softmax, page-mass export, and split-KV
partial-state merging. The dense selected graph path also has an explicit
no-input refresh branch; a B=256 current-source server startup no longer falls
through to the reference row-ID assertion.

## Raw geometry and tile results

The CUDA fixture uses 24 Q heads, 4 KV heads (GQA group size 6), head width
256, 530 selected rows, logical-to-physical page permutation with gaps, a
one-row and 17-row tail, native positions, a poisoned future row, and output
canaries. It checks Q=64/65/256/257/512 against the serial oracle, including
page mass and split-KV state. The complete output is in
`.wiretail/execution/evidence/raw/SPEED25_05_PREFILL_KERNELS_cuda.txt`.

| Shape | tile_q16 | tile_q32 | tile_q64 | measured winner |
| --- | ---: | ---: | ---: | --- |
| Q=64 | 4.606 ms | 4.718 ms | 8.882 ms | direct/tile_q16 |
| Q=256 | 13.392 ms | 13.376 ms | 16.543 ms | direct/tile_q32 |
| Q=512 | 25.092 ms | 25.179 ms | 23.313 ms | direct/tile_q64 |

The balanced default remains `tile_q32`; the dispatcher accepts all three
bounded choices. Q=512 serial control was 21.635 ms and split control was
21.202 ms. These are kernel/fixture timings, not packed+dense end-to-end
claims.

## Verification

- `cmake --build build --target test-kv-attention-view test-kv-attention-execution -j2` — pass.
- `ctest --test-dir build -R 'kv-attention-(view|execution)' --output-on-failure` — 2/2 pass.
- `cmake --build build-cuda --target llama-server -j2` — pass.
- `cmake --build build-cuda --target test-cuda-fattn-paged-turbo4 -j2` — pass.
- `build-cuda/bin/test-cuda-fattn-paged-turbo4` — pass, exit 0, RTX 4080.
- Current-source bundle activated through the established service; `/health` returned `{"status":"ok"}`.
- One V6 micro request with prompt 2048/context 4096/ubatch 256 — driver status pass; no generated SSE timing sample was exported.
- `git diff --check` — pass.

## Provenance and deferred verification

The uncommitted source diff hash at receipt creation is
`8e9f787e380881be13a32b1f27093dd011bbc8ca7be300793a177e56c6bb6b69`.
The current-source server bundle and live raw directory are recorded in the
JSON receipt. Nsight Compute attached but could not access GPU performance
counters (`ERR_NVGPUCTRPERM`), so occupancy/register/spill evidence is
deferred. A matched packed-versus-dense total-cost experiment is also
deferred; the raw fixture establishes correctness and direct-kernel shape
coverage, while task 25-06 owns full-model batching.
