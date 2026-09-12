# SPEED25_07_HOST evidence

Status: complete.

The CUDA host-publication probe used one selected page backed by 32 CUDA
source units. Publication was queued asynchronously; the catalog had zero live
pages before the worker completion and one committed page afterward. The
committed version was 7 and the catalog bytes passed byte equality against the
fixture.

Measured CUDA result: 16,384 payload bytes copied once, 16,384 canonical
pageable bytes, 24,320 catalog metadata bytes, a 131,072-byte bounded staging
ring, 32 submitted D2H chunks, zero backpressure waits, and 32 event
completions. Payload amplification was 1.0. The host result reported zero
catalog-pinned bytes because the bounded ring owns the staging allocation.

Raw output: `raw/SPEED25_07_HOST_cuda.txt`.

Before/after counts: catalog live pages 0 -> 1; queued pages 0 -> 1 before
drain and one committed page after drain. The focused test wall time was about
0.073 seconds. No model throughput claim is made: the required 4K prompt and
128-token pressure comparison is deferred to 25-08.

Commands and results:

- `cmake --build build --target test-kv-pager llama-server -j2` — pass.
- `cmake --build build-cuda --target test-kv-pager test-vbr-artifact-capture test-cuda-fattn-paged-turbo4 llama-server -j2` — pass.
- `./build/bin/test-kv-pager` — pass.
- `./build-cuda/bin/test-kv-pager` — pass; measured output in raw receipt.
- `./build-cuda/bin/test-vbr-artifact-capture` — pass.
- `./build-cuda/bin/test-cuda-fattn-paged-turbo4` — pass.
- `ctest --test-dir build-cuda -R 'kv-pager|vbr-artifact-capture' --output-on-failure` — 2/2 pass.
- `git diff --check` — pass.
