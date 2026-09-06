# Phase 21 prefill evidence — task 21-02

## Result

The serialized three-token safety workaround is removed. Selective/exact
prefill now uses bounded 16-token Turbo4 query tiles, fences before advancing
at contiguous 256-token page ends, and fences once for the final tile. The CUDA
Turbo4 paged kernel accepts query tiles 1 through 16. A 32-token planner shape
is intentionally routed to the tested selected-reference fallback with the
explicit reason `bounded Turbo4 selected reference for unsupported direct query
tile`.

The direct CUDA path and dense/reference parity fixtures pass. The live Qwen
probe did not satisfy the direct capability predicate, so the model-backed
selective route was selected-reference and remains slow. The speed target is
not claimed.

## Verification

- `cmake --build build-cuda --target test-kv-attention-execution test-cuda-fattn-paged-turbo4 llama-server -j2` — pass.
- `build-cuda/bin/test-kv-attention-execution` and `build-cuda/bin/test-cuda-fattn-paged-turbo4` — pass.
- Focused CUDA/CPU CTest expression — 6/6 pass.
- `python3 -m unittest discover -s tools/server/bench -p 'test_*.py'` — 46 pass.
- `git diff --check` — pass.

The CUDA fixture covers 1/2/3/4/8/16 query tiles, rejected 17/32 shapes,
partial tails, page mass, logical gaps/permutations, native causal positions,
partial state, host upload, and split-wave `[m,l,o]` merge. The planner fixture
covers 1/3/8/16/32 bounded shapes and checks the explicit fallback reason.

## Bounded live diagnostic

Raw records are under `/tmp/21-02-prefill-wave`. Off and observe completed at
4,148 and 8,244 prompt tokens at approximately 1.85–1.90K prompt tokens/s.
The corrected selective 4K request completed at 14.393 prompt tokens/s with
18 graph submissions/completions and 18 waits for 4,148 tokens; its route was
selected-reference with zero direct routes. The selective 8K request was
bounded to 750 seconds, reached 5,120 tokens at 10.5 tokens/s, and timed out;
that result is recorded as a diagnostic limitation rather than a performance
claim. GPU utilization samples, graph counts, wait time, and route counters are
captured beside each raw case.

The managed service was restored and health-checked after the probe using the
20-07 runtime bundle, 262,144 context, selective mode, 256-token pages, four
hot pages, and native GPU Turbo4 MTP. No corpus campaign was run.

## Deferred verification

The live direct route remains deferred because the current model-backed runtime
capability predicate rejects that binding; task 21-03 owns the selective
cold-page promotion/routing follow-up. The selective 8K reference case timed
out at its explicit diagnostic bound. Full-context and corpus-quality checks
remain deferred to later packets.
