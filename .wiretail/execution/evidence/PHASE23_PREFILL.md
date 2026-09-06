# Phase 23 prefill route repair

Task 23-03 identified and repaired the live Qwen selected-reference admission
bottleneck. The current `qwen35` GGUF reports 24 query heads, 4 KV heads, and
256-wide K/V heads. The direct Turbo4 implementation already computes the KV
head as `query_head / (query_heads / kv_heads)`, but the context, execution
owner, prefill chunk admission, and CUDA shape checks still required a ratio of
4. That stale condition forced the valid Qwen ratio-6 path onto
`selected_reference`.

The gates now accept divisible GQA ratios. The 16-token query tile remains the
CUDA kernel bound; a 32-token tile retains the explicit bounded reference
fallback. Fallback diagnostics identify the first rejected backend, phase,
geometry, storage, model-feature, or slab-geometry condition.

## Verification

- `cmake --build build-cuda --target test-kv-attention-execution test-cuda-fattn-paged-turbo4 -j2` — passed.
- Focused ctest for execution and CUDA paged Turbo4 — 2/2 passed.
- `cmake --build build-cuda --target llama-server test-kv-attention-exact test-kv-pager -j2` — passed.
- Regression ctest for execution, exact attention, pager, and CUDA paged Turbo4 — 4/4 passed.
- The CUDA fixture used the Qwen3.5 `24:4` geometry and passed page permutation,
  native-position gaps/masks/tails, canaries, query tiles 1–16, page mass,
  split-KV state merge, and partial-state waves.
- `git diff --check` — passed.

## Measured finding

The pre-change Phase 21 startup receipts at 20K, 40K, 60K, and 100K all report
`kv_pager_route{route="selected reference"} 1`, with one prefill reference
route and zero direct routes. The source-level geometry measurement explains
that route: the model is 24:4, not 4:1. The deterministic post-change route
fixture selects `selected_direct` for 24:4 and the CUDA numerical fixture passes.
No live prompt-rate improvement is claimed until the same 4K/8K prompts and one
20K warmup are run with the newly built server.

## Deferred verification

The existing successful 262144-token candidate remained loaded on the RTX 4080;
it was not stopped and port 8092 was not touched. Consequently, the live 4K/8K
off/observe/reference/direct comparison, detailed launch/synchronization and
throughput profile, dense parity, and one progress-aware 20K warmup remain
deferred to an authorized GPU lifecycle transition.
