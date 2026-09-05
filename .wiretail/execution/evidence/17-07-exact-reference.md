# 17-07 exact/reference evidence

## Portable correctness

The exact planner now accepts an explicit logical address span with sparse
logical page IDs, while coverage still rejects duplicate visits and requires
every page present in the immutable inventory. Resident identity is resolved
by logical page ID rather than vector order. Cold waves retain bounded staging,
native page positions, online-softmax `(m,l,o)` merging, and measured upload /
wait callback time.

The focused fixture covers:

- permuted resident and cold pages;
- an intentional logical gap;
- a 17-token tail;
- serial and double-buffered cold waves;
- duplicate, missing, stale, and host-backing refusal paths;
- direct-vs-merged online-softmax output equality.

## Commands

```text
cmake --build build --target llama test-kv-attention-exact test-kv-attention-execution -j2
  passed

cmake --build build --target llama-server -j2
  passed

ctest --test-dir build -R '^(test-kv-attention-exact|test-kv-attention-execution|test-kv-attention-view|test-kv-pager|test-kv-residency)$' --output-on-failure
  5/5 passed

git diff --check
  passed
```

The existing bounded CUDA Turbo4 fixture remains the source of direct
dequantization, native causal mask, page-table permutation, and partial-state
hardware evidence. Its live graph integration is not asserted by this task.

## Deferred verification

No linked current-tree CUDA model run or production per-layer graph callback
registration was available. The following remain deferred: host-catalog to
staging-slot upload through the live CUDA event/fence path, per-layer partial
state binding and merge in a Qwen graph, dense-versus-selected-all-versus-exact
logit parity over the V4 corpus, and CUDA sanitizer evidence for the integrated
callback path. Unsupported geometry continues to fail closed and now exposes
`exact_refusal_reason` in server pager metrics.
