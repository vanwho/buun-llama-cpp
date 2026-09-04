# Cluster 10b — bounded prefill and live exact page waves

Tasks: `10-04`, `10-05`.

Purpose: complete prompt ingestion and connect the exact all-page oracle to real host/catalog/CUDA
callbacks.

Carry forward:

- prefill must be chunked and bounded by selected physical rows plus current query block; graph/scratch
  cannot scale to the full logical frontier on GPU;
- prefill writes/seals pages in order while attention uses native positions and the complete required
  causal set for the chosen mode; selected mode may use the reference path until an optimized supported
  path is proven;
- first decode after prefill, page-aligned and odd tails, table changes between chunks, and cancellation
  during seal must be covered;
- exact mode visits each valid logical page once, stages cold Turbo4 waves through bounded slots, computes
  per-layer/head `(m,l,o)` state on GPU, and merges stably without an F16 full cache;
- exact shares the real catalog, transfer streams, identity checks, masks, positions, and graph fences;
  no fake callback may satisfy the model-backed gate;
- exact is measured separately and may be slow, but its output/logit comparison is mandatory.

Read: plan sections 8.10, 22–23; handoffs 03-04, 06-05, 09-04–10-03;
`src/llama-kv-attention-exact.*`, attention execution/view/operator, model graph/prefill scheduling, and
exact/CUDA fixtures.

Exit artifacts: real multi-chunk prefill tests and timings, a model-backed exact route at bounded and
maximum contexts, dense-vs-selected-all-vs-exact numerical evidence, page coverage ledgers, staging peak,
and graph/scratch scaling evidence.
