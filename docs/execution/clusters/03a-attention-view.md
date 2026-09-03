# Cluster 03a — compact attention view and operator plumbing

Tasks: `03-01`, `03-02`.

Purpose: make attention consume a bounded selected physical view with native logical positions before
writing the optimized Turbo4 CUDA kernel.

Carry forward:

- attention width and scratch scale with selected physical rows, never the 262K logical frontier;
- compact slot order is independent from native logical position/RoPE/mask order;
- table epoch is a graph-reuse key and snapshot pages stay pinned until graph completion;
- selected-all-pages and an explicit gather reference are correctness oracles;
- operator metadata must remain backend-neutral and validate gaps, tails, dimensions, and duplicates.

Read: plan sections 8.3, 8.5, 9.4–9.5, 12.2, 17; fixed-window handoff; KV memory interface, graph
builders, sparse FA plumbing from upstream `8e93a9773`, and position fixes from `36b101543`.

Exit artifacts: compact reference view plus validated page-table/native-position graph operator contract.
Do not optimize CUDA loaders in this cluster.
