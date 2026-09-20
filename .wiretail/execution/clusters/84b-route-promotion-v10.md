# 84b — route and promotion mechanics diagnostics

Revision: `hotpath-v10-20260914`. Task: `84-02`.

Separate route/residency mechanics from MTP correctness. Prove automatic and
diagnostic route behavior and the selector → mailbox → H2D → publication →
target-use chain with a bounded two-document A/B/A workload. The compact
packed bridge is diagnostic only; the desired production path is direct paged
Turbo4 CUDA consumption without per-graph K/V compaction.
