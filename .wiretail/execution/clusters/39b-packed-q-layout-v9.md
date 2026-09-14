# Cluster 39b — packed Q/layout boundary

Revision: `hotpath-v9-20260914`. Task: `39-02`.

Purpose: check the packed selected-route Q transform/grouping and its execution
metadata at the existing multi-KV-head Turbo4 fixture. Change code only for a
reproduced mismatch, preserving mature packed attention and full-L GPU
native-MTP. Stop at the first layout mismatch; do not optimize unrelated
latency or restore the historical direct path.

Context is the phase-38 review, the 39-01 handoff, named graph/execution
symbols, and the existing focused attention view/execution fixtures.
