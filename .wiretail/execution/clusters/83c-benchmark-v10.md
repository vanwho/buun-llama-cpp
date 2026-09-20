# 83c — freeze a stable phase-83 native-MTP/pager diagnostic matrix

Revision: `hotpath-v10-20260914`. Task: `83-03`.

Do not spend another long run on the broad speed matrix. Establish one small,
immutable q0 coordinate across all-GPU, CPU-main-KV, selected pager,
feature-off, and native-MTP placements, with B=U=128 first and U64 only as a
secondary causal row. Capture token parity, acceptance, route counters, and
identity so later tasks can distinguish common model failure, MTP state failure,
and pager route failure.
