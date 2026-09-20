# 85c — promote the direct paged Turbo4 selective route

Revision: `hotpath-v10-20260914`. Task: `85-03`.

After native-MTP parity is repaired, make the persistent physical hot-pool
page table eligible for automatic multi-page selective attention. Keep dense
and reference paths as guarded correctness fallbacks and do not reintroduce a
compact `selected_packed` bridge or per-graph K/V copies.
