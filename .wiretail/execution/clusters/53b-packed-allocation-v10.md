# 53b-packed-allocation-v10

Revision: `hotpath-v10-20260914`. Task: `53-02`.

Repair the separately measured packed selected-attention allocation and CUDA
dynamic-shared-memory failure. Preserve the mature packed FA route, pager
ownership, full-L GPU native-MTP, and failure cleanup. The smallest fixture is
`L262144/H16384/B128/U64`; no scale run is useful until it reaches a request.

