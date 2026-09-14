# 61b — repair the alternate full-L packed graph boundary

Revision: `hotpath-v10-20260914`. Task: `61-02`.

Use the current V10 contracts and 60-01 handoff. Consume the repaired VBR
boundary from 61-01, then isolate packed-cache ownership and CUDA launch
resources at L262144/H16384/B128/U64 before scheduling any occupancy claim.
