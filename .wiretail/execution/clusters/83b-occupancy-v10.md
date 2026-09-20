# 83b — advance the phase-81 occupancy frontier

Revision: `hotpath-v10-20260914`. Task: `83-02`.

Advance the measured full-L occupancy frontier with an allocation ledger and
a bounded packed-route probe. Preserve exact stop semantics. The probe has a
12-minute wall budget and stops after three 30-second polls without durable-C
progress, or immediately on timeout/OOM/segfault; it is not a speed campaign
and must not delay the stable MTP/pager diagnostic matrix.
