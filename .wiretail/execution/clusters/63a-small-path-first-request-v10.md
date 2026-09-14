# 63a — repair the measured small-path first-request boundary

Revision: `hotpath-v10-20260914`. Task: `63-01`.

Repair only the exact-token L8192/H4096/B128/U64 first-request stall preserved
by the phase-61 summary. The packed selected-attention allocation/submission
boundary is the measured owner; do not broaden this task into paging policy,
MTP tuning, or a new scheduler.
