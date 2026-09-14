# 57a — repair native-MTP observation

Revision: `hotpath-v10-20260914`. Task: `57-01`.

Repair the benchmark observation path that drops the server's native-MTP
Prometheus counters. Preserve request-scoped deltas and explicit missing-data
classification; do not infer acceptance from response timing or journal text.
