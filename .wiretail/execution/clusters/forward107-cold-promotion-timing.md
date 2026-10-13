# Phase 107 — Paired cold-page promotion timing

Revision: `verified-cold-promotion-cost-20261013`.
Task: `107-01`. Execution/assessment: GPT-6 Luna High.

106-00 filled its rows but kept its chosen page 0 resident. It produced no
same-context hot/cold timing pairs. 105-21 previously observed real promotions,
but its transfer counters alone do not measure the extra request latency.
Those results remain historical; do not reopen them or load their raw history.

107-01 uses the latest already deployed ACO build, frozen once, and a fresh
server for each geometry. Build an A source region larger than the entire hot
pool, then a B pressure region, and choose the final A question from an actual
host-backed cold page in A. A cannot all remain resident because A exceeds H.
Measure cold recall first, then the identical question with its promoted target
hot. Re-evaluate the final question in both cases using the existing production
query rewind. Use three fresh-fill repetitions per geometry. Raw evidence stays
external. Read this cluster, the current task, and bounded source symbols only.

No source tuning, diagnostic attention routes, forced page IDs, artificial
answers, or per-token instrumentation belongs in this measurement task.
Setup defects are repaired in the helper. A completed retrieval miss is data;
an all-resident skip is an invalid setup and cannot satisfy this task.
The packet defines the exact commands, comparison validity and failure handling.
