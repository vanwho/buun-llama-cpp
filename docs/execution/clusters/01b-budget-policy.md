# Cluster 01b — admission budget and pure policy

Tasks: `01-03`, `01-04`.

Purpose: prove capacity arithmetic and deterministic retrieval/retention decisions without transfers or
live backend mutation.

Carry forward:

- reserve weights/fixed state, full 256K Turbo4 MTP, graph/Turbo4 scratch, routing/staging, and headroom
  before rounding remaining VRAM down to full 4.125-MiB target pages;
- 304 pages/77,824 tokens is a desired measured result, never an overcommit instruction;
- initial disjoint pool counts are 96 recent, 32 structural, 140 historical, 36 transient;
- mandatory pins win; overflow is a typed refusal;
- cold retrieval needs summaries/anchors/exploration; retention can use observed attention;
- policy scoring is deterministic and coefficients wait for normalized trace evidence.

Read: plan sections 3, 8.5–8.7, 11 Phase 01, 17; cluster 01a handoffs; fit/cache-budget code and pure
retention precedents. Do not add CUDA or public CLI.

Exit artifacts: reconciled budget ledger and one shared pure policy/replay function with adversarial
tests that later live code calls unchanged.
