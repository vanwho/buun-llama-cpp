# Cluster 04b — hotset controller and asynchronous scheduler

Tasks: `04-03`, `04-04`.

Purpose: combine proven routing/retention evidence into deterministic bounded intents, then execute them
with predictive asynchronous transfers and backpressure.

Carry forward:

- exact initial capacity split is 96 recent, 32 structural, 140 historical, 36 transient after
  deduplication; total remains budget-derived;
- mandatory/current/in-flight/speculative pins are never silently evicted;
- normalized EMA/peak/frequency/recency/fault costs and hysteresis drive victim order; trace evidence
  precedes fixed coefficients;
- policy is a pure function shared by replay/live paths; scheduler owns transfer queues/events;
- dedicated upload stream and bounded/double-buffered staging overlap future-token work;
- late pages follow an explicit wait/old-set/fallback policy and partial pages never publish;
- clean eviction has zero D2H; obsolete intents cancel idempotently.

Read: plan sections 3.2, 8.6–8.8, 11 Phase 04, 17; cluster 04a and 02b handoffs. Do not begin
server/MTP integration.

Exit artifacts: deterministic controller and bounded scheduler with trace replay, overlap timeline,
prefetch/fault/churn counters, cancellation, and backpressure tests.
