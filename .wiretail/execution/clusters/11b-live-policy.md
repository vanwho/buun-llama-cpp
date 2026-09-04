# Cluster 11b — dynamic hot tables, asynchronous prefetch, and atomic lifecycle

Tasks: `11-04`, `11-05`.

Purpose: turn retrieval and retention evidence into safe live residency changes and overlap those changes
with useful model work.

Carry forward:

- controller capacity is the live `H`; mandatory pages are inserted first and dynamic ratios fill the
  remainder exactly after deduplication;
- retrieval candidates and retention evidence remain separate. Structural pins, recent pages, bounded
  exploration, hysteresis, fault/dirty cost, and attention evidence have explicit decision reasons;
- policy publishes only complete target tables. Required misses may wait within a budget, reuse the prior
  hot table, or temporarily use a safe larger union when capacity permits; partial data is never visible;
- prefetch uses dedicated streams/events and bounded staging, predicts future candidates, coalesces page
  runs where profitable, measures overlap, and enforces backpressure;
- server slot reuse, sequence operations, prompt artifacts/checkpoints, speculative accept/reject,
  recurrent state, MTP frontier, cancellation, and teardown share one generation-safe operation;
- start with one slot/sequence if required by capability gating. Multi-slot is enabled only after a real
  isolation/concurrency proof.

Read: plan sections 8.8–8.9 and 22; handoffs 04-03–05-04, 09-05, 11-01–11-03;
policy/prefetch/residency transactions, hybrid memory, speculative code, server slot/cache lifecycle,
and checkpoint tests.

Exit gate: warm focus remains stable, a cold needle promotion becomes usable on a later token without a
synchronous full-cache copy, focus shifts do not corrupt state, target/MTP/recurrent rollback agrees,
stale completions cannot publish after slot reuse, and raw counters demonstrate real compute/transfer
overlap.
