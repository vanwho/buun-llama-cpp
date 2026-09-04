# Cluster 06a — consolidated correctness and fault campaign

Tasks: `06-01`, `06-02`.

Purpose: close the CPU/fake-backend and CUDA correctness matrices before performance claims.

Carry forward:

- use existing approved test targets/files unless maintainer instructions explicitly allow a new one;
- test feature-off behavior, identity/ABA, map bijection, budgets, seal/reseal, zero-D2H eviction,
  failure at every transaction phase, router/policy traces, hybrid/MTP/server rollback;
- CUDA compares dense T4, gather reference, paged T4, telemetry on/off, physical permutations/gaps/tails,
  graph epochs, prefill/decode, and sanitizer fixtures;
- missing hardware is deferred with exact commands, never reported as passed;
- fix implementation defects in owning code and update its handoff rather than weakening tolerance.

Read: plan section 12 and decision ledger; all implementation handoffs; current test registration and
repository instructions. Avoid benchmark tuning or quality-threshold changes.

Exit artifacts: matrix indexed by invariant, command, result, platform, raw output, and any permitted
deferred check.
