# Cluster 14a — release correctness, dynamic ladder, and held-out quality

Tasks: `14-01`, `14-02`, `14-03`.

Purpose: certify the release candidate without deferring model-backed work or retuning after seeing the
held-out results.

Carry forward:

- run all focused and broad CPU tests, CUDA backend/operator tests, memcheck/racecheck where applicable,
  fault injection, and feature-off regression on the release SHA;
- run a context ladder from small all-pages-fit through intermediate, odd-tail, and model-maximum values;
  for each, prove MTP rows equal resolved target context, MTP is Turbo4/GPU, `H` is freshly derived,
  target/host/device bytes reconcile, and startup/teardown returns memory;
- validate explicit budgets/headroom and auto mode, too-small refusal, user hot-page upper bounds, and no
  fixed capacity signature across contexts;
- run dense-versus-selected-all and exact comparisons where dense fits, then exact-versus-selective on
  the held-out corpus at maximum context;
- report expected-answer score, routing recall, captured attention mass, output/logit divergence,
  perplexity/KL sample, MTP acceptance, every failure example, and ablations;
- use the phase-13 parameter/config hash unchanged. A quality failure is a real failure and returns to
  implementation; do not alter answers or thresholds.

Read: plan sections 21–24; all phase 13 handoffs; CPU/CUDA matrices; exact evidence; frozen corpus and
release-candidate config.

Exit artifacts: raw test/sanitizer logs, machine-readable context ladder, allocation ledgers, teardown
snapshots, held-out quality records, exact coverage records, and a pass/block decision. These tasks cannot
be marked deferred.
