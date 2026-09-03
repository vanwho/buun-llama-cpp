# Cluster 08b — frozen benchmark and corrected controls

Tasks: `08-04`, `08-05`.

Purpose: freeze quality and speed measurement before selection parameters are tuned, then record
same-build controls that prove context-sized MTP GPU placement.

Carry forward:

- historical Fast-profile results are provenance only; they are not a capacity target;
- create `pager-corpus-v2` with calibration and held-out splits, immutable expected answers/checkers,
  page-distance metadata, hashes, and workloads for warm focus, cold needles, competing needles,
  system/tool anchors, repository facts, conversation referents, focus shifts, churn, and recent control;
- freeze gates from plan sections 23–24 before any selective result exists;
- baseline contexts include small all-pages-fit, an intermediate context, an odd/non-page-aligned test,
  and the model maximum. Exact values are derived from model metadata and harness parameters, not
  production constants;
- compare same-context ordinary CPU target K/V with MTP forced GPU, feature-off, observe where available,
  and the largest safe all-GPU context discovered by a fresh ladder;
- record MTP rows/types/backend/bytes for every context, full resource manifests, raw requests, per-trial
  timing, service state, and restoration health;
- all server-specific corpus/config/result data belongs under `/srv/ai`; only generic harness support and
  immutable hashes/summaries belong in the fork.

Read: plan sections 21, 23, 24; `/srv/ai/benchmarks/{README.md,benchmarks.md,run-profile-benchmark.sh}`;
baseline results/reference; handoffs 00-03, 06-03, 06-04, 08-02, 08-03.

Operational rule: capture the active profile and health, use the established controlled restart path,
ensure `ai-long-memory.service` supplies 8091, never touch 8092, and restore the initial profile even on
failure. Passwordless sudo is available.

Exit artifacts: frozen corpus/gate manifest, generic benchmark schema support, raw control directories,
machine-readable summary, and a handoff that distinguishes completed controls from later selective runs.
