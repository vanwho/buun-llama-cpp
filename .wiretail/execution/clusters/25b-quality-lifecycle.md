# Cluster 25b — live quality, cold identity and lifecycle

Tasks: `25-03`, `25-04`. Model policy: Luna High.

Read `POST17_IMPLEMENTATION_STRATEGY.md`, `BENCHMARK_PROTOCOL_V5.md` C/E/F,
cookbook C3–C5, the 24-01 handoff, and the direct dependency handoff. Reuse only
the immutable phase-25 deployment. First establish cold-page physical identity,
exact/selected-all parity and a nonzero quality denominator including a newly
hashed deterministic multi-hop case. Then exercise lifecycle behavior above the
measured hot capacity.

Every accepted row requires complete atomic request telemetry. Preserve failed
attempts, minimize and repair causal live failures, and never turn absent
telemetry, zero movement, skipped rows or old-runtime results into success.
