# Cluster 04a — cold-page routing and attention telemetry

Tasks: `04-01`, `04-02`.

Purpose: produce the two independent evidence streams used by policy: all-page retrieval scores and
observed resident-page retention scores.

Carry forward:

- routing summaries exist for all valid pages so a cold page can be rediscovered;
- initial summary candidate is 4–8 representative rotated K vectors on selected layers/heads, but its
  precision/shape remains measured;
- anchors and exploration remain explicit even with a good router;
- FA reduces sum attention mass and recent peak on GPU; controller transfer is bounded to about 4 KiB
  for 1,024 floats, never an attention matrix;
- observations are keyed by page/table generation and stale results are dropped;
- `off` has no work/allocation and telemetry on/off preserves attention output.

Read: plan sections 2, 8.6–8.7, 12, 17; Phase 01 policy handoff and Phase 03 kernel handoffs;
`ggml/src/ggml-cuda/top-k.cu` and bounded observer metrics. Do not select victims or move pages yet.

Exit artifacts: measured summary builder/scorer and allocation-free page telemetry with recall,
normalization candidates, byte cost, latency, and overhead evidence.
