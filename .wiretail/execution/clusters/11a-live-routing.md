# Cluster 11a — production all-page retrieval and attention telemetry

Tasks: `11-01`, `11-02`, `11-03`.

Purpose: give selective mode a quality-safe way to rediscover cold pages and retain genuinely attended
resident pages.

Carry forward:

- every sealed page gets a compact routing summary derived from the same logical Turbo4 content and
  invalidated by the same identity/representation generation;
- measure candidate summary shapes on calibration traces rather than assuming one. Start with the prior
  representative/rotated-K seam, then compare centroid, extrema/upper-bound, and multi-vector summaries
  only when they preserve bounded storage and computation;
- query-to-summary scoring covers all logical pages, applies top-K plus structural/recent/mandatory union,
  and runs on the cheaper measured CPU or GPU path. It does not copy full K or attention matrices;
- attention telemetry is reduced on GPU by logical page/layer/head, copied to the controller in bounded
  form after events complete, and tagged by table/page generations;
- missing cold-page observation is unknown. Only participating pages update EMA/peak/frequency;
- record routing recall, captured exact/dense attention mass, summary bytes, scoring time, reduction time,
  copy bytes, sample cadence, and total observe overhead.

Read: plan sections 8.6–8.7, 23; handoffs 04-01–04-03, 06-05, 08-04, 10-01–10-05;
`src/llama-kv-{routing-summary,attention-telemetry,policy}.*`, CUDA attention reduction, server token/turn
metadata, and frozen calibration corpus.

Gate: held-back cold pages are retrievable in live model traces, stale samples are rejected, telemetry
does not alter attention output, and observe overhead remains within the frozen checkpoint budget or is
returned to optimization with raw profiles.
