# Phase 21 quality checkpoint

## Result

The bounded representative matrix is **not measured**. The managed 8080
candidate accepted the first warm preflight request, but its authenticated
`/metrics` response exported no `llamacpp:kv_pager_*` samples. The existing
fail-fast harness recorded `telemetry_refusal` and stopped before sending the
remaining preflight requests or any quality-matrix request. This is a
capability result, not a model-quality failure.

Raw live receipt: `PHASE21_QUALITY.live-preflight.json`.

## Frozen matrix

The planned calibration matrix was six representative cases repeated once in
each of the exact, selected-all, and selective labels: `warm-focus`,
`cold-early`, `cold-middle`, `cold-end`, `competing-a`, and `focus-shift`.
Sampling was T=0, seed 42, thinking off, and max output 8 tokens. The corpus
fixture is pager-corpus-v4 with a derived acceptance context of 22,016 tokens.

| Control | Numerator | Denominator | Planned | Omitted-page label | Sparse-policy label |
| --- | ---: | ---: | ---: | --- | --- |
| Exact | 0 | 0 | 6 | not_observed | not_observed |
| Selected-all | 0 | 0 | 6 | not_observed | not_observed |
| Selective | 0 | 0 | 6 | not_observed | not_observed |

The zero denominators mean no response had the required route, selected-page,
residency, movement, timing, placement, and hot-budget telemetry. No answer,
selected page ID, host checksum, promotion, or sparse-policy behavior was
scored.

## Preflight and repair owner

- Executed: one warm request (`warm-focus`), HTTP 200, actual occupied prompt
  tokens 4,088.
- Not executed after fail-fast: `cold_needle` and `selected_all` preflight
  probes, then all 18 matrix requests.
- Missing evidence: route; selected/physical/logical pages; host-valid rows;
  H2D/D2H useful and aligned bytes; faults; evictions; queue/copy/wait times;
  target/MTP placement; hot-page budget; monotonic snapshot timestamp.
- Repair owner: managed `llama-server` deployment and its pager telemetry
  exporter. The next run must expose authenticated `llamacpp:kv_pager_*`
  metrics and pass the three-request preflight before quality scoring.

No selective failure is attributed to model quality. No speed authorization or
256K capacity claim is made. The exact resume command and omitted secrets are
in the machine-readable receipt.

## Verification

The local benchmark contract tests and state validator are recorded in the
21-05 handoff. Port 8092 was not contacted or modified.
