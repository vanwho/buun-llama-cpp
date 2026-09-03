# 13-05 policy calibration and release candidate

The production policy now has an explicit capacity-relative release candidate.
It keeps `H` runtime-derived: recent, structural, historical, and transient
page counts remain `auto`, while ratios and minima are applied only after
mandatory pages have been inserted. The complete machine-readable record is
[`POLICY_CALIBRATION_13-05.json`](POLICY_CALIBRATION_13-05.json).

## Selected configuration

| Area | Release value |
| --- | --- |
| Recent / structural / historical / transient ratios | 40% / 10% / 35% / 15% |
| Automatic minima | recent 1 page, transient 1 page; structural and historical 0 |
| Normalized retention weights | EMA 50%, recent peak 20%, frequency 15%, recency 15% |
| Hysteresis | `100000` normalized score units |
| Summary and exploration | top-K 8, exploration budget 2 |
| Attention telemetry cadence | every 4 tokens |
| Predictive prefetch depth | 2 pages |
| Hot-page and recent-token caps | `auto` |

Rounding is deterministic: mandatory pages consume capacity first, minima are
clamped to the remaining capacity, floor shares are assigned, and remainder
pages are assigned in recent/structural/historical/transient order. At very
small `H`, unavailable minima are simply not allocated. If `H >= L`, the
controller selects all logical pages.

The release configuration hash is
`2497f112026c71f904ac115cded4dd290a42bfd52a18ba202dc7337e5f88276b`. It covers
the complete pager/policy payload, the frozen corpus hash, and the freshly
rebuilt local Release CPU `libllama` binary hash recorded in the JSON artifact.
The CUDA binary refresh was attempted but interrupted during generated
template compilation, so no CUDA binary or model benchmark is claimed.

## Bounded replay and ablations

The focused deterministic replay exercised admitted capacities
`1, 2, 3, 5, 8, 16, 304`, including mandatory-page subtraction, minima below
capacity, exact target sizing, weighted normalized evidence, and invalid
zero-weight rejection. The release candidate, the prior equal-evidence
baseline, and a quality-first exploration candidate all remained bounded and
deterministic.

| Component | Replay result | Live quality/speed |
| --- | --- | --- |
| Structural pins | Mandatory path preserved | deferred |
| Recent region | Capacity-relative quota preserved | deferred |
| Summary top-K | Deterministic candidate union | deferred |
| Exploration | Bounded deterministic slice | deferred |
| Attention retention | Weighted normalized score | deferred |
| Hysteresis | Deterministic previous-target retention | deferred |
| Prefetch | Existing bounded scheduler coverage | deferred |

The replay result is not a model quality result. Held-out answers and results
were not read.

## Pareto and integrated speed status

No live selective pager trial was available: the current benchmark controls
are `off`/all-GPU and the server does not expose a configured selective pager
route with the required counters. Therefore routing recall, captured exact
attention mass, task score, perplexity/KL, MTP acceptance under paging, cold
latency, observe overhead, and the 3x CPU-KV checkpoint remain `null` in the
JSON. The candidate is locked for phase-14 wiring, but it is not represented
as a completed quality or speed gate.

Prior profiler-driven evidence is preserved in the 13-02, 13-03, and 13-04
raw artifacts. The reserved live output directory is
`/srv/ai/paged-kv/results/13-05/`; no live raw output was produced by this
task.
