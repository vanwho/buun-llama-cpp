# Evidence index V2

This release-facing ledger records the attention-aware KV pager’s operator
boundary, measured evidence, failed gates, and deferred checks. Raw artifacts
are external evidence; their machine paths are not portable command defaults.

## Portable contracts

- [Server operator contract](../../../tools/server/README.md)
- [Speculative/MTP contract](../../../docs/speculative.md)
- [Benchmark adapter contract](../../../tools/server/bench/README.md)
- [Frozen benchmark schema](PAGER_BENCHMARK_CONTRACT_V2.json)
- [V2 upstream slice map](../upstream/SLICE_MAP_V2.md)

The supported pager boundary is causal Qwen3.8-family Turbo4 K/V on the
selected CUDA route, with single-slot, batch-one decode coverage. Modes are
`off`, `observe`, `selective`, and `exact`; unsupported geometry, placement,
stale identity, missing backing, dirty eviction, or insufficient budget fails
closed. Page size, hot capacity, host budget, and safety headroom are runtime
ledger results; no fixed hot-page count is an operator promise.

Native MTP uses the resolved target per-sequence context, is Turbo4/GPU
resident, and is outside the target pager victim set. VRAM owns the selected
target window and ordinary model/runtime reservations; host RAM owns canonical
sealed-page backing and metadata; pinned memory owns only the bounded transfer
ring. Recurrent state and MTP are separate allocations.

Metrics use `llamacpp:kv_pager_mode` plus labeled identity gauges and scalar
`llamacpp:kv_pager_<field>` gauges. Missing telemetry means `not_configured`,
not zero.

## Repository evidence

| area | evidence |
| --- | --- |
| CPU release matrix | [CPU_TEST_MATRIX_V2](CPU_TEST_MATRIX_V2.md) |
| CUDA release matrix and fault coverage | [CUDA_TEST_MATRIX_V2](CUDA_TEST_MATRIX_V2.md) |
| operator/parser surface | [OPERATOR_SURFACE_MATRIX](OPERATOR_SURFACE_MATRIX.md) |
| dynamic context and capacity ladder | [DYNAMIC_CAPACITY_LADDER](DYNAMIC_CAPACITY_LADDER.md), [JSON manifest](DYNAMIC_CAPACITY_LADDER.json) |
| frozen corpus V3 validation | [PAGER_CORPUS_V3_VALIDATION](PAGER_CORPUS_V3_VALIDATION.md), [contract](PAGER_BENCHMARK_CONTRACT_V2.json) |
| historical controls/calibration | [BASELINE_V2](BASELINE_V2.md), [policy calibration](POLICY_CALIBRATION_13-05.md), [performance profile](PERFORMANCE_PROFILE.md) |
| quality acceptance | [V2 report](QUALITY_ACCEPTANCE_V2.md), [V2 manifest](QUALITY_ACCEPTANCE_V2.json) |
| performance/selective speed | [V2 report](PERFORMANCE_ACCEPTANCE_V2.md), [V2 manifest](PERFORMANCE_ACCEPTANCE_V2.json) |
| soak/churn/fault/cancellation | [V2 report](SOAK_ACCEPTANCE_V2.md), [V2 manifest](SOAK_ACCEPTANCE_V2.json) |
| final disposition | [FINAL_ACCEPTANCE_V2](FINAL_ACCEPTANCE_V2.md), [JSON](FINAL_ACCEPTANCE_V2.json) |

Release logs are [CPU](release-cpu-14-01.log), [CUDA](release-cuda-14-01.log),
[fault](release-faults-14-01.log), and
[sanitizer](release-sanitizer-14-01.log). These are deterministic release
checks, not model-backed quality or speed claims.

## Acceptance disposition

The former V2 gates remain historical blocked diagnostics. They are not replay
instructions for the current execution. Phase 17 owns corrective implementation
and fresh quality/performance/soak evidence; phase 18 reads only its compact
summary and decides whether the overall goal was reached. A dense control result
is not an acceptance result, and a historical profile is not a speed claim.

Corrective V3 artifacts are indexed below; they do not rewrite V2:

| gate | raw root | manifest | manifest SHA-256 |
| --- | --- | --- | --- |
| quality | `/srv/ai/paged-kv/results/15-07-quality-20260904T163000Z/` | `quality-gate-v3.manifest.json` | `9ccd97f27fdaab0216a2d03f662e8f12bc48f44b3b7677f39d9390d1ed0d50b1` |
| performance | `/srv/ai/paged-kv/results/15-08-performance-20260904T144503Z/` | `performance-v3.manifest.json` | `7fa2a5f9fafc16909250b8eadfe98263718c70c80b50ccc9e0da5db6aebeb603` |
| performance paired controls | same performance root | `paired-controls-v3.manifest.json` | `badacb7ead79bb1b138251a0031676d77bb9bd569133e3780219737211a2c8a2` |
| bounded soak | `/srv/ai/paged-kv/results/15-09-soak-20260904T181000Z-bounded/` | `soak-v3.manifest.json` | `5c735e8af6f4c711f5dea21eaf5cd29eb6edfb1f6c6e44d9da8a4cc47653eefb` |

V3 statuses are historical `failed_dense_control`,
`failed_floor_and_runtime_telemetry_boundary`, and
`lifecycle_passed_acceptance_blocked`. No V3 result promotes a blocked gate.

## Current review chain

Phase 17 produces `PHASE17_QUALITY`, `PHASE17_PERFORMANCE`,
`PHASE17_SOAK`, and `PHASE17_BENCHMARK_SUMMARY` manifests. Task 18-01 reads
that summary and writes `PHASE18_REVIEW`; if the goal is not reached, it adds a
new measured remediation phase and the next benchmark-only review task. These
current artifacts intentionally contain concise pointers rather than the old
16-03 audit narrative.

## Shared provenance

| item | SHA-256 / value |
| --- | --- |
| model | `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199` |
| tokenizer | `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199` |
| release binary | `4cb1a90011d2d5b486b3f83b2ce3c9863eb65b55119b290fa52c8e70c1854626` |
| release config | `2497f112026c71f904ac115cded4dd290a42bfd52a18ba202dc7337e5f88276b` |
| V2 corpus | `e8cd002b2a6215a5003db7b8c204602bba1879a5bdf91a2a9ef9b0a2e3ff0e35` |
| V3 corpus file | `ff62534953f9fc73e616cf050c7b6df367e1c9f7f048443700167067dab83e5f` |
| V3 corpus logical hash | `cbf1e3bf727f0f52e8d0e922a68af9cd70bf7ded48b440ae2ef9328921ac52e` |

## Deferred verification

No new hardware or service verification is claimed by this documentation
task. Model-backed selective/native-MTP quality, speed, and full pager soak
remain blocked by the linked raw manifests. Upstream synchronization and
final clean-worktree publication remain the authorized Git owner’s action;
the [16-01 handoff](../handoffs/16-01.md) records that boundary.
