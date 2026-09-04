# Evidence index V2

This is the release-facing index for the attention-aware KV pager. It is an
evidence ledger, not a claim that every gate passed. Machine paths below are
intentional raw-artifact pointers and must not be copied into portable source
or operator commands.

## Scope and source contracts

- [Work state](../WORK_STATE.json) — task and acceptance contract.
- [V2 slice map](../upstream/SLICE_MAP_V2.md) — reviewable implementation
  slices and upstream handoff.
- [server operator contract](../../../tools/server/README.md) and
  [speculative/MTP contract](../../../docs/speculative.md).
- [benchmark adapter contract](../../../tools/server/bench/README.md) and
  [frozen schema](PAGER_BENCHMARK_CONTRACT_V2.json).

The supported boundary is Qwen3.8-family causal Turbo4 K/V on the selected
CUDA route, with single-slot/batch-one decode coverage. Modes are `off`,
`observe`, `selective`, and `exact`; unsupported geometry, backend, placement,
stale identity, missing backing, dirty eviction, or insufficient budget fails
closed. Hot capacity is derived from the runtime ledger. Native MTP uses the
resolved target context, is Turbo4/GPU-resident, and is outside the target
victim set.

## Provenance

The acceptance manifests share these exact identity values:

| item | value |
| --- | --- |
| model SHA256 | `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199` |
| tokenizer SHA256 | `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199` |
| release binary SHA256 | `4cb1a90011d2d5b486b3f83b2ce3c9863eb65b55119b290fa52c8e70c1854626` |
| release config SHA256 | `2497f112026c71f904ac115cded4dd290a42bfd52a18ba202dc7337e5f88276b` |
| corpus schema/hash | `pager-corpus-v2` / `e8cd002b2a6215a5003db7b8c204602bba1879a5bdf91a2a9ef9b0a2e3ff0e35` |

## Release and deterministic checks

- [CPU matrix V2](CPU_TEST_MATRIX_V2.md), [CUDA matrix V2](CUDA_TEST_MATRIX_V2.md)
- [operator surface matrix](OPERATOR_SURFACE_MATRIX.md)
- [release CPU log](release-cpu-14-01.log), [CUDA log](release-cuda-14-01.log),
  [fault log](release-faults-14-01.log), and [sanitizer log](release-sanitizer-14-01.log)
- [dynamic context ladder](DYNAMIC_CAPACITY_LADDER.md) and its
  [JSON manifest](DYNAMIC_CAPACITY_LADDER.json)

The ladder proves runtime-derived sizing and explicit refusal boundaries. Its
measured rows are historical controls, not a preferred hot count or portable
capacity promise.

## Acceptance ledger

| gate | result | evidence and interpretation |
| --- | --- | --- |
| quality | **BLOCKED** | [QUALITY_ACCEPTANCE_V2](QUALITY_ACCEPTANCE_V2.md): corpus fixtures did not produce the required dense/selective acceptance rows; exact failed closed. |
| selective/exact speed | **BLOCKED** | [PERFORMANCE_ACCEPTANCE_V2](PERFORMANCE_ACCEPTANCE_V2.md): maximum-context startup refused and the smaller selective trial hit speculative rollback; no ratio is claimed. |
| soak/churn | **BLOCKED** | [SOAK_ACCEPTANCE_V2](SOAK_ACCEPTANCE_V2.md): pager soak depends on the blocked performance prerequisite; ordinary control, cancellation, restart, and restoration observations remain recorded. |
| final release acceptance | **DEFERRED/BLOCKED** | [FINAL_ACCEPTANCE](FINAL_ACCEPTANCE.md) and the three V2 manifests; phase-14 failures are explicit and are not hidden as a packaging pass. |

Controls, calibration, held-out acceptance, exact-mode checks, and selective
speed are separate evidence classes. Failed startup, fault, cancellation,
rollback, and churn artifacts remain part of the record in the linked raw
roots.

## Raw roots and exact manifest pointers

| artifact | raw root | manifest |
| --- | --- | --- |
| capacity ladder | `/srv/ai/paged-kv/results/14-02-ladder-20260904T004547Z` | [JSON](DYNAMIC_CAPACITY_LADDER.json) |
| quality | `/srv/ai/paged-kv/results/14-03-quality-20260904T012500Z` | [JSON](QUALITY_ACCEPTANCE_V2.json) |
| performance | `/srv/ai/paged-kv/results/14-04-speed-20260904T020000Z` | [JSON](PERFORMANCE_ACCEPTANCE_V2.json) |
| soak | `/srv/ai/paged-kv/results/14-05-soak-20260904T041000Z` | [JSON](SOAK_ACCEPTANCE_V2.json) |

The exact raw files include the release configuration, request envelopes,
metrics snapshots, resource ledgers, and failure/churn traces. They are
external evidence, not repository portability inputs.

## Deferred verification

No specialized hardware check is newly deferred by this documentation task.
The quality, selective-speed, and pager-soak gates above remain blocked by
their recorded runtime outcomes. Upstream re-sync and final branch verification
remain deferred to the authorized Git owner as recorded in the historical
[handoff 16-01](../handoffs/16-01.md).
