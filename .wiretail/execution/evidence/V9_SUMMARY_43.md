# Phase-43 compact benchmark summary

Result: aggregation passed, but natural cold eligibility and all scale gates
remain unmet.

## Status

| Check | Result |
| --- | --- |
| `active_campaign_met` | `false`; the L8192 gate stopped before 32K |
| `overall_256k_demonstrated` | `false`; no 256K run or total was measured |
| `natural_cold_proven` | `false`; no candidate, promotion, useful H2D, or cold-page use |
| `fullL_gpu_turbo4_mtp` | `true`; CUDA, Turbo4 target/draft K/V, GPU draft, native MTP at L8192 |
| `measured_cpu_speedup` | `null`; the CPU-main-KV route is not an equivalent-throughput claim |

## Identity, units, and geometry

The selected route is `selected_packed`, with `L8192/H4096/A2048/B128/U64`
and 256-token pages. Target and draft K/V are Turbo4; draft is on GPU and MTP
is native. The invariant `selected <= A_by_layer <= H <= L` passed for 5006
occupied tokens, 5048 recall-prompt tokens, and 5014 cached rows. Rates are
tok/s, durations are milliseconds, `_us` fields are microseconds, bytes are
integer bytes, and ratios are dimensionless selected/control ratios.

## Current-Q and natural cold use

The producer emitted 36 attention samples and 36 attention-mass samples with
565 us publication time; all four dropped counters were zero. Publication did
not produce a candidate or promotion. In the incremental C>H fixture, recall
and append were coherent and 8 output tokens completed, but transfer submitted,
useful H2D bytes, and selected cold pages were all zero. This is a failed
selection/use gate, not proof of cold selection quality, so scale was not run.

## Matched controls

Three samples were taken for each original prompt q0/q1/q2, with 128 committed
output tokens per request and zero errors.

| Profile | Prefill tok/s q0/q1/q2 | Decode tok/s q0/q1/q2 | MTP proposed/accepted |
| --- | --- | --- | --- |
| Selected packed | 658.631 / 648.633 / 674.276 | 34.188 / 34.136 / 33.875 | 2994/9, 2994/9, 2994/9 |
| CPU-main-KV, MTP GPU | 225.639 / 206.771 / 212.768 | 9.986 / 10.072 / 10.074 | null |
| All-GPU | 652.429 / 624.277 / 635.396 | 36.693 / 36.818 / 36.833 | null |

Selected/all-GPU decode ratios were `0.931 / 0.927 / 0.920`; descriptive
selected/CPU-main-KV ratios were `3.423 / 3.389 / 3.365`. No CPU speedup is
claimed because the receipt does not define equivalent throughput.

## Memory, failures, and next actions

Measured bytes: target allocated/valid `69,206,016/67,415,040`, host valid
`64,880,640`, weights `13,695,551,488`, MTP compute `268,435,456`, packed
workspace `69,206,016`, and headroom `201,326,592`. Full-256K total is `null`.

- Measurement: reduced but not failed; current-Q samples/publication counters are nonzero.
- Runtime: not observed; all requests completed coherently with zero control errors.
- Selection quality: failed gate; no candidate, promotion, useful H2D, or selected cold use.

1. Instrument or repair candidate/promotion selection after current-Q publication, then rerun the C>H fixture.
2. Keep the explicit CPU-main-KV boundary and `measured_cpu_speedup: null` unless equivalence is defined.
3. Reopen bounded 32K, then conditional 128K validation only after natural cold use is proven.

Raw results root: `/srv/ai/paged-kv/results/v9/43-02/`.

