# Phase-45 compact benchmark summary

Result: aggregation passed, but natural-cold eligibility and all scale gates
remain unmet. The first natural run had a recoverable pager transaction
failure; the fresh managed recovery completed the bounded fixture.

## Status

| Check | Result |
| --- | --- |
| `active_campaign_met` | `false`; L8192 stopped before 32K |
| `overall_256k_demonstrated` | `false`; no 256K run or total was measured |
| `natural_cold_proven` | `false`; no candidate, promotion, useful H2D, or cold-page use |
| `fullL_gpu_turbo4_mtp` | `true`; CUDA, Turbo4 target/draft K/V, GPU draft, native MTP at L8192 |
| `measured_cpu_speedup` | `null`; CPU-main-KV is not an equivalent-throughput claim |

## Identity, units, and geometry

The selected route is `selected_packed`, with `L8192/H4096/A2048/B128/U64`
and 256-token pages. Rates are tok/s, durations are milliseconds, `_us` fields
are microseconds, bytes are integer bytes, and ratios are dimensionless.
The invariant `selected <= A_by_layer <= H <= L` passed; the fixture reached
5004 occupied tokens before recall and a 5052-token recall prompt.

## Current-Q and natural cold use

The selected-packed producer emitted 26 attention and 26 attention-mass samples.
No candidate or promotion was published. The recovery recall and append were
coherent with 8 committed tokens, but transfer submitted, useful H2D bytes,
and selected cold pages used were all zero. This failed the selection/use gate,
so scale was not run.

## Matched q0/q1/q2 controls

Each profile has three samples for original prompts q0/q1/q2, 128 committed
tokens per request, and zero errors.

| Profile | Prefill tok/s q0/q1/q2 | Decode tok/s q0/q1/q2 | MTP proposed/accepted q0/q1/q2 |
| --- | --- | --- | --- |
| Selected packed | 304.668 / 578.084 / 601.078 | 34.784 / 34.440 / 34.169 | 245/3, 249/1, 251/0 |
| CPU-main-KV, GPU weights/MTP | 153.641 / 204.738 / 213.470 | 10.122 / 10.085 / 10.153 | 251/0, 251/0, 251/0 |
| All-GPU, pager off | 301.605 / 579.710 / 599.277 | 36.530 / 36.669 / 36.782 | 251/0, 251/0, 251/0 |

Selected/all-GPU decode ratios are `0.952 / 0.939 / 0.929`; selected/CPU
ratios are `3.437 / 3.415 / 3.365`. These are descriptive only.

## Placement, memory, failures, and actions

Selected packed places target K/V in the selected-packed route with native GPU
MTP. The CPU control places target K/V on CPU while weights, draft, and MTP
remain on GPU. The all-GPU control places target K/V on GPU with the pager off.

Measured bytes: target allocated/valid `69,206,016/68,175,360`, host valid
`64,880,640`, weights `13,695,551,488`, MTP compute `268,435,456`, packed
workspace `69,206,016`, and headroom `201,326,592`. Full-256K total is `null`.

- Measurement: pass; 26 current-Q samples and positive publication observed.
- Runtime: recovered after failure; the first run failed at C=2807 during pager batch-write reservation, then recovery passed.
- Selection quality: failed gate; candidate, promotion, useful H2D, and selected cold use were absent. Owners include `publish_completed`, `capture_kv_routing_query`, `apply_pager_live_policy`, and prefetch `poll`/`take_ready`.

1. Repair or instrument candidate/promotion selection, then rerun the bounded C>H fixture.
2. Keep `measured_cpu_speedup: null` unless equivalent throughput is defined.
3. Run 32K and conditional 128K only after natural cold use passes.

Raw results root: `/srv/ai/paged-kv/results/v9/45-02/`.
