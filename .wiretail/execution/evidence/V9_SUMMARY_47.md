# Phase-47 compact benchmark summary

Result: the bounded L8192 implementation and natural append/recall completed,
but natural-cold eligibility and all scale gates remain unmet.

## Status

| Check | Result |
| --- | --- |
| `active_campaign_met` | `false`; stopped at L8192 before 32K |
| `overall_256k_demonstrated` | `false`; no scale run |
| `natural_cold_proven` | `false`; no candidate, promotion, useful H2D, or cold-page use |
| `fullL_gpu_turbo4_mtp` | `true`; CUDA, full L8192 fixture, Turbo4 target/draft K/V, native GPU MTP |
| `measured_cpu_speedup` | `null`; controls are not equivalent-throughput measurements |

## Identity, units, and geometry

The route is `selected_packed`, with `L8192/H4096/A2048/B128/U64` and
256-token pages. Rates are tok/s, durations are milliseconds, `_us` fields are
microseconds, bytes are integer bytes, and ratios are dimensionless selected
divided by the named control. The invariant `selected <= A_by_layer <= H <= L`
passed; the fixture reached 5004 occupied tokens before a 5050-token recall.

## Current-Q and natural cold use

The selected-packed producer emitted 2 attention and 2 attention-mass samples.
No candidate or promotion was published. Recall and append were coherent, but
transfer submitted, useful H2D bytes, and selected cold pages used were all
zero. The selection/use gate therefore failed and scale was not run.

## q0/q1/q2 rates and MTP

Selected packed has q0/q1/q2 measurements; the retry's CPU and all-GPU controls
have q0 only, so q1/q2 remain explicit `null` values.

| Profile | Prefill tok/s q0/q1/q2 | Decode tok/s q0/q1/q2 | MTP proposed/accepted q0/q1/q2 |
| --- | --- | --- | --- |
| Selected packed | 667.720 / 275.603 / 598.854 | 34.247 / 34.482 / 34.367 | 249/1, 247/2, 249/1 |
| CPU-main-KV, GPU weights/MTP | 222.810 / null / null | 10.074 / null / null | 251/0, null, null |
| All-GPU, pager off | 651.141 / null / null | 36.641 / null / null | 251/0, null, null |

Selected/all-GPU decode ratios are `0.935 / 0.941 / 0.938`; selected/CPU q0
decode ratio is `3.400`. These are descriptive only.

## Placement, memory, failures, and actions

Selected packed places target K/V in the selected-packed route with native GPU
MTP. The CPU control places target K/V on CPU while weights, draft, and MTP
remain on GPU. The all-GPU control places target K/V on GPU with the pager off.
No memory category samples were reported, so memory fields remain `null`.

- Measurement: pass; positive current-Q samples and coherent recall/append.
- Runtime: pass; no runtime failure in the bounded fixture.
- Selection quality: failed gate; candidate, promotion, useful H2D, and selected cold use were absent.

1. Repair or instrument candidate/promotion selection, then rerun bounded L8192 natural use.
2. Keep the control boundaries explicit and `measured_cpu_speedup` null.
3. Run 32K and conditional 128K only after natural cold use passes.

Raw results root: `/srv/ai/paged-kv/results/v9/47-02/`.
