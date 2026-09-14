# Phase-39 compact benchmark summary

Result: aggregation passed, but the capability remains unmet. The selected
packed route is identified and stable in the bounded L8192 run; natural cold
selection/promotion/use is still unproven, so scale remains gated.

## Status

| Check | Result |
| --- | --- |
| `active_campaign_met` | `false`; 32K/128K were not run behind the failed natural-cold gate |
| `overall_256k_demonstrated` | `false`; no 256K process or allocation was launched |
| `natural_cold_proven` | `false`; current-Q sample, candidate, promotion, and cold use are zero/absent |
| `fullL_gpu_turbo4_mtp` | `true`; CUDA target/draft Turbo4, GPU draft, native MTP, L8192 |
| `measured_cpu_speedup` | `null`; current CPU-main-KV control was not run |

## Identity, units, and geometry

Candidate source commit is `09b41e8a1881a570f633eab301a5aaad3719ec49`; the
resolved model SHA-256 is `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
The measured route is `selected_packed`, with target and draft K/V `turbo4`,
GPU draft, native MTP, 256-token pages, `L8192/H4096/A2048/B128/U64`, and
`selected <= A <= H <= L`. Rates are tok/s, durations are seconds unless
named `_ms`/`_us`, bytes are integer bytes, and ratios are dimensionless.

## Natural cold proof

The incremental C>H fixture used 5004 occupied tokens, 5052 recall prompt
tokens, and 5017 cached rows. Recall and append were coherent, with 20
committed output tokens and MTP 147 proposed/42 accepted. However,
`attention_samples=0`, candidate publication was false, `transfer_submitted=0`,
`h2d_useful_bytes=0`, and `selected_cold_pages_used=0`. Therefore this is a
measurement-boundary failure, not proof of poor selection. It is not eligible
for 32K or 128K.

## Matched controls

Three fresh original prompts were measured for selected packed and pager-off
all-GPU controls, each with 128 committed output tokens:

| Profile | Prefill tok/s q0/q1/q2 | Decode tok/s q0/q1/q2 | MTP accepted |
| --- | --- | --- | --- |
| Selected packed | 297.858 / 276.983 / 587.384 | 35.278 / 34.365 / 33.999 | 4% / 81% / 0% |
| CPU-main-KV | not run; launcher lacks `--no-kv-offload` | not run | 0 samples |
| All-GPU | 316.026 / 579.076 / 598.879 | 36.485 / 36.629 / 36.735 | 0 / 0 / 0 |

Selected/all-GPU decode ratios are 0.9669 / 0.9387 / 0.9251 (median
0.9387). No CPU speedup is claimed. Runtime errors were zero.

## Memory and failure classification

Measured bytes: target allocated/valid `69,206,000/68,378,100`, canonical
host `82,182,100`, host valid `64,880,600`, weights `13,695,600,000`, MTP
compute `268,435,000`, packed workspace `69,206,000`, and headroom
`201,327,000`. Full-256K totals are null: larger-context draft, catalogue,
graph, scratch, and working-set allocations were not measured.

- Measurement failure: current-Q publication remained absent at
  `build_attn_inp_kv`, `publish_kv_attention_telemetry`, and
  `publish_completed`.
- Runtime failure: not observed; no crash, non-finite output, or completion
  error occurred.
- Selection quality: unmeasured; no candidate or target cold-page use existed.

## Next actions (three measured actions)

1. Repair/instrument current-Q publication and rerun the C>H fixture; stop if
   samples, candidate, or useful H2D remain zero.
2. Add the explicit no-KV-offload launcher boundary and rerun the three CPU
   controls; retain a null speedup until measured.
3. Reopen 32K, then conditionally 128K, only after candidate publication,
   attention samples, useful H2D, selected cold-page use, and coherent recall
   all pass.

Raw results: `/srv/ai/paged-kv/results/v9/39-03/`. The focused telemetry,
execution, and pager-model tests passed; scale verification is deferred.
