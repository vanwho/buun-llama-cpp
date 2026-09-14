# Phase-41 compact benchmark summary

Result: aggregation passed, but the capability remains unmet. The explicit
CPU-main-KV boundary now has three matched samples. Natural current-Q
publication, cold promotion/use, and all scale gates remain unproven.

## Status

| Check | Result |
| --- | --- |
| `active_campaign_met` | `false`; 32K/128K were not run behind the failed natural-cold gate |
| `overall_256k_demonstrated` | `false`; no 256K process or allocation was launched |
| `natural_cold_proven` | `false`; current-Q sample, candidate, promotion, and cold use are zero/absent |
| `fullL_gpu_turbo4_mtp` | `true`; CUDA target/draft Turbo4, GPU draft, native MTP, L8192 |
| `measured_cpu_speedup` | `null`; the CPU-main-KV samples are not claimed as an equivalent throughput comparison |

## Identity, units, and geometry

The source commit is `5a9ca5b0eabcb3e02a9cae45d8aac0d083400fc2`; the server,
model, and benchmark receipt hashes are recorded in the JSON summary. The
route is `selected_packed`, with target and draft K/V `turbo4`, GPU draft,
native MTP, 256-token pages, `L8192/H4096/A2048/B128/U64`, and
`selected <= A <= H <= L`. Rates are tok/s, benchmark durations are ms,
bytes are integer bytes, and ratios are dimensionless selected/control ratios.

## Natural cold proof

The incremental C>H fixture used 5004 occupied tokens, 5052 recall prompt
tokens, and 5017 cached rows. Recall and append were coherent; 20 output
tokens completed with 19 proposed and 9 accepted MTP tokens. However,
`attention_samples=0`, candidate publication was false, `transfer_submitted=0`,
`h2d_useful_bytes=0`, and `selected_cold_pages_used=0`. This is a
measurement-boundary failure, not evidence of poor selection, and is not
eligible for 32K or 128K.

## Matched controls

Three fresh original prompts were measured for each profile, with 128
committed output tokens:

| Profile | Prefill tok/s q0/q1/q2 | Decode tok/s q0/q1/q2 | MTP proposed/accepted q0/q1/q2 |
| --- | --- | --- | --- |
| Selected packed | 604.105 / 589.051 / 613.198 | 33.760 / 33.611 / 32.983 | 745/4 / 747/3 / 753/0 |
| CPU-main-KV | 227.193 / 205.642 / 210.079 | 9.924 / 9.918 / 9.987 | 753/0 / 753/0 / 753/0 |
| All-GPU | 609.938 / 575.537 / 610.571 | 36.715 / 36.828 / 36.846 | 753/0 / 753/0 / 753/0 |

Selected/all-GPU decode ratios are 0.9198 / 0.9121 / 0.8952. The paired
selected/CPU-main-KV ratios are 3.4012 / 3.3882 / 3.3029, but no speedup is
claimed because the receipt does not define that comparison as equivalent.
Runtime errors were zero.

## Memory, failures, and next actions

Measured bytes: target allocated/valid `69,206,000/2,618,880`, host valid `0`,
weights `13,695,600,000`, MTP compute `268,435,000`, packed workspace
`69,206,000`, and headroom `201,327,000`. Full-256K totals are null.

- Measurement failure: current-Q publication remained absent at
  `build_attn_inp_kv`, `publish_kv_attention_telemetry`, and `publish_completed`.
- Runtime failure: not observed; all requests completed coherently.
- Selection quality: unmeasured; no candidate or target cold-page use existed.

1. Repair/instrument current-Q publication and rerun the C>H fixture; stop if samples, candidate, or useful H2D remain zero.
2. Preserve the explicit CPU-main-KV boundary for future matched comparisons; retain null speedup until equivalence is established in the receipt contract.
3. Reopen 32K, then conditionally 128K, only after candidate publication, attention samples, useful H2D, selected cold-page use, and coherent recall all pass.

Raw results: `/srv/ai/paged-kv/results/v9/41-03/`. Focused CUDA tests and the
bounded controls passed; scale verification is deferred.
