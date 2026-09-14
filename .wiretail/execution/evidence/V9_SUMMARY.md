# V9 compact benchmark summary

Result: the aggregation is complete, but v9 capability is not met. The small
selected route is a real matched measurement; natural cold promotion and the
32K/128K campaign are not proven and must remain gated.

## Status booleans

| Boolean | Value | Meaning |
| --- | --- | --- |
| `active_campaign_met` | `false` | The up-to-128K live campaign did not start because the natural cold gate failed. |
| `overall_256k_demonstrated` | `false` | No 256K runtime was launched or demonstrated. |
| `natural_cold_proven` | `false` | No current-Q candidate, useful H2D promotion, or cold target use was published. |
| `fullL_gpu_turbo4_mtp` | `true` | The measured selected identity and integration receipt retain full-L GPU draft and Turbo4 K/V for target and draft with native MTP. |
| `measured_cpu_speedup` | `true` | Three matched selected/CPU-main-KV decode ratios are 3.20–3.53×. This is not an overall product-success claim. |

The requested `V9_INTEGRATION.json` alias is absent; the immediate result
`.wiretail/execution/evidence/33-03-integration.json` was used and indexed.
The compact index is `V9_SUMMARY_INDEX.json`.

## Identity and contract

The selected small candidate was `/srv/ai/paged-kv/results/v9/36-01/20260913T224500Z-candidate/bin/llama-server`, PID 3626509, health 200, CUDA target, GPU MTP, page capacity 16, and 4096 attention tokens. The resolved model was `Qwen3.8-27B-UD-IQ4_XS.gguf` with SHA-256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.

Measured selected geometry is `L8192/C6144/H4096/A4096/B128/U128`; the cold proof is `L8192/C5377→5415/H4096/A2048/B128/U64`. Both satisfy `selected ≤ A ≤ H ≤ L`, use 256-token pages, and retain target/draft Turbo4 with `draft_L=L`, GPU draft, and native MTP. The 32K and 128K tuples are requested coordinates only; their runtime fields remain null.

## Original-three-prompt comparison

All nine fresh requests completed 128 committed output tokens. Rates are tok/s;
each row is q0/q1/q2, with three samples per profile.

| Profile / occupied frontier | Prefill | Committed decode | MTP proposed/accepted |
| --- | --- | --- | --- |
| Selected H4096/A4096, B/U128/128 at C6144 | 214.23 / 215.64 / 214.91 | 43.68 / 39.61 / 42.33 | 66/60, 72/55, 67/59 |
| CPU-main-KV, `--no-kv-offload` | 723.49 / 729.51 / 733.70 | 12.37 / 12.39 / 12.64 | 126/0, 126/0, 126/0 |
| All-GPU control, pager off | 1436.45 / 1445.61 / 1440.24 | 40.30 / 40.11 / 40.32 | 126/0, 126/0, 126/0 |

Selected/CPU-main-KV decode is 3.5298× / 3.1971× / 3.3486×, median
3.4164×. Selected/all-GPU decode is 1.0840× / 0.9875× / 1.0497×, median
1.0504×. Selected prefill median is 0.2946× CPU-KV and 0.1492× all-GPU;
selected TTFT median is 3.3924× CPU-KV and 6.6901× all-GPU. These are paired
measurements, not dense-equivalent sparse-attention claims.

## Cached append, work, and memory

The bounded cold proof reached `C5377→5415` with 5381 cached tokens and 34 new
prefill tokens, then committed 15 output tokens at 29.8662 tok/s after 0.468757
s. It proposed 16 and accepted 7 MTP tokens. The answer and append-after-recall
were coherent (`cedar`), but this was not natural cold use.

Its selected-packed work was 34,603,008 pack bytes, 90,832,896 host-copy
bytes, zero H2D bytes, zero selected cold pages, 115 graph builds, zero replays,
115 router refreshes, 4,896,310 µs queue time, zero copy time, and 109,555 µs
wait time. It reported zero attention samples, ten skipped samples, zero
submitted transfers, and zero transfer completions.

Peak measured categories were target-valid 64,965,120 B, host-valid
64,880,640 B, host-pageable 90,832,896 B, and packed workspace 69,206,016 B;
safety headroom was 201,326,592 B. CPU RSS is null. The foundation fixture
reported packed workspace 69,206,000 B, dequant scratch 8,388,610 B, and graph
268,435,000 B. Its first append+attention event was 22.969280 ms versus
1.246208–1.299456 ms for later tails; tail H2D was only 0.001728–0.005888 ms.

## Diagnosis

- Measurement failure: the current-Q → catalogue → mailbox edge did not
  publish. `attention_samples=0` and zero H2D make promotion unmeasured; they
  do not show that the host copy or hardware failed.
- Runtime: no selected small-request crash, non-finite result, or completion
  failure was recorded. Recall success is separate from selection quality.
- Selection quality: unmeasured. Normal selection had no candidate; the
  A4096/recent512 diagnostic selected all 16 pages and cannot prove ranking.
- Measured latency dominators are first-use append/page-table plus graph/kernel
  warmup, and cold receipt graph/router churn. Small per-category attribution
  is unavailable and remains null.

## 256K ledger estimate (not a demonstration)

For `L=C=262144` and 256-token pages (1024 logical pages), proportional host
history from the measured C5415 receipt is estimated at 3,140,917,912 B. With
H held at 4096, measured target-valid remains 64,965,120 B; an H16384
sensitivity is 259,860,480 B. Holding measured A4096 workspace, dequant, and
graph categories gives 69,206,016 B, 8,388,610 B, and 268,435,000 B.

The partial sum, 3,551,912,658 B, is not a total: full-L draft GPU bytes,
all-history catalogue bytes, weights, recurrent state, transfer ring, and
larger-context graph/scratch growth were not measured. The source catalogue
formula is `llama_context_catalogue_reserve_bytes` plus
`llama_kv_routing_summary_device_layout::make`, but its model geometry is not
in the selected receipts. Before any 256K claim, prove the actual full-L GPU
Turbo4 native-MTP allocation/headroom, catalogue and working-set ledger, and
natural cold promotion/use.

## Next actions (maximum three)

1. Repair current-Q sample publication through
   `llm_graph_context::build_attn_inp_kv`,
   `llama_context::publish_kv_attention_telemetry`, and
   `llama_kv_attention_telemetry::publish_completed`. Cheapest discriminating
   test: focused telemetry tests plus one bounded C>H natural recall requiring
   a non-sentinel candidate and `attention_samples>0`. Stop before 32K if it
   still produces zero samples/candidates/H2D.
2. Check packed Q transform/grouping in `llm_graph_context::build_attn_mha`,
   `llm_graph_context::build_attn`, and
   `llama_kv_attention_execution::prepare` with the existing CUDA
   multi-KV-head selected-packed fixture at A4096/U128. Stop at the first
   layout mismatch; do not revive the superseded direct-kernel path.
3. After those checks pass, rerun the natural proof and original-three-prompt
   comparison through `llama_kv_cache::apply_pager_live_policy` and
   `llama_context::get_kv_pager_metrics`; only then reopen the bounded 32K gate
   and conditionally the 128K pilot.

## Evidence and deferred checks

Receipts: `V9_SUMMARY.json`, `V9_SUMMARY_INDEX.json`,
`V9_FOUNDATION.json`, `V9_COLD_PROOF.json`, `V9_SMALL.json`, `V9_32K.json`,
`V9_128K_PILOT.json`, `V9_128K.json`, and the integration fallback above.
No new live campaign was launched by this aggregation task. 32K/128K remain
honest `not_run` findings; 256K remains an estimate/design extension.
