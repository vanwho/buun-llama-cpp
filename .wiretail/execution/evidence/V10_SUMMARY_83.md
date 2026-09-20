# Phase-83 summary

Revision: `hotpath-v10-20260914`. This summary aggregates only the phase-83
answer-quality, occupancy, and stable diagnostic receipts. Requested, observed,
and immutable identities remain separate.

## Result

Full `L262144` allocation was admitted, but occupied `C262144` was not
established: the bounded packed probe stopped at durable `C29134` and live
`C29149` for `operator_bounded_wall_budget_stop_after_C29134`. The independent
answer oracle failed all three A/B/A rows: each returned 32 slash tokens and
zero of three expected nonces matched.

The stable matrix has 16 q0 rows at `L8192/C256` and `L8192/C6143`. Native
output matched its same-placement feature-off control in every comparison, so
there is no native-only token divergence in this matrix. All-GPU and
CPU-main-KV controls and native rows share the slash-output failure. Selected
pager rows instead produce `DIAG-83-03`; that is a selected-pager placement
difference, not proof of answer correctness or physical promotion.

## Identities and claims

The requested diagnostic primary was `B128/U128`, with an explicit `B128/U64`
secondary. The launcher observed `B128/U64` for all rows. The stable observed
coordinate is `L8192/C6143/H8192/A4096/B128/U64`, with 256-token pages,
Turbo4 target K/V, and GPU Turbo4 native-MTP K/V. The immutable candidate,
model, and source identities are recorded in the JSON summary.

Full-L allocation measured target `138412032` bytes, native-MTP `276955136`,
packed workspace `138412032`, packed dequant `16777216`, charged
`15965452416`, reserved `2068443264`, and headroom `201326592`. The occupancy
probe completed eight requests with packed route counts `473/8/112` for
prefill/decode/native-MTP verification and `593` accepted route overrides.

The A/B/A oracle used expected values `EMBER-QUARTZ-914`, `CIRRUS-VAULT-268`,
and `EMBER-QUARTZ-914`; observed values were slash output in all rows. Its
first divergent token is position 0 relative to each expected nonce sequence.
This quality failure is not relabelled as promotion, target-graph use, or
speed evidence.

Native MTP produced positive draft work in all native rows. All-GPU and
CPU-main-KV rows were `123/0` drafted/accepted per request. Selected pager was
`16/0` at C256 and `8/4` at C6143, the latter 50 percent. Feature-off rows
have no MTP denominator by design.

## Stable diagnosis matrix

`first divergent token` is relative to the same-placement feature-off/native
pair. A dash means the pair was token-identical; it does not mean the output
was correct. Every non-pass row has an exact measured SSE pointer.

| Row | L/C/H/A/B/U observed | Failure classification | Native draft/accepted | First divergent token | Exact raw pointer |
|---|---|---|---:|---:|---|
| all_gpu_feature_off/secondary-U64/C256 | 8192/256/-/-/128/64 | common model/output | 0/0 | - | `/srv/ai/paged-kv/results/v10/83-03/20260920T155646Z-stable-matrix/all_gpu_feature_off/secondary-U64/C256/measured/raw.sse` |
| all_gpu_feature_off/secondary-U64/C6143 | 8192/6143/-/-/128/64 | common model/output | 0/0 | - | `/srv/ai/paged-kv/results/v10/83-03/20260920T155646Z-stable-matrix/all_gpu_feature_off/secondary-U64/C6143/measured/raw.sse` |
| all_gpu_native_mtp/primary-U128/C256 | 8192/256/-/-/128/64 | common model/output; native-MTP-only zero acceptance; setup U128->U64 | 123/0 | - | `/srv/ai/paged-kv/results/v10/83-03/20260920T155646Z-stable-matrix/all_gpu_native_mtp/primary-U128/C256/measured/raw.sse` |
| all_gpu_native_mtp/primary-U128/C6143 | 8192/6143/-/-/128/64 | common model/output; native-MTP-only zero acceptance; setup U128->U64 | 123/0 | - | `/srv/ai/paged-kv/results/v10/83-03/20260920T155646Z-stable-matrix/all_gpu_native_mtp/primary-U128/C6143/measured/raw.sse` |
| all_gpu_native_mtp/secondary-U64/C256 | 8192/256/-/-/128/64 | common model/output; native-MTP-only zero acceptance | 123/0 | - | `/srv/ai/paged-kv/results/v10/83-03/20260920T155646Z-stable-matrix/all_gpu_native_mtp/secondary-U64/C256/measured/raw.sse` |
| all_gpu_native_mtp/secondary-U64/C6143 | 8192/6143/-/-/128/64 | common model/output; native-MTP-only zero acceptance | 123/0 | - | `/srv/ai/paged-kv/results/v10/83-03/20260920T155646Z-stable-matrix/all_gpu_native_mtp/secondary-U64/C6143/measured/raw.sse` |
| cpu_main_kv_gpu_native_mtp/primary-U128/C256 | 8192/256/-/-/128/64 | common model/output; native-MTP-only zero acceptance; setup U128->U64 | 123/0 | - | `/srv/ai/paged-kv/results/v10/83-03/20260920T155646Z-stable-matrix/cpu_main_kv_gpu_native_mtp/primary-U128/C256/measured/raw.sse` |
| cpu_main_kv_gpu_native_mtp/primary-U128/C6143 | 8192/6143/-/-/128/64 | common model/output; native-MTP-only zero acceptance; setup U128->U64 | 123/0 | - | `/srv/ai/paged-kv/results/v10/83-03/20260920T155646Z-stable-matrix/cpu_main_kv_gpu_native_mtp/primary-U128/C6143/measured/raw.sse` |
| cpu_main_kv_gpu_native_mtp/secondary-U64/C256 | 8192/256/-/-/128/64 | common model/output; native-MTP-only zero acceptance | 123/0 | - | `/srv/ai/paged-kv/results/v10/83-03/20260920T155646Z-stable-matrix/cpu_main_kv_gpu_native_mtp/secondary-U64/C256/measured/raw.sse` |
| cpu_main_kv_gpu_native_mtp/secondary-U64/C6143 | 8192/6143/-/-/128/64 | common model/output; native-MTP-only zero acceptance | 123/0 | - | `/srv/ai/paged-kv/results/v10/83-03/20260920T155646Z-stable-matrix/cpu_main_kv_gpu_native_mtp/secondary-U64/C6143/measured/raw.sse` |
| selected_pager_native_mtp/primary-U128/C256 | 8192/256/8192/4096/128/64 | selected-pager-only; native-MTP-only zero acceptance; setup U128->U64 | 16/0 | - | `/srv/ai/paged-kv/results/v10/83-03/20260920T155646Z-stable-matrix/selected_pager_native_mtp/primary-U128/C256/measured/raw.sse` |
| selected_pager_native_mtp/primary-U128/C6143 | 8192/6143/8192/4096/128/64 | parity pass; native acceptance measured; setup U128->U64 | 8/4 | - | `/srv/ai/paged-kv/results/v10/83-03/20260920T155646Z-stable-matrix/selected_pager_native_mtp/primary-U128/C6143/measured/raw.sse` |
| selected_pager_native_mtp/secondary-U64/C256 | 8192/256/8192/4096/128/64 | selected-pager-only; native-MTP-only zero acceptance | 16/0 | - | `/srv/ai/paged-kv/results/v10/83-03/20260920T155646Z-stable-matrix/selected_pager_native_mtp/secondary-U64/C256/measured/raw.sse` |
| selected_pager_native_mtp/secondary-U64/C6143 | 8192/6143/8192/4096/128/64 | parity pass; native acceptance measured | 8/4 | - | `/srv/ai/paged-kv/results/v10/83-03/20260920T155646Z-stable-matrix/selected_pager_native_mtp/secondary-U64/C6143/measured/raw.sse` |
| selected_pager_feature_off/secondary-U64/C256 | 8192/256/8192/4096/128/64 | selected-pager-only | 0/0 | - | `/srv/ai/paged-kv/results/v10/83-03/20260920T155646Z-stable-matrix/selected_pager_feature_off/secondary-U64/C256/measured/raw.sse` |
| selected_pager_feature_off/secondary-U64/C6143 | 8192/6143/8192/4096/128/64 | selected-pager-only | 0/0 | - | `/srv/ai/paged-kv/results/v10/83-03/20260920T155646Z-stable-matrix/selected_pager_feature_off/secondary-U64/C6143/measured/raw.sse` |

No row has a native-vs-control divergent token. The route counters are retained
in `V10_83-03-summary.json`; pager-off route fields are explicitly not
applicable. The launcher U128 mismatch is an observation/setup failure, not a
native-MTP acceptance result.

## Phase-84 prerequisites

Before broad benchmarking, phase 84 must:

- explain and rerun the all-GPU and CPU-main-KV zero-acceptance state;
- repair or explicitly account for the U128 request being loaded as U64;
- capture cold eligibility, submission, H2D completion, table publication, and
  completed target-graph-use IDs for selected-pager rows;
- keep nonce correctness, native-MTP acceptance, physical promotion, and
  target-graph use as separate claims.

The stable coordinate is `L8192/C6143/H8192/A4096/B128/U64` with selected
pager, Turbo4 target K/V, and GPU Turbo4 native-MTP K/V. The three-prompt speed
matrix, speed ratios, physical-promotion/target-graph claims, and full occupied
`C262144` remain prohibited until the matrix passes.

## Sources

- `.wiretail/execution/evidence/V10_83-01.json`
- `.wiretail/execution/evidence/V10_83-02.json`
- `.wiretail/execution/evidence/V10_83-03.json`
- `.wiretail/execution/evidence/V10_83-03-summary.json`
