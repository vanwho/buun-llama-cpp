# V10 phase-59 benchmark summary

- Result: **current_findings**
- Revision: `hotpath-v10-20260914`
- Matched geometry: L=8192, C=6144, H=8192, A=4096, B=128, U=64

## Identity and placement

- Requested model/mode: `qwen-3.8` / `selective`.
- Observed bundle: `/srv/ai/paged-kv/results/v10/59-01/candidate-bundle/bin/llama-server`; model hash `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
- Target and draft placement are recorded independently: Turbo4 target on CUDA and Turbo4 native draft on GPU.

## Rates and request-scoped MTP

| mode | prefill samples/median tok/s | decode samples/median tok/s |
|---|---:|---:|
| selective_native | 9 / 169.23090228992217 | 9 / 12.677138246965708 |
| all_gpu_control | 9 / 1255.3911941173153 | 9 / 38.07416132736045 |
| cpu_main_kv_gpu_draft | 9 / 445.7631604608077 | 9 / 10.404467157974276 |

- Native selective rows retain request-scoped draft/accepted denominators (59/0 per row); controls are explicitly feature-off.
- Cached append is `not_run` because the campaign stopped at the full-L boundary.

## Promotion, quality, and capacity boundaries

- Controlled and organic physical promotion are separate `not_run` findings; answer quality is also separate and `not_run`.
- 32K and 128K pilots are `not_run`.
- 256K allocation/startup is `measured` with actual occupied C1000; occupied C262144 is `failed` after resume frontier mismatch and is not inferred from allocation.

See the JSON for every row, reason, denominator, identity field, and checksum.
