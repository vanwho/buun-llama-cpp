# V9_SMALL

Task 36-04 completed the matched small comparison at L8192/C6144 with the
three original prompts, token-counted by `run-final-curve.py`, fresh slots,
temperature 0, seed 42, thinking off, and 128 committed output tokens.

| Profile | q0/q1/q2 prefill tok/s | q0/q1/q2 decode tok/s | MTP draft/accepted |
| --- | --- | --- | --- |
| Selected H4096/A4096, B/U128/128 | 214.23 / 215.64 / 214.91 | 43.68 / 39.61 / 42.33 | 66/60, 72/55, 67/59 |
| CPU-main-KV, `--no-kv-offload` | 723.49 / 729.51 / 733.70 | 12.37 / 12.39 / 12.64 | 126/0, 126/0, 126/0 |
| All-GPU control, pager off | 1436.45 / 1445.61 / 1440.24 | 40.30 / 40.11 / 40.32 | 126/0, 126/0, 126/0 |

Selected decode divided by CPU-KV decode was 3.20–3.53x (three matched
samples). Selected decode divided by all-GPU decode was 0.987–1.084x. Prefill
and TTFT ratios are retained in `V9_SMALL.json`; the selected route has the
expected selective-path overhead and is not presented as a dense-equivalent
speed win.

The selected candidate bundle, executable, model, target/draft Turbo4 types,
native GPU MTP, and all loaded DSO identities matched across runs. The final
loaded selected process was independently health-checked on port 8080 with
page capacity 16 and attention tokens 4096.

Cold-proof linkage is deliberately separate from performance: the preceding
36-03 incremental proof recalled `cedar-orbit-17`, but published no natural
cold candidate and transferred zero H2D bytes. Thus recall passed while
natural cold promotion remains unproven.

Raw records:

- `/srv/ai/paged-kv/results/v9/36-04/20260915T010000Z-selected-h4096-a4096-u128`
- `/srv/ai/paged-kv/results/v9/36-04/20260915T013000Z-cpu-kv-h4096-control`
- `/srv/ai/paged-kv/results/v9/36-04/20260915T020000Z-allgpu-h8192-control`
