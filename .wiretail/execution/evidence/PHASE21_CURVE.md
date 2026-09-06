# Phase 21 final Turbo4 context curve — task 21-07

## Result

The final speed curve is **incomplete with no completed speed rows**. The
automatic budget-derived startup ladder passed at 20K, 40K, 60K, and 100K,
but the first contextual 20K warmup reached only 3,072 of 19,600 prompt
tokens in 134.30 seconds and was cancelled without a response. Automatic
native-MTP startup failed at 175K and 262,144. No shorter context, fixed-hot
page run, or historical control was substituted.

Machine-readable evidence is [`PHASE21_CURVE.json`](PHASE21_CURVE.json).
Raw artifacts remain at:

- `/srv/ai/paged-kv/results/21-07-startup-ladder-20260906T0726Z/`
- `/srv/ai/paged-kv/results/21-07-curve-20k-20260906T0725Z/`

## Startup findings

| Requested context | Auto startup | Logical pages | Target allocation | Native MTP rows | Speed trials |
| ---: | --- | ---: | ---: | ---: | --- |
| 20,000 | pass | 79 | 341,705,000 B | 20,224 | incomplete warmup |
| 40,000 | pass | 157 | 679,084,000 B | 40,192 | not measured |
| 60,000 | pass | 235 | 1,016,460,000 B | 60,160 | not measured |
| 100,000 | pass | 391 | 1,691,220,000 B | 100,096 | not measured |
| 175,000 | failed: recurrent-state allocation | — | — | — | not measured |
| 262,144 | failed: CUDA compute-buffer allocation | — | — | — | not measured |

Successful startup rows report CUDA target Turbo4 K/V, GPU Turbo4 MTP, 256-token
pages, and automatic hot sizing. They are allocation findings only;
startup `selected_page_count=1` is not occupied-history or movement evidence.

## Speed and telemetry

There are zero completed measured trials. Every per-question median,
dispersion, TTFT, pp/tg rate, output count, MTP acceptance, route fraction,
selected/host/physical row count, transfer byte count, queue/copy/wait time,
graph-launch count, and memory peak is therefore `null`, with a reason in the
JSON receipt. Zero transfer is not claimed: no completed contextual
before/after telemetry exists.

Prepared-warm, cold-load, and fixed-262K growing-history diagnostics remain
separate. The existing 21-06 fixed-four-page 262K allocation and near-full
prefill timeout was not pooled into this curve.

## Runtime state and provenance

After the automatic ladder, the healthy 262K fixed-four-page Turbo4/MTP
candidate was restored and left loaded for dependent work. It is explicitly
not a valid budget-derived curve row. The 8092 CPU service was not touched.

The run used T=0, seed 42, thinking off, the three unchanged benchmark
questions, deterministic neutral context material, one 40-token discarded
warmup, and three requested 400-token streamed trials per question. Binary,
model, corpus, GPU, and raw artifact hashes are in the JSON receipt.

## Deferred verification

The complete six-point per-question speed campaign, cold-load run, growing
262K history, and movement/component telemetry are deferred until the prefill
path and automatic 175K/262K startup allocation failures are repaired. CPU-KV
and all-GPU matched controls are deferred to 21-08.
