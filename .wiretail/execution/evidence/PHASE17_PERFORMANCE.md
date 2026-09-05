# Phase 17 performance benchmark

Decision: **BLOCKED for acceptance**. The feasible 8,192-token probe produced
reproducible diagnostic measurements, but the required CPU-KV and safe
all-GPU Turbo4-MTP denominators were unavailable. The exact/native reference
probe returned HTTP 500. The 3x floor and 5x/70% targets are therefore null,
not inferred from the partial data.

## Provenance and protocol

All successful diagnostic modes used the same CUDA server binary
(`d306009f...14e361db`), Qwen3.8-27B UD-IQ4_XS model, V4 corpus provenance,
greedy sampling (`T=0`, seed 42), 8,192 server context, and 256-token pages.
The immutable V4 corpus hash is
`37111506c8ddc7f9d04086121797cc9c2b7a2842cc1a56ef4b33f506c2252007`.

The direct authenticated probe used the three canonical short prompts, ten
trials per prompt, 30 requests per successful mode, `max_tokens=16`, and
non-streaming JSON. This differs from the canonical runner's one discarded
40-token warmup plus three streamed 400-token measurements, so these are
diagnostic stability results rather than canonical acceptance results. The
full V4 resolved contexts were not exercised.

## Measured diagnostic results

| Mode | Requests | Decode p50 / p95 / p99 (tok/s) | TTFT p50 / p95 / p99 (ms) | MTP acceptance |
| --- | ---: | ---: | ---: | ---: |
| Feature off, MTP off | 30 | 50.0637 / 50.0798 / 50.0885 | 37.5805 / 50.1055 / 63.7236 | null |
| Observe, native Turbo4/GPU MTP | 30 | 111.0067 / 111.2590 / 111.3247 | 38.4250 / 52.4897 / 81.8741 | 290/300 = 96.6667% |
| Selective, native Turbo4/GPU MTP | 30 | 74.2253 / 103.3370 / 103.4232 | 38.3750 / 52.1252 / 83.3216 | 260/360 = 72.2222% |
| Exact, native Turbo4/GPU MTP | 1 | — | — | HTTP 500 |

The descriptive same-context ratios are selective/feature-off `1.482615x` and
observe/feature-off `2.217307x`. They are not gate ratios: feature-off is
full-GPU K/V with MTP disabled, not ordinary CPU-KV offload or a safe all-GPU
Turbo4-MTP denominator.

## Runtime telemetry

The selective snapshot measured 32 logical pages, 32-page capacity, 256
tokens/page, one resident page, zero host pages, 138,412,000 target bytes,
8,192 native MTP rows, and GPU Turbo4 target/draft K/V. It recorded zero pager
faults, prefetch hits, H2D/D2H bytes, queued transfers, transfer waits, and
overlap microseconds; 221 pager waits were recorded. These short prompts did
not cause host-page transfers, so they do not establish the intended hot-page
speedup. Full metric snapshots are retained in
`raw/telemetry-summary.json`.

The exact probe planned one resident page and returned HTTP 500 before a usable
reference record. Its request, response, HTTP metadata, service journal,
metrics, GPU snapshot, and RAM snapshot are retained under
`raw/exact-native-8192/`.

## Missing segments and failed hypotheses

Ordinary CPU-KV, safe all-GPU Turbo4-MTP, cold-needle, focus-shift, churn, and
the full V4 performance matrix were not run. No historical results were
substituted. The initial canonical-wrapper attempt is retained as an exit-127
setup failure; direct authenticated runs were used to obtain local diagnostic
data. The exact/native HTTP 500 and the absence of pager transfers on short
inputs are also preserved as failed hypotheses in the JSON manifest.

Machine-readable per-request statistics and all raw response/metric files are
outside Git at
`/srv/ai/paged-kv/results/17-09-performance-20260905T020000Z/`; its
`SHA256SUMS` was checked after generation. See
`PHASE17_PERFORMANCE.json` for the complete gate/nullability record.
