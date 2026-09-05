# Initial phase-17 benchmark baseline

## Status

This is the compact initial baseline for corrective task **17-12**. It is not
a phase-18 acceptance verdict and does not claim the overall Turbo4 pager goal
was reached.

The three phase artifacts agree on the model, tokenizer, CUDA binary, V4
corpus hash, 256-token page size, and greedy seed-42 sampling. They do not
provide one fully coherent runtime identity: 17-08/09 record repository head
`817b405f…`, 17-10 records `5d5276ca…`; profile metadata is incomplete in the
first two manifests; and the runs cover 8,192, 16,384, and the V4 ceiling of
22,016 tokens. These are preserved as baseline gaps.

Machine-readable summary: [`PHASE17_BENCHMARK_SUMMARY.json`](PHASE17_BENCHMARK_SUMMARY.json).

## Measured baseline

Quality is incomplete: the dense control matched 11/11 attempted cases but
stopped at 11/24. The 8,192-token native recovery probe returned HTTP 200 but
matched 0/1 exact answers. Selected-all, exact, selective full scoring,
output/logit parity, routing recall, attention mass, perplexity, KL, and
ablations are unmeasured.

Performance is diagnostic only. At 8,192 tokens, selective/native decode p50
was 74.2253 tokens/s, feature-off was 50.0637 tokens/s, and observe/native was
111.0067 tokens/s. The descriptive ratios are 1.482615x and 2.217307x; they
are not the required CPU-KV or safe-all-GPU denominators. The 3x floor, 5x
target, and 70%-of-all-GPU ratio are null.

Native MTP placement was Turbo4/GPU for target and draft. Diagnostic MTP
acceptance was 72.2222% selective and 96.6667% observe. These figures do not
establish quality parity.

The 16,384-token soak passed lifecycle operations and bounded observation:
checkpoint SHA round trip, cancellation release, clear/recovery, restart
health, 45 workload samples, and a 30-second post-restart idle plateau. Pager
bookkeeping reached 12 host pages and 13 resident pages. This is distinct from
physical transfer evidence: faults, evictions, transfer submissions, and H2D/
D2H useful bytes all remained zero. Concurrent transport and slot release
passed, but the retained HTTP-200 output anomaly and failed exact-marker retry
leave semantic isolation unmeasured.

## Goal matrix

| Goal | Baseline status | Observed result |
| --- | --- | --- |
| V4 quality parity | Unmeasured | Full 24-case oracle matrix not completed |
| Full V4 context coverage | Unmeasured | Corpus ceiling 22,016; diagnostics reached at most 16,384 |
| Native-MTP capacity at corpus context | Unmeasured | 8,192 started; 16,384 later soaked but was previously startup-failed |
| Required 3x speed floor | Unmeasured | CPU-KV denominator unavailable |
| 5x / 70% targets | Unmeasured | Safe all-GPU denominator unavailable |
| MTP placement/acceptance | Measured diagnostic | Turbo4/GPU; 72.2222% selective, 96.6667% observe |
| Pager residency bookkeeping | Measured | Host/resident page counts and byte ledger captured |
| Physical transfer/fault/eviction path | Unmeasured | Counters were zero; no positive event evidence |
| Lifecycle safety/resource bound | Measured bounded observation | Save/restore, cancel, clear, restart, health, and idle plateau passed |
| Concurrent semantic isolation | Unmeasured | HTTP/slot behavior passed; deterministic output oracle did not |
| Overall goal | Unmeasured | Required quality, pressure, denominator, transfer, and lifecycle gates are not all complete |

## Raw evidence

- 17-08 quality: `/srv/ai/paged-kv/results/17-08-quality-20260905T032000Z/`
- 17-09 performance: `/srv/ai/paged-kv/results/17-09-performance-20260905T020000Z/`
- 17-10 soak: `/srv/ai/paged-kv/results/17-10-soak-20260905T021200Z/`

Each root contains its own `SHA256SUMS`; the JSON summary links the manifests
and the exact phase artifacts. No large logs or historical narratives are
duplicated here.

## Blocked reasons for 17-12

Repair work must establish a contract-valid context through the V4 ceiling,
complete the exact quality matrix, provide ordinary CPU-KV and safe all-GPU
denominators, and generate positive cold/focus/churn transfer telemetry. It
must also add a deterministic concurrent semantic oracle and retain coherent
repository/profile/context identity. Thresholds are unchanged.
