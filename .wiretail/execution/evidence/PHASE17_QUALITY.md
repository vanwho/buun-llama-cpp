# Phase 17 quality benchmark

Decision: BLOCKED. The full V4 quality matrix has no acceptance claim.

The immutable `pager-corpus-v4` contains 24 cases: 12 calibration and 12
held-out, with measured contexts from 4,096 through 22,016 tokens. Provenance
is fixed to Qwen3.8-27B UD-IQ4_XS, model/tokenizer SHA256
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`, corpus
hash `37111506c8ddc7f9d04086121797cc9c2b7a2842cc1a56ef4b33f506c2252007`, and
CUDA binary SHA256
`d306009f51f0ebb554752d9c41fabb51c796f9d708eef977cc3cc09014e361db`.

## Results

| Run | Configuration | Result | Raw evidence |
| --- | --- | --- | --- |
| Dense control | 16,384, pager off, MTP off | 11/24 cases completed before the diagnostic client was interrupted; 11/11 attempted exact matches | `raw/dense-16384` |
| Native recovery probe | 8,192, selective, native Turbo4/GPU MTP | startup passed; 1 HTTP 200 probe, exact checker failed | `raw/native-8192` |
| Native ladder | 77,824 -> 16,384 -> 8,192 | first two failed CUDA allocation; 8,192 startup passed | `raw/native-mtp-ladder.json` |
| Selected-all reference | required full corpus | not run after dense-control gate | manifest |
| Exact oracle | required full corpus | not run after dense-control gate | manifest |
| Selective attention | required full corpus/native MTP | not run after dense-control gate | manifest |

The startup failures are recorded from the managed service: the 77,824 and
16,384 native-MTP attempts could not allocate an 82.01 MiB compute buffer with
67.8 MiB free. The smaller 8,192 recovery committed 8,192 Turbo4 MTP rows
(8,781,824 bytes) on the GPU, proving the ladder recovery path but not the
full-corpus quality contract.

No output/logit parity, routing recall, attention-mass, MTP acceptance, exact
coverage, perplexity, KL, focus-shift, churn, or ablation gate is claimed.
The exact checker and thresholds were frozen before observation.

Raw results and SHA256 pointers are retained outside Git at
`/srv/ai/paged-kv/results/17-08-quality-20260905T032000Z/`.
