# 105-03b attempt01 — partial split-layout long-context findings

Status: `incomplete`, not a PASS receipt. The occupancy snapshot records C=114687 against target250000 (L262144/H51200, 200 hot pages, B1024/U256, split-document-query layout). Ten requests completed; the next returned HTTP500:

```text
decode() failed: KV pager batch write reservation failed: transaction sequence=0 positions=114687..114942 rows=256 frozen_history_pages=0 phase=1
```

This is a runtime failure, not a semantic miss. Root cause is unknown; a failure-only-logging build and tiny repro are in progress. Cleanup restored the previously healthy service. Final A2, occupied-context canonical12, and both additional source-specific fact requests were not reached. Do not infer ranking, promotion, or full occupancy.

## Candidate and partial performance

Candidate source was repo HEAD `503aec68e4c920a4586941fdb7c60446a6ce0d82` plus graph-arena repair delta `a079452f56788049109cbef6404f233a41f75e08`; it was not built from a079 HEAD. Geometry was L262144/H51200, Turbo4 target plus GPU Turbo4 MTP, B1024/U256. Model SHA-256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`; executable `9c95e267c7daf768ecf77f2cb03c2a23e4e2f100b9db12773f66369f36d05b65`; libllama `5a03a591c7d0125a9469274bc747cf819fa801b9d580bda4294fa885b841e4bb`; server-impl `efa611a672ab73931f8d834cd6d7017fc4e702acc2307f9ccebb1c1384f6b1c0`; CUDA `f8b66c3fa9b17cc77e08678dba8407b4c2998362278e6e796f9d5cf67de42867`. Full loaded-DSO identity is preserved in the external candidate artifact. The ten-row summary gives median fresh/processed input throughput870/872 tok/s, decode45.5 tok/s, and native-MTP acceptance72.3%.

The rates below use each request's `timings.prompt_ms`: fresh tok/s is fresh input tokens divided by prompt time; processed tok/s is `timings.prompt_n` divided by prompt time. Medians are870 fresh and872 processed tok/s. The earlier ~375 fresh tok/s result came from a different whole-message-replay run, not a strict paired baseline. In the paired small-context B-layout comparison, the split layout measured1236.61 fresh tok/s versus703.20 for legacy.

| # | Stage | C after | Fresh input | GPU processed input (`prompt_n`) | Fresh tok/s | Processed tok/s | Decode tok/s | MTP accepted/drafted |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | A1 | 4293 | 3894 | 4030 | 1217.9 | 1260.4 | 81.2 | 225/347 |
| 2 | B | 16586 | 11894 | 11991 | 1217.0 | 1226.9 | 88.5 | 247/302 |
| 3 | F2 | 28845 | 11860 | 11892 | 926.0 | 928.5 | 79.2 | 238/322 |
| 4 | F3 | 41146 | 11902 | 11934 | 743.7 | 745.7 | 79.0 | 246/304 |
| 5 | F4 | 53442 | 11897 | 11929 | 634.3 | 636.0 | 47.1 | 243/311 |
| 6 | F5 | 65726 | 11885 | 11917 | 904.4 | 906.8 | 43.5 | 233/330 |
| 7 | F6 | 77987 | 11900 | 11932 | 878.0 | 880.4 | 42.1 | 205/312 |
| 8 | F7 | 90221 | 11867 | 11899 | 861.5 | 863.8 | 40.8 | 209/316 |
| 9 | F8 | 102469 | 11887 | 11919 | 847.4 | 849.7 | 43.9 | 218/286 |
| 10 | F9 | 114687 | 11859 | 11891 | 829.9 | 832.1 | 39.3 | 201/316 |

Separately, prior 105-03 attempt02 completed its larger trajectory and correctly answered primary A2 (`<ff fe>`, `invalid_utf8=true`). That result is not evidence that this attempt completed, nor proof of physical rank/promotion.

## External raw artifact hashes

Append-only source: `/srv/ai/paged-kv/results/forward/105-03b/attempt-01/`.

| Artifact | SHA-256 |
| --- | --- |
| `occupied-frontier.json` | `9ea9b0c3174f0608da1743290799dddcfa126ce80e7d76f888ff275acf324a04` |
| `incremental-state.json` | `bc88b705066a599f17df386d98f886423e9d41099a5d15548fef5bd433eceef1` |
| `candidate-identity.json` | `d5f71c6cde438521e6c5e0d05bdbb1a39e08393b626d3abfeca1058e34f84839` |
| `raw-000010.sse` | `34a760a9b9cd598a2d767928e9159ad548e22b4e577507940a31ff876f2b7611` |
| `driver-console.log` | `05cce07e502930b58deb2cc5a7acb7da020ac3cb483779ed11e34709dd8da2f3` |
| `lifecycle-cleanup.txt` | `dc48334711452ac5545365fcf264e5da218dbd3dc17b55a5e2915e5e389f6112` |

## Attempt02 — complete planned measurements; target short by 707

The default full occupancy checker and extra-facts checker both pass for this
attempt. All scheduled requests completed at C=249293 (707 below the requested
250000), with HTTP200 A2, 12/12 occupied-base canonical rows, and two
occupied-base extra fact requests. The driver retains `status=incomplete` and
`goal_status=goal_miss` for the short numeric frontier; the checker reports
`execution_status=complete` and validates the completed scope. The 707-token
gap is disclosed, not treated as a validation failure or exact-C gate.
Optional physical target-use/rank telemetry remains unknown; semantic success
does not establish physical cold residency or promotion.

Run source provenance is repo HEAD
`8f8faebacedff769e48a3ce80b7544d886a3a049` plus dirty
`src/llama-kv-pager.cpp` (not a later commit identity). Build provenance
records compiled-source delta `f493a4817f92590acbe510d0051516d9f8dfd90445ce67f42f47d35f940bce99`;
the execution-time source snapshot records `src/llama-kv-pager.cpp` SHA-256
`e171c1894a3740dd0872c38e79805a2ff16e093142461da00397afaa85da93b4`. The
report records source dirty fingerprint
`85f92af631fc2b9b620167fcd5b5b72d775bb6b73bdf23e44535462319d882ed`. The actual
server command and complete loaded DSO identities are in the hashed candidate
artifact. Model SHA-256 is
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`;
server `2d46af95cc6151096a026845d763947cb9d4041b857acb1515b962db418a4af4`,
libllama `d9c93b62edaff5322108d684c1dd9f95d71eeeffb953afec351855200f3247dd`,
server-impl `efa611a672ab73931f8d834cd6d7017fc4e702acc2307f9ccebb1c1384f6b1c0`,
CUDA `f8b66c3fa9b17cc77e08678dba8407b4c2998362278e6e796f9d5cf67de42867`.
Geometry remains L262144/H51200, 200 hot pages ×256, B1024/U256, Turbo4
target and GPU Turbo4 MTP.

Rates use `timings.prompt_ms`: fresh rate is `fresh_tokens / prompt_ms`,
processed input is `timings.prompt_n` (GPU-executed rows), and processed rate
is `prompt_n / prompt_ms`. The full-history API `usage.prompt_tokens` is not a
GPU-processed token count. The 21 fill requests had median fresh/processed
rates 787.27/789.39 tok/s (median decode 41.35 tok/s); A2's short new question
was 62 fresh /115 processed tokens at 21.61/40.08 tok/s.

| # | Stage | C after | Fresh input | GPU processed (`prompt_n`) | Fresh tok/s | Processed tok/s | Decode TG/s | MTP accepted/drafted |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | A1 | 4293 | 3894 | 4030 | 1224.41 | 1267.18 | 82.61 | 228/341 |
| 2 | B | 16586 | 11894 | 11991 | 1214.51 | 1224.41 | 86.99 | 244/308 |
| 3 | F2 | 28845 | 11860 | 11892 | 923.66 | 926.16 | 77.76 | 235/328 |
| 4 | F3 | 41146 | 11902 | 11934 | 742.95 | 744.94 | 75.67 | 239/318 |
| 5 | F4 | 53442 | 11897 | 11929 | 632.38 | 634.08 | 47.14 | 243/311 |
| 6 | F5 | 65726 | 11885 | 11917 | 901.12 | 903.54 | 43.77 | 233/331 |
| 7 | F6 | 78023 | 11900 | 11932 | 875.53 | 877.89 | 42.01 | 227/340 |
| 8 | F7 | 90289 | 11867 | 11899 | 855.96 | 858.27 | 42.39 | 231/335 |
| 9 | F8 | 102559 | 11890 | 11922 | 842.12 | 844.39 | 43.79 | 226/308 |
| 10 | F9 | 114725 | 11859 | 11891 | 827.17 | 829.41 | 41.35 | 177/260 |
| 11 | F10 | 126960 | 11899 | 11931 | 815.90 | 818.10 | 38.13 | 184/304 |
| 12 | F11 | 139151 | 11891 | 11923 | 801.08 | 803.24 | 45.16 | 186/228 |
| 13 | F12 | 151398 | 11904 | 11936 | 787.27 | 789.39 | 40.23 | 198/290 |
| 14 | F13 | 163684 | 11895 | 11927 | 776.95 | 779.04 | 38.93 | 223/336 |
| 15 | F14 | 175920 | 11903 | 11935 | 765.06 | 767.12 | 39.99 | 195/276 |
| 16 | F15 | 188182 | 11863 | 11895 | 752.69 | 754.72 | 37.47 | 225/346 |
| 17 | F16 | 200212 | 11798 | 11830 | 739.33 | 741.33 | 38.57 | 134/196 |
| 18 | F17 | 212511 | 11900 | 11932 | 730.85 | 732.81 | 36.20 | 221/353 |
| 19 | F18 | 224801 | 11891 | 11923 | 719.32 | 721.25 | 31.89 | 200/398 |
| 20 | F19 | 236943 | 11891 | 11923 | 708.60 | 710.51 | 33.68 | 134/234 |
| 21 | F20 | 249075 | 11878 | 11910 | 699.11 | 700.99 | 38.17 | 150/208 |
| 22 | A2 | 249293 | 62 | 115 | 21.61 | 40.08 | 33.86 | 84/144 |

### Occupied-base canonical matrix

Each row is the median of three measured requests after one 40-token warmup;
MTP acceptance is derived from request-scoped accepted/drafted pairs. The
generated content is coherent for each prompt (merge implementation, mmap vs
read explanation, directory-watch script); these canonical prompts have no
separate semantic scorer in the checker.

| Prompt | Measured decode median | MTP acceptance median | Pairs (accepted/drafted) | Semantic content review |
| --- | ---: | ---: | --- | --- |
| Merge two sorted lists | 43.32 tok/s | 88.28% | 113/128, 113/128, 113/128 | Correct merge with docstring in all three |
| mmap vs read | 31.17 tok/s | 48.91% (pooled 48.78%) | 90/184, 77/172, 72/134 | Correct distinction in all three |
| Directory watch script | 40.75 tok/s | 79.78% | 228/302, 217/272, 231/286 | Scripts watch for and print newly created files |

Primary A2 naturally stopped and correctly states invalid UTF-8 `ff fe` prints
`<ff fe>` and sets `invalid_utf8=true`. Its markers are recognized by the
default findings checker; the answer does not prove cold physical residency.
Both extra facts are `semantic_status=met`, natural EOS: `CONTRIBUTING.md`
requires a reproducible issue and a regression test failing before/passing
after (source SHA
`cf80093cca52e66dc1592175c3a2bcb1acf6aec1e30adaf212c5bedf0967da0e`, byte
span2021..2190); batched-bench shared N_KV is `PP + B * TG` (source SHA
`db9158e439e1d31e7b931d65fd50771093ef6667e14d617fc6d2fde6798bc12c`, byte
span137..342).

### Attempt02 external evidence hashes

Append-only source:
`/srv/ai/paged-kv/results/forward/105-03b/attempt-02/`.

| Artifact | SHA-256 |
| --- | --- |
| `occupied-frontier.json` | `d7811d038527482253424ce191a50397d5fc02db1fca3cc0bd6a3c19703c2e88` |
| `incremental-state.json` | `d1f7c0b1a901b6c2a9a73452df4212503d860eb714dbc1f8462d1c6351b5d893` |
| `candidate-identity.json` | `3d9166c3b1bd14f72275c2deaa4ee0bec73c61b1f4ad413a68ead97e8ece7290` |
| `build-provenance.txt` | `060cab5cadb997b219fd2bf3b0b1cad42eeffd7e5165e2df6337fead3f460397` |
| `run-source-hashes.txt` | `04d14b39e6b9140b1f98fb0e81a9d43bbe3ebf66cd92433e5fc981563f6185c4` |
| `extra-facts-findings.json` | `9fc6b9466159b8e1bdf11eb6dc2945f0305eabe25f8ac2ede364f2d06198a7a4` |
| `validated-occupancy-findings.json` | `c80aba3af7450ddc7222349008d9df69ee1e669a81ced5acc14929d5717ed6e5` |
| `validated-extra-facts-findings.json` | `76663fa45a7d63c9337cfd6e136232f4b81b8be2994d44474850dc5b1ad246dd` |
| A2 raw SSE `raw-000021.sse` | `9a5d4028cfde8ecd0cce64397af0aad7ffd06f86fbe1f6576504ad81c4931462` |
| Extra fact raw SSEs `extra-question-00/01.sse` | `acf5ccac31a6a2d5f431b46a3fa76231f958780bfda6a33c1939f3f8f33339e1` / `376ff1ae77417c700d0ad6258f00195d3b0af8bdefb916bf80fa1dd09766eb8c` |
