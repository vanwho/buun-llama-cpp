# Forward measured summary — refreshed for 105-04

Overall goal status is **not assessed**. The completed evidence includes a candidate-matched 105-03b occupied-base canonical matrix and correct semantic recalls, while the matched CPU-KV speed comparison is still missing. That gap does not establish a production defect. The 105-03b numeric trajectory ended 707 tokens below its requested C=250000 frontier; execution completed and its checker passed. Exact C=L remains unproven.

## Current 105-03b candidate and geometry

Attempt02 used source HEAD `8f8faebacedff769e48a3ce80b7544d886a3a049` plus the captured dirty pager diff (dirty fingerprint `85f92af631fc2b9b620167fcd5b5b72d775bb6b73bdf23e44535462319d882ed`; compiled delta `f493a4817f92590acbe510d0051516d9f8dfd90445ce67f42f47d35f940bce99`). The raw build identity stays attached to that HEAD-plus-diff candidate. Executable SHA-256 is `2d46af95cc6151096a026845d763947cb9d4041b857acb1515b962db418a4af4`, libllama `d9c93b62edaff5322108d684c1dd9f95d71eeeffb953afec351855200f3247dd`, server implementation `efa611a672ab73931f8d834cd6d7017fc4e702acc2307f9ccebb1c1384f6b1c0`, CUDA `f8b66c3fa9b17cc77e08678dba8407b4c2998362278e6e796f9d5cf67de42867`, and model `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`. Full loaded DSO identity and actual server argv are in the hashed candidate artifact.

| L | Occupied C | H | R/G | B/U | KV and MTP | Result |
|---:|---:|---:|---:|---:|---|---|
| 262144 | 249293 | 51200 | 50944/256 | 1024/256 | Turbo4 target KV; full-L GPU Turbo4 MTP | Execution complete; 707 short of requested C=250000; C=L unproven |

The old 105-03 trajectory and 105-03a paired layout remain separately identified findings. The older 103-04 canonical, geometry, replay, and memory sections retained in the JSON are historical candidate-bound results; no candidate identity is blended across them.

## 105-03b occupied-context curve

The 21 bulk rows below use fresh tokens divided by `timings.prompt_ms` for useful ingestion, and `timings.prompt_n` divided by the same duration for GPU-executed input. `usage.prompt_tokens` contains cached history and is not executed input. Each MTP value is request-local accepted/drafted.

| Stage | C after | Fresh | Processed (`prompt_n`) | Fresh tok/s | Processed tok/s | Decode tok/s | MTP |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 4293 | 3894 | 4030 | 1224.41 | 1267.18 | 82.61 | 228/341 |
| B | 16586 | 11894 | 11991 | 1214.51 | 1224.41 | 86.99 | 244/308 |
| F2 | 28845 | 11860 | 11892 | 923.66 | 926.16 | 77.76 | 235/328 |
| F3 | 41146 | 11902 | 11934 | 742.95 | 744.94 | 75.67 | 239/318 |
| F4 | 53442 | 11897 | 11929 | 632.38 | 634.08 | 47.14 | 243/311 |
| F5 | 65726 | 11885 | 11917 | 901.12 | 903.54 | 43.77 | 233/331 |
| F6 | 78023 | 11900 | 11932 | 875.53 | 877.89 | 42.01 | 227/340 |
| F7 | 90289 | 11867 | 11899 | 855.96 | 858.27 | 42.39 | 231/335 |
| F8 | 102559 | 11890 | 11922 | 842.12 | 844.39 | 43.79 | 226/308 |
| F9 | 114725 | 11859 | 11891 | 827.17 | 829.41 | 41.35 | 177/260 |
| F10 | 126960 | 11899 | 11931 | 815.90 | 818.10 | 38.13 | 184/304 |
| F11 | 139151 | 11891 | 11923 | 801.08 | 803.24 | 45.16 | 186/228 |
| F12 | 151398 | 11904 | 11936 | 787.27 | 789.39 | 40.23 | 198/290 |
| F13 | 163684 | 11895 | 11927 | 776.95 | 779.04 | 38.93 | 223/336 |
| F14 | 175920 | 11903 | 11935 | 765.06 | 767.12 | 39.99 | 195/276 |
| F15 | 188182 | 11863 | 11895 | 752.69 | 754.72 | 37.47 | 225/346 |
| F16 | 200212 | 11798 | 11830 | 739.33 | 741.33 | 38.57 | 134/196 |
| F17 | 212511 | 11900 | 11932 | 730.85 | 732.81 | 36.20 | 221/353 |
| F18 | 224801 | 11891 | 11923 | 719.32 | 721.25 | 31.89 | 200/398 |
| F19 | 236943 | 11891 | 11923 | 708.60 | 710.51 | 33.68 | 134/234 |
| F20 | 249075 | 11878 | 11910 | 699.11 | 700.99 | 38.17 | 150/208 |

Across the 21 fill rows, median fresh/processed rates were 787.27/789.39 tok/s and median decode was 41.35 tok/s. The short A2 tail added 62 fresh tokens and executed 115 prompt tokens in 2869.059 ms (21.61/40.08 tok/s); it correctly answered `<ff fe>` with `invalid_utf8=true` and is not a bulk-prefill point. The final 699.11 tok/s fresh rate differs from 700.99 tok/s processed because the denominators are 11878 fresh versus 11910 executed tokens.

## Canonical generation and retrieval

The occupied prefix was reused for all 12 canonical rows: one 40-token warmup and three measured requests for each prompt, with reasoning off. All responses were coherent; request-local medians and accepted/drafted pairs are:

| Prompt | Median decode tok/s | Median MTP acceptance | Accepted/drafted per measured row |
|---|---:|---:|---|
| Merge two sorted lists | 43.32 | 88.28% | 113/128, 113/128, 113/128 |
| mmap vs read | 31.17 | 48.91% (48.78% pooled) | 90/184, 77/172, 72/134 |
| Directory watcher | 40.75 | 79.78% | 228/302, 217/272, 231/286 |

The primary A2 fact and both additional earlier-source facts were correct with natural EOS. The extra facts were the `CONTRIBUTING.md` bug-fix PR requirements and `PP + B * TG` in the batched-bench README. These semantic answers do not certify physical cold-page residency, rank, transfer, or target use; that optional identity-bound witness remains unknown.

105-02a also established semantically successful selected and dense exact-fact responses. The selected response recorded the page cold/host-backed before recall, rank/admission/mapping observations, and a correct answer. Its identity-bound physical-transfer/target-use witness remains unknown; this is not a retrieval failure or completion gate. The 105-03 primary recall and 105-02a recall are separate candidate-bound evidence.

## Route, replay, and memory

The current live route was `probe-rerank`. Occupied-base replay was coherent for canonical and extra-fact requests. Selector, replay, and promotion costs were not measured separately; a matched route comparison and candidate-matched CPU-KV speed control are missing. The missing comparison does not imply a known implementation defect.

Current attempt02 memory counters (bytes) are candidate-bound: target pool capacity/allocation 865075000; target resident 856424000 and valid 855563000; full-L GPU MTP 276955000 bytes / 262144 rows; graph 198435000; scratch high-water 54067200; host pageable 4212050000, host valid 855563000, pinned 0; packed storage 54067200 and workspace/dequant reported 0; device used/free 16041500000/678691000. Reserved headroom 201327000 and catalogue 100876000 are not observed allocations. Do not add overlapping categories; unreported categories remain unknown.

## Goal rows

All 15 rows are retained. This refresh records current evidence without re-auditing every historical phase.

| Goal | Current compact finding |
|---:|---|
| 1 | Partial: L-sized allocation and C=249293 measured; exact C=L unproven and requested C short by 707. |
| 2 | Met: current L=262144 with H=51200; historical 128K evidence stays separate. |
| 3 | Semantic success: 105-02a selected/dense and 105-03b facts correct; physical witness unknown. |
| 4 | Historical boundary parity and replay findings retained; no new bitwise claim. |
| 5 | Historical frozen-history findings retained within their tested scopes. |
| 6 | Historical zero-H2D observation remains interval-scoped; current physical transfer witness unknown. |
| 7 | Ring implementation/tests are historical; current numeric rollback counter not measured. |
| 8 | Not reassessed in current scale receipts. |
| 9 | Current full-L GPU Turbo4 MTP canonical medians are 88.28%, 48.91%, 79.78%. |
| 10 | Current host pageable/valid/pinned counters are recorded in the candidate-bound memory snapshot. |
| 11 | Unknown: matched current candidate CPU-KV comparison is missing. |
| 12 | 105-03a paired small-layout B measured 1236.61 fresh tok/s split versus 703.20 legacy; 105-03b large curve is separate. |
| 13 | Historical replay/PCIe findings remain scoped; current scale-wide relation unknown. |
| 14 | Partial: large-context curve, canonical, semantic, and memory findings exist; C=L is unproven and CPU-KV comparison missing. |
| 15 | Not reassessed for the current candidate. |

The full 105-03b findings and validation artifacts are hashed in `FORWARD_FINAL_SUMMARY.json` and `.wiretail/execution/evidence/V10_105-03b.json`. The 105-03 large trajectory/primary fact and 105-03a paired layout remain in their own compact handoffs; attempt01's 105-03b reservation failure remains a separate runtime finding with unknown cause.
