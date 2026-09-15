# Phase-78 benchmark summary

Revision: `hotpath-v10-20260914`. This summary aggregates only the phase-78
receipts and raw manifests. Requested, observed, and immutable identities are
kept separate.

## Result

Phase 78 measured the verified C≈6144 selected-native coordinate, the matched
three-placement benchmark, cached-prefix reuse, and a full-L allocation with
bounded partial occupancy. Full occupied C262144 was not established. No
phase-78 answer-quality or physical-promotion campaign ran.

| Requested row | Status | Finding |
| --- | --- | --- |
| Answer quality | not_run | No phase-78 quality campaign was executed |
| Controlled physical promotion | not_run | No controlled model-query promotion scenario was executed |
| Organic physical promotion | not_run | No organic cold-to-hot promotion scenario was executed |
| Target-graph use | not_run | No phase-78 promotion graph evidence exists |
| Cached append 64 | measured | C6207, cache_n6143, 64 new tokens; 75.334 tok/s in 78-02 |
| Cached append 256 | measured | C6399, cache_n6146, 253 tokenizer-realized new tokens; 90.523 tok/s in 78-02 |
| Cold prefill | measured | Verified coordinate plus 27-row matched matrix |
| Committed decode | measured | Verified coordinate stopped at EOS after 3 tokens plus matched matrix |
| Native MTP denominator | measured | Request-scoped deltas retained, including zero-work and 50% decode row |
| Full-L allocation | measured | L262144 admitted with 138412032 target and 276955136 draft bytes |
| Occupied C262144 | failed | Wall-bounded frontier stopped at durable C33952/live C33963 |

## Identity, geometry, placement, and bytes

The requested coordinate was L8192/C6144/H4096/A2048/B128/U64 with 256-token
pages. The verified coordinate observed L8192/H8192/A4096/B128/U64 and
C6143, C6207, and C6399 frontiers. The matched matrix observed C30, C27, and
C28 with cache_n26, 23, and 24; these are not relabeled as C6144.

The resolved model was `Qwen3.8-27B-UD-IQ4_XS.gguf`, SHA-256
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
Selected-native used CUDA Turbo4/Turbo4 target KV and GPU Turbo4/Turbo4 native
draft KV. The CPU-main-KV control used CPU target KV with GPU draft; the
all-GPU control used CUDA target KV with GPU draft. The full-L allocation
measured target 138412032 bytes, draft 276955136 bytes, packed storage 0,
packed workspace 138412032, charged 15965452416, reserved 2068443264, and
headroom 201326592 bytes. Each 256-token page occupied 4325376 bytes.

## Rates, cache reuse, and MTP

The verified coordinate measured cold prefill 181.244 tok/s over 6143 tokens,
cached append-64 75.334 tok/s, cached append-256 90.523 tok/s, and committed
decode 30.450 tok/s over the 3 tokens emitted before EOS. The matched matrix
used three measured trials for each of three prompts in each placement. Its
selected-native prefill medians were 101.965, 108.716, and 104.499 tok/s, and
decode medians were 32.784, 33.046, and 32.486 tok/s. CPU-main-KV/GPU-draft
prefill/decode medians were 36.222/10.060, 36.719/10.063, and 36.063/10.006;
all-GPU/GPU-draft medians were 103.864/36.776, 110.229/36.782, and
109.999/36.777 tok/s.

Native MTP used request-scoped Prometheus deltas. The verified coordinate had
0/0 draft/accepted for cold and both append rows, and 2 drafted/1 accepted
(50%) for committed decode. The matched selected-native rows totaled 5337
drafted and 9 accepted tokens; feature-off controls totaled 7155 drafted and
0 accepted and are not native-MTP capability evidence.

## Occupancy and boundaries

The occupancy pilot completed seven successful turns and reached durable
C33952/live C33963 under L262144/H8192/A4096/B128/U64. The operator wall
budget stopped the run there, so occupied C262144 remains a failed finding;
the measured allocation is not relabeled as occupancy. No answer correctness,
controlled promotion, organic promotion, or target-graph-use claim is inferred
from these performance and allocation runs. Raw roots and phase receipts are
listed in `V10_SUMMARY_78.json`.
