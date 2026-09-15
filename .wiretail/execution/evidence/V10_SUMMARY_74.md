# Phase-74 benchmark summary

Revision: `hotpath-v10-20260914`. This summary aggregates only the phase-74
receipts and their raw manifests. Requested, observed, and immutable identities
remain separate.

## Result

Phase 74 measured the organic answer-quality/promotion path, cached-append
coordinates, the matched q0/q1/q2 benchmark, native-MTP request denominators,
and a bounded L262144 occupancy pilot. Full-L allocation passed, but occupied
C262144 did not.

| Requested row | Status | Finding |
| --- | --- | --- |
| Answer quality | measured / failed row | A and A-again correct; B returned `COBALT-ELM-117` instead of `COBALT-MAPLE-731` |
| Physical promotion | measured | Organic A-again promoted cold logical page 0; H2D useful/aligned bytes 4,325,376 |
| Target-graph use | measured | Published mapping was consumed by the completed target graph |
| Cached append 64 | measured | Non-MTP cache reuse at `cache_n=6143`; native-MTP row reused `cache_n=6130` |
| Cached append 256 | measured row / no cached-rate claim | Both rows completed with `cache_n=0`; they did not reuse the prefix cache |
| Cold prefill | measured | Three trials per prompt at observed C39–42 selected/native and C27–30 controls |
| Committed decode | measured | Three trials per prompt, 128 output tokens |
| Native MTP | measured | Selected benchmark: 2,229 drafted / 15 accepted across nine request-scoped denominators; append MTP: 6 / 3 across three rows |
| L262144 allocation | measured | Target 138,412,032 bytes; native-MTP 276,955,136 bytes; admission passed at C0 |
| Occupied C262144 | failed | Bounded pilot stopped at durable C12,408 / live C12,535 after the operator wall-budget boundary |

The verified controlled coordinate is L8192/C6144/H4096/A2048/B128/U64 with
256-token pages and 4,325,376-byte pages. The matched benchmark observed
H8192/A2048/B128/U64, not the requested C6144 rate coordinate. The occupancy
pilot observed L262144/H8192/A4096/B128/U64. These are separate claims.

## Placement and rates

Selected/native used CUDA Turbo4 target K/V, GPU Turbo4 draft K/V, and selective
paging. The CPU-main-KV control used CPU target KV with GPU weights/draft and
feature-off MTP. The all-GPU control used CUDA target KV, GPU draft, feature-off
MTP, and no pager.

Matched median cold-prefill rates (q0/q1/q2 tok/s) were selected/native
897.551/877.469/883.939, CPU-main-KV/GPU-draft 304.897/279.096/291.965, and
all-GPU/GPU-draft 686.845/650.540/673.449. Committed-decode medians were
33.349/33.574/33.576, 17.893/17.868/17.886, and 49.378/49.286/49.341
respectively. These are short-prompt rates, not C6144 rates.

## Identity and boundaries

The resolved model was `Qwen3.8-27B-UD-IQ4_XS.gguf`, SHA-256
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`, and the
loaded candidate executable SHA-256 was
`76404a3ace49a8f77a3567a8b26e98ba57c4e074827f7db7aab55d875eea3cae`. The
source commits and raw roots are listed in `V10_SUMMARY_74.json`.

Answer correctness is reported separately from physical promotion: a correct
A-again answer does not prove promotion by itself, and the B mismatch does not
erase the separately observed target-graph use. Full-L allocation is not
occupied-C262144 evidence. No historical rate, allocation, or mapping was
used to fill a missing phase-74 row.
