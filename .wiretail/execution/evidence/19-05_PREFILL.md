# 19-05 direct tiled selective prefill evidence

Classification: implementation and deterministic CUDA/host verification passed
locally. The model-backed two-length prefill checkpoint is deferred because the
shared RTX 4080 runtime workload was retained.

## Implementation

The direct paged Turbo4 kernel is now a bounded query-tile kernel. One CTA per
query head rotates one, two, or three Q vectors, traverses the compressed page
descriptor list once, decodes each valid physical K/V row once, and updates
each query's online softmax state with its own native causal position. Page
tails, physical-slot remapping, logical gaps, page-mass telemetry, and
`[m,l,o]` partial state retain explicit query/head strides.

Selective Qwen35/Qwen35MOE Turbo4 prefill is admitted to the direct route.
`prefill_ubatch_size` caps that supported path at the three-query tile while
retaining the physical-page bound; unsupported models and exact mode retain
their existing physical-window/reference behavior. Direct graph storage is
descriptor/position/mask/query metadata plus resident scratch, not a dense
full-context F16 target or unbounded dequantized gather. Selective prefill is
an explicitly bounded approximation; exact full-history prefill remains the
19-06 task.

## Verification

| Command | Result |
|---|---|
| `cmake --build build-cuda --target test-cuda-fattn-paged-turbo4 test-kv-attention-execution llama -j2` | pass |
| `build-cuda/bin/test-cuda-fattn-paged-turbo4` | pass; 1/2/3-query tiles, q4 refusal, tails, causal masks, page mass, partial state |
| `build-cuda/bin/test-kv-attention-execution` | pass; selective prefill direct counters and fallback coverage |
| focused 7-test CTest set | 7/7 pass |
| `build-cuda/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 -j 2` | 2948/2948 pass |
| `git diff --check` | pass |

The direct fixture measured 19.154 ms for its deterministic 529-row, four-Q
head case. The host fixture recorded four direct selective-prefill prepares;
its bounded scratch high-water example was 468 rows. Machine-readable hashes
and the exact receipt are in `19-05_PREFILL.json`.

## Deferred verification

No model-backed Qwen prefill/MTP invocation was started or disturbed. After GPU
isolation, run the runtime-supported Qwen command at two occupied lengths,
record progress timestamps/ETAs, CUDA allocation and event data, route counts,
and compare all-selected outputs with the 18-05 oracle. Do not infer exact
full-history equivalence from this selective-prefill receipt.

## Exact next action

Task 19-06 should implement the live exact GPU page-wave reference and its
full-history prefill binding; preserve the three-query selective direct tile
and its explicit fallback boundary.
