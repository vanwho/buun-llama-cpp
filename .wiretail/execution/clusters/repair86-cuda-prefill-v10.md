# repair86 - long-context CUDA prefill repair

Revision: `hotpath-v10-20260914`. Amendment: `repair86-20260921`.

This cluster addresses only the first failing long-context invariant from the
repair85 review: `ggml_cuda_turbo_prefill_attend` returns CUDA invalid argument
before the first committed token at L32768/H16384/B128/U64. Preserve Turbo4
transformed-domain addressing, full-L native GPU draft capacity, and the
existing page-table/direct consumer. A passing allocation probe is not a
request or occupancy proof.
