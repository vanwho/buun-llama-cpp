#pragma once

#include "ggml.h"

// Keep legacy dynamic/indexed suffix captures unchanged for independent
// oracles. Production spread probes cover the user question, not only its
// last eight tokens (often punctuation or generic answer-format instructions).
enum { GGML_KV_QUERY_PROBES_SPREAD = 2 };

#ifdef __CUDACC__
__host__ __device__
#endif
static inline int64_t ggml_kv_query_probe_target(
        int64_t begin, int64_t end, int slot, int mode) {
    if (begin < 0 || end <= begin || slot < 0 || slot >= 4) return -1;
    if (mode != GGML_KV_QUERY_PROBES_SPREAD) {
        const int64_t target = end - (INT64_C(1) << slot);
        return target >= begin ? target : -1;
    }
    const int64_t span = end - begin - 1;
    const int64_t targets[4] = {
        end - 1, begin + span / 4, begin + span / 2, end - 1 - span / 4,
    };
    // Short queries need fewer probes, not duplicated channels/weights.
    for (int earlier = 0; earlier < slot; ++earlier) {
        if (targets[earlier] == targets[slot]) return -1;
    }
    return targets[slot];
}

#ifdef __cplusplus
extern "C" {
#endif
GGML_API struct ggml_tensor * ggml_kv_query_probes_spread(
        struct ggml_context * ctx, struct ggml_tensor * q,
        struct ggml_tensor * probes, struct ggml_tensor * validity,
        int64_t generation, int64_t query_start, int64_t query_end,
        const int32_t rows[4]);
#ifdef __cplusplus
}
#endif
