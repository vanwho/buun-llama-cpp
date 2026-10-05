#include "kv-page-select.cuh"
#include "fattn-common.cuh"
#include "ggml-impl.h"

#include <cmath>

namespace {

__global__ void kv_page_rerank_kernel(
        const float * probes, size_t q_nb0, size_t q_nb1, size_t q_nb2,
        const char * resident_keys, size_t rk_nb1, size_t rk_nb2, int resident_rows, int resident_streams,
        const char * staged_keys, size_t sk_nb1, size_t sk_nb2, int staged_rows, int staged_streams,
        const int64_t * descriptors, size_t desc_nb1, const int64_t * identity,
        const int64_t * validity, float * state,
        size_t state_nb0, size_t state_nb1, size_t state_nb2, size_t state_nb3,
        float * output, size_t out_nb0, size_t out_nb1, size_t out_nb2,
        int pages, int heads, int n_probes, int dim, int kv_heads, float scale, float softcap) {
    const int page = blockIdx.x;
    const int head = blockIdx.y;
    const int probe = blockIdx.z;
    const int64_t * desc = (const int64_t *) ((const char *) descriptors + page * desc_nb1);
    const int64_t logical = desc[0], slot = desc[1], rows = desc[2], stream = desc[3];
    const int64_t page_size = desc[4], first_position = desc[8], key_set = desc[9];
    const char * keys = key_set == 0 ? resident_keys : staged_keys;
    const size_t k_nb1 = key_set == 0 ? rk_nb1 : sk_nb1;
    const size_t k_nb2 = key_set == 0 ? rk_nb2 : sk_nb2;
    const int key_rows = key_set == 0 ? resident_rows : staged_rows;
    const int streams = key_set == 0 ? resident_streams : staged_streams;
    const bool eligible = desc[7] != 0 && logical >= 0 && slot >= 0 && rows > 0 &&
        (key_set == 0 || key_set == 1) && page_size > 0 && rows <= page_size && stream >= 0 && stream < streams &&
        slot <= (key_rows - rows) / page_size && desc[5] == identity[0] && desc[6] == identity[1] &&
        validity[5 + probe] != 0;
    float * out = (float *) ((char *) output + page * out_nb0 + head * out_nb1 + probe * out_nb2);
    if (threadIdx.x == 0) *out = -INFINITY;
    if (!eligible) return;

    __shared__ float warp_sum[8];
    __shared__ float lse_m, lse_s;
    const int kv_head = head / (heads / kv_heads);
    const float old_m = *(const float *) ((const char *) state + page * state_nb1 + head * state_nb2 + probe * state_nb3);
    const float old_s = *(const float *) ((const char *) state + state_nb0 + page * state_nb1 + head * state_nb2 + probe * state_nb3);
    if (threadIdx.x == 0) { lse_m = -INFINITY; lse_s = 0.0f; }
    __syncthreads();

    for (int64_t row = 0; row < rows; ++row) {
        if (first_position + row > validity[1 + probe]) continue;
        const char * row_data = keys + stream * k_nb2 + (slot * page_size + row) * k_nb1;
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        float dot = 0.0f;
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            const int element = kv_head * dim + d;
            const block_turbo4_0 * block = (const block_turbo4_0 *) row_data + element / QK_TURBO4;
            const int within = element % QK_TURBO4;
            const uint8_t index = (within & 1) ? (block->qs[within / 2] >> 4) : (block->qs[within / 2] & 0xF);
            const float key = d_turbo_centroids_4bit_fattn[index] * __half2float(block->norm);
            const float q = *(const float *) ((const char *) probes + d * q_nb0 + head * q_nb1 + probe * q_nb2);
            dot += q * key;
        }
        for (int delta = 16; delta > 0; delta >>= 1) dot += __shfl_down_sync(0xffffffff, dot, delta);
        if (lane == 0) warp_sum[warp] = dot;
        __syncthreads();
        if (threadIdx.x == 0) {
            float score = 0.0f;
            for (int w = 0; w < blockDim.x / 32; ++w) score += warp_sum[w];
            score *= scale;
            if (softcap > 0.0f) score = softcap * tanhf(score);
            if (isfinite(score)) {
                if (score > lse_m) { lse_s = lse_s * expf(lse_m - score) + 1.0f; lse_m = score; }
                else lse_s += expf(score - lse_m);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && (old_s > 0.0f || lse_s > 0.0f)) {
        const float m = fmaxf(old_m, lse_m);
        const float s = (old_s > 0.0f ? old_s * expf(old_m - m) : 0.0f) +
            (lse_s > 0.0f ? lse_s * expf(lse_m - m) : 0.0f);
        state[page * state_nb1 / sizeof(float) + head * state_nb2 / sizeof(float) +
                probe * state_nb3 / sizeof(float)] = m;
        state[state_nb0 / sizeof(float) + page * state_nb1 / sizeof(float) +
                head * state_nb2 / sizeof(float) + probe * state_nb3 / sizeof(float)] = s;
        *out = m + logf(s);
    } else if (threadIdx.x == 0 && old_s > 0.0f) {
        *out = old_m + logf(old_s);
    }
}

} // namespace

void ggml_cuda_op_kv_page_rerank(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * probes = dst->src[0], * resident_keys = dst->src[1], * staged_keys = dst->src[2];
    const ggml_tensor * desc = dst->src[3], * identity = dst->src[4];
    const ggml_tensor * validity = dst->src[5], * state = dst->src[6];
    const int pages = int(dst->ne[0]), heads = int(dst->ne[1]), n_probes = int(dst->ne[2]);
    const int dim = int(probes->ne[0]), kv_heads = int(resident_keys->ne[0] / dim);
    kv_page_rerank_kernel<<<dim3(pages, heads, n_probes), 256, 0, ctx.stream()>>>(
        (const float *) probes->data, probes->nb[0], probes->nb[1], probes->nb[2],
        (const char *) resident_keys->data, resident_keys->nb[1], resident_keys->nb[2],
        int(resident_keys->ne[1]), int(resident_keys->ne[2]),
        (const char *) staged_keys->data, staged_keys->nb[1], staged_keys->nb[2],
        int(staged_keys->ne[1]), int(staged_keys->ne[2]),
        (const int64_t *) desc->data, desc->nb[1],
        (const int64_t *) identity->data, (const int64_t *) validity->data, (float *) state->data,
        state->nb[0], state->nb[1], state->nb[2], state->nb[3], (float *) dst->data,
        dst->nb[0], dst->nb[1], dst->nb[2], pages, heads, n_probes, dim, kv_heads,
        ggml_get_op_params_f32(dst, 0), ggml_get_op_params_f32(dst, 1));
    CUDA_CHECK(cudaGetLastError());
}
