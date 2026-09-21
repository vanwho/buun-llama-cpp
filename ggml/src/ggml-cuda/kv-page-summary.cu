#include "kv-page-summary.cuh"

#include "fattn-common.cuh"
#include "ggml-impl.h"

#include <cfloat>
#include <cmath>

namespace {

__device__ __forceinline__ half summary_lower(float value) {
    return __float2half_rd(value);
}

__device__ __forceinline__ half summary_upper(float value) {
    return __float2half_ru(value);
}

__global__ void kv_page_summary_kernel(
        const char * k, size_t k_nb1, size_t k_nb2,
        const int64_t * metadata, size_t metadata_nb0, size_t metadata_nb1,
        const half * catalogue, size_t catalogue_nb0, size_t catalogue_nb1,
        size_t catalogue_nb2, size_t catalogue_nb3,
        half * output, size_t output_nb0, size_t output_nb1,
        size_t output_nb2, size_t output_nb3,
        int n_pages, int n_heads, int dim, int page_size, int k_rows, int k_streams) {
    const int page = blockIdx.x;
    const int head = blockIdx.y;
    if (page >= n_pages || head >= n_heads) return;

    const int64_t * page_data = (const int64_t *)((const char *) metadata +
            page * metadata_nb1);
    const int64_t physical_slot = page_data[4];
    const int64_t valid_rows = page_data[1];
    const int64_t stream = page_data[5];
    const bool update = page_data[6] != 0 && page_data[7] != 0 &&
        physical_slot >= 0 && valid_rows > 0 && valid_rows <= page_size &&
        stream >= 0 && stream < k_streams &&
        physical_slot <= (k_rows - valid_rows) / page_size;

    for (int coord = threadIdx.x; coord < dim; coord += blockDim.x) {
        const char * source = (const char *) catalogue + coord * catalogue_nb0 +
            head * catalogue_nb2 + page * catalogue_nb3;
        char * destination = (char *) output + coord * output_nb0 +
            head * output_nb2 + page * output_nb3;
        if (!update) {
            *(half *) destination = *(const half *) source;
            *(half *)(destination + output_nb1) =
                *(const half *)(source + catalogue_nb1);
            continue;
        }

        float minimum = FLT_MAX;
        float maximum = -FLT_MAX;
        bool invalid = false;
        for (int row = 0; row < valid_rows; ++row) {
            const char * row_data = k + stream * k_nb2 +
                (physical_slot * page_size + row) * k_nb1;
            const int element = head * dim + coord;
            const block_turbo4_0 * block = (const block_turbo4_0 *) row_data +
                element / QK_TURBO4;
            const int within = element % QK_TURBO4;
            const uint8_t index = (within & 1)
                ? (block->qs[within / 2] >> 4) : (block->qs[within / 2] & 0xF);
            const float value = d_turbo_centroids_4bit_fattn[index] *
                __half2float(block->norm);
            if (!isfinite(value)) {
                invalid = true;
            } else {
                minimum = fminf(minimum, value);
                maximum = fmaxf(maximum, value);
            }
        }
        if (invalid || !isfinite(minimum) || !isfinite(maximum)) {
            // Poison invalid metadata instead of manufacturing an apparently
            // eligible zero interval. The selector is finite/ordering gated.
            *(half *) destination = __float2half(NAN);
            *(half *)(destination + output_nb1) = __float2half(NAN);
        } else {
            *(half *) destination = summary_lower(minimum);
            *(half *)(destination + output_nb1) = summary_upper(maximum);
        }
    }
}

} // namespace

void ggml_cuda_op_kv_page_summary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k = dst->src[0];
    const ggml_tensor * metadata = dst->src[1];
    const ggml_tensor * catalogue = dst->src[2];
    const int page_size = ggml_get_op_params_i32(dst, 0);
    const int n_pages = int(dst->ne[3]);
    const int n_heads = int(dst->ne[2]);
    const int dim = int(dst->ne[0]);
    kv_page_summary_kernel<<<dim3(n_pages, n_heads, 1), 256, 0, ctx.stream()>>>(
            (const char *) k->data, k->nb[1], k->nb[2],
            (const int64_t *) metadata->data, metadata->nb[0], metadata->nb[1],
            (const half *) catalogue->data, catalogue->nb[0], catalogue->nb[1],
            catalogue->nb[2], catalogue->nb[3], (half *) dst->data,
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], n_pages, n_heads,
            dim, page_size, int(k->ne[1]), int(k->ne[2]));
    CUDA_CHECK(cudaGetLastError());
}
