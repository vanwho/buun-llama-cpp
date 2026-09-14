#include "kv-page-select.cuh"

#include "ggml-impl.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>

#ifdef GGML_CUDA_USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif

namespace {

__device__ bool page_select_eligible(
        const int64_t * metadata, size_t metadata_nb0, size_t metadata_nb1,
        const int32_t * membership, size_t membership_nb0,
        const int64_t * query, size_t query_nb0,
        int page, int expected_membership, int page_size, int metadata_fields) {
    const int64_t position_begin = *(const int64_t *)((const char *) metadata + 0 * metadata_nb0 + page * metadata_nb1);
    const int64_t valid_length = *(const int64_t *)((const char *) metadata + 1 * metadata_nb0 + page * metadata_nb1);
    const int64_t sequence_generation = *(const int64_t *)((const char *) metadata + 2 * metadata_nb0 + page * metadata_nb1);
    const int64_t page_generation = *(const int64_t *)((const char *) metadata + 3 * metadata_nb0 + page * metadata_nb1);
    const int64_t query_position = *(const int64_t *)((const char *) query + 0 * query_nb0);
    const int64_t query_sequence_generation = *(const int64_t *)((const char *) query + 1 * query_nb0);
    const int64_t snapshot_generation = *(const int64_t *)((const char *) query + 2 * query_nb0);
    const int64_t refresh_enabled = *(const int64_t *)((const char *) query + 3 * query_nb0);
    const int membership_value = *(const int *)((const char *) membership + page * membership_nb0);
    const int64_t summary_ready = metadata_fields >= 7
        ? *(const int64_t *)((const char *) metadata + 6 * metadata_nb0 + page * metadata_nb1) : 1;
    return refresh_enabled && membership_value == expected_membership && valid_length > 0 &&
        valid_length <= page_size &&
        summary_ready != 0 &&
        sequence_generation == query_sequence_generation && page_generation > 0 &&
        page_generation <= snapshot_generation && position_begin >= 0 &&
        position_begin < query_position && valid_length <= query_position - position_begin;
}

__global__ void page_select_scores(
        const float * q, size_t q_nb0, size_t q_nb1, size_t q_nb2,
        const half * bounds, size_t bounds_nb0, size_t bounds_nb1,
        size_t bounds_nb2, size_t bounds_nb3,
        const int64_t * metadata, size_t metadata_nb0, size_t metadata_nb1,
        const int32_t * membership, size_t membership_nb0,
        const int64_t * query, size_t query_nb0, int metadata_fields, int page_size,
        float * scores, int n_pages, int d, int n_q_heads, int n_kv_heads,
        int query_row) {
    const int page = blockIdx.x;
    if (page >= n_pages) return;

    __shared__ float reduction[256];
    __shared__ int invalid_reduction[256];
    if (!page_select_eligible(metadata, metadata_nb0, metadata_nb1,
            membership, membership_nb0, query, query_nb0, page, 1, page_size,
            metadata_fields) &&
        !page_select_eligible(metadata, metadata_nb0, metadata_nb1,
            membership, membership_nb0, query, query_nb0, page, 0, page_size,
            metadata_fields)) {
        if (threadIdx.x == 0) scores[page] = -INFINITY;
        return;
    }
    float best = -INFINITY;
    const int group = n_q_heads / n_kv_heads;
    for (int kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
        for (int q_head = kv_head * group; q_head < (kv_head + 1) * group; ++q_head) {
            float partial = 0.0f;
            int invalid = 0;
            const char * q_row = (const char *) q + q_head * q_nb1 + query_row * q_nb2;
            for (int coord = threadIdx.x; coord < d; coord += blockDim.x) {
                const float qi = *(const float *)(q_row + coord * q_nb0);
                const char * b = (const char *) bounds + coord * bounds_nb0 +
                    kv_head * bounds_nb2 + page * bounds_nb3;
                const float lo = __half2float(*(const half *) b);
                const float hi = __half2float(*(const half *) (b + bounds_nb1));
                if (!isfinite(qi) || !isfinite(lo) || !isfinite(hi) || lo > hi) {
                    invalid = 1;
                } else {
                    partial += qi >= 0.0f ? qi * hi : qi * lo;
                }
            }
            reduction[threadIdx.x] = partial;
            invalid_reduction[threadIdx.x] = invalid;
            __syncthreads();
            for (int width = blockDim.x / 2; width > 0; width >>= 1) {
                if (threadIdx.x < width) {
                    invalid_reduction[threadIdx.x] |= invalid_reduction[threadIdx.x + width];
                }
                __syncthreads();
            }
            for (int width = blockDim.x / 2; width > 0; width >>= 1) {
                if (threadIdx.x < width) reduction[threadIdx.x] += reduction[threadIdx.x + width];
                __syncthreads();
            }
            if (threadIdx.x == 0 && invalid_reduction[0] == 0 && isfinite(reduction[0])) {
                best = max(best, reduction[0]);
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        scores[page] = best;
    }
}

__device__ uint32_t page_select_descending_score_key(float score) {
    const uint32_t bits = __float_as_uint(score);
    // Map finite IEEE-754 values to monotonically increasing unsigned values,
    // then invert the score portion for descending order. The page ID in the
    // low word makes ties stable without relying on CUB's unspecified tie
    // behavior.
    const uint32_t ordered = (bits & UINT32_C(0x80000000)) != 0
        ? ~bits : bits ^ UINT32_C(0x80000000);
    return UINT32_MAX - ordered;
}

__global__ void page_select_build_keys(
        const float * scores,
        const int64_t * metadata, size_t metadata_nb0, size_t metadata_nb1,
        const int32_t * membership, size_t membership_nb0,
        const int64_t * query, size_t query_nb0, int metadata_fields,
        uint64_t * resident_keys, uint64_t * cold_keys,
        int32_t * resident_ids, int32_t * cold_ids,
        int n_pages, int page_size) {
    const int page = blockIdx.x * blockDim.x + threadIdx.x;
    if (page >= n_pages) return;
    const uint64_t invalid = UINT64_MAX;
    const float score = scores[page];
    const uint64_t key = isfinite(score)
        ? (uint64_t(page_select_descending_score_key(score)) << 32) |
            uint32_t(page)
        : invalid;
    resident_keys[page] = page_select_eligible(metadata, metadata_nb0, metadata_nb1,
            membership, membership_nb0, query, query_nb0, page, 1, page_size,
            metadata_fields) ? key : invalid;
    cold_keys[page] = page_select_eligible(metadata, metadata_nb0, metadata_nb1,
            membership, membership_nb0, query, query_nb0, page, 0, page_size,
            metadata_fields) ? key : invalid;
    resident_ids[page] = page;
    cold_ids[page] = page;
}

__global__ void page_select_clear_output(int32_t * output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = -1;
}

__global__ void page_select_emit(
        const uint64_t * keys, const int32_t * ids,
        int32_t * output, int begin, int count, int n_pages) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[begin + index] = index < n_pages && keys[index] != UINT64_MAX
        ? ids[index] : -1;
}

__global__ void page_select_rank(
        const float * scores,
        const int64_t * metadata, size_t metadata_nb0, size_t metadata_nb1,
        const int32_t * membership, size_t membership_nb0,
        const int64_t * query, size_t query_nb0,
        int32_t * output, int n_pages, int begin, int count,
        int expected_membership, int page_size, int metadata_fields) {
    __shared__ float best_score[256];
    __shared__ int best_page[256];
    for (int rank = 0; rank < count; ++rank) {
        float local_score = -FLT_MAX;
        int local_page = -1;
        for (int page = threadIdx.x; page < n_pages; page += blockDim.x) {
            if (!page_select_eligible(metadata, metadata_nb0, metadata_nb1,
                    membership, membership_nb0, query, query_nb0, page, expected_membership,
                    page_size, metadata_fields)) continue;
            if (!isfinite(scores[page])) continue;
            bool selected = false;
            for (int prior = 0; prior < rank; ++prior) {
                if (output[begin + prior] == page) {
                    selected = true;
                    break;
                }
            }
            if (!selected && (local_page < 0 || scores[page] > local_score ||
                    (scores[page] == local_score && page < local_page))) {
                local_score = scores[page];
                local_page = page;
            }
        }
        best_score[threadIdx.x] = local_score;
        best_page[threadIdx.x] = local_page;
        __syncthreads();
        for (int width = blockDim.x / 2; width > 0; width >>= 1) {
            if (threadIdx.x < width) {
                const int other = best_page[threadIdx.x + width];
                if (other >= 0 && (best_page[threadIdx.x] < 0 ||
                        best_score[threadIdx.x + width] > best_score[threadIdx.x] ||
                        (best_score[threadIdx.x + width] == best_score[threadIdx.x] && other < best_page[threadIdx.x]))) {
                    best_score[threadIdx.x] = best_score[threadIdx.x + width];
                    best_page[threadIdx.x] = other;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) output[begin + rank] = best_page[0];
        __syncthreads();
    }
}

} // namespace

void ggml_cuda_op_kv_page_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * bounds = dst->src[1];
    const ggml_tensor * metadata = dst->src[2];
    const ggml_tensor * membership = dst->src[3];
    const ggml_tensor * query = dst->src[4];
    const int k_resident = ggml_get_op_params_i32(dst, 0);
    const int k_cold = ggml_get_op_params_i32(dst, 1);
    const int page_size = ggml_get_op_params_i32(dst, 2);
    const int query_row = ggml_get_op_params_i32(dst, 3);
    const int n_pages = bounds->ne[3];
    const int metadata_fields = metadata->ne[0];
    const int output_count = k_resident + k_cold;
    ggml_cuda_pool_alloc<float> scores(ctx.pool(), n_pages);
    cudaStream_t stream = ctx.stream();
    int32_t * output = (int32_t *) dst->data;
    if (output_count > 0) {
        page_select_clear_output<<<(output_count + 255) / 256, 256, 0, stream>>>(
                output, output_count);
        CUDA_CHECK(cudaGetLastError());
    }
    page_select_scores<<<n_pages, 256, 0, stream>>>(
        (const float *) q->data, q->nb[0], q->nb[1], q->nb[2],
        (const half *) bounds->data, bounds->nb[0], bounds->nb[1], bounds->nb[2], bounds->nb[3],
        (const int64_t *) metadata->data, metadata->nb[0], metadata->nb[1],
        (const int32_t *) membership->data, membership->nb[0],
        (const int64_t *) query->data, query->nb[0], metadata_fields, page_size,
        scores.get(), n_pages,
        q->ne[0], q->ne[1], bounds->ne[2], query_row);
    CUDA_CHECK(cudaGetLastError());

#ifdef GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<uint64_t> resident_keys_in(ctx.pool(), n_pages);
    ggml_cuda_pool_alloc<uint64_t> resident_keys_out(ctx.pool(), n_pages);
    ggml_cuda_pool_alloc<uint64_t> cold_keys_in(ctx.pool(), n_pages);
    ggml_cuda_pool_alloc<uint64_t> cold_keys_out(ctx.pool(), n_pages);
    ggml_cuda_pool_alloc<int32_t> resident_ids_in(ctx.pool(), n_pages);
    ggml_cuda_pool_alloc<int32_t> resident_ids_out(ctx.pool(), n_pages);
    ggml_cuda_pool_alloc<int32_t> cold_ids_in(ctx.pool(), n_pages);
    ggml_cuda_pool_alloc<int32_t> cold_ids_out(ctx.pool(), n_pages);
    page_select_build_keys<<<(n_pages + 255) / 256, 256, 0, stream>>>(
        scores.get(),
        (const int64_t *) metadata->data, metadata->nb[0], metadata->nb[1],
        (const int32_t *) membership->data, membership->nb[0],
        (const int64_t *) query->data, query->nb[0], metadata_fields,
        resident_keys_in.get(), cold_keys_in.get(),
        resident_ids_in.get(), cold_ids_in.get(), n_pages, page_size);
    CUDA_CHECK(cudaGetLastError());

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
        resident_keys_in.get(), resident_keys_out.get(), resident_ids_in.get(),
        resident_ids_out.get(), n_pages, 0, 64, stream));
    size_t cold_temp_storage_bytes = 0;
    CUDA_CHECK(DeviceRadixSort::SortPairs(nullptr, cold_temp_storage_bytes,
        cold_keys_in.get(), cold_keys_out.get(), cold_ids_in.get(),
        cold_ids_out.get(), n_pages, 0, 64, stream));
    temp_storage_bytes = std::max(temp_storage_bytes, cold_temp_storage_bytes);
    ggml_cuda_pool_alloc<uint8_t> temp_storage(ctx.pool(), temp_storage_bytes);
    CUDA_CHECK(DeviceRadixSort::SortPairs(temp_storage.get(), temp_storage_bytes,
        resident_keys_in.get(), resident_keys_out.get(), resident_ids_in.get(),
        resident_ids_out.get(), n_pages, 0, 64, stream));
    CUDA_CHECK(DeviceRadixSort::SortPairs(temp_storage.get(), temp_storage_bytes,
        cold_keys_in.get(), cold_keys_out.get(), cold_ids_in.get(),
        cold_ids_out.get(), n_pages, 0, 64, stream));
    if (k_resident > 0) {
        page_select_emit<<<(k_resident + 255) / 256, 256, 0, stream>>>(
            resident_keys_out.get(), resident_ids_out.get(), output, 0,
            k_resident, n_pages);
        CUDA_CHECK(cudaGetLastError());
    }
    if (k_cold > 0) {
        page_select_emit<<<(k_cold + 255) / 256, 256, 0, stream>>>(
            cold_keys_out.get(), cold_ids_out.get(), output, k_resident,
            k_cold, n_pages);
        CUDA_CHECK(cudaGetLastError());
    }
#else
    if (k_resident > 0) {
        page_select_rank<<<1, 256, 0, stream>>>(scores.get(),
            (const int64_t *) metadata->data, metadata->nb[0], metadata->nb[1],
            (const int32_t *) membership->data, membership->nb[0],
            (const int64_t *) query->data, query->nb[0], output, n_pages, 0, k_resident, 1, page_size,
            metadata_fields);
        CUDA_CHECK(cudaGetLastError());
    }
    if (k_cold > 0) {
        page_select_rank<<<1, 256, 0, stream>>>(scores.get(),
            (const int64_t *) metadata->data, metadata->nb[0], metadata->nb[1],
            (const int32_t *) membership->data, membership->nb[0],
            (const int64_t *) query->data, query->nb[0], output, n_pages, k_resident, k_cold, 0, page_size,
            metadata_fields);
        CUDA_CHECK(cudaGetLastError());
    }
#endif
}
