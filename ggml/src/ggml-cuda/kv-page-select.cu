#include "kv-page-select.cuh"

#include "ggml-impl.h"

#include <cfloat>

namespace {

__device__ bool page_select_eligible(
        const int64_t * metadata, size_t metadata_nb0, size_t metadata_nb1,
        const int32_t * membership, size_t membership_nb0,
        const int64_t * query, size_t query_nb0,
        int page, int expected_membership, int page_size) {
    const int64_t position_begin = *(const int64_t *)((const char *) metadata + 0 * metadata_nb0 + page * metadata_nb1);
    const int64_t valid_length = *(const int64_t *)((const char *) metadata + 1 * metadata_nb0 + page * metadata_nb1);
    const int64_t sequence_generation = *(const int64_t *)((const char *) metadata + 2 * metadata_nb0 + page * metadata_nb1);
    const int64_t page_generation = *(const int64_t *)((const char *) metadata + 3 * metadata_nb0 + page * metadata_nb1);
    const int64_t query_position = *(const int64_t *)((const char *) query + 0 * query_nb0);
    const int64_t query_sequence_generation = *(const int64_t *)((const char *) query + 1 * query_nb0);
    const int64_t snapshot_generation = *(const int64_t *)((const char *) query + 2 * query_nb0);
    const int64_t refresh_enabled = *(const int64_t *)((const char *) query + 3 * query_nb0);
    const int membership_value = *(const int *)((const char *) membership + page * membership_nb0);
    return refresh_enabled && membership_value == expected_membership && valid_length > 0 &&
        valid_length <= page_size &&
        sequence_generation == query_sequence_generation && page_generation > 0 &&
        page_generation <= snapshot_generation && position_begin >= 0 &&
        position_begin < query_position && valid_length <= query_position - position_begin;
}

__global__ void page_select_scores(
        const float * q, size_t q_nb0, size_t q_nb1, size_t q_nb2,
        const half * bounds, size_t bounds_nb0, size_t bounds_nb1,
        size_t bounds_nb2, size_t bounds_nb3,
        float * scores, int n_pages, int d, int n_q_heads, int n_kv_heads,
        int query_row) {
    const int page = blockIdx.x;
    if (page >= n_pages) return;

    __shared__ float reduction[256];
    float best = -FLT_MAX;
    const int group = n_q_heads / n_kv_heads;
    for (int kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
        for (int q_head = kv_head * group; q_head < (kv_head + 1) * group; ++q_head) {
            float partial = 0.0f;
            const char * q_row = (const char *) q + q_head * q_nb1 + query_row * q_nb2;
            for (int coord = threadIdx.x; coord < d; coord += blockDim.x) {
                const float qi = *(const float *)(q_row + coord * q_nb0);
                const char * b = (const char *) bounds + coord * bounds_nb0 +
                    kv_head * bounds_nb2 + page * bounds_nb3;
                const float lo = __half2float(*(const half *) b);
                const float hi = __half2float(*(const half *) (b + bounds_nb1));
                partial += qi >= 0.0f ? qi * hi : qi * lo;
            }
            reduction[threadIdx.x] = partial;
            __syncthreads();
            for (int width = blockDim.x / 2; width > 0; width >>= 1) {
                if (threadIdx.x < width) reduction[threadIdx.x] += reduction[threadIdx.x + width];
                __syncthreads();
            }
            if (threadIdx.x == 0) best = max(best, reduction[0]);
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        // Eligibility is checked by the ranking kernels. Keeping this first
        // pass independent of metadata lets a refreshed catalogue reuse the
        // same cooperative score reduction without synchronizing the host.
        scores[page] = best;
    }
}

__global__ void page_select_rank(
        const float * scores,
        const int64_t * metadata, size_t metadata_nb0, size_t metadata_nb1,
        const int32_t * membership, size_t membership_nb0,
        const int64_t * query, size_t query_nb0,
        int32_t * output, int n_pages, int begin, int count,
        int expected_membership, int page_size) {
    __shared__ float best_score[256];
    __shared__ int best_page[256];
    for (int rank = 0; rank < count; ++rank) {
        float local_score = -FLT_MAX;
        int local_page = -1;
        for (int page = threadIdx.x; page < n_pages; page += blockDim.x) {
            if (!page_select_eligible(metadata, metadata_nb0, metadata_nb1,
                    membership, membership_nb0, query, query_nb0, page, expected_membership, page_size)) continue;
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
    ggml_cuda_pool_alloc<float> scores(ctx.pool(), n_pages);
    cudaStream_t stream = ctx.stream();
    page_select_scores<<<n_pages, 256, 0, stream>>>(
        (const float *) q->data, q->nb[0], q->nb[1], q->nb[2],
        (const half *) bounds->data, bounds->nb[0], bounds->nb[1], bounds->nb[2], bounds->nb[3],
        scores.get(), n_pages,
        q->ne[0], q->ne[1], bounds->ne[2], query_row);
    CUDA_CHECK(cudaGetLastError());

    int32_t * output = (int32_t *) dst->data;
    if (k_resident > 0) {
        page_select_rank<<<1, 256, 0, stream>>>(scores.get(),
            (const int64_t *) metadata->data, metadata->nb[0], metadata->nb[1],
            (const int32_t *) membership->data, membership->nb[0],
            (const int64_t *) query->data, query->nb[0], output, n_pages, 0, k_resident, 1, page_size);
        CUDA_CHECK(cudaGetLastError());
    }
    if (k_cold > 0) {
        page_select_rank<<<1, 256, 0, stream>>>(scores.get(),
            (const int64_t *) metadata->data, metadata->nb[0], metadata->nb[1],
            (const int32_t *) membership->data, membership->nb[0],
            (const int64_t *) query->data, query->nb[0], output, n_pages, k_resident, k_cold, 0, page_size);
        CUDA_CHECK(cudaGetLastError());
    }
}
