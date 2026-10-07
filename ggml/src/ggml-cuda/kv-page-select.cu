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

__global__ void query_accumulate_reset(float * sum, size_t n, size_t plane, int64_t * count,
        const int64_t * control) {
    const int64_t turn_id = control[0];
    if (count[1] != turn_id) {
        for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
            // Unsampled probes must not become artificial zero-score winners.
            sum[i] = i < plane ? 0.0f : __int_as_float(0x7fffffff);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
        count[0] = 0;
        count[1] = turn_id;
        }
    }
}

__global__ void query_accumulate_rows(const float * q, size_t q_nb0, size_t q_nb1,
        size_t q_nb2, const int64_t * positions, size_t pos_nb0, float * sum,
        size_t sum_nb0, size_t sum_nb1, size_t sum_nb2, int probes, int64_t * count,
        const int64_t * control, int d, int heads, int rows) {
    const int64_t query_start = control[1];
    const int64_t query_end = control[2];
    const int64_t tail_length = query_end > query_start ? min(int64_t(32), query_end - query_start) : 0;
    const int64_t tail_start = query_end - tail_length;
    const int64_t probe_positions[3] = { tail_start,
        tail_start + (tail_length - 1) / 2, query_end - 1 };
    const int64_t linear = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = int64_t(d) * heads;
    if (linear < total) {
        const int head = linear / d;
        const int coord = linear % d;
        float value = 0.0f;
        for (int row = 0; row < rows; ++row) {
            const int64_t pos = *(const int64_t *)((const char *) positions + row * pos_nb0);
            if (pos < query_start || pos >= query_end) continue;
            const char * q_row = (const char *) q + head * q_nb1 + row * q_nb2;
            const float qi = *(const float *)(q_row + coord * q_nb0);
            value += qi;
            for (int probe = 1; probe < probes; ++probe) {
                if (pos == probe_positions[probe - 1]) {
                    *(float *)((char *) sum + head * sum_nb1 + probe * sum_nb2 + coord * sum_nb0) = qi;
                }
            }
        }
        char * sum_row = (char *) sum + head * sum_nb1;
        *(float *)(sum_row + coord * sum_nb0) += value;
    }
    if (linear == 0) {
        int64_t added = 0;
        for (int row = 0; row < rows; ++row) {
            const int64_t pos = *(const int64_t *)((const char *) positions + row * pos_nb0);
            added += pos >= query_start && pos < query_end;
        }
        count[0] += added;
    }
}

__global__ void query_accumulate_mean(const float * sum, size_t sum_nb0, size_t sum_nb1, size_t sum_nb2,
        const int64_t * count, float * out, size_t out_nb0, size_t out_nb1, size_t out_nb2,
        int d, int heads, int probes) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(d) * heads * probes) return;
    const int probe = i / (int64_t(d) * heads);
    const int head = (i / d) % heads;
    const int coord = i % d;
    const char * s = (const char *) sum + head * sum_nb1 + probe * sum_nb2 + coord * sum_nb0;
    char * o = (char *) out + head * out_nb1 + probe * out_nb2 + coord * out_nb0;
    const int64_t n = count[0];
    *(float *)o = n > 0 ? *(const float *)s / (probe == 0 ? float(n) : 1.0f)
        : probes > 1 ? __int_as_float(0x7fffffff) : 0.0f;
}

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
        int n_q_rows, int query_row, int scorer_mode) {
    const int page = blockIdx.x;
    if (page >= n_pages) return;

    __shared__ float reduction[256];
    __shared__ int invalid_reduction[256];
    __shared__ int positive_unbounded[256];
    __shared__ int negative_unbounded[256];
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
          const int probes = scorer_mode == 2 && query_row < 0 ? n_q_rows : 1;
          for (int probe = 0; probe < probes; ++probe) {
            float partial = 0.0f;
            int invalid = 0;
            int positive_inf = 0;
            int negative_inf = 0;
            for (int coord = threadIdx.x; coord < d; coord += blockDim.x) {
                float qi = 0.0f;
                const int row_begin = scorer_mode == 2 && query_row < 0 ? probe : query_row < 0 ? 0 : query_row;
                const int row_end = scorer_mode == 2 && query_row < 0 ? probe + 1 : query_row < 0 ? n_q_rows : query_row + 1;
                for (int row = row_begin; row < row_end; ++row) {
                    const char * q_row = (const char *) q + q_head * q_nb1 + row * q_nb2;
                    qi += *(const float *)(q_row + coord * q_nb0);
                }
                if (query_row < 0) qi /= float(row_end - row_begin);
                const char * b = (const char *) bounds + coord * bounds_nb0 +
                    kv_head * bounds_nb2 + page * bounds_nb3;
                if (scorer_mode == 1 || scorer_mode == 2) {
                    const float mean = __half2float(*(const half *)(b + 2 * bounds_nb1));
                    if (!isfinite(qi) || !isfinite(mean)) invalid = 1;
                    else partial += qi * mean;
                    continue;
                }
                const float lo = __half2float(*(const half *) b);
                const float hi = __half2float(*(const half *) (b + bounds_nb1));
                if (!isfinite(qi) || isnan(lo) || isnan(hi) || lo > hi) {
                    invalid = 1;
                } else if (qi == 0.0f) {
                    // A zero query coordinate contributes zero even when an
                    // outward-rounded half bound is infinite.
                } else if (isinf(qi >= 0.0f ? hi : lo)) {
                    const float bound = qi >= 0.0f ? hi : lo;
                    const bool positive = signbit(qi) == signbit(bound);
                    if (positive) positive_inf = 1;
                    else negative_inf = 1;
                } else {
                    partial += qi >= 0.0f ? qi * hi : qi * lo;
                }
            }
            reduction[threadIdx.x] = partial;
            invalid_reduction[threadIdx.x] = invalid;
            positive_unbounded[threadIdx.x] = positive_inf;
            negative_unbounded[threadIdx.x] = negative_inf;
            __syncthreads();
            for (int width = blockDim.x / 2; width > 0; width >>= 1) {
                if (threadIdx.x < width) {
                    invalid_reduction[threadIdx.x] |= invalid_reduction[threadIdx.x + width];
                    positive_unbounded[threadIdx.x] |= positive_unbounded[threadIdx.x + width];
                    negative_unbounded[threadIdx.x] |= negative_unbounded[threadIdx.x + width];
                }
                __syncthreads();
            }
            for (int width = blockDim.x / 2; width > 0; width >>= 1) {
                if (threadIdx.x < width) reduction[threadIdx.x] += reduction[threadIdx.x + width];
                __syncthreads();
            }
            if (threadIdx.x == 0 && invalid_reduction[0] == 0) {
                const float score = positive_unbounded[0]
                    ? __int_as_float(0x7f800000)
                    : negative_unbounded[0] ? -__int_as_float(0x7f800000) : reduction[0];
                if (scorer_mode == 0 ? !isnan(score) : isfinite(score)) best = max(best, score);
            }
            __syncthreads();
          }
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
    const uint64_t key = !isnan(score) && score != -__int_as_float(0x7f800000)
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
            if (isnan(scores[page]) || scores[page] == -__int_as_float(0x7f800000)) continue;
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

__global__ void page_select_report_failure(
        const float * scores,
        const int64_t * metadata, size_t metadata_nb0, size_t metadata_nb1,
        const int32_t * membership, size_t membership_nb0,
        const int64_t * query, size_t query_nb0,
        int32_t * output, int n_pages, int cold_begin, int cold_count,
        int page_size, int metadata_fields) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || cold_count == 0 ||
            output[cold_begin] != -1) return;
    bool eligible = false;
    bool rankable = false;
    for (int page = 0; page < n_pages; ++page) {
        if (!page_select_eligible(metadata, metadata_nb0, metadata_nb1,
                membership, membership_nb0, query, query_nb0, page, 0,
                page_size, metadata_fields)) continue;
        eligible = true;
        const float score = scores[page];
        rankable |= !isnan(score) && score != -__int_as_float(0x7f800000);
    }
    output[cold_begin] = eligible ? (rankable ? -1 : -2) : -3;
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
    const int diagnostic_mode = ggml_get_op_params_i32(dst, 4);
    const int scorer_mode = ggml_get_op_params_i32(dst, 5);
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
        q->ne[0], q->ne[1], bounds->ne[2], q->ne[2], query_row, scorer_mode);
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
    if (diagnostic_mode) {
        page_select_report_failure<<<1, 1, 0, stream>>>(scores.get(),
            (const int64_t *) metadata->data, metadata->nb[0], metadata->nb[1],
            (const int32_t *) membership->data, membership->nb[0],
            (const int64_t *) query->data, query->nb[0], output, n_pages,
            k_resident, k_cold, page_size, metadata_fields);
        CUDA_CHECK(cudaGetLastError());
    }
}

void ggml_cuda_op_kv_query_accumulate(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * positions = dst->src[1];
    ggml_tensor * sum = dst->src[2];
    ggml_tensor * count = dst->src[3];
    const ggml_tensor * control = dst->src[4];
    cudaStream_t stream = ctx.stream();
    const int64_t n = q->ne[0] * q->ne[1];
    const int threads = 256;
    const int blocks = int((n + threads - 1) / threads);
    query_accumulate_reset<<<1, threads, 0, stream>>>(
            (float *) sum->data, size_t(ggml_nelements(sum)), size_t(n),
            (int64_t *) count->data, (const int64_t *) control->data);
    CUDA_CHECK(cudaGetLastError());
    query_accumulate_rows<<<blocks, threads, 0, stream>>>(
            (const float *) q->data, q->nb[0], q->nb[1], q->nb[2],
            (const int64_t *) positions->data, positions->nb[0],
            (float *) sum->data, sum->nb[0], sum->nb[1], sum->nb[2], sum->ne[2], (int64_t *) count->data,
            (const int64_t *) control->data, q->ne[0], q->ne[1], q->ne[2]);
    CUDA_CHECK(cudaGetLastError());
    const int output_blocks = int((n * sum->ne[2] + threads - 1) / threads);
    query_accumulate_mean<<<output_blocks, threads, 0, stream>>>(
            (const float *) sum->data, sum->nb[0], sum->nb[1], sum->nb[2], (const int64_t *) count->data,
            (float *) dst->data, dst->nb[0], dst->nb[1], dst->nb[2], q->ne[0], q->ne[1], sum->ne[2]);
    CUDA_CHECK(cudaGetLastError());
}
