#include "kv-page-select.cuh"

#include "ggml-impl.h"
#include "ggml-kv-query-probes.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstring>

#ifdef GGML_CUDA_USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif

namespace {

__global__ void query_accumulate_reset(float * sum, size_t n, int64_t * count,
        const int64_t * control) {
    const int64_t turn_id = control[0];
    if (count[1] != turn_id) {
        for (size_t i = threadIdx.x; i < n; i += blockDim.x) sum[i] = 0.0f;
        __syncthreads();
        if (threadIdx.x == 0) {
        count[0] = 0;
        count[1] = turn_id;
        }
    }
}

__global__ void query_accumulate_rows(const float * q, size_t q_nb0, size_t q_nb1,
        size_t q_nb2, const int64_t * positions, size_t pos_nb0, float * sum,
        size_t sum_nb0, size_t sum_nb1, int64_t * count,
        const int64_t * control, int d, int heads, int rows) {
    const int64_t query_start = control[1];
    const int64_t query_end = control[2];
    const int64_t linear = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = int64_t(d) * heads;
    if (linear < total) {
        const int head = linear / d;
        const int coord = linear % d;
        float value = 0.0f;
        int64_t added = 0;
        for (int row = 0; row < rows; ++row) {
            const int64_t pos = *(const int64_t *)((const char *) positions + row * pos_nb0);
            if (pos < query_start || pos >= query_end) continue;
            const char * q_row = (const char *) q + head * q_nb1 + row * q_nb2;
            value += *(const float *)(q_row + coord * q_nb0);
            added++;
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

__global__ void query_accumulate_mean(const float * sum, size_t sum_nb0, size_t sum_nb1,
        const int64_t * count, float * out, size_t out_nb0, size_t out_nb1,
        int d, int heads) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(d) * heads) return;
    const int head = i / d;
    const int coord = i % d;
    const char * s = (const char *) sum + head * sum_nb1 + coord * sum_nb0;
    char * o = (char *) out + head * out_nb1 + coord * out_nb0;
    const int64_t n = count[0];
    *(float *)o = n > 0 ? *(const float *)s / float(n) : 0.0f;
}

__global__ void query_probes_capture(
        const float * q, size_t q_nb0, size_t q_nb1, size_t q_nb2,
        const int64_t * positions, size_t pos_nb0,
        float * probes, size_t probes_nb0, size_t probes_nb1, size_t probes_nb2,
        int64_t * validity, int64_t generation, int64_t query_start, int64_t query_end,
        int4 probe_rows, int indexed,
        float * out, size_t out_nb0, size_t out_nb1, size_t out_nb2,
        int d, int heads, int rows) {
    if (threadIdx.x == 0 && validity[0] != generation) {
        validity[0] = generation;
        for (int slot = 0; slot < 4; ++slot) {
            validity[1 + slot] = ggml_kv_query_probe_target(query_start, query_end, slot, indexed);
            validity[5 + slot] = 0;
        }
    }
    __syncthreads();
    const int64_t total = int64_t(d) * heads * 4;
    for (int64_t i = threadIdx.x; i < total; i += blockDim.x) {
        const int slot = i / (int64_t(d) * heads);
        const int head = (i / d) % heads;
        const int coord = i % d;
        const int64_t target = ggml_kv_query_probe_target(query_start, query_end, slot, indexed);
        if (target >= query_start) {
            const int indexed_rows[4] = { probe_rows.x, probe_rows.y, probe_rows.z, probe_rows.w };
            int matched_row = indexed ? indexed_rows[slot] : -1;
            if (!indexed) {
                for (int row = 0; row < rows; ++row) {
                    const int64_t pos = *(const int64_t *)((const char *) positions + row * pos_nb0);
                    if (pos == target) matched_row = row;
                }
            }
            if (matched_row >= 0 && matched_row < rows) {
                const char * src = (const char *) q + head * q_nb1 + matched_row * q_nb2 + coord * q_nb0;
                char * dst = (char *) probes + slot * probes_nb2 + head * probes_nb1 + coord * probes_nb0;
                *(float *) dst = *(const float *) src;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int slot = 0; slot < 4; ++slot) {
            const int64_t target = ggml_kv_query_probe_target(query_start, query_end, slot, indexed);
            if (target < query_start) continue;
            const int indexed_rows[4] = { probe_rows.x, probe_rows.y, probe_rows.z, probe_rows.w };
            if (indexed) {
                if (indexed_rows[slot] >= 0 && indexed_rows[slot] < rows) {
                    validity[1 + slot] = target;
                    validity[5 + slot] = 1;
                }
            } else {
                for (int row = 0; row < rows; ++row) {
                    const int64_t pos = *(const int64_t *)((const char *) positions + row * pos_nb0);
                    if (pos == target) {
                        validity[1 + slot] = target;
                        validity[5 + slot] = 1;
                        break;
                    }
                }
            }
        }
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < total; i += blockDim.x) {
        const int slot = i / (int64_t(d) * heads);
        const int head = (i / d) % heads;
        const int coord = i % d;
        const char * src = (const char *) probes + slot * probes_nb2 + head * probes_nb1 + coord * probes_nb0;
        char * dst = (char *) out + slot * out_nb2 + head * out_nb1 + coord * out_nb0;
        *(float *) dst = *(const float *) src;
    }
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
            float partial = 0.0f;
            int invalid = 0;
            int positive_inf = 0;
            int negative_inf = 0;
            for (int coord = threadIdx.x; coord < d; coord += blockDim.x) {
                float qi = 0.0f;
                const int row_begin = query_row < 0 ? 0 : query_row;
                const int row_end = query_row < 0 ? n_q_rows : query_row + 1;
                for (int row = row_begin; row < row_end; ++row) {
                    const char * q_row = (const char *) q + q_head * q_nb1 + row * q_nb2;
                    qi += *(const float *)(q_row + coord * q_nb0);
                }
                if (query_row < 0) qi /= float(row_end - row_begin);
                const char * b = (const char *) bounds + coord * bounds_nb0 +
                    kv_head * bounds_nb2 + page * bounds_nb3;
                if (scorer_mode == 1) {
                    const float mean = __half2float(*(const half *)(b + 2 * bounds_nb1));
                    if (!isfinite(mean)) invalid = 1;
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
                if (!isnan(score)) best = max(best, score);
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

__device__ float page_rank_score(const float * probes, size_t probe_nb0,
        size_t probe_nb1, size_t probe_nb2, const half * bounds,
        size_t bounds_nb0, size_t bounds_nb1, size_t bounds_nb2,
        size_t bounds_nb3, int64_t * metadata, size_t metadata_nb0,
        size_t metadata_nb1, const int32_t * membership, size_t membership_nb0,
        const int64_t * query, size_t query_nb0, int metadata_fields,
        int page_size, int page, int region, int source, int head, int probe,
        int kv_head, int dim, float scale) {
    if (!page_select_eligible(metadata, metadata_nb0, metadata_nb1,
            membership, membership_nb0, query, query_nb0, page,
            region == 0 ? 1 : 0, page_size, metadata_fields)) return nanf("");
    double score = 0.0;
    for (int d = 0; d < dim; ++d) {
        const float q = *(const float *) ((const char *) probes +
                d * probe_nb0 + head * probe_nb1 + probe * probe_nb2) * scale;
        if (!isfinite(q)) return nanf("");
        if (q == 0.0f) continue;
        const int bound_index = source == 0 ? (q > 0.0f ? 1 : 0) : 2;
        const float value = __half2float(*(const half *) ((const char *) bounds +
                d * bounds_nb0 + bound_index * bounds_nb1 + kv_head * bounds_nb2 + page * bounds_nb3));
        if (isnan(value) || (source == 0 && q > 0.0f &&
                __half2float(*(const half *) ((const char *) bounds + d * bounds_nb0 + kv_head * bounds_nb2 + page * bounds_nb3)) > value)) return nanf("");
        if (source == 0 && q < 0.0f &&
                __half2float(*(const half *) ((const char *) bounds + d * bounds_nb0 + bounds_nb1 + kv_head * bounds_nb2 + page * bounds_nb3)) < value) return nanf("");
        score += double(q) * double(value);
    }
    return float(score);
}

__global__ static void page_rank_clear(float * peak, float * sum, uint32_t * count,
        uint32_t * channels, int32_t * representatives, int representative_count,
        ggml_kv_page_rank_record * output, int pages, int output_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = 4 * pages;
    if (i < total) { peak[i] = 0.0f; sum[i] = 0.0f; count[i] = 0; }
    if (i < 4) channels[i] = 0;
    if (i < representative_count) representatives[i] = -1;
    if (i < output_count) output[i] = {-1, 0, 0.0f, 0.0f};
}

__global__ static void page_rank_normalize_accumulate(
        const float * probes, size_t probe_nb0, size_t probe_nb1, size_t probe_nb2,
        const half * bounds, size_t bounds_nb0, size_t bounds_nb1,
        size_t bounds_nb2, size_t bounds_nb3,
        const int64_t * metadata, size_t metadata_nb0, size_t metadata_nb1,
        const int32_t * membership, size_t membership_nb0,
        const int64_t * query, size_t query_nb0, const int64_t * validity,
        int metadata_fields, int page_size, int pages, int dim, int heads,
        int kv_heads, int n_probes, uint32_t probe_mask, float scale,
        float * peak, float * sum, uint32_t * count, uint32_t * channels,
        int32_t * representatives) {
    const int context = blockIdx.x;
    const int probe = context % n_probes;
    const int head = (context / n_probes) % heads;
    const int region = (context / (n_probes * heads)) % 2;
    const int source = context / (n_probes * heads * 2);
    const int combo = source * 2 + region;
    const int64_t * valid = validity;
    if (!(probe_mask & (1u << probe)) || !valid[5 + probe]) return;
    const int kv_head = head / (heads / kv_heads);
    __shared__ float values[256];
    __shared__ float sums[256];
    __shared__ int valid_counts[256];
    __shared__ int inf_counts[256];
    __shared__ float best_probabilities[256];
    __shared__ int best_pages[256];
    float local_max = -INFINITY;
    int local_valid = 0, local_inf = 0;
    for (int page = threadIdx.x; page < pages; page += blockDim.x) {
        const float value = page_rank_score(probes, probe_nb0, probe_nb1, probe_nb2,
                bounds, bounds_nb0, bounds_nb1, bounds_nb2, bounds_nb3,
                (int64_t *) metadata, metadata_nb0, metadata_nb1, membership,
                membership_nb0, query, query_nb0, metadata_fields, page_size,
                page, region, source, head, probe, kv_head, dim, scale);
        if (!isnan(value)) {
            ++local_valid;
            local_inf += value == INFINITY;
            local_max = fmaxf(local_max, value);
        }
    }
    values[threadIdx.x] = local_max;
    valid_counts[threadIdx.x] = local_valid;
    inf_counts[threadIdx.x] = local_inf;
    __syncthreads();
    for (int offset = 128; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            values[threadIdx.x] = fmaxf(values[threadIdx.x], values[threadIdx.x + offset]);
            valid_counts[threadIdx.x] += valid_counts[threadIdx.x + offset];
            inf_counts[threadIdx.x] += inf_counts[threadIdx.x + offset];
        }
        __syncthreads();
    }
    const float maximum = values[0];
    const int total_valid = valid_counts[0], total_inf = inf_counts[0];
    if (threadIdx.x == 0 && total_valid > 0) atomicAdd(channels + combo, 1u);
    float local_sum = 0.0f;
    if (total_inf == 0 && maximum != -INFINITY) {
        for (int page = threadIdx.x; page < pages; page += blockDim.x) {
            const float value = page_rank_score(probes, probe_nb0, probe_nb1, probe_nb2,
                    bounds, bounds_nb0, bounds_nb1, bounds_nb2, bounds_nb3,
                    (int64_t *) metadata, metadata_nb0, metadata_nb1, membership,
                    membership_nb0, query, query_nb0, metadata_fields, page_size,
                    page, region, source, head, probe, kv_head, dim, scale);
            if (isfinite(value)) local_sum += expf(value - maximum);
        }
    }
    sums[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = 128; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    const float denominator = sums[0];
    float local_best_probability = -1.0f;
    int local_best_page = -1;
    for (int page = threadIdx.x; page < pages; page += blockDim.x) {
        const float value = page_rank_score(probes, probe_nb0, probe_nb1, probe_nb2,
                bounds, bounds_nb0, bounds_nb1, bounds_nb2, bounds_nb3,
                (int64_t *) metadata, metadata_nb0, metadata_nb1, membership,
                membership_nb0, query, query_nb0, metadata_fields, page_size,
                page, region, source, head, probe, kv_head, dim, scale);
        if (isnan(value)) continue;
        float probability;
        if (total_inf != 0) probability = value == INFINITY ? 1.0f / total_inf : 0.0f;
        else if (maximum == -INFINITY) probability = 1.0f / total_valid;
        else probability = isfinite(value) && denominator > 0.0f ? expf(value - maximum) / denominator : 0.0f;
        if (probability > local_best_probability || (probability == local_best_probability &&
                (local_best_page < 0 || page < local_best_page))) {
            local_best_probability = probability;
            local_best_page = page;
        }
        const int index = combo * pages + page;
        atomicMax((unsigned int *) (peak + index), __float_as_uint(probability));
        atomicAdd(sum + index, probability);
        atomicAdd(count + index, 1u);
    }
    best_probabilities[threadIdx.x] = local_best_probability;
    best_pages[threadIdx.x] = local_best_page;
    __syncthreads();
    for (int offset = 128; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            const float other_probability = best_probabilities[threadIdx.x + offset];
            const int other_page = best_pages[threadIdx.x + offset];
            if (other_probability > best_probabilities[threadIdx.x] ||
                    (other_probability == best_probabilities[threadIdx.x] && other_page >= 0 &&
                     (best_pages[threadIdx.x] < 0 || other_page < best_pages[threadIdx.x]))) {
                best_probabilities[threadIdx.x] = other_probability;
                best_pages[threadIdx.x] = other_page;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int rep_index = combo * heads * n_probes + head * n_probes + probe;
        representatives[rep_index] = best_pages[0];
    }
}

__device__ static uint32_t page_rank_descending_key(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t ordered = (bits & UINT32_C(0x80000000)) != 0
        ? ~bits : bits ^ UINT32_C(0x80000000);
    return UINT32_MAX - ordered;
}

__global__ static void page_rank_build_sort_keys(const float * peak, const float * sum,
        const uint32_t * count, const uint32_t * channels,
        uint32_t * peak_keys, uint32_t * mean_keys,
        int32_t * ids, int pages) {
    const int combo = blockIdx.y;
    const int page = blockIdx.x * blockDim.x + threadIdx.x;
    if (page >= pages) return;
    const int index = combo * pages + page;
    const bool valid = count[index] != 0;
    peak_keys[index] = valid ? page_rank_descending_key(peak[index]) : UINT32_MAX;
    const float mean = valid && channels[combo] != 0 ? sum[index] / channels[combo] : 0.0f;
    mean_keys[index] = valid ? page_rank_descending_key(mean) : UINT32_MAX;
    ids[index] = page;
}

__global__ static void page_rank_build_peak_keys_from_mean(const float * peak,
        const uint32_t * count, const int32_t * mean_ids, uint32_t * peak_keys,
        int pages, int combo) {
    const int rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank >= pages) return;
    const int32_t page = mean_ids[rank];
    const int index = combo * pages + page;
    const bool valid = count[index] != 0;
    peak_keys[rank] = valid
        ? page_rank_descending_key(peak[index]) : UINT32_MAX;
}

__global__ static void page_rank_bounded_order(const float * peak, const float * sum,
        const uint32_t * count, const uint32_t * channels,
        int * ordered, int pages, int top_k) {
    const int combo = blockIdx.x;
    if (threadIdx.x != 0) return;
    for (int rank = 0; rank < min(pages, top_k); ++rank) {
        int best = -1;
        for (int page = 0; page < pages; ++page) {
            const int index = combo * pages + page;
            if (count[index] == 0) continue;
            bool selected = false;
            for (int prior = 0; prior < rank; ++prior) selected |= ordered[combo * pages + prior] == page;
            if (selected) continue;
            if (best < 0) { best = page; continue; }
            const int best_index = combo * pages + best;
            const float page_mean = channels[combo] ? sum[index] / channels[combo] : 0.0f;
            const float best_mean = channels[combo] ? sum[best_index] / channels[combo] : 0.0f;
            if (peak[index] > peak[best_index] || (peak[index] == peak[best_index] &&
                    (page_mean > best_mean || (page_mean == best_mean && page < best)))) best = page;
        }
        if (best >= 0) ordered[combo * pages + rank] = best;
    }
}

__global__ static void page_rank_emit(const float * peak, const float * sum,
        const uint32_t * count, const uint32_t * channels, const int * ordered,
        const int32_t * representatives, int heads, int n_probes,
        const int64_t * metadata, size_t metadata_nb0, size_t metadata_nb1,
        const int32_t * membership, size_t membership_nb0,
        const int64_t * query, size_t query_nb0, int metadata_fields, int page_size,
        ggml_kv_page_rank_record * output, int pages, int k_resident, int k_cold) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int region = 0; region < 2; ++region) {
        const int output_begin = region == 0 ? 0 : k_resident;
        const int width = region == 0 ? k_resident : k_cold;
        const int mean_combo = 2 + region;
        const int bound_combo = region;
        int source_count[2] = {0, 0};
        for (int page = 0; page < pages; ++page) {
            source_count[0] += count[mean_combo * pages + page] != 0;
            source_count[1] += count[bound_combo * pages + page] != 0;
        }
        int eligible = 0;
        for (int page = 0; page < pages; ++page) {
            eligible += page_select_eligible(metadata, metadata_nb0, metadata_nb1,
                    membership, membership_nb0, query, query_nb0, page,
                    region == 0 ? 1 : 0, page_size, metadata_fields);
        }
        const int limit = min(width, eligible);
        const int seed = (limit + 1) / 2;
        int written = 0;
        if (region == 1) {
            for (int source = 0; source < 2; ++source) {
                const int combo = source * 2 + region;
                for (int head = 0; head < heads; ++head) for (int probe = 0; probe < n_probes; ++probe) {
                    const int rep_index = combo * heads * n_probes + head * n_probes + probe;
                    const int page = representatives[rep_index];
                    if (page < 0 || page >= pages) continue;
                    const int index = combo * pages + page;
                    bool duplicate = false;
                    for (int i = 0; i < written; ++i) duplicate |= output[output_begin + i].logical_page == page;
                    const ggml_kv_page_rank_record candidate = {page, 1u, peak[index],
                        channels[combo] ? sum[index] / channels[combo] : 0.0f};
                    if (duplicate) continue;
                    if (written < limit) {
                        output[output_begin + written++] = candidate;
                    } else {
                        int worst = 0;
                        for (int i = 1; i < written; ++i) {
                            const auto a = output[output_begin + i], b = output[output_begin + worst];
                            if (a.peak_probability < b.peak_probability ||
                                    (a.peak_probability == b.peak_probability &&
                                     (a.mean_probability < b.mean_probability ||
                                      (a.mean_probability == b.mean_probability && a.logical_page > b.logical_page)))) worst = i;
                        }
                        const auto old = output[output_begin + worst];
                        if (candidate.peak_probability > old.peak_probability ||
                                (candidate.peak_probability == old.peak_probability &&
                                 (candidate.mean_probability > old.mean_probability ||
                                  (candidate.mean_probability == old.mean_probability && candidate.logical_page < old.logical_page)))) {
                            output[output_begin + worst] = candidate;
                        }
                    }
                }
            }
        }
        for (int source = 0; source < 2; ++source) {
            const int combo = source == 0 ? bound_combo : mean_combo;
            const int source_eligible = source_count[source == 0 ? 1 : 0];
            for (int rank = 0; rank < seed && rank < source_eligible && written < limit; ++rank) {
                const int page = ordered[combo * pages + rank];
                bool duplicate = false;
                for (int i = 0; i < written; ++i) duplicate |= output[output_begin + i].logical_page == page;
                if (!duplicate) {
                    const int index = combo * pages + page;
                    output[output_begin + written++] = {page, 1u, peak[index], channels[combo] ? sum[index] / channels[combo] : 0.0f};
                }
            }
        }
        int cursors[2] = {seed, seed};
        bool take_bound = true;
        while (written < limit && (cursors[0] < source_count[1] || cursors[1] < source_count[0])) {
            const int source = take_bound ? 0 : 1;
            const int combo = source == 0 ? bound_combo : mean_combo;
            const int source_eligible = source_count[source == 0 ? 1 : 0];
            if (cursors[source] < source_eligible) {
                const int page = ordered[combo * pages + cursors[source]];
                bool duplicate = false;
                for (int i = 0; i < written; ++i) duplicate |= output[output_begin + i].logical_page == page;
                if (!duplicate) {
                    const int index = combo * pages + page;
                    output[output_begin + written++] = {page, 1u, peak[index], channels[combo] ? sum[index] / channels[combo] : 0.0f};
                }
                ++cursors[source];
            }
            take_bound = !take_bound;
        }
        for (int i = 1; i < written; ++i) {
            const ggml_kv_page_rank_record value = output[output_begin + i];
            int j = i;
            while (j > 0) {
                const auto previous = output[output_begin + j - 1];
                if (previous.peak_probability > value.peak_probability ||
                        (previous.peak_probability == value.peak_probability &&
                         (previous.mean_probability > value.mean_probability ||
                          (previous.mean_probability == value.mean_probability && previous.logical_page < value.logical_page)))) break;
                output[output_begin + j] = previous;
                --j;
            }
            output[output_begin + j] = value;
        }
    }
}

void ggml_cuda_op_kv_page_rank(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * bounds = dst->src[1];
    const ggml_tensor * metadata = dst->src[2];
    const ggml_tensor * membership = dst->src[3];
    const ggml_tensor * query = dst->src[4];
    const ggml_tensor * probes = dst->src[5];
    const ggml_tensor * validity = dst->src[6];
    const int k_resident = ggml_get_op_params_i32(dst, 0);
    const int k_cold = ggml_get_op_params_i32(dst, 1);
    const int page_size = ggml_get_op_params_i32(dst, 2);
    const uint32_t probe_mask = uint32_t(ggml_get_op_params_i32(dst, 4));
    const float scale = ggml_get_op_params_f32(dst, 5);
    const int pages = int(bounds->ne[3]);
    const int heads = int(probes->ne[1]);
    const int n_probes = int(probes->ne[2]);
    const int output_count = k_resident + k_cold;
    ggml_cuda_pool_alloc<float> peak(ctx.pool(), size_t(4) * pages);
    ggml_cuda_pool_alloc<float> sum(ctx.pool(), size_t(4) * pages);
    ggml_cuda_pool_alloc<uint32_t> count(ctx.pool(), size_t(4) * pages);
    ggml_cuda_pool_alloc<uint32_t> channels(ctx.pool(), 4);
    const int representative_count = 4 * heads * n_probes;
    ggml_cuda_pool_alloc<int32_t> representatives(ctx.pool(), representative_count);
    ggml_cuda_pool_alloc<int> ordered(ctx.pool(), size_t(4) * pages);
    ggml_cuda_pool_alloc<uint32_t> peak_keys_in(ctx.pool(), size_t(4) * pages);
    ggml_cuda_pool_alloc<uint32_t> peak_keys_out(ctx.pool(), size_t(4) * pages);
    ggml_cuda_pool_alloc<uint32_t> mean_keys_in(ctx.pool(), size_t(4) * pages);
    ggml_cuda_pool_alloc<uint32_t> mean_keys_out(ctx.pool(), size_t(4) * pages);
    ggml_cuda_pool_alloc<int32_t> ids_in(ctx.pool(), size_t(4) * pages);
    ggml_cuda_pool_alloc<int32_t> ids_out(ctx.pool(), size_t(4) * pages);
    const int clear_count = max(max(4 * pages, output_count), representative_count);
    page_rank_clear<<<(clear_count + 255) / 256, 256, 0, ctx.stream()>>>(
        peak.get(), sum.get(), count.get(), channels.get(), representatives.get(), representative_count,
        (ggml_kv_page_rank_record *) dst->data, pages, output_count);
    CUDA_CHECK(cudaGetLastError());
    page_rank_normalize_accumulate<<<2 * 2 * heads * n_probes, 256, 0, ctx.stream()>>>(
        (const float *) probes->data, probes->nb[0], probes->nb[1], probes->nb[2],
        (const half *) bounds->data, bounds->nb[0], bounds->nb[1], bounds->nb[2], bounds->nb[3],
        (const int64_t *) metadata->data, metadata->nb[0], metadata->nb[1],
        (const int32_t *) membership->data, membership->nb[0],
        (const int64_t *) query->data, query->nb[0],
        (const int64_t *) validity->data, int(metadata->ne[0]), page_size,
        pages, int(bounds->ne[0]), heads, int(bounds->ne[2]), n_probes,
        probe_mask, scale, peak.get(), sum.get(), count.get(), channels.get(), representatives.get());
    CUDA_CHECK(cudaGetLastError());
    page_rank_build_sort_keys<<<dim3((pages + 127) / 128, 4), 128, 0, ctx.stream()>>>(
        peak.get(), sum.get(), count.get(), channels.get(), peak_keys_in.get(), mean_keys_in.get(), ids_in.get(), pages);
    CUDA_CHECK(cudaGetLastError());
#ifdef GGML_CUDA_USE_CUB
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
        mean_keys_in.get(), mean_keys_out.get(), ids_in.get(), ids_out.get(), pages, 0, 32, ctx.stream()));
    size_t peak_storage_bytes = 0;
    CUDA_CHECK(DeviceRadixSort::SortPairs(nullptr, peak_storage_bytes,
        peak_keys_in.get(), peak_keys_out.get(), ids_in.get(), ids_out.get(), pages, 0, 32, ctx.stream()));
    temp_storage_bytes = max(temp_storage_bytes, peak_storage_bytes);
    ggml_cuda_pool_alloc<uint8_t> temp_storage(ctx.pool(), temp_storage_bytes);
    for (int combo = 0; combo < 4; ++combo) {
        const size_t offset = size_t(combo) * pages;
        CUDA_CHECK(DeviceRadixSort::SortPairs(temp_storage.get(), temp_storage_bytes,
            mean_keys_in.get() + offset, mean_keys_out.get() + offset,
            ids_in.get() + offset, ids_out.get() + offset, pages, 0, 32, ctx.stream()));
        page_rank_build_peak_keys_from_mean<<<(pages + 127) / 128, 128, 0, ctx.stream()>>>(
            peak.get(), count.get(), ids_out.get() + offset, peak_keys_in.get() + offset, pages, combo);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(DeviceRadixSort::SortPairs(temp_storage.get(), temp_storage_bytes,
            peak_keys_in.get() + offset, peak_keys_out.get() + offset,
            ids_out.get() + offset, ordered.get() + offset, pages, 0, 32, ctx.stream()));
    }
#else
    // Non-CUB builds retain a bounded top-K rank table; they never allocate
    // or compute a quadratic all-pages ordering.
    page_rank_bounded_order<<<4, 128, 0, ctx.stream()>>>(
        peak.get(), sum.get(), count.get(), channels.get(), ordered.get(), pages, max(k_resident, k_cold));
    CUDA_CHECK(cudaGetLastError());
#endif
    page_rank_emit<<<1, 1, 0, ctx.stream()>>>(peak.get(), sum.get(), count.get(), channels.get(), ordered.get(),
        representatives.get(), heads, n_probes,
        (const int64_t *) metadata->data, metadata->nb[0], metadata->nb[1],
        (const int32_t *) membership->data, membership->nb[0],
        (const int64_t *) query->data, query->nb[0], int(metadata->ne[0]), page_size,
        (ggml_kv_page_rank_record *) dst->data, pages, k_resident, k_cold);
    CUDA_CHECK(cudaGetLastError());
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
            (float *) sum->data, size_t(ggml_nelements(sum)),
            (int64_t *) count->data, (const int64_t *) control->data);
    CUDA_CHECK(cudaGetLastError());
    query_accumulate_rows<<<blocks, threads, 0, stream>>>(
            (const float *) q->data, q->nb[0], q->nb[1], q->nb[2],
            (const int64_t *) positions->data, positions->nb[0],
            (float *) sum->data, sum->nb[0], sum->nb[1], (int64_t *) count->data,
            (const int64_t *) control->data, q->ne[0], q->ne[1], q->ne[2]);
    CUDA_CHECK(cudaGetLastError());
    query_accumulate_mean<<<blocks, threads, 0, stream>>>(
            (const float *) sum->data, sum->nb[0], sum->nb[1], (const int64_t *) count->data,
            (float *) dst->data, dst->nb[0], dst->nb[1], q->ne[0], q->ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_kv_query_probes(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * positions = dst->src[1];
    ggml_tensor * probes = dst->src[2];
    ggml_tensor * validity = dst->src[3];
    ggml_kv_query_probe_params controls{};
    memcpy(&controls, dst->op_params, sizeof(controls));
    query_probes_capture<<<1, 256, 0, ctx.stream()>>>(
            (const float *) q->data, q->nb[0], q->nb[1], q->nb[2],
            positions != nullptr ? (const int64_t *) positions->data : nullptr,
            positions != nullptr ? positions->nb[0] : sizeof(int64_t),
            (float *) probes->data, probes->nb[0], probes->nb[1], probes->nb[2],
            (int64_t *) validity->data, controls.generation, controls.query_start, controls.query_end,
            make_int4(controls.rows[0], controls.rows[1], controls.rows[2], controls.rows[3]), controls.indexed,
            (float *) dst->data, dst->nb[0], dst->nb[1], dst->nb[2],
            int(q->ne[0]), int(q->ne[1]), int(q->ne[2]));
    CUDA_CHECK(cudaGetLastError());
}
