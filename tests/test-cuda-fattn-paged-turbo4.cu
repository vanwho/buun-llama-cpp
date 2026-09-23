#include "ggml-cuda/fattn-paged-turbo4.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-backend-impl.h"
#include "ggml-turbo-wht.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static void cuda_check(cudaError_t status, const char * what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::abort();
    }
}

static float time_paged_attention(
        ggml_backend_t backend,
        const ggml_cuda_fattn_turbo4_paged_params & params,
        cudaStream_t stream,
        cudaEvent_t start,
        cudaEvent_t stop,
        uint32_t warmups,
        uint32_t iterations) {
    for (uint32_t i = 0; i < warmups; ++i) {
        assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) ==
            ggml_cuda_fattn_turbo4_paged_status::ok);
    }
    cuda_check(cudaEventRecord(start, stream), "timing warmup fence");
    cuda_check(cudaEventSynchronize(start), "timing warmup synchronize");
    cuda_check(cudaEventRecord(start, stream), "timing start record");
    for (uint32_t i = 0; i < iterations; ++i) {
        assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) ==
            ggml_cuda_fattn_turbo4_paged_status::ok);
    }
    cuda_check(cudaEventRecord(stop, stream), "timing stop record");
    cuda_check(cudaEventSynchronize(stop), "timing stop synchronize");
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "timing readback");
    return elapsed_ms / iterations;
}

static void fill_turbo4_page(std::vector<uint8_t> & storage, size_t page_offset, uint8_t nibble) {
    const size_t row_bytes = 2 * sizeof(block_turbo4_0);
    for (uint32_t row = 0; row < 256; ++row) {
        for (uint32_t block = 0; block < 2; ++block) {
            auto * turbo = reinterpret_cast<block_turbo4_0 *>(storage.data() + page_offset +
                row * row_bytes + block * sizeof(block_turbo4_0));
            *reinterpret_cast<uint16_t *>(&turbo->norm) = 0x3c00; // fp16 1.0 bits
            for (uint32_t i = 0; i < sizeof(turbo->qs); ++i) {
                turbo->qs[i] = uint8_t(nibble | (nibble << 4));
            }
        }
    }
}

static float turbo4_host_value(const std::vector<uint8_t> & storage, size_t row_offset, uint32_t d) {
    static const float centroids[16] = {
        -0.241556f, -0.182907f, -0.143047f, -0.111065f,
        -0.083317f, -0.058069f, -0.034311f, -0.011353f,
         0.011353f,  0.034311f,  0.058069f,  0.083317f,
         0.111065f,  0.143047f,  0.182907f,  0.241556f,
    };
    const block_turbo4_0 * block = reinterpret_cast<const block_turbo4_0 *>(
        storage.data() + row_offset) + d / QK_TURBO4;
    const uint32_t j = d % QK_TURBO4;
    const uint8_t index = (j & 1) ? block->qs[j / 2] >> 4 : block->qs[j / 2] & 0x0F;
    return centroids[index] * 1.0f;
}

static void fill_interleaved_turbo4_page(
        std::vector<uint8_t> & storage, size_t page_offset, size_t row_bytes,
        size_t head_bytes, uint32_t kv_heads, uint32_t seed) {
    for (uint32_t row = 0; row < 256; ++row) {
        for (uint32_t head = 0; head < kv_heads; ++head) {
            const size_t row_offset = page_offset + size_t(row) * row_bytes + size_t(head) * head_bytes;
            for (uint32_t block = 0; block < 2; ++block) {
                auto * turbo = reinterpret_cast<block_turbo4_0 *>(storage.data() + row_offset +
                    size_t(block) * sizeof(block_turbo4_0));
                *reinterpret_cast<uint16_t *>(&turbo->norm) = 0x3c00;
                for (uint32_t i = 0; i < sizeof(turbo->qs); ++i) {
                    const uint32_t d = block * QK_TURBO4 + i * 2;
                    const uint8_t lo = uint8_t((seed + row * 3 + head * 5 + d) & 0xf);
                    const uint8_t hi = uint8_t((seed + row * 7 + head * 3 + d + 1) & 0xf);
                    turbo->qs[i] = uint8_t(lo | (hi << 4));
                }
            }
        }
    }
}

static std::vector<float> cpu_interleaved_turbo4_oracle(
        const std::vector<float> & q_storage, size_t q_head_stride, size_t q_query_stride,
        const std::vector<uint8_t> & k_storage, const std::vector<uint8_t> & v_storage,
        const ggml_cuda_fattn_turbo4_page * pages, uint32_t n_pages, uint32_t n_rows,
        const std::vector<int64_t> & native_positions, const std::vector<uint8_t> & native_mask,
        const std::vector<int64_t> & query_positions, uint32_t query_count,
        uint32_t n_head_q, uint32_t n_head_kv, size_t head_bytes, size_t row_bytes,
        size_t page_stride, float scale) {
    std::vector<float> result(size_t(query_count) * n_head_q * 256, 0.0f);
    const uint32_t gqa_ratio = n_head_q / n_head_kv;
    for (uint32_t query = 0; query < query_count; ++query) {
        for (uint32_t head = 0; head < n_head_q; ++head) {
            float q_rot[256];
            const char * q_head = reinterpret_cast<const char *>(q_storage.data()) +
                size_t(query) * q_query_stride + size_t(head) * q_head_stride;
            std::memcpy(q_rot, q_head, sizeof(q_rot));
            for (uint32_t group = 0; group < 2; ++group) {
                ggml_turbo_wht_transform_128(q_rot + group * 128, 0);
            }

            const uint32_t kv_head = head / gqa_ratio;
            float max_score = -std::numeric_limits<float>::infinity();
            uint32_t valid_count = 0;
            for (uint32_t page_index = 0; page_index < n_pages; ++page_index) {
                const auto & page = pages[page_index];
                for (uint32_t row = 0; row < page.row_count; ++row) {
                    const uint32_t compact_row = page.compact_row_begin + row;
                    if (compact_row >= n_rows || native_mask[compact_row] == 0 ||
                            native_positions[compact_row] > query_positions[query]) {
                        continue;
                    }
                    const size_t row_offset = size_t(page.source_physical_slot) * page_stride +
                        size_t(row) * row_bytes + size_t(kv_head) * head_bytes;
                    float score = 0.0f;
                    for (uint32_t d = 0; d < 256; ++d) {
                        score += q_rot[d] * turbo4_host_value(k_storage, row_offset, d);
                    }
                    max_score = std::max(max_score, score * scale);
                    ++valid_count;
                }
            }
            if (valid_count == 0) {
                continue;
            }
            float sum = 0.0f;
            float * output = result.data() + (size_t(query) * n_head_q + head) * 256;
            for (uint32_t page_index = 0; page_index < n_pages; ++page_index) {
                const auto & page = pages[page_index];
                for (uint32_t row = 0; row < page.row_count; ++row) {
                    const uint32_t compact_row = page.compact_row_begin + row;
                    if (compact_row >= n_rows || native_mask[compact_row] == 0 ||
                            native_positions[compact_row] > query_positions[query]) {
                        continue;
                    }
                    const size_t row_offset = size_t(page.source_physical_slot) * page_stride +
                        size_t(row) * row_bytes + size_t(kv_head) * head_bytes;
                    float score = 0.0f;
                    for (uint32_t d = 0; d < 256; ++d) {
                        score += q_rot[d] * turbo4_host_value(k_storage, row_offset, d);
                    }
                    const float weight = std::exp(score * scale - max_score);
                    sum += weight;
                    for (uint32_t d = 0; d < 256; ++d) {
                        output[d] += weight * turbo4_host_value(v_storage, row_offset, d);
                    }
                }
            }
            if (sum > 0.0f) {
                for (uint32_t d = 0; d < 256; ++d) output[d] /= sum;
            }
        }
    }
    return result;
}

static void run_multigroup_turbo4_numerics(ggml_backend_t backend) {
    constexpr uint32_t n_head_q = 24;
    constexpr uint32_t n_head_kv = 4;
    constexpr uint32_t n_physical_pages = 5;
    constexpr uint32_t max_query_tokens = 128;
    constexpr uint32_t n_rows = 513;
    constexpr size_t head_bytes = 2 * sizeof(block_turbo4_0);
    constexpr size_t row_bytes = n_head_kv * head_bytes;
    constexpr size_t page_stride = 256 * row_bytes;
    constexpr size_t q_head_stride = (256 + 8) * sizeof(float);
    constexpr size_t q_query_stride = (n_head_q * q_head_stride) + 16 * sizeof(float);
    constexpr size_t output_head_stride = (256 + 4) * sizeof(float);
    constexpr size_t output_query_stride = n_head_q * output_head_stride;

    const ggml_cuda_fattn_turbo4_page pages[3] = {
        { 0, 3,   0, 256,   0 },
        { 1, 0, 256, 256, 256 },
        { 2, 4, 512,   1, 512 },
    };
    assert(ggml_cuda_fattn_turbo4_page_table_valid(pages, 3, n_rows, 256, n_physical_pages));
    const ggml_cuda_fattn_turbo4_page pages_257[3] = {
        { 0, 3,   0, 256,   0 },
        { 1, 0, 256,   1, 256 },
        { 2, 4, 257, 256, 512 },
    };
    assert(ggml_cuda_fattn_turbo4_page_table_valid(pages_257, 2, 257, 256, n_physical_pages));
    const auto make_row_lookup = [](const ggml_cuda_fattn_turbo4_page * page_table,
            uint32_t page_count, uint32_t row_count) {
        std::vector<ggml_cuda_fattn_turbo4_row_lookup> result(row_count,
            { UINT32_MAX, UINT32_MAX });
        for (uint32_t page_index = 0; page_index < page_count; ++page_index) {
            for (uint32_t page_row = 0; page_row < page_table[page_index].row_count; ++page_row) {
                result[page_table[page_index].compact_row_begin + page_row] = {
                    page_index, page_row };
            }
        }
        return result;
    };
    const std::vector<ggml_cuda_fattn_turbo4_row_lookup> row_lookup =
        make_row_lookup(pages, 3, n_rows);
    const std::vector<ggml_cuda_fattn_turbo4_row_lookup> row_lookup_257 =
        make_row_lookup(pages_257, 2, n_rows);

    std::vector<uint8_t> k_host(size_t(n_physical_pages) * page_stride, 0xff);
    std::vector<uint8_t> v_host(k_host.size(), 0xff);
    for (uint32_t slot = 0; slot < n_physical_pages; ++slot) {
        fill_interleaved_turbo4_page(k_host, size_t(slot) * page_stride, row_bytes,
            head_bytes, n_head_kv, 1 + slot * 11);
        fill_interleaved_turbo4_page(v_host, size_t(slot) * page_stride, row_bytes,
            head_bytes, n_head_kv, 7 + slot * 13);
    }

    std::vector<float> q_host(size_t(max_query_tokens) * q_query_stride / sizeof(float), 0.0f);
    for (uint32_t query = 0; query < max_query_tokens; ++query) {
        for (uint32_t head = 0; head < n_head_q; ++head) {
            float * q = reinterpret_cast<float *>(reinterpret_cast<char *>(q_host.data()) +
                size_t(query) * q_query_stride + size_t(head) * q_head_stride);
            for (uint32_t d = 0; d < 256; ++d) {
                q[d] = 0.03f * std::sin(float((query + 1) * (head + 3) * (d + 5)) * 0.017f);
            }
        }
    }
    std::vector<int64_t> native_positions(n_rows);
    std::vector<uint8_t> native_mask(n_rows, 1);
    for (uint32_t row = 0; row < n_rows; ++row) native_positions[row] = int64_t(row);
    native_positions[17] = 401;
    native_positions[300] = 901;
    native_mask[33] = 0;
    native_mask[470] = 0;
    std::vector<int64_t> query_positions(max_query_tokens, 1000);

    float * q_device = nullptr;
    char * k_device = nullptr;
    char * v_device = nullptr;
    float * output_device = nullptr;
    ggml_cuda_fattn_turbo4_page * pages_device = nullptr;
    ggml_cuda_fattn_turbo4_row_lookup * row_lookup_device = nullptr;
    int64_t * native_positions_device = nullptr;
    uint8_t * native_mask_device = nullptr;
    int64_t * query_positions_device = nullptr;
    uint32_t * active_pages_device = nullptr;
    uint32_t * active_rows_device = nullptr;
    cuda_check(cudaMalloc(&q_device, q_host.size() * sizeof(float)), "multigroup q allocation");
    cuda_check(cudaMalloc(&k_device, k_host.size()), "multigroup k allocation");
    cuda_check(cudaMalloc(&v_device, v_host.size()), "multigroup v allocation");
    cuda_check(cudaMalloc(&output_device, size_t(max_query_tokens) * output_query_stride), "multigroup output allocation");
    cuda_check(cudaMalloc(&pages_device, sizeof(pages)), "multigroup pages allocation");
    cuda_check(cudaMalloc(&row_lookup_device, row_lookup.size() * sizeof(row_lookup[0])), "multigroup row lookup allocation");
    cuda_check(cudaMalloc(&native_positions_device, native_positions.size() * sizeof(int64_t)), "multigroup positions allocation");
    cuda_check(cudaMalloc(&native_mask_device, native_mask.size()), "multigroup mask allocation");
    cuda_check(cudaMalloc(&query_positions_device, query_positions.size() * sizeof(int64_t)), "multigroup query positions allocation");
    cuda_check(cudaMalloc(&active_pages_device, sizeof(uint32_t)), "multigroup active pages allocation");
    cuda_check(cudaMalloc(&active_rows_device, sizeof(uint32_t)), "multigroup active rows allocation");
    cuda_check(cudaMemcpy(q_device, q_host.data(), q_host.size() * sizeof(float), cudaMemcpyHostToDevice), "multigroup q copy");
    cuda_check(cudaMemcpy(k_device, k_host.data(), k_host.size(), cudaMemcpyHostToDevice), "multigroup k copy");
    cuda_check(cudaMemcpy(v_device, v_host.data(), v_host.size(), cudaMemcpyHostToDevice), "multigroup v copy");
    cuda_check(cudaMemcpy(pages_device, pages, sizeof(pages), cudaMemcpyHostToDevice), "multigroup pages copy");
    cuda_check(cudaMemcpy(row_lookup_device, row_lookup.data(), row_lookup.size() * sizeof(row_lookup[0]), cudaMemcpyHostToDevice), "multigroup row lookup copy");
    cuda_check(cudaMemcpy(native_positions_device, native_positions.data(), native_positions.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "multigroup native positions copy");
    cuda_check(cudaMemcpy(native_mask_device, native_mask.data(), native_mask.size(), cudaMemcpyHostToDevice), "multigroup native mask copy");
    cuda_check(cudaMemcpy(query_positions_device, query_positions.data(), query_positions.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "multigroup query positions copy");

    uint32_t active_pages = 3;
    uint32_t active_rows = n_rows;
    ggml_cuda_fattn_turbo4_paged_params params;
    params.q = q_device;
    params.output = output_device;
    params.q_head_stride_bytes = q_head_stride;
    params.q_query_stride_bytes = q_query_stride;
    params.output_head_stride_bytes = output_head_stride;
    params.output_query_stride_bytes = output_query_stride;
    params.type_k = GGML_TYPE_TURBO4_0;
    params.type_v = GGML_TYPE_TURBO4_0;
    params.head_dim_k = 256;
    params.head_dim_v = 256;
    params.k = k_device;
    params.v = v_device;
    params.k_row_stride_bytes = row_bytes;
    params.k_head_stride_bytes = head_bytes;
    params.k_page_stride_bytes = page_stride;
    params.v_row_stride_bytes = row_bytes;
    params.v_head_stride_bytes = head_bytes;
    params.v_page_stride_bytes = page_stride;
    params.pages_host = pages;
    params.pages_device = pages_device;
    params.compact_row_lookup_device = row_lookup_device;
    params.compact_row_lookup_capacity = n_rows;
    params.native_positions_device = native_positions_device;
    params.native_mask_device = native_mask_device;
    params.query_positions_device = query_positions_device;
    params.active_page_count_host = &active_pages;
    params.active_row_count_host = &active_rows;
    params.active_page_count_device = active_pages_device;
    params.active_row_count_device = active_rows_device;
    params.page_capacity = 3;
    params.row_capacity = n_rows;
    params.n_pages = 3;
    params.n_physical_pages = n_physical_pages;
    params.n_rows = n_rows;
    params.n_head_q = n_head_q;
    params.n_head_kv = n_head_kv;
    params.n_batch = 1;
    params.scale = 1.0f / std::sqrt(256.0f);
    params.explicit_native_metadata = true;
    cuda_check(cudaMemcpy(active_pages_device, &active_pages, sizeof(active_pages), cudaMemcpyHostToDevice), "multigroup active pages copy");
    cuda_check(cudaMemcpy(active_rows_device, &active_rows, sizeof(active_rows), cudaMemcpyHostToDevice), "multigroup active rows copy");

    const uint32_t query_counts[] = { 1, 2, 3, 4, 16, 17, 64, 128 };
    for (const uint32_t query_count : query_counts) {
        params.n_query_tokens = query_count;
        const std::vector<float> oracle = cpu_interleaved_turbo4_oracle(q_host,
            q_head_stride, q_query_stride, k_host, v_host, pages, 3, n_rows,
            native_positions, native_mask, query_positions, query_count, n_head_q,
            n_head_kv, head_bytes, row_bytes, page_stride, params.scale);
        std::vector<float> canary(size_t(max_query_tokens) * output_query_stride / sizeof(float),
            std::numeric_limits<float>::quiet_NaN());
        cuda_check(cudaMemcpy(output_device, canary.data(), canary.size() * sizeof(float), cudaMemcpyHostToDevice), "multigroup output poison");
        assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
        assert(ggml_cuda_fattn_turbo4_paged_last_dispatch_was_mma());
        cuda_check(cudaDeviceSynchronize(), "multigroup attention synchronize");
        std::vector<float> output(canary.size());
        cuda_check(cudaMemcpy(output.data(), output_device, output.size() * sizeof(float), cudaMemcpyDeviceToHost), "multigroup output readback");
        for (uint32_t query = 0; query < max_query_tokens; ++query) {
            for (uint32_t head = 0; head < n_head_q; ++head) {
                const float * actual = reinterpret_cast<const float *>(reinterpret_cast<const char *>(output.data()) +
                    size_t(query) * output_query_stride + size_t(head) * output_head_stride);
                for (uint32_t d = 0; d < 256; ++d) {
                    if (query < query_count) {
                        const float expected = oracle[(size_t(query) * n_head_q + head) * 256 + d];
                        assert(std::isfinite(actual[d]));
                        assert(std::fabs(actual[d] - expected) < 3.0e-3f);
                    } else {
                        assert(std::isnan(actual[d]));
                    }
                }
            }
        }
    }

    std::vector<int64_t> no_rows(query_positions.size(), -1);
    cuda_check(cudaMemcpy(query_positions_device, no_rows.data(), no_rows.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "multigroup no-row positions copy");
    params.n_query_tokens = 16;
    std::vector<float> zero_canary(size_t(max_query_tokens) * output_query_stride / sizeof(float), 17.0f);
    cuda_check(cudaMemcpy(output_device, zero_canary.data(), zero_canary.size() * sizeof(float), cudaMemcpyHostToDevice), "multigroup zero output poison");
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
    cuda_check(cudaDeviceSynchronize(), "multigroup no-row synchronize");
    cuda_check(cudaMemcpy(zero_canary.data(), output_device, zero_canary.size() * sizeof(float), cudaMemcpyDeviceToHost), "multigroup no-row output readback");
    for (uint32_t query = 0; query < params.n_query_tokens; ++query) {
        for (uint32_t head = 0; head < n_head_q; ++head) {
            const float * actual = reinterpret_cast<const float *>(reinterpret_cast<const char *>(zero_canary.data()) +
                size_t(query) * output_query_stride + size_t(head) * output_head_stride);
            for (uint32_t d = 0; d < 256; ++d) {
                if (actual[d] != 0.0f) {
                    std::fprintf(stderr, "multigroup no-row mismatch: query=%u head=%u d=%u value=%.9g\n",
                        query, head, d, actual[d]);
                    std::abort();
                }
            }
        }
    }
    cuda_check(cudaMemcpy(query_positions_device, query_positions.data(), query_positions.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "multigroup query positions restore");

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    active_pages = 2;
    active_rows = 257;
    cuda_check(cudaMemcpy(pages_device, pages_257, sizeof(pages_257), cudaMemcpyHostToDevice), "multigroup graph initial pages");
    cuda_check(cudaMemcpy(row_lookup_device, row_lookup_257.data(), row_lookup_257.size() * sizeof(row_lookup_257[0]), cudaMemcpyHostToDevice), "multigroup graph initial row lookup");
    cuda_check(cudaMemcpy(active_pages_device, &active_pages, sizeof(active_pages), cudaMemcpyHostToDevice), "multigroup graph initial page count");
    cuda_check(cudaMemcpy(active_rows_device, &active_rows, sizeof(active_rows), cudaMemcpyHostToDevice), "multigroup graph initial row count");
    params.pages_host = pages_257;
    params.n_query_tokens = 16;
    const cudaStream_t stream = static_cast<ggml_backend_cuda_context *>(backend->context)->stream();
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
    cuda_check(cudaStreamSynchronize(stream), "multigroup graph warmup");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "multigroup graph begin capture");
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
    assert(ggml_cuda_fattn_turbo4_paged_last_dispatch_was_mma());
    cuda_check(cudaStreamEndCapture(stream, &graph), "multigroup graph end capture");
    cuda_check(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), "multigroup graph instantiate");
    const std::vector<float> replay_canary(
        size_t(max_query_tokens) * output_query_stride / sizeof(float),
        std::numeric_limits<float>::quiet_NaN());
    const auto replay = [&](const ggml_cuda_fattn_turbo4_page * replay_pages, uint32_t page_count,
            uint32_t row_count, const char * label) {
        active_pages = page_count;
        active_rows = row_count;
        const std::vector<float> expected = cpu_interleaved_turbo4_oracle(q_host,
            q_head_stride, q_query_stride, k_host, v_host, replay_pages, page_count,
            row_count, native_positions, native_mask, query_positions, 16, n_head_q,
            n_head_kv, head_bytes, row_bytes, page_stride, params.scale);
        cuda_check(cudaMemcpyAsync(output_device, replay_canary.data(),
            replay_canary.size() * sizeof(float), cudaMemcpyHostToDevice, stream), label);
        cuda_check(cudaMemcpyAsync(pages_device, replay_pages, sizeof(pages), cudaMemcpyHostToDevice, stream), label);
        const auto & replay_lookup = page_count == 2 ? row_lookup_257 : row_lookup;
        cuda_check(cudaMemcpyAsync(row_lookup_device, replay_lookup.data(), replay_lookup.size() * sizeof(replay_lookup[0]), cudaMemcpyHostToDevice, stream), label);
        cuda_check(cudaMemcpyAsync(active_pages_device, &active_pages, sizeof(active_pages), cudaMemcpyHostToDevice, stream), label);
        cuda_check(cudaMemcpyAsync(active_rows_device, &active_rows, sizeof(active_rows), cudaMemcpyHostToDevice, stream), label);
        cuda_check(cudaGraphLaunch(graph_exec, stream), label);
        cuda_check(cudaStreamSynchronize(stream), label);
        std::vector<float> actual(replay_canary.size());
        cuda_check(cudaMemcpy(actual.data(), output_device,
            actual.size() * sizeof(float), cudaMemcpyDeviceToHost), label);
        for (uint32_t query = 0; query < max_query_tokens; ++query) {
            for (uint32_t head = 0; head < n_head_q; ++head) {
                const float * values = reinterpret_cast<const float *>(
                    reinterpret_cast<const char *>(actual.data()) +
                    size_t(query) * output_query_stride + size_t(head) * output_head_stride);
                for (uint32_t d = 0; d < 256; ++d) {
                    if (query < 16) {
                        const float expected_value = expected[(size_t(query) * n_head_q + head) * 256 + d];
                        assert(std::isfinite(values[d]));
                        assert(std::fabs(values[d] - expected_value) < 3.0e-3f);
                    } else {
                        assert(std::isnan(values[d]));
                    }
                }
            }
        }
    };
    replay(pages_257, 2, 257, "multigroup graph replay partial");
    replay(pages, 3, n_rows, "multigroup graph replay full");
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);

    cudaFree(active_rows_device);
    cudaFree(active_pages_device);
    cudaFree(query_positions_device);
    cudaFree(native_mask_device);
    cudaFree(native_positions_device);
    cudaFree(pages_device);
    cudaFree(row_lookup_device);
    cudaFree(output_device);
    cudaFree(v_device);
    cudaFree(k_device);
    cudaFree(q_device);
    std::fprintf(stderr, "cuda_multigroup_turbo4_numerics: passed (Q=1,2,3,4,16,17,64,128; GQA=24/4; interleaved rows)\n");
    std::fprintf(stderr, "cuda_paged_graph_capture_replay: passed (active rows 257 -> 513; stable capacities)\n");
}

// This is deliberately independent of the CUDA implementation.  The fixture
// uses zero Q, so every unmasked causal row has equal weight; the oracle still
// decodes the same Turbo4 V bytes and follows the compact/native page metadata.
static std::vector<float> cpu_selected_oracle(
        const std::vector<uint8_t> & v_storage,
        const ggml_cuda_fattn_turbo4_page * pages,
        uint32_t n_pages,
        const std::vector<int64_t> & native_positions,
        const std::vector<uint8_t> & native_mask,
        const std::vector<int64_t> & query_positions,
        uint32_t n_rows,
        uint32_t n_head_q,
        uint32_t n_head_kv,
        uint32_t n_physical_pages,
        size_t page_stride,
        uint32_t query_count,
        bool causal) {
    std::vector<float> result(size_t(query_count) * n_head_q * 256, 0.0f);
    const uint32_t gqa_ratio = n_head_q / n_head_kv;
    for (uint32_t query = 0; query < query_count; ++query) {
        for (uint32_t head = 0; head < n_head_q; ++head) {
            const uint32_t kv_head = head / gqa_ratio;
            const size_t head_offset = size_t(kv_head) * n_physical_pages * page_stride;
            uint32_t valid_rows = 0;
            float * output = result.data() + (size_t(query) * n_head_q + head) * 256;
            for (uint32_t page_index = 0; page_index < n_pages; ++page_index) {
                const auto & page = pages[page_index];
                for (uint32_t row = 0; row < page.row_count; ++row) {
                    const uint32_t compact_row = page.compact_row_begin + row;
                    if (compact_row >= n_rows || native_mask[compact_row] == 0) {
                        continue;
                    }
                    const int64_t native_position = native_positions[compact_row];
                    if (causal && native_position > query_positions[query]) {
                        continue;
                    }
                    const size_t row_offset = head_offset + size_t(page.source_physical_slot) * page_stride +
                        size_t(row) * 2 * sizeof(block_turbo4_0);
                    for (uint32_t d = 0; d < 256; ++d) {
                        output[d] += turbo4_host_value(v_storage, row_offset, d);
                    }
                    ++valid_rows;
                }
            }
            if (valid_rows != 0) {
                for (uint32_t d = 0; d < 256; ++d) {
                    output[d] /= valid_rows;
                }
            }
        }
    }
    return result;
}

static std::vector<uint8_t> pack_selected_storage(
        const std::vector<uint8_t> & storage,
        const ggml_cuda_fattn_turbo4_page * pages,
        uint32_t n_pages,
        uint32_t n_rows,
        uint32_t n_head_kv,
        uint32_t n_physical_pages,
        size_t page_stride,
        size_t row_bytes) {
    std::vector<uint8_t> packed(size_t(n_head_kv) * n_rows * row_bytes, 0);
    for (uint32_t head = 0; head < n_head_kv; ++head) {
        const size_t source_head = size_t(head) * n_physical_pages * page_stride;
        const size_t target_head = size_t(head) * n_rows * row_bytes;
        for (uint32_t page_index = 0; page_index < n_pages; ++page_index) {
            const auto & page = pages[page_index];
            for (uint32_t row = 0; row < page.row_count; ++row) {
                const size_t source = source_head + size_t(page.source_physical_slot) * page_stride +
                    size_t(row) * row_bytes;
                const size_t target = target_head + size_t(page.compact_row_begin + row) * row_bytes;
                std::memcpy(packed.data() + target, storage.data() + source, row_bytes);
            }
        }
    }
    return packed;
}

static float time_dense_fa(
        ggml_backend_t backend,
        const std::vector<float> & q_host,
        const std::vector<uint8_t> & k_storage,
        const std::vector<uint8_t> & v_storage,
        const std::vector<uint8_t> & packed_k,
        const std::vector<uint8_t> & packed_v,
        const ggml_cuda_fattn_turbo4_page * pages,
        uint32_t n_pages,
        uint32_t n_rows,
        uint32_t n_physical_pages,
        uint32_t n_head_q,
        uint32_t n_head_kv,
        size_t page_stride,
        size_t row_bytes,
        bool pack_rows) {
    const uint32_t query_count = uint32_t(q_host.size() / (size_t(n_head_q) * 256));
    ggml_init_params init_params = { 64u * 1024u * 1024u, nullptr, true };
    ggml_context * ctx = ggml_init(init_params);
    assert(ctx != nullptr);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, query_count, n_head_q);
    ggml_tensor * k_source = nullptr;
    ggml_tensor * v_source = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * indices = nullptr;
    std::vector<int32_t> index_host;
    if (pack_rows) {
        const uint32_t physical_rows = n_physical_pages * 256;
        k_source = ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, 256, physical_rows, n_head_kv);
        v_source = ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, 256, physical_rows, n_head_kv);
        indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, size_t(n_rows) * n_head_kv);
        index_host.resize(size_t(n_rows) * n_head_kv);
        for (uint32_t head = 0; head < n_head_kv; ++head) {
            const int32_t base = int32_t(head * physical_rows);
            uint32_t out = 0;
            for (uint32_t page_index = 0; page_index < n_pages; ++page_index) {
                const auto & page = pages[page_index];
                for (uint32_t row = 0; row < page.row_count; ++row) {
                    index_host[size_t(head) * n_rows + out++] = base +
                        int32_t(page.source_physical_slot * 256 + row);
                }
            }
        }
        k = ggml_reshape_3d(ctx, ggml_get_rows(ctx, k_source, indices), 256, n_rows, n_head_kv);
        v = ggml_reshape_3d(ctx, ggml_get_rows(ctx, v_source, indices), 256, n_rows, n_head_kv);
    } else {
        k = ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, 256, n_rows, n_head_kv);
        v = ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, 256, n_rows, n_head_kv);
    }
    ggml_tensor * output = ggml_flash_attn_ext(ctx, q, k, v, nullptr,
        1.0f / std::sqrt(256.0f), 0.0f, 0.0f);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    if (!ggml_cuda_flash_attn_ext_supported(0, output)) {
        std::fprintf(stderr, "dense Turbo4 FA control unsupported for Q=%u\n", query_count);
        ggml_free(ctx);
        return -1.0f;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);

    std::vector<float> dense_q(q_host.size());
    for (uint32_t query = 0; query < query_count; ++query) {
        for (uint32_t head = 0; head < n_head_q; ++head) {
            std::memcpy(dense_q.data() + (size_t(head) * query_count + query) * 256,
                q_host.data() + (size_t(query) * n_head_q + head) * 256, 256 * sizeof(float));
        }
    }
    ggml_backend_tensor_set(q, dense_q.data(), 0, dense_q.size() * sizeof(float));
    if (pack_rows) {
        ggml_backend_tensor_set(k_source, k_storage.data(), 0, k_storage.size());
        ggml_backend_tensor_set(v_source, v_storage.data(), 0, v_storage.size());
        ggml_backend_tensor_set(indices, index_host.data(), 0, index_host.size() * sizeof(index_host[0]));
    } else {
        ggml_backend_tensor_set(k, packed_k.data(), 0, packed_k.size());
        ggml_backend_tensor_set(v, packed_v.data(), 0, packed_v.size());
    }

    const cudaStream_t stream = static_cast<ggml_backend_cuda_context *>(backend->context)->stream();
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaEventCreate(&start), "dense timing start allocation");
    cuda_check(cudaEventCreate(&stop), "dense timing stop allocation");
    for (uint32_t i = 0; i < 5; ++i) {
        assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    }
    cuda_check(cudaEventRecord(start, stream), "dense timing start record");
    for (uint32_t i = 0; i < 20; ++i) {
        assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    }
    cuda_check(cudaEventRecord(stop, stream), "dense timing stop record");
    cuda_check(cudaEventSynchronize(stop), "dense timing stop synchronize");
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "dense timing readback");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return elapsed_ms / 20.0f;
}

static void run_split_tail_growth_regression(ggml_backend_t backend) {
    constexpr uint32_t n_head_q = 24;
    constexpr uint32_t n_head_kv = 4;
    constexpr uint32_t n_query_tokens = 64;
    constexpr uint32_t page_capacity = 128;
    constexpr uint32_t physical_pages = 64;
    constexpr uint32_t row_capacity = page_capacity * 256;
    // The production graph currently reserves one partition. Matching that
    // geometry is important for the Q=1 cold-append reproduction: a larger
    // scratch arena changes both the active partition count and kernel path.
    constexpr uint32_t partition_capacity = 1;
    constexpr size_t row_bytes = 2 * sizeof(block_turbo4_0);
    constexpr size_t page_stride = 256 * row_bytes;

    std::vector<ggml_cuda_fattn_turbo4_page> pages(page_capacity, {
        UINT32_MAX, UINT32_MAX, 0, 0, -1 });
    std::vector<uint8_t> k_host(size_t(n_head_kv) * physical_pages * page_stride, 0);
    std::vector<uint8_t> v_host(k_host.size(), 0);
    for (uint32_t head = 0; head < n_head_kv; ++head) {
        for (uint32_t slot = 0; slot < physical_pages; ++slot) {
            fill_turbo4_page(k_host, (size_t(head) * physical_pages + slot) * page_stride, 8);
            fill_turbo4_page(v_host, (size_t(head) * physical_pages + slot) * page_stride, 9);
        }
    }
    std::vector<float> q_host(size_t(n_query_tokens) * n_head_q * 256, 0.0f);
    std::vector<int64_t> query_positions(n_query_tokens, 1400);
    std::vector<float> split_scratch(
        size_t(partition_capacity) * n_query_tokens * n_head_q * (2 + 256) +
        size_t(partition_capacity) * n_query_tokens * n_head_q * page_capacity * 2, 0.0f);

    float * q_device = nullptr;
    char * k_device = nullptr;
    char * v_device = nullptr;
    float * output_device = nullptr;
    float * page_mass_device = nullptr;
    ggml_cuda_fattn_turbo4_page * pages_device = nullptr;
    int64_t * query_positions_device = nullptr;
    float * split_device = nullptr;
    uint32_t * active_pages_device = nullptr;
    uint32_t * active_rows_device = nullptr;
    uint32_t * active_tail_device = nullptr;
    uint64_t * selection_generation_device = nullptr;
    cuda_check(cudaMalloc(&q_device, q_host.size() * sizeof(float)), "split q allocation");
    cuda_check(cudaMalloc(&k_device, k_host.size()), "split k allocation");
    cuda_check(cudaMalloc(&v_device, v_host.size()), "split v allocation");
    cuda_check(cudaMalloc(&output_device, q_host.size() * sizeof(float)), "split output allocation");
    cuda_check(cudaMalloc(&page_mass_device,
        size_t(page_capacity) * n_head_q * n_query_tokens * sizeof(float)),
        "split page mass allocation");
    cuda_check(cudaMalloc(&pages_device, pages.size() * sizeof(pages[0])), "split page allocation");
    cuda_check(cudaMalloc(&query_positions_device,
        query_positions.size() * sizeof(query_positions[0])), "split query allocation");
    cuda_check(cudaMalloc(&split_device, split_scratch.size() * sizeof(float)), "split scratch allocation");
    cuda_check(cudaMalloc(&active_pages_device, sizeof(uint32_t)), "split active pages allocation");
    cuda_check(cudaMalloc(&active_rows_device, sizeof(uint32_t)), "split active rows allocation");
    cuda_check(cudaMalloc(&active_tail_device, sizeof(uint32_t)), "split active tail allocation");
    cuda_check(cudaMalloc(&selection_generation_device, sizeof(uint64_t)), "split generation allocation");
    cuda_check(cudaMemcpy(q_device, q_host.data(), q_host.size() * sizeof(float), cudaMemcpyHostToDevice),
        "split q copy");
    cuda_check(cudaMemcpy(k_device, k_host.data(), k_host.size(), cudaMemcpyHostToDevice), "split k copy");
    cuda_check(cudaMemcpy(v_device, v_host.data(), v_host.size(), cudaMemcpyHostToDevice), "split v copy");
    cuda_check(cudaMemcpy(query_positions_device, query_positions.data(),
        query_positions.size() * sizeof(query_positions[0]), cudaMemcpyHostToDevice), "split query copy");
    const uint64_t generation = 1;
    cuda_check(cudaMemcpy(selection_generation_device, &generation, sizeof(generation), cudaMemcpyHostToDevice),
        "split generation copy");

    const cudaStream_t stream = static_cast<ggml_backend_cuda_context *>(backend->context)->stream();
    uint32_t active_pages = 0;
    uint32_t active_rows = 0;
    uint32_t active_tail = 0;
    ggml_cuda_fattn_turbo4_paged_params params;
    params.q = q_device;
    params.output = output_device;
    params.page_mass = page_mass_device;
    params.page_mass_head_stride_bytes = size_t(page_capacity) * sizeof(float);
    params.page_mass_query_stride_bytes =
        params.page_mass_head_stride_bytes * n_head_q;
    params.page_mass_logical_count = page_capacity;
    params.reduce_page_mass = true;
    params.q_head_stride_bytes = 256 * sizeof(float);
    params.q_query_stride_bytes = n_head_q * params.q_head_stride_bytes;
    params.output_head_stride_bytes = 256 * sizeof(float);
    params.output_query_stride_bytes = n_head_q * params.output_head_stride_bytes;
    params.type_k = GGML_TYPE_TURBO4_0;
    params.type_v = GGML_TYPE_TURBO4_0;
    params.head_dim_k = 256;
    params.head_dim_v = 256;
    params.k = k_device;
    params.v = v_device;
    params.k_row_stride_bytes = row_bytes;
    params.k_head_stride_bytes = size_t(physical_pages) * page_stride;
    params.k_page_stride_bytes = page_stride;
    params.v_row_stride_bytes = row_bytes;
    params.v_head_stride_bytes = size_t(physical_pages) * page_stride;
    params.v_page_stride_bytes = page_stride;
    params.pages_host = pages.data();
    params.pages_device = pages_device;
    params.query_positions_device = query_positions_device;
    params.split_kv_scratch = split_device;
    const size_t split_state_head_stride = size_t(2 + 256) * sizeof(float);
    const size_t split_state_query_stride = split_state_head_stride * n_head_q;
    const size_t split_state_partition_stride = split_state_query_stride * n_query_tokens;
    params.split_kv_partition_capacity = partition_capacity;
    params.split_kv_page_count = page_capacity;
    params.split_kv_partition_stride_bytes = split_state_partition_stride;
    params.split_kv_page_state = reinterpret_cast<float *>(
        reinterpret_cast<char *>(split_device) + partition_capacity * split_state_partition_stride);
    params.split_kv_page_state_head_stride_bytes = size_t(page_capacity) * 2 * sizeof(float);
    params.split_kv_page_state_query_stride_bytes =
        params.split_kv_page_state_head_stride_bytes * n_head_q;
    params.split_kv_page_state_partition_stride_bytes =
        params.split_kv_page_state_query_stride_bytes * n_query_tokens;
    params.n_pages = page_capacity;
    params.n_physical_pages = physical_pages;
    params.n_rows = row_capacity;
    params.n_head_q = n_head_q;
    params.n_head_kv = n_head_kv;
    params.n_query_tokens = n_query_tokens;
    params.n_batch = 1;
    params.scale = 1.0f / std::sqrt(256.0f);
    params.page_capacity = page_capacity;
    params.row_capacity = row_capacity;
    params.active_page_count_host = &active_pages;
    params.active_row_count_host = &active_rows;
    params.active_page_count_device = active_pages_device;
    params.active_row_count_device = active_rows_device;
    params.active_tail_length_device = active_tail_device;
    params.selection_generation_device = selection_generation_device;
    params.explicit_native_metadata = false;

    cudaEvent_t tail_copy_start = nullptr;
    cudaEvent_t tail_copy_stop = nullptr;
    cudaEvent_t append_start = nullptr;
    cudaEvent_t append_stop = nullptr;
    cuda_check(cudaEventCreate(&tail_copy_start), "split tail copy start allocation");
    cuda_check(cudaEventCreate(&tail_copy_stop), "split tail copy stop allocation");
    cuda_check(cudaEventCreate(&append_start), "split append start allocation");
    cuda_check(cudaEventCreate(&append_stop), "split append stop allocation");

    const auto append = [&](uint32_t page_count, uint32_t rows) {
        pages.assign(page_capacity, { UINT32_MAX, UINT32_MAX, 0, 0, -1 });
        uint32_t remaining = rows;
        for (uint32_t page = 0; page < page_count; ++page) {
            const uint32_t count = std::min<uint32_t>(remaining, 256);
            pages[page] = { page, page, page * 256, count, int64_t(page * 256) };
            remaining -= count;
        }
        assert(remaining == 0 && ggml_cuda_fattn_turbo4_page_table_valid(
            pages.data(), page_count, rows));
        active_pages = page_count;
        active_rows = rows;
        active_tail = pages[page_count - 1].row_count;
        cuda_check(cudaEventRecord(tail_copy_start, stream), "split tail copy timing start record");
        cuda_check(cudaMemcpyAsync(active_tail_device, &active_tail, sizeof(active_tail),
            cudaMemcpyHostToDevice, stream), "split timed active tail copy");
        cuda_check(cudaEventRecord(tail_copy_stop, stream), "split tail copy timing stop record");
        cuda_check(cudaEventSynchronize(tail_copy_stop), "split tail copy timing synchronize");
        float tail_copy_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&tail_copy_ms, tail_copy_start, tail_copy_stop),
            "split tail copy timing readback");
        std::fprintf(stderr, "split incremental tail H2D: %.6f ms (tail=%u rows, KV heads=%u)\n",
            tail_copy_ms, active_tail, n_head_kv);
        cuda_check(cudaEventRecord(append_start, stream), "split append timing start record");
        cuda_check(cudaMemcpyAsync(pages_device, pages.data(), pages.size() * sizeof(pages[0]),
            cudaMemcpyHostToDevice, stream), "split append page copy");
        cuda_check(cudaMemcpyAsync(active_pages_device, &active_pages, sizeof(active_pages),
            cudaMemcpyHostToDevice, stream), "split append active pages copy");
        cuda_check(cudaMemcpyAsync(active_rows_device, &active_rows, sizeof(active_rows),
            cudaMemcpyHostToDevice, stream), "split append active rows copy");
        cuda_check(cudaMemcpyAsync(active_tail_device, &active_tail, sizeof(active_tail),
            cudaMemcpyHostToDevice, stream), "split append active tail copy");
        const auto status = ggml_cuda_flash_attn_ext_paged_turbo4(backend, params);
        if (status != ggml_cuda_fattn_turbo4_paged_status::ok) {
            std::fprintf(stderr, "split tail append failed: pages=%u rows=%u status=%s\n",
                page_count, rows, ggml_cuda_fattn_turbo4_paged_status_name(status));
            std::abort();
        }
        cuda_check(cudaEventRecord(append_stop, stream), "split append timing stop record");
        cuda_check(cudaEventSynchronize(append_stop), "split append timing synchronize");
        float append_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&append_ms, append_start, append_stop),
            "split append timing readback");
        std::fprintf(stderr, "split incremental append+attention: %.6f ms (rows=%u, pages=%u, KV heads=%u)\n",
            append_ms, rows, page_count, n_head_kv);
    };
    append(5, 1203);
    append(5, 1267);
    append(6, 1331);
    cuda_check(cudaDeviceSynchronize(), "split tail growth and page reuse");

    std::vector<float> output(q_host.size());
    cuda_check(cudaMemcpy(output.data(), output_device, output.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "split output readback");
    for (const float value : output) assert(std::isfinite(value));

    cudaEventDestroy(append_stop);
    cudaEventDestroy(append_start);
    cudaEventDestroy(tail_copy_stop);
    cudaEventDestroy(tail_copy_start);

    cudaFree(selection_generation_device);
    cudaFree(active_tail_device);
    cudaFree(active_rows_device);
    cudaFree(active_pages_device);
    cudaFree(split_device);
    cudaFree(query_positions_device);
    cudaFree(pages_device);
    cudaFree(output_device);
    cudaFree(page_mass_device);
    cudaFree(v_device);
    cudaFree(k_device);
    cudaFree(q_device);
}

static void time_large_prefill_cases(
        ggml_backend_t backend,
        cudaStream_t stream,
        cudaEvent_t timing_start,
        cudaEvent_t timing_stop) {
    constexpr uint32_t n_head_q = 4;
    constexpr uint32_t n_head_kv = 1;
    constexpr uint32_t max_query_tokens = 128;
    constexpr size_t row_bytes = 2 * sizeof(block_turbo4_0);
    constexpr size_t page_stride = 256 * row_bytes;

    for (const uint32_t n_rows : { 2048u, 4096u, 8192u }) {
        const uint32_t n_pages = n_rows / 256;
        const uint32_t n_physical_pages = n_pages;
        std::vector<ggml_cuda_fattn_turbo4_page> pages(n_pages);
        for (uint32_t page = 0; page < n_pages; ++page) {
            pages[page] = { page, page, page * 256, 256, int64_t(page * 256) };
        }
        assert(ggml_cuda_fattn_turbo4_page_table_valid(pages.data(), n_pages, n_rows));

        std::vector<uint8_t> k_host(size_t(n_physical_pages) * page_stride, 0);
        std::vector<uint8_t> v_host(size_t(n_physical_pages) * page_stride, 0);
        for (uint32_t page = 0; page < n_pages; ++page) {
            fill_turbo4_page(k_host, size_t(page) * page_stride, 8);
            fill_turbo4_page(v_host, size_t(page) * page_stride, 9);
        }
        std::vector<float> q_host(size_t(max_query_tokens) * n_head_q * 256, 0.0f);
        std::vector<int64_t> native_positions(n_rows);
        for (uint32_t row = 0; row < n_rows; ++row) native_positions[row] = row;
        std::vector<int64_t> query_positions(max_query_tokens);
        for (uint32_t query = 0; query < max_query_tokens; ++query) {
            query_positions[query] = int64_t(n_rows - max_query_tokens + query);
        }

        float * q_device = nullptr;
        char * k_device = nullptr;
        char * v_device = nullptr;
        char * output_device = nullptr;
        ggml_cuda_fattn_turbo4_page * pages_device = nullptr;
        int64_t * native_positions_device = nullptr;
        int64_t * query_positions_device = nullptr;
        cuda_check(cudaMalloc(&q_device, q_host.size() * sizeof(float)), "large q allocation");
        cuda_check(cudaMalloc(&k_device, k_host.size()), "large k allocation");
        cuda_check(cudaMalloc(&v_device, v_host.size()), "large v allocation");
        cuda_check(cudaMalloc(&output_device, q_host.size() * sizeof(float)), "large output allocation");
        cuda_check(cudaMalloc(&pages_device, pages.size() * sizeof(pages[0])), "large page allocation");
        cuda_check(cudaMalloc(&native_positions_device, native_positions.size() * sizeof(native_positions[0])),
            "large positions allocation");
        cuda_check(cudaMalloc(&query_positions_device, query_positions.size() * sizeof(query_positions[0])),
            "large query positions allocation");
        cuda_check(cudaMemcpy(q_device, q_host.data(), q_host.size() * sizeof(float), cudaMemcpyHostToDevice),
            "large q copy");
        cuda_check(cudaMemcpy(k_device, k_host.data(), k_host.size(), cudaMemcpyHostToDevice), "large k copy");
        cuda_check(cudaMemcpy(v_device, v_host.data(), v_host.size(), cudaMemcpyHostToDevice), "large v copy");
        cuda_check(cudaMemcpy(pages_device, pages.data(), pages.size() * sizeof(pages[0]), cudaMemcpyHostToDevice),
            "large page copy");
        cuda_check(cudaMemcpy(native_positions_device, native_positions.data(),
            native_positions.size() * sizeof(native_positions[0]), cudaMemcpyHostToDevice), "large positions copy");
        cuda_check(cudaMemcpy(query_positions_device, query_positions.data(),
            query_positions.size() * sizeof(query_positions[0]), cudaMemcpyHostToDevice),
            "large query positions copy");

        ggml_cuda_fattn_turbo4_paged_params params;
        params.q = q_device;
        params.output = reinterpret_cast<float *>(output_device);
        params.q_head_stride_bytes = 256 * sizeof(float);
        params.q_query_stride_bytes = n_head_q * params.q_head_stride_bytes;
        params.output_head_stride_bytes = 256 * sizeof(float);
        params.output_query_stride_bytes = n_head_q * params.output_head_stride_bytes;
        params.type_k = GGML_TYPE_TURBO4_0;
        params.type_v = GGML_TYPE_TURBO4_0;
        params.head_dim_k = 256;
        params.head_dim_v = 256;
        params.k = k_device;
        params.v = v_device;
        params.k_row_stride_bytes = row_bytes;
        params.k_head_stride_bytes = size_t(n_physical_pages) * page_stride;
        params.k_page_stride_bytes = page_stride;
        params.v_row_stride_bytes = row_bytes;
        params.v_head_stride_bytes = size_t(n_physical_pages) * page_stride;
        params.v_page_stride_bytes = page_stride;
        params.pages_host = pages.data();
        params.pages_device = pages_device;
        params.native_positions_device = native_positions_device;
        params.query_positions_device = query_positions_device;
        params.n_pages = n_pages;
        params.n_physical_pages = n_physical_pages;
        params.n_rows = n_rows;
        params.n_head_q = n_head_q;
        params.n_head_kv = n_head_kv;
        params.n_batch = 1;
        params.scale = 1.0f / std::sqrt(256.0f);

        const std::vector<uint8_t> packed_k = pack_selected_storage(k_host, pages.data(),
            n_pages, n_rows, n_head_kv, n_physical_pages, page_stride, row_bytes);
        const std::vector<uint8_t> packed_v = pack_selected_storage(v_host, pages.data(),
            n_pages, n_rows, n_head_kv, n_physical_pages, page_stride, row_bytes);
        for (const uint32_t query_count : { 16u, 64u, 128u }) {
            params.n_query_tokens = query_count;
            const float direct_ms = time_paged_attention(backend, params, stream,
                timing_start, timing_stop, 5, 20);
            const std::vector<float> q_timing(q_host.begin(),
                q_host.begin() + size_t(query_count) * n_head_q * 256);
            const float contiguous_ms = time_dense_fa(backend, q_timing, k_host, v_host,
                packed_k, packed_v, pages.data(), n_pages, n_rows, n_physical_pages,
                n_head_q, n_head_kv, page_stride, row_bytes, false);
            const float packed_ms = time_dense_fa(backend, q_timing, k_host, v_host,
                packed_k, packed_v, pages.data(), n_pages, n_rows, n_physical_pages,
                n_head_q, n_head_kv, page_stride, row_bytes, true);
            assert(direct_ms >= 0.0f && contiguous_ms >= 0.0f && packed_ms >= 0.0f);
            std::fprintf(stderr, "large timing table (A=%u, U=%u, pages=%u, warmups=5, iterations=20)\n",
                n_rows, query_count, n_pages);
            std::fprintf(stderr, "  fused paged MMA direct: %.3f ms\n", direct_ms);
            std::fprintf(stderr, "  contiguous Turbo4 FA:   %.3f ms\n", contiguous_ms);
            std::fprintf(stderr, "  non-contiguous pack+FA: %.3f ms\n", packed_ms);
        }

        cudaFree(query_positions_device);
        cudaFree(native_positions_device);
        cudaFree(pages_device);
        cudaFree(output_device);
        cudaFree(v_device);
        cudaFree(k_device);
        cudaFree(q_device);
    }
}

static void run_pager_layer_slot_stride_case(
        ggml_backend_t backend, uint32_t query_count, uint32_t active_rows) {
    // Match the live cold-append page map: a full first page and a 91-row
    // tail, selected from slots 0 and 1. Each layer occupies one contiguous
    // physical-page run in the shared slab, so the attention page stride is
    // this layer's 135,168-byte page payload. The aggregate pager slot size
    // includes every layer and both K/V sides and is not an attention stride.
    // The production single-query direct graph uses the MMA consumer. The
    // Q=64 case keeps split scratch to cover the separate cooperative path.
    constexpr uint32_t n_head_q = 24;
    constexpr uint32_t n_head_kv = 4;
    constexpr uint32_t active_pages = 2;
    // The 8192-token production graph reserves 32 logical pages even while
    // this request has only two selected resident pages.
    constexpr uint32_t page_capacity = 32;
    constexpr uint32_t row_capacity = 8192;
    constexpr uint32_t physical_slots = 4;
    constexpr size_t head_bytes = 2 * sizeof(block_turbo4_0);
    constexpr size_t row_bytes = n_head_kv * head_bytes;
    constexpr size_t layer_page_stride = 256 * row_bytes;
    constexpr size_t q_head_stride = 256 * sizeof(float);

    std::vector<ggml_cuda_fattn_turbo4_page> pages(page_capacity,
        { UINT32_MAX, UINT32_MAX, 0, 0, -1 });
    pages[0] = { 0, 0,   0, 256,   0 };
    pages[1] = { 1, 1, 256, active_rows - 256, 256 };
    assert(ggml_cuda_fattn_turbo4_page_table_valid(
        pages.data(), active_pages, active_rows, 256, physical_slots));

    std::vector<ggml_cuda_fattn_turbo4_row_lookup> row_lookup(row_capacity,
        { UINT32_MAX, UINT32_MAX });
    for (uint32_t row = 0; row < active_rows; ++row) {
        row_lookup[row] = row < 256
            ? ggml_cuda_fattn_turbo4_row_lookup{ 0, row }
            : ggml_cuda_fattn_turbo4_row_lookup{ 1, row - 256 };
    }
    std::vector<uint8_t> k_host(physical_slots * layer_page_stride, 0);
    std::vector<uint8_t> v_host(k_host.size(), 0);
    fill_interleaved_turbo4_page(k_host, 0, row_bytes, head_bytes, n_head_kv, 3);
    fill_interleaved_turbo4_page(k_host, layer_page_stride, row_bytes,
        head_bytes, n_head_kv, 17);
    fill_interleaved_turbo4_page(v_host, 0, row_bytes, head_bytes, n_head_kv, 11);
    fill_interleaved_turbo4_page(v_host, layer_page_stride, row_bytes,
        head_bytes, n_head_kv, 23);

    std::vector<float> q_host(size_t(query_count) * n_head_q * 256, 0.0f);
    for (uint32_t query = 0; query < query_count; ++query) {
        for (uint32_t head = 0; head < n_head_q; ++head) {
            for (uint32_t d = 0; d < 256; ++d) {
                q_host[(size_t(query) * n_head_q + head) * 256 + d] =
                    0.04f * std::sin(float((query + 1) * (head + 1) * (d + 3)) * 0.013f);
            }
        }
    }
    std::vector<int64_t> native_positions(row_capacity, -1);
    std::vector<uint8_t> native_mask(row_capacity, 0);
    for (uint32_t row = 0; row < active_rows; ++row) {
        native_positions[row] = row;
        native_mask[row] = 1;
    }
    std::vector<int64_t> query_positions(query_count);
    for (uint32_t query = 0; query < query_count; ++query) {
        query_positions[query] = active_rows - 1 + query;
    }
    std::vector<float> expected = cpu_interleaved_turbo4_oracle(q_host,
        q_head_stride, size_t(n_head_q) * q_head_stride, k_host, v_host,
        pages.data(), active_pages, active_rows, native_positions, native_mask,
        query_positions, query_count, n_head_q, n_head_kv, head_bytes, row_bytes,
        layer_page_stride, 1.0f / std::sqrt(256.0f));

    float * q_device = nullptr;
    char * k_device = nullptr;
    char * v_device = nullptr;
    float * output_device = nullptr;
    ggml_cuda_fattn_turbo4_page * pages_device = nullptr;
    ggml_cuda_fattn_turbo4_row_lookup * row_lookup_device = nullptr;
    int64_t * native_positions_device = nullptr;
    uint8_t * native_mask_device = nullptr;
    int64_t * query_positions_device = nullptr;
    uint32_t * active_pages_device = nullptr;
    uint32_t * active_rows_device = nullptr;
    uint32_t * active_tail_device = nullptr;
    uint64_t * selection_generation_device = nullptr;
    float * split_device = nullptr;
    constexpr uint32_t split_partition_capacity = 1;
    const size_t split_partition_stride = size_t(query_count) * n_head_q *
        (2 + 256) * sizeof(float);
    const size_t split_page_state_stride = size_t(query_count) * n_head_q *
        page_capacity * 2 * sizeof(float);
    cuda_check(cudaMalloc(&q_device, q_host.size() * sizeof(float)), "slot-stride q allocation");
    cuda_check(cudaMalloc(&k_device, k_host.size()), "slot-stride k allocation");
    cuda_check(cudaMalloc(&v_device, v_host.size()), "slot-stride v allocation");
    cuda_check(cudaMalloc(&output_device, expected.size() * sizeof(float)), "slot-stride output allocation");
    cuda_check(cudaMalloc(&pages_device, pages.size() * sizeof(pages[0])), "slot-stride pages allocation");
    cuda_check(cudaMalloc(&row_lookup_device, row_lookup.size() * sizeof(row_lookup[0])), "slot-stride lookup allocation");
    cuda_check(cudaMalloc(&native_positions_device, native_positions.size() * sizeof(int64_t)), "slot-stride positions allocation");
    cuda_check(cudaMalloc(&native_mask_device, native_mask.size()), "slot-stride mask allocation");
    cuda_check(cudaMalloc(&query_positions_device, query_positions.size() * sizeof(int64_t)), "slot-stride query allocation");
    cuda_check(cudaMalloc(&active_pages_device, sizeof(uint32_t)), "slot-stride active pages allocation");
    cuda_check(cudaMalloc(&active_rows_device, sizeof(uint32_t)), "slot-stride active rows allocation");
    cuda_check(cudaMalloc(&active_tail_device, sizeof(uint32_t)), "slot-stride active tail allocation");
    cuda_check(cudaMalloc(&selection_generation_device, sizeof(uint64_t)), "slot-stride generation allocation");
    if (query_count != 1) {
        cuda_check(cudaMalloc(&split_device,
            size_t(split_partition_capacity) * (split_partition_stride + split_page_state_stride)),
            "slot-stride split scratch allocation");
    }
    cuda_check(cudaMemcpy(q_device, q_host.data(), q_host.size() * sizeof(float), cudaMemcpyHostToDevice), "slot-stride q copy");
    cuda_check(cudaMemcpy(k_device, k_host.data(), k_host.size(), cudaMemcpyHostToDevice), "slot-stride k copy");
    cuda_check(cudaMemcpy(v_device, v_host.data(), v_host.size(), cudaMemcpyHostToDevice), "slot-stride v copy");
    cuda_check(cudaMemcpy(pages_device, pages.data(), pages.size() * sizeof(pages[0]), cudaMemcpyHostToDevice), "slot-stride pages copy");
    cuda_check(cudaMemcpy(row_lookup_device, row_lookup.data(), row_lookup.size() * sizeof(row_lookup[0]), cudaMemcpyHostToDevice), "slot-stride lookup copy");
    cuda_check(cudaMemcpy(native_positions_device, native_positions.data(), native_positions.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "slot-stride positions copy");
    cuda_check(cudaMemcpy(native_mask_device, native_mask.data(), native_mask.size(), cudaMemcpyHostToDevice), "slot-stride mask copy");
    cuda_check(cudaMemcpy(query_positions_device, query_positions.data(), query_positions.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "slot-stride query copy");
    cuda_check(cudaMemcpy(active_pages_device, &active_pages, sizeof(active_pages), cudaMemcpyHostToDevice), "slot-stride active pages copy");
    cuda_check(cudaMemcpy(active_rows_device, &active_rows, sizeof(active_rows), cudaMemcpyHostToDevice), "slot-stride active rows copy");
    const uint32_t active_tail = active_rows - 256;
    constexpr uint64_t selection_generation = 1;
    cuda_check(cudaMemcpy(active_tail_device, &active_tail, sizeof(active_tail), cudaMemcpyHostToDevice), "slot-stride active tail copy");
    cuda_check(cudaMemcpy(selection_generation_device, &selection_generation, sizeof(selection_generation), cudaMemcpyHostToDevice), "slot-stride generation copy");

    ggml_cuda_fattn_turbo4_paged_params params;
    params.q = q_device;
    params.output = output_device;
    params.q_head_stride_bytes = q_head_stride;
    params.q_query_stride_bytes = size_t(n_head_q) * q_head_stride;
    params.output_head_stride_bytes = 256 * sizeof(float);
    params.output_query_stride_bytes = size_t(n_head_q) * params.output_head_stride_bytes;
    params.type_k = GGML_TYPE_TURBO4_0;
    params.type_v = GGML_TYPE_TURBO4_0;
    params.head_dim_k = 256;
    params.head_dim_v = 256;
    params.k = k_device;
    params.v = v_device;
    params.k_row_stride_bytes = row_bytes;
    params.k_head_stride_bytes = head_bytes;
    params.k_page_stride_bytes = layer_page_stride;
    params.v_row_stride_bytes = row_bytes;
    params.v_head_stride_bytes = head_bytes;
    params.v_page_stride_bytes = layer_page_stride;
    params.pages_host = pages.data();
    params.pages_device = pages_device;
    params.compact_row_lookup_device = row_lookup_device;
    params.compact_row_lookup_capacity = row_capacity;
    params.native_positions_device = native_positions_device;
    params.native_mask_device = native_mask_device;
    params.query_positions_device = query_positions_device;
    params.active_page_count_host = &active_pages;
    params.active_row_count_host = &active_rows;
    params.active_page_count_device = active_pages_device;
    params.active_row_count_device = active_rows_device;
    params.active_tail_length_device = active_tail_device;
    params.selection_generation_device = selection_generation_device;
    params.split_kv_scratch = split_device;
    params.split_kv_partition_stride_bytes = split_device != nullptr ? split_partition_stride : 0;
    params.split_kv_page_state = split_device != nullptr
        ? reinterpret_cast<float *>(reinterpret_cast<char *>(split_device) + split_partition_stride)
        : nullptr;
    params.split_kv_page_state_head_stride_bytes = split_device != nullptr
        ? size_t(page_capacity) * 2 * sizeof(float) : 0;
    params.split_kv_page_state_query_stride_bytes =
        params.split_kv_page_state_head_stride_bytes * n_head_q;
    params.split_kv_page_state_partition_stride_bytes = split_device != nullptr
        ? split_page_state_stride : 0;
    params.split_kv_partition_capacity = split_device != nullptr ? split_partition_capacity : 0;
    params.split_kv_page_count = split_device != nullptr ? page_capacity : 0;
    params.page_capacity = page_capacity;
    params.row_capacity = row_capacity;
    params.n_pages = page_capacity;
    params.n_physical_pages = physical_slots;
    params.n_rows = row_capacity;
    params.n_head_q = n_head_q;
    params.n_head_kv = n_head_kv;
    params.n_query_tokens = query_count;
    params.n_batch = 1;
    params.scale = 1.0f / std::sqrt(256.0f);
    params.explicit_native_metadata = true;
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) ==
        ggml_cuda_fattn_turbo4_paged_status::ok);
    assert(ggml_cuda_fattn_turbo4_paged_last_dispatch_was_mma() == (query_count == 1));
    cuda_check(cudaDeviceSynchronize(), "slot-stride attention synchronize");
    std::vector<float> actual(expected.size());
    cuda_check(cudaMemcpy(actual.data(), output_device, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "slot-stride output readback");
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(std::isfinite(actual[i]));
        assert(std::fabs(actual[i] - expected[i]) < 3.0e-3f);
    }

    cudaFree(active_rows_device);
    cudaFree(active_pages_device);
    cudaFree(active_tail_device);
    cudaFree(selection_generation_device);
    cudaFree(split_device);
    cudaFree(query_positions_device);
    cudaFree(native_mask_device);
    cudaFree(native_positions_device);
    cudaFree(row_lookup_device);
    cudaFree(pages_device);
    cudaFree(output_device);
    cudaFree(v_device);
    cudaFree(k_device);
    cudaFree(q_device);
    std::fprintf(stderr,
        "cuda_pager_layer_slot_stride_regression: passed (Q=%u, pages=2, rows=%u, "
        "layer_page_stride=%zu, layer_slab_bytes=%zu)\n",
        query_count, active_rows, layer_page_stride, k_host.size());
}

static void run_pager_layer_slot_stride_regression(ggml_backend_t backend) {
    run_pager_layer_slot_stride_case(backend, 64, 347);
    // Cold B stalls on a one-query decode attention node after a 1,007-row
    // target append. Its direct attention view still carries two selected
    // pages, so exercise the same two-page tail with Q=1 as well.
    run_pager_layer_slot_stride_case(backend, 1, 347);
}

static void run_cuda_prefill_long_context_regression(ggml_backend_t backend) {
    constexpr uint32_t head_dim = 256;
    constexpr uint32_t query_tokens = 64;
    constexpr uint32_t query_heads = 6;
    constexpr uint32_t kv_heads = 1;
    constexpr uint32_t kv_rows = 32768;
    constexpr size_t row_bytes = 2 * sizeof(block_turbo4_0);

    const size_t kv_bytes = size_t(kv_rows) * row_bytes;
    std::vector<uint8_t> k_host(kv_bytes, 0);
    std::vector<uint8_t> v_host(kv_bytes, 0);
    for (uint32_t page = 0; page < kv_rows / 256; ++page) {
        fill_turbo4_page(k_host, size_t(page) * 256 * row_bytes, 8);
        fill_turbo4_page(v_host, size_t(page) * 256 * row_bytes, 9);
    }

    ggml_init_params init_params = { 32u * 1024u * 1024u, nullptr, true };
    ggml_context * ctx = ggml_init(init_params);
    assert(ctx != nullptr);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, query_tokens, query_heads);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, head_dim, kv_rows, kv_heads);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, head_dim, kv_rows, kv_heads);
    ggml_tensor * output = ggml_flash_attn_ext(ctx, q, k, v, nullptr,
        1.0f / std::sqrt(float(head_dim)), 0.0f, 0.0f);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, output);
    assert(ggml_cuda_flash_attn_ext_supported(0, output));

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);
    std::vector<float> q_host(size_t(query_heads) * query_tokens * head_dim, 0.0f);
    std::vector<float> output_host(ggml_nelements(output), NAN);
    ggml_backend_tensor_set(q, q_host.data(), 0, q_host.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_host.data(), 0, k_host.size());
    ggml_backend_tensor_set(v, v_host.data(), 0, v_host.size());

    const size_t f16_scratch_bytes = size_t(head_dim) * kv_rows * kv_heads * sizeof(uint16_t);
    std::fprintf(stderr,
        "cuda_prefill_long_context_regression: Q=[D=%u,U=%u,H=%u], K/V=[D=%u,L=%u,H=%u], "
        "q_nb=[%zu,%zu,%zu], kv_nb=[%zu,%zu,%zu], f16_scratch=[K=%zu,V=%zu], "
        "native_dynamic_shared_bytes=0\n",
        head_dim, query_tokens, query_heads, head_dim, kv_rows, kv_heads,
        q->nb[0], q->nb[1], q->nb[2], k->nb[0], k->nb[1], k->nb[2],
        f16_scratch_bytes, f16_scratch_bytes);

    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    cuda_check(cudaDeviceSynchronize(), "long-context Turbo4 prefill synchronize");
    ggml_backend_tensor_get(output, output_host.data(), 0, output_host.size() * sizeof(float));
    for (const float value : output_host) {
        assert(std::isfinite(value));
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::fprintf(stderr,
        "cuda_prefill_long_context_regression: passed (L=%u,U=%u,GQA=%u:%u)\n",
        kv_rows, query_tokens, query_heads, kv_heads);
}

int main(int argc, char ** argv) {
    const bool run_timing = argc == 2 && std::string(argv[1]) == "--timing";
    const bool run_long_context = argc == 2 &&
        std::string(argv[1]) == "--prefill-long-context-regression";
    assert(argc == 1 || run_timing || run_long_context);
    assert(ggml_cuda_fattn_turbo4_page_table_valid(nullptr, 0, 0) == false);
    assert(ggml_cuda_fattn_turbo4_query_tile_for_count(1) == 1);
    assert(ggml_cuda_fattn_turbo4_query_tile_for_count(2) == 2);
    assert(ggml_cuda_fattn_turbo4_query_tile_for_count(3) == 4);
    assert(ggml_cuda_fattn_turbo4_query_tile_for_count(4) == 4);
    assert(ggml_cuda_fattn_turbo4_query_tile_for_count(5) == 8);
    assert(ggml_cuda_fattn_turbo4_query_tile_for_count(8) == 8);
    assert(ggml_cuda_fattn_turbo4_query_tile_for_count(9) == 16);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    assert(backend != nullptr);

    if (run_long_context) {
        run_cuda_prefill_long_context_regression(backend);
        ggml_backend_free(backend);
        return 0;
    }

    run_multigroup_turbo4_numerics(backend);
    run_pager_layer_slot_stride_regression(backend);
    run_split_tail_growth_regression(backend);

    // Keep the fixture small enough to coexist with a loaded full-model
    // candidate. The non-trivial GQA ratio also exercises the runtime path.
    constexpr uint32_t n_head_q = 4;
    constexpr uint32_t n_head_kv = 1;
    constexpr uint32_t n_pages = 4;
    constexpr uint32_t n_rows = 530;
    // B is intentionally larger than one CTA tile. The dispatcher must keep
    // shared memory bounded while covering both the 64->65 and 256->257
    // boundaries plus the largest bounded sweep shape.
    const uint32_t max_query_tokens = run_timing ? 128 : 64;
    constexpr size_t row_bytes = 2 * sizeof(block_turbo4_0);
    constexpr size_t page_stride = 256 * row_bytes;
    constexpr uint32_t n_physical_pages = 8;

    // The selected order is logical 3, 0, 1 while physical slots are 5, 1, 7.
    // Logical page 2 is a one-row tail, the final selected page is a 17-row
    // tail, and native positions are supplied in compact order rather than
    // physical order.
    const ggml_cuda_fattn_turbo4_page pages[n_pages] = {
        { 3, 5,   0,  17, 768 },
        { 0, 1,  17, 256,   0 },
        { 1, 7, 273, 256, 256 },
        { 2, 3, 529,   1, 512 },
    };
    assert(ggml_cuda_fattn_turbo4_page_table_valid(pages, n_pages, n_rows));
    assert(ggml_cuda_fattn_turbo4_page_table_valid(
        pages, n_pages, n_rows, 256, n_physical_pages));
    // The selected table is deliberately non-contiguous in both logical page
    // order and physical slot order. It is valid because compact rows remain
    // contiguous; corrupting either the slot bound or compact range must fail
    // before a CUDA launch.
    auto invalid_slot = std::vector<ggml_cuda_fattn_turbo4_page>(pages, pages + n_pages);
    invalid_slot[0].source_physical_slot = n_physical_pages;
    assert(!ggml_cuda_fattn_turbo4_page_table_valid(
        invalid_slot.data(), n_pages, n_rows, 256, n_physical_pages));
    auto invalid_compact = std::vector<ggml_cuda_fattn_turbo4_page>(pages, pages + n_pages);
    invalid_compact[1].compact_row_begin += 1;
    assert(!ggml_cuda_fattn_turbo4_page_table_valid(
        invalid_compact.data(), n_pages, n_rows, 256, n_physical_pages));

    std::vector<uint8_t> k_host(n_head_kv * n_physical_pages * page_stride, 0);
    std::vector<uint8_t> v_host(n_head_kv * n_physical_pages * page_stride, 0);
    for (uint32_t kv_head = 0; kv_head < n_head_kv; ++kv_head) {
        const size_t head_offset = size_t(kv_head) * n_physical_pages * page_stride;
        fill_turbo4_page(k_host, head_offset + 5 * page_stride, 8);
        fill_turbo4_page(k_host, head_offset + 1 * page_stride, 8);
        fill_turbo4_page(k_host, head_offset + 7 * page_stride, 8);
        fill_turbo4_page(k_host, head_offset + 3 * page_stride, 8);
        fill_turbo4_page(v_host, head_offset + 5 * page_stride, 8);
        fill_turbo4_page(v_host, head_offset + 1 * page_stride, 9);
        fill_turbo4_page(v_host, head_offset + 7 * page_stride, 10);
        fill_turbo4_page(v_host, head_offset + 3 * page_stride, 11);
    }

    std::vector<float> q_host(max_query_tokens * n_head_q * 256, 0.0f);
    std::vector<int64_t> native_positions;
    std::vector<uint8_t> native_mask(n_rows, 1);
    native_positions.reserve(n_rows);
    for (uint32_t row = 0; row < 17; ++row) native_positions.push_back(768 + row);
    for (uint32_t row = 0; row < 256; ++row) native_positions.push_back(row);
    for (uint32_t row = 0; row < 256; ++row) native_positions.push_back(256 + row);
    native_positions.push_back(512);
    native_positions[528] = 1200; // causal rejection must use native metadata.
    native_mask[17] = 0;            // mask one compact row in the permuted page.
    std::vector<int64_t> query_positions_host(max_query_tokens, 1200);
    query_positions_host[0] = 1000;
    query_positions_host[1] = 1100;
    for (uint32_t query = 2; query < max_query_tokens; ++query) {
        query_positions_host[query] = 1200;
    }

    float * q_device = nullptr;
    char * k_device = nullptr;
    char * v_device = nullptr;
    ggml_cuda_fattn_turbo4_page * pages_device = nullptr;
    int64_t * native_positions_device = nullptr;
    uint8_t * native_mask_device = nullptr;
    int64_t * query_position_device = nullptr;
    float * output_device = nullptr;
    float * page_mass_device = nullptr;
    float * partial_state_device = nullptr;
    uint16_t * routing_min_device = nullptr;
    uint16_t * routing_max_device = nullptr;
    uint32_t * routing_indices_device = nullptr;
    uint32_t * routing_count_device = nullptr;
    float * routing_scores_device = nullptr;
    cuda_check(cudaMalloc(&q_device, q_host.size() * sizeof(float)), "q allocation");
    cuda_check(cudaMalloc(&k_device, k_host.size()), "k allocation");
    cuda_check(cudaMalloc(&v_device, v_host.size()), "v allocation");
    cuda_check(cudaMalloc(&pages_device, sizeof(pages)), "page table allocation");
    cuda_check(cudaMalloc(&native_positions_device, native_positions.size() * sizeof(int64_t)), "positions allocation");
    cuda_check(cudaMalloc(&native_mask_device, native_mask.size()), "mask allocation");
    cuda_check(cudaMalloc(&query_position_device,
        query_positions_host.size() * sizeof(query_positions_host[0])), "query position allocation");
    cuda_check(cudaMalloc(&output_device, q_host.size() * sizeof(float)), "output allocation");
    cuda_check(cudaMalloc(&page_mass_device, max_query_tokens * n_head_q * 4 * sizeof(float)), "mass allocation");
    cuda_check(cudaMalloc(&partial_state_device, max_query_tokens * n_head_q * (2 + 256) * sizeof(float)), "partial state allocation");
    constexpr uint32_t routing_subblocks = 4;
    constexpr uint32_t routing_top_k = 2;
    constexpr size_t routing_page_stride = size_t(routing_subblocks) * 256 * sizeof(uint16_t);
    std::vector<uint16_t> routing_min(n_pages * routing_page_stride / sizeof(uint16_t), 0);
    std::vector<uint16_t> routing_max = routing_min;
    const uint16_t routing_scores_host[] = { 0x4400, 0x4200, 0x4000, 0x3c00 };
    for (uint32_t page = 0; page < n_pages; ++page) {
        for (uint32_t block = 0; block < routing_subblocks; ++block) {
            routing_max[size_t(page) * routing_page_stride / sizeof(uint16_t) +
                size_t(block) * 256 * 2] = routing_scores_host[page];
        }
    }
    cuda_check(cudaMalloc(&routing_min_device, routing_min.size() * sizeof(uint16_t)), "routing min allocation");
    cuda_check(cudaMalloc(&routing_max_device, routing_max.size() * sizeof(uint16_t)), "routing max allocation");
    cuda_check(cudaMalloc(&routing_indices_device, size_t(max_query_tokens) * n_head_kv * routing_top_k * sizeof(uint32_t)), "routing index allocation");
    cuda_check(cudaMalloc(&routing_count_device, size_t(max_query_tokens) * n_head_kv * sizeof(uint32_t)), "routing count allocation");
    cuda_check(cudaMalloc(&routing_scores_device, size_t(max_query_tokens) * n_head_kv * routing_top_k * sizeof(float)), "routing score allocation");
    cuda_check(cudaMemcpy(q_device, q_host.data(), q_host.size() * sizeof(float), cudaMemcpyHostToDevice), "q copy");
    cuda_check(cudaMemcpy(k_device, k_host.data(), k_host.size(), cudaMemcpyHostToDevice), "k copy");
    cuda_check(cudaMemcpy(v_device, v_host.data(), v_host.size(), cudaMemcpyHostToDevice), "v copy");
    cuda_check(cudaMemcpy(pages_device, pages, sizeof(pages), cudaMemcpyHostToDevice), "page table copy");
    cuda_check(cudaMemcpy(native_positions_device, native_positions.data(), native_positions.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "positions copy");
    cuda_check(cudaMemcpy(native_mask_device, native_mask.data(), native_mask.size(), cudaMemcpyHostToDevice), "mask copy");
    cuda_check(cudaMemcpy(query_position_device, query_positions_host.data(),
        query_positions_host.size() * sizeof(query_positions_host[0]), cudaMemcpyHostToDevice), "query position copy");
    cuda_check(cudaMemcpy(routing_min_device, routing_min.data(), routing_min.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "routing min copy");
    cuda_check(cudaMemcpy(routing_max_device, routing_max.data(), routing_max.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "routing max copy");

    ggml_cuda_fattn_turbo4_paged_params params;
    params.q = q_device;
    params.output = output_device;
    params.q_head_stride_bytes = 256 * sizeof(float);
    params.q_query_stride_bytes = n_head_q * params.q_head_stride_bytes;
    params.output_head_stride_bytes = 256 * sizeof(float);
    params.output_query_stride_bytes = n_head_q * params.output_head_stride_bytes;
    params.type_k = GGML_TYPE_TURBO4_0;
    params.type_v = GGML_TYPE_TURBO4_0;
    params.head_dim_k = 256;
    params.head_dim_v = 256;
    params.k = k_device;
    params.v = v_device;
    params.k_row_stride_bytes = row_bytes;
    params.k_head_stride_bytes = page_stride * n_physical_pages;
    params.k_page_stride_bytes = page_stride;
    params.v_row_stride_bytes = row_bytes;
    params.v_head_stride_bytes = page_stride * n_physical_pages;
    params.v_page_stride_bytes = page_stride;
    params.pages_host = pages;
    params.pages_device = pages_device;
    params.native_positions_device = native_positions_device;
    params.native_mask_device = native_mask_device;
    params.query_positions_device = query_position_device;
    params.n_pages = n_pages;
    params.n_physical_pages = n_physical_pages;
    params.n_rows = n_rows;
    params.n_head_q = n_head_q;
    params.n_head_kv = n_head_kv;
    params.n_query_tokens = 1;
    params.n_batch = 1;
    params.scale = 1.0f / std::sqrt(256.0f);
    params.routing_range_min = routing_min_device;
    params.routing_range_max = routing_max_device;
    params.routing_selected_indices = routing_indices_device;
    params.routing_selected_count = routing_count_device;
    params.routing_selected_scores = routing_scores_device;
    params.routing_range_page_stride_bytes = routing_page_stride;
    params.routing_selected_stride_bytes = routing_top_k * sizeof(uint32_t);
    params.routing_count_stride_bytes = sizeof(uint32_t);
    params.routing_subblocks = routing_subblocks;
    params.routing_vector_dim = 256;
    params.routing_top_k = routing_top_k;

    std::vector<float> routing_q(q_host.size(), 0.0f);
    for (uint32_t head = 0; head < n_head_q; ++head) routing_q[size_t(head) * 256] = 1.0f;
    cuda_check(cudaMemcpy(q_device, routing_q.data(), routing_q.size() * sizeof(float), cudaMemcpyHostToDevice), "routing query copy");
    assert(ggml_cuda_flash_attn_ext_paged_turbo4_route(
            *static_cast<ggml_backend_cuda_context *>(backend->context), params) ==
        ggml_cuda_fattn_turbo4_paged_status::ok);
    cuda_check(cudaDeviceSynchronize(), "routing selector");
    uint32_t selected_indices[routing_top_k] = {};
    uint32_t selected_count = 0;
    cuda_check(cudaMemcpy(selected_indices, routing_indices_device,
        sizeof(selected_indices), cudaMemcpyDeviceToHost), "routing indices readback");
    cuda_check(cudaMemcpy(&selected_count, routing_count_device,
        sizeof(selected_count), cudaMemcpyDeviceToHost), "routing count readback");
    assert(selected_count == routing_top_k);
    assert(selected_indices[0] == 0 && selected_indices[1] == 1);
    std::fill(q_host.begin(), q_host.end(), 0.0f);
    cuda_check(cudaMemcpy(q_device, q_host.data(), q_host.size() * sizeof(float), cudaMemcpyHostToDevice), "routing query restore copy");
    // The direct attention oracle below intentionally covers the complete
    // fixture page list; route consumption is exercised by the selector call
    // above and the production dispatcher accepts these fields optionally.
    params.routing_range_min = nullptr;
    params.routing_range_max = nullptr;
    params.routing_selected_indices = nullptr;
    params.routing_selected_count = nullptr;
    params.routing_selected_scores = nullptr;

    params.type_k = GGML_TYPE_F16;
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::unsupported_type);
    params.type_k = GGML_TYPE_TURBO4_0;
    params.head_dim_v = 128;
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::unsupported_shape);
    params.head_dim_v = 256;

    cudaEvent_t timing_start = nullptr;
    cudaEvent_t timing_stop = nullptr;
    const cudaStream_t stream = static_cast<ggml_backend_cuda_context *>(backend->context)->stream();
    cuda_check(cudaEventCreate(&timing_start), "timing start allocation");
    cuda_check(cudaEventCreate(&timing_stop), "timing stop allocation");
    const float elapsed_ms = run_timing
        ? time_paged_attention(backend, params, stream, timing_start, timing_stop, 5, 20)
        : 0.0f;
    if (!run_timing) {
        assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) ==
            ggml_cuda_fattn_turbo4_paged_status::ok);
    }
    std::fprintf(stderr, "paged Turbo4 query tile: %.3f ms (Q=%u, %u Q heads, %u KV heads, 530 selected rows)\n",
        elapsed_ms, params.n_query_tokens, n_head_q, n_head_kv);
    cuda_check(cudaDeviceSynchronize(), "direct page attention");
    std::vector<float> output_without_mass(q_host.size());
    cuda_check(cudaMemcpy(output_without_mass.data(), output_device, output_without_mass.size() * sizeof(float), cudaMemcpyDeviceToHost), "output readback");

    // The repaired serial implementation is an explicit oracle only. Verify
    // the production cooperative result against it for the one-query shape.
    params.reference_kernel = true;
    const std::vector<float> reference_canary(q_host.size(), -12345.0f);
    cuda_check(cudaMemcpy(output_device, reference_canary.data(),
        reference_canary.size() * sizeof(float), cudaMemcpyHostToDevice), "reference output canary copy");
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) ==
        ggml_cuda_fattn_turbo4_paged_status::ok);
    cuda_check(cudaDeviceSynchronize(), "reference page attention");
    std::vector<float> reference_output(q_host.size());
    cuda_check(cudaMemcpy(reference_output.data(), output_device,
        reference_output.size() * sizeof(float), cudaMemcpyDeviceToHost), "reference output readback");
    for (size_t i = 0; i < size_t(n_head_q) * 256; ++i) {
        assert(std::fabs(output_without_mass[i] - reference_output[i]) < 2.0e-6f);
    }
    params.reference_kernel = false;

    constexpr float c8 = 0.011353f;
    constexpr float c9 = 0.034311f;
    constexpr float c10 = 0.058069f;
    constexpr float c11 = 0.083365f;
    const float expected_528 = (17.0f * c8 + 255.0f * c9 + 255.0f * c10 + c11) / 528.0f;
    const float expected_529 = (17.0f * c8 + 255.0f * c9 + 256.0f * c10 + c11) / 529.0f;

    // Multiquery verification exercises the direct fused MMA prefill tiles for
    // Q16/Q64 (and Q128 in timing mode). Verify each query's causal position
    // and guard the following output query with a canary so adjacent query
    // results cannot alias.
    const std::vector<float> output_canary(q_host.size(), -12345.0f);
    const std::vector<uint32_t> query_counts = run_timing
        ? std::vector<uint32_t>{ 1, 2, 3, 4, 5, 8, 16, 64, 128 }
        : std::vector<uint32_t>{ 1, 2, 3, 4, 5, 8, 16, 64 };
    const std::vector<float> oracle = cpu_selected_oracle(v_host, pages, n_pages,
        native_positions, native_mask,
        query_positions_host,
        n_rows, n_head_q, n_head_kv, n_physical_pages, page_stride, max_query_tokens, true);
    for (const uint32_t query_count : query_counts) {
        cuda_check(cudaMemcpy(output_device, output_canary.data(),
            output_canary.size() * sizeof(float), cudaMemcpyHostToDevice), "output canary copy");
        params.n_query_tokens = query_count;
        assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
        cuda_check(cudaDeviceSynchronize(), "multiquery page attention");
        std::vector<float> multiquery_output(q_host.size());
        cuda_check(cudaMemcpy(multiquery_output.data(), output_device,
            multiquery_output.size() * sizeof(float), cudaMemcpyDeviceToHost), "multiquery output readback");
        for (uint32_t query = 0; query < max_query_tokens; ++query) {
            for (uint32_t head = 0; head < n_head_q; ++head) {
                for (uint32_t d = 0; d < 256; ++d) {
                    const size_t index = (size_t(query) * n_head_q + head) * 256 + d;
                    if (query < query_count) {
                        assert(std::fabs(multiquery_output[index] - oracle[index]) < 2.0e-4f);
                    } else {
                        assert(multiquery_output[index] == -12345.0f);
                    }
                }
            }
        }
    }
    params.n_query_tokens = 0;
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::unsupported_shape);
    params.n_query_tokens = 1;

    // An all-masked row must produce a finite zero output and zero page mass;
    // it must not read an uninitialized online-softmax state.
    const int64_t all_masked_query[] = { -1 };
    cuda_check(cudaMemcpy(query_position_device, all_masked_query, sizeof(all_masked_query),
        cudaMemcpyHostToDevice), "all-masked query position copy");
    cuda_check(cudaMemcpy(output_device, output_canary.data(), output_canary.size() * sizeof(float),
        cudaMemcpyHostToDevice), "all-masked output canary copy");
    params.reduce_page_mass = true;
    params.page_mass = page_mass_device;
    params.page_mass_head_stride_bytes = 4 * sizeof(float);
    params.page_mass_query_stride_bytes = n_head_q * params.page_mass_head_stride_bytes;
    params.page_mass_logical_count = 4;
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
    cuda_check(cudaDeviceSynchronize(), "all-masked page attention");
    std::vector<float> all_masked_output(q_host.size());
    std::vector<float> all_masked_mass(size_t(max_query_tokens) * n_head_q * 4);
    cuda_check(cudaMemcpy(all_masked_output.data(), output_device,
        all_masked_output.size() * sizeof(float), cudaMemcpyDeviceToHost), "all-masked output readback");
    cuda_check(cudaMemcpy(all_masked_mass.data(), page_mass_device,
        all_masked_mass.size() * sizeof(float), cudaMemcpyDeviceToHost), "all-masked mass readback");
    for (uint32_t d = 0; d < n_head_q * 256; ++d) {
        assert(all_masked_output[d] == 0.0f);
    }
    for (uint32_t d = 0; d < n_head_q * 4; ++d) {
        assert(all_masked_mass[d] == 0.0f);
    }
    cuda_check(cudaMemcpy(query_position_device, query_positions_host.data(),
        query_positions_host.size() * sizeof(query_positions_host[0]),
        cudaMemcpyHostToDevice), "query position restore copy");

    // Reuse the same resident physical page as a one-page tail case. This
    // exercises the compact bound independently of the many-page permutation.
    params.n_pages = 1;
    params.n_rows = 17;
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
    cuda_check(cudaDeviceSynchronize(), "one-page tail attention");
    std::vector<float> one_page_output(q_host.size());
    cuda_check(cudaMemcpy(one_page_output.data(), output_device, one_page_output.size() * sizeof(float), cudaMemcpyDeviceToHost), "one-page output readback");
    for (uint32_t head = 0; head < n_head_q; ++head) {
        for (uint32_t d = 0; d < 256; ++d) {
            if (std::fabs(one_page_output[head * 256 + d] - 0.011353f) >= 2.0e-6f) {
                std::fprintf(stderr, "one-page output mismatch: head=%u d=%u value=%.9g\n",
                    head, d, one_page_output[head * 256 + d]);
                std::abort();
            }
        }
    }

    // Two-append lifetime regression.  The first append leaves a partial
    // logical page in slot 3.  The second append grows the tail while the
    // descriptor is copied on the same CUDA stream and moves that tail to
    // slot 5.  A rejected descriptor update is then rolled back, and the
    // freed slot 3 is reused by the next append.  Keep the launches queued
    // until the final fence: this is the small CUDA analogue of decode()
    // entering memory_update() before the previous graph's fence.
    const std::array<ggml_cuda_fattn_turbo4_page, 4> append_one = {{
        { 0, 1, 0,   256, 0 },
        { 1, 3, 256, 64, 256 },
        { UINT32_MAX, UINT32_MAX, 0, 0, 0 },
        { UINT32_MAX, UINT32_MAX, 0, 0, 0 },
    }};
    const std::array<ggml_cuda_fattn_turbo4_page, 4> append_two = {{
        { 0, 1, 0,   256, 0 },
        { 1, 5, 256, 128, 256 },
        { UINT32_MAX, UINT32_MAX, 0, 0, 0 },
        { UINT32_MAX, UINT32_MAX, 0, 0, 0 },
    }};
    const std::array<ggml_cuda_fattn_turbo4_page, 4> append_bad = {{
        { 0, 1, 0,   256, 0 },
        { 1, 1, 256, 192, 256 }, // duplicate physical slot: reject and roll back
        { UINT32_MAX, UINT32_MAX, 0, 0, 0 },
        { UINT32_MAX, UINT32_MAX, 0, 0, 0 },
    }};
    const std::array<ggml_cuda_fattn_turbo4_page, 4> append_three = {{
        { 0, 1, 0,   256, 0 },
        { 1, 3, 256, 192, 256 }, // slot 3 is reused after append_two
        { UINT32_MAX, UINT32_MAX, 0, 0, 0 },
        { UINT32_MAX, UINT32_MAX, 0, 0, 0 },
    }};
    assert(ggml_cuda_fattn_turbo4_page_table_valid(append_one.data(), 2, 320));
    assert(ggml_cuda_fattn_turbo4_page_table_valid(append_two.data(), 2, 384));
    assert(!ggml_cuda_fattn_turbo4_page_table_valid(append_bad.data(), 2, 448));
    assert(ggml_cuda_fattn_turbo4_page_table_valid(append_three.data(), 2, 448));
    params.pages_host = append_one.data();
    params.n_pages = 2;
    params.n_rows = 320;
    params.n_query_tokens = 1;
    cuda_check(cudaMemcpyAsync(pages_device, append_one.data(), sizeof(append_one),
        cudaMemcpyHostToDevice, stream), "append one page table copy");
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) ==
        ggml_cuda_fattn_turbo4_paged_status::ok);

    params.pages_host = append_two.data();
    params.n_rows = 384;
    cuda_check(cudaMemcpyAsync(pages_device, append_two.data(), sizeof(append_two),
        cudaMemcpyHostToDevice, stream), "append two page table copy");
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) ==
        ggml_cuda_fattn_turbo4_paged_status::ok);

    params.pages_host = append_bad.data();
    params.n_rows = 448;
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) ==
        ggml_cuda_fattn_turbo4_paged_status::invalid_page_table);

    params.pages_host = append_three.data();
    cuda_check(cudaMemcpyAsync(pages_device, append_three.data(), sizeof(append_three),
        cudaMemcpyHostToDevice, stream), "append three page table copy");
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) ==
        ggml_cuda_fattn_turbo4_paged_status::ok);
    cuda_check(cudaDeviceSynchronize(), "two-append tail rollback and slot reuse");
    std::vector<float> append_output(q_host.size());
    cuda_check(cudaMemcpy(append_output.data(), output_device,
        append_output.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "two-append output readback");
    for (const float value : append_output) assert(std::isfinite(value));

    params.pages_host = pages;
    cuda_check(cudaMemcpy(pages_device, pages, sizeof(pages), cudaMemcpyHostToDevice),
        "restore baseline page table after append regression");
    params.n_pages = n_pages;
    params.n_rows = n_rows;
    params.n_query_tokens = max_query_tokens;

    // Decode and native-MTP verification use the shape-sized default rather
    // than paying for the prefill tile. Keep these timings separate from the
    // large-query tile sweep below.
    const uint32_t decode_query_counts[] = { 1, 2, 3, 5 };
    for (const uint32_t query_count : decode_query_counts) {
        params.n_query_tokens = query_count;
        cuda_check(cudaEventRecord(timing_start, stream), "decode shape timing start record");
        assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
        cuda_check(cudaEventRecord(timing_stop, stream), "decode shape timing stop record");
        cuda_check(cudaEventSynchronize(timing_stop), "decode shape timing stop synchronize");
        float decode_shape_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&decode_shape_ms, timing_start, timing_stop), "decode shape timing readback");
        std::fprintf(stderr, "paged Turbo4 decode shape: %.3f ms (Q=%u, default tile, 530 selected rows)\n",
            decode_shape_ms, query_count);
    }
    params.n_query_tokens = max_query_tokens;

    params.reduce_page_mass = true;
    params.page_mass = page_mass_device;
    params.page_mass_head_stride_bytes = 4 * sizeof(float);
    params.page_mass_query_stride_bytes = n_head_q * params.page_mass_head_stride_bytes;
    params.page_mass_logical_count = 4;
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
    cuda_check(cudaDeviceSynchronize(), "direct page attention with mass");
    std::vector<float> output_with_mass(q_host.size());
    std::vector<float> page_mass(max_query_tokens * n_head_q * 4);
    cuda_check(cudaMemcpy(output_with_mass.data(), output_device, output_with_mass.size() * sizeof(float), cudaMemcpyDeviceToHost), "output readback with mass");
    cuda_check(cudaMemcpy(page_mass.data(), page_mass_device, page_mass.size() * sizeof(float), cudaMemcpyDeviceToHost), "mass readback");

    for (size_t i = 0; i < n_head_q * 256; ++i) {
        assert(std::isfinite(output_without_mass[i]));
        // The ordinary route now uses the fused MMA shape dispatch while the
        // page-mass route intentionally stays on the descriptor-aware
        // reduction path; compare their equivalent outputs within FP32
        // accumulation tolerance.
        assert(std::fabs(output_without_mass[i] - output_with_mass[i]) < 1.0e-4f);
    }
    for (uint32_t query = 0; query < max_query_tokens; ++query) {
        const float expected = query >= 2 ? expected_529 : expected_528;
        const float denominator = query >= 2 ? 529.0f : 528.0f;
        for (uint32_t head = 0; head < n_head_q; ++head) {
            for (uint32_t d = 0; d < 256; ++d) {
                const size_t index = (size_t(query) * n_head_q + head) * 256 + d;
                assert(std::fabs(output_with_mass[index] - expected) < 2.0e-6f);
            }
            const size_t mass_base = (size_t(query) * n_head_q + head) * 4;
            assert(std::fabs(page_mass[mass_base + 0] - 255.0f / denominator) < 1.0e-5f);
            const float expected_page1 = query >= 2 ? 256.0f : 255.0f;
            assert(std::fabs(page_mass[mass_base + 1] - expected_page1 / denominator) < 1.0e-5f);
            assert(std::fabs(page_mass[mass_base + 2] - 1.0f / denominator) < 1.0e-5f);
            assert(std::fabs(page_mass[mass_base + 3] - 17.0f / denominator) < 1.0e-5f);
        }
    }

    // The split-KV path uses the same [m,l,o] contract as exact page waves.
    // Exercise the serial fallback, two partitions, and the shape-selected
    // three-partition case against the serial result, including global page
    // mass after the merge.
    constexpr uint32_t split_capacity = 2;
    constexpr uint32_t split_page_count = n_pages;
    const size_t split_state_head_stride = (2 + 256) * sizeof(float);
    const size_t split_state_query_stride = n_head_q * split_state_head_stride;
    const size_t split_state_partition_stride = max_query_tokens * split_state_query_stride;
    const size_t split_page_head_stride = split_page_count * 2 * sizeof(float);
    const size_t split_page_query_stride = n_head_q * split_page_head_stride;
    const size_t split_page_partition_stride = max_query_tokens * split_page_query_stride;
    float * split_state_device = nullptr;
    float * split_page_state_device = nullptr;
    cuda_check(cudaMalloc(&split_state_device,
        split_capacity * split_state_partition_stride), "split state allocation");
    cuda_check(cudaMalloc(&split_page_state_device,
        split_capacity * split_page_partition_stride), "split page state allocation");
    const std::vector<float> serial_mass = page_mass;
    const std::vector<float> serial_output = output_with_mass;

    // Bounded launch sweep for the direct query tile. Keep the largest tile
    // as the graph default, but retain measurements for the two smaller
    // candidates so tile choice is evidence-driven rather than hard-coded by
    // the title of the task.
    const uint32_t query_tiles[] = { 16, 32, 64 };
    const uint32_t timed_query_counts[] = { 16, 64 };
    for (const uint32_t query_count : timed_query_counts) {
        params.n_query_tokens = query_count;
        for (const uint32_t query_tile : query_tiles) {
            params.query_tile_tokens = query_tile;
            if (!run_timing) {
                continue;
            }
            cuda_check(cudaEventRecord(timing_start, stream), "query tile timing start record");
            assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
            cuda_check(cudaEventRecord(timing_stop, stream), "query tile timing stop record");
            cuda_check(cudaEventSynchronize(timing_stop), "query tile timing stop synchronize");
            float query_tile_ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&query_tile_ms, timing_start, timing_stop), "query tile timing readback");
            std::fprintf(stderr, "paged Turbo4 tile sweep: %.3f ms (tile_q %u, %u Q tokens, 530 selected rows)\n",
                query_tile_ms, query_tile, query_count);
        }
    }
    params.query_tile_tokens = 0;
    params.n_query_tokens = max_query_tokens;
    params.split_kv_scratch = split_state_device;
    params.split_kv_partition_stride_bytes = split_state_partition_stride;
    params.split_kv_page_state = split_page_state_device;
    params.split_kv_page_state_head_stride_bytes = split_page_head_stride;
    params.split_kv_page_state_query_stride_bytes = split_page_query_stride;
    params.split_kv_page_state_partition_stride_bytes = split_page_partition_stride;
    params.split_kv_partition_capacity = split_capacity;
    params.split_kv_page_count = split_page_count;
    for (uint32_t requested_partitions = 1; requested_partitions <= split_capacity; ++requested_partitions) {
        params.split_kv_partition_capacity = requested_partitions;
        cuda_check(cudaEventRecord(timing_start, stream), "split timing start record");
        assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
        cuda_check(cudaEventRecord(timing_stop, stream), "split timing stop record");
        cuda_check(cudaEventSynchronize(timing_stop), "split timing stop synchronize");
        float split_elapsed_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&split_elapsed_ms, timing_start, timing_stop), "split timing readback");
        std::fprintf(stderr, "paged Turbo4 split-KV: %.3f ms (requested capacity %u)\n",
            split_elapsed_ms, requested_partitions);
        cuda_check(cudaDeviceSynchronize(), "split-KV page attention");
        std::vector<float> split_output(output_with_mass.size());
        std::vector<float> split_mass(page_mass.size());
        cuda_check(cudaMemcpy(split_output.data(), output_device,
            split_output.size() * sizeof(float), cudaMemcpyDeviceToHost), "split output readback");
        cuda_check(cudaMemcpy(split_mass.data(), page_mass_device,
            split_mass.size() * sizeof(float), cudaMemcpyDeviceToHost), "split mass readback");
        for (size_t i = 0; i < serial_output.size(); ++i) {
            assert(std::fabs(split_output[i] - serial_output[i]) < 2.0e-6f);
        }
        for (size_t i = 0; i < serial_mass.size(); ++i) {
            assert(std::isfinite(split_mass[i]));
            if (requested_partitions == 1) {
                assert(std::fabs(split_mass[i] - serial_mass[i]) < 1.0e-5f);
            }
        }
    }

    // Capture same-shape reference and cooperative controls after CUDA module
    // warm-up.  The first timing above includes first-use compilation on some
    // drivers, so this pair is the useful kernel-only comparison for the
    // receipt.
    params.split_kv_scratch = nullptr;
    params.split_kv_partition_stride_bytes = 0;
    params.split_kv_page_state = nullptr;
    params.split_kv_page_state_head_stride_bytes = 0;
    params.split_kv_page_state_query_stride_bytes = 0;
    params.split_kv_page_state_partition_stride_bytes = 0;
    params.split_kv_partition_capacity = 0;
    params.split_kv_page_count = 0;
    params.reference_kernel = true;
    cuda_check(cudaEventRecord(timing_start, stream), "serial comparison timing start record");
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
    cuda_check(cudaEventRecord(timing_stop, stream), "serial comparison timing stop record");
    cuda_check(cudaEventSynchronize(timing_stop), "serial comparison timing stop synchronize");
    float serial_comparison_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&serial_comparison_ms, timing_start, timing_stop), "serial comparison timing readback");
    std::fprintf(stderr, "paged Turbo4 reference control: %.3f ms (%u Q tokens, 530 selected rows)\n",
        serial_comparison_ms, max_query_tokens);

    params.reference_kernel = false;
    params.split_kv_scratch = split_state_device;
    params.split_kv_partition_stride_bytes = split_state_partition_stride;
    params.split_kv_page_state = split_page_state_device;
    params.split_kv_page_state_head_stride_bytes = split_page_head_stride;
    params.split_kv_page_state_query_stride_bytes = split_page_query_stride;
    params.split_kv_page_state_partition_stride_bytes = split_page_partition_stride;
    params.split_kv_partition_capacity = split_capacity;
    params.split_kv_page_count = split_page_count;
    cuda_check(cudaEventRecord(timing_start, stream), "split comparison timing start record");
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
    cuda_check(cudaEventRecord(timing_stop, stream), "split comparison timing stop record");
    cuda_check(cudaEventSynchronize(timing_stop), "split comparison timing stop synchronize");
    float split_comparison_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&split_comparison_ms, timing_start, timing_stop), "split comparison timing readback");
    std::fprintf(stderr, "paged Turbo4 split control: %.3f ms (%u Q tokens, 530 selected rows, capacity %u)\n",
        split_comparison_ms, max_query_tokens, split_capacity);

    params.split_kv_scratch = nullptr;
    params.split_kv_partition_stride_bytes = 0;
    params.split_kv_page_state = nullptr;
    params.split_kv_page_state_head_stride_bytes = 0;
    params.split_kv_page_state_query_stride_bytes = 0;
    params.split_kv_page_state_partition_stride_bytes = 0;
    params.split_kv_partition_capacity = 0;
    params.split_kv_page_count = 0;

    // The exact page-wave path consumes unnormalized [m, l, o] state. It may
    // omit the normalized output entirely, which keeps each cold wave's
    // staging bounded to Turbo4 pages plus this per-head partial state.
    params.output = nullptr;
    params.output_head_stride_bytes = 0;
    params.partial_state = partial_state_device;
    params.partial_state_head_stride_bytes = (2 + 256) * sizeof(float);
    params.partial_state_query_stride_bytes = n_head_q * params.partial_state_head_stride_bytes;
    params.write_partial_state = true;
    params.reduce_page_mass = false;
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
    cuda_check(cudaDeviceSynchronize(), "partial page attention");
    std::vector<float> partial_state(max_query_tokens * n_head_q * (2 + 256));
    cuda_check(cudaMemcpy(partial_state.data(), partial_state_device,
        partial_state.size() * sizeof(float), cudaMemcpyDeviceToHost), "partial state readback");
    for (uint32_t query = 0; query < max_query_tokens; ++query) {
        for (uint32_t head = 0; head < n_head_q; ++head) {
            const float * state = partial_state.data() +
                (size_t(query) * n_head_q + head) * (2 + 256);
            assert(std::fabs(state[0]) < 1.0e-6f);
            assert(std::fabs(state[1] - (query >= 2 ? 529.0f : 528.0f)) < 1.0e-4f);
            for (uint32_t d = 0; d < 256; ++d) assert(std::isfinite(state[2 + d]));
        }
    }

    // Split the same logical coverage into two waves and merge the first
    // device-owned [m,l,o] state into the second wave.  The second descriptor
    // list starts at compact row 17, so this also checks that merge uses the
    // supplied logical row spans rather than assuming every wave starts at 0.
    params.pages_host = pages;
    params.pages_device = pages_device;
    params.n_pages = 1;
    params.n_rows = 17;
    params.partial_state_input = nullptr;
    params.merge_partial_state = false;
    params.output = nullptr;
    params.output_head_stride_bytes = 0;
    params.output_query_stride_bytes = 0;
    // Exercise the explicit noncaptured cold-upload boundary. The fixture
    // payload is identical to the resident backing, so the numerical oracle
    // remains unchanged while the H2D staging copy is still required.
    params.host_upload = k_host.data();
    params.host_upload_bytes = k_host.size();
    params.upload_destination = k_device;
    params.upload_capacity_bytes = k_host.size();
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
    cuda_check(cudaDeviceSynchronize(), "first split wave state");

    const ggml_cuda_fattn_turbo4_page second_wave_pages[2] = {
        { 0, 1, 0, 256, 0 },
        { 1, 7, 256, 256, 256 },
    };
    ggml_cuda_fattn_turbo4_page * second_wave_pages_device = nullptr;
    cuda_check(cudaMalloc(&second_wave_pages_device, sizeof(second_wave_pages)),
        "second wave page table allocation");
    cuda_check(cudaMemcpy(second_wave_pages_device, second_wave_pages,
        sizeof(second_wave_pages), cudaMemcpyHostToDevice), "second wave page table copy");
    params.pages_host = second_wave_pages;
    params.pages_device = second_wave_pages_device;
    params.n_pages = 2;
    params.n_rows = 512;
    params.native_positions_device += 17;
    params.native_mask_device += 17;
    params.partial_state_input = partial_state_device;
    params.partial_state_input_head_stride_bytes = params.partial_state_head_stride_bytes;
    params.partial_state_input_query_stride_bytes = params.partial_state_query_stride_bytes;
    params.merge_partial_state = true;
    params.output = output_device;
    params.output_head_stride_bytes = 256 * sizeof(float);
    params.output_query_stride_bytes = n_head_q * params.output_head_stride_bytes;
    const auto merged_status = ggml_cuda_flash_attn_ext_paged_turbo4(backend, params);
    if (merged_status != ggml_cuda_fattn_turbo4_paged_status::ok) {
        std::fprintf(stderr, "merged split wave status: %s\n",
            ggml_cuda_fattn_turbo4_paged_status_name(merged_status));
        std::abort();
    }
    cuda_check(cudaDeviceSynchronize(), "merged split wave state");
    std::vector<float> merged_output(q_host.size());
    cuda_check(cudaMemcpy(merged_output.data(), output_device,
        merged_output.size() * sizeof(float), cudaMemcpyDeviceToHost), "merged output readback");
    cuda_check(cudaMemcpy(partial_state.data(), partial_state_device,
        partial_state.size() * sizeof(float), cudaMemcpyDeviceToHost), "merged state readback");
    const float merged_expected_527 = (17.0f * c8 + 255.0f * c9 + 255.0f * c10) / 527.0f;
    const float merged_expected_528 = (17.0f * c8 + 255.0f * c9 + 256.0f * c10) / 528.0f;
    for (uint32_t query = 0; query < max_query_tokens; ++query) {
        const float expected = query >= 2 ? merged_expected_528 : merged_expected_527;
        for (uint32_t head = 0; head < n_head_q; ++head) {
            const float * state = partial_state.data() +
                (size_t(query) * n_head_q + head) * (2 + 256);
            assert(std::fabs(state[1] - (query >= 2 ? 528.0f : 527.0f)) < 1.0e-4f);
            for (uint32_t d = 0; d < 256; ++d) {
                const size_t index = (size_t(query) * n_head_q + head) * 256 + d;
                assert(std::fabs(merged_output[index] - expected) < 2.0e-6f);
                assert(std::isfinite(state[2 + d]));
            }
        }
    }

    if (run_timing) {
        // Restore the full selected descriptor after the split-wave checks so
        // all three measurements use the same Q/K/V bytes and row order.
        params.pages_host = pages;
        params.pages_device = pages_device;
        params.native_positions_device = native_positions_device;
        params.native_mask_device = native_mask_device;
        params.n_pages = n_pages;
        params.n_rows = n_rows;
        params.n_query_tokens = max_query_tokens;
        params.output = output_device;
        params.output_head_stride_bytes = 256 * sizeof(float);
        params.output_query_stride_bytes = n_head_q * params.output_head_stride_bytes;
        params.page_mass = nullptr;
        params.page_mass_head_stride_bytes = 0;
        params.page_mass_query_stride_bytes = 0;
        params.page_mass_logical_count = 0;
        params.reduce_page_mass = false;
        params.partial_state = nullptr;
        params.partial_state_input = nullptr;
        params.write_partial_state = false;
        params.merge_partial_state = false;
        params.query_tile_tokens = 0;

        const std::vector<uint8_t> packed_k = pack_selected_storage(k_host, pages,
            n_pages, n_rows, n_head_kv, n_physical_pages, page_stride, row_bytes);
        const std::vector<uint8_t> packed_v = pack_selected_storage(v_host, pages,
            n_pages, n_rows, n_head_kv, n_physical_pages, page_stride, row_bytes);
        for (const uint32_t query_count : { 16u, 64u, max_query_tokens }) {
            params.n_query_tokens = query_count;
            const float direct_ms = time_paged_attention(backend, params, stream,
                timing_start, timing_stop, 5, 20);
            const std::vector<float> q_timing(q_host.begin(),
                q_host.begin() + size_t(query_count) * n_head_q * 256);
            const float contiguous_ms = time_dense_fa(backend, q_timing, k_host, v_host,
                packed_k, packed_v, pages, n_pages, n_rows, n_physical_pages,
                n_head_q, n_head_kv, page_stride, row_bytes, false);
            const float packed_ms = time_dense_fa(backend, q_timing, k_host, v_host,
                packed_k, packed_v, pages, n_pages, n_rows, n_physical_pages,
                n_head_q, n_head_kv, page_stride, row_bytes, true);
            std::fprintf(stderr, "timing table (U=%u, selected rows=%u, warmups=5, iterations=20)\n",
                query_count, n_rows);
            std::fprintf(stderr, "  fused paged MMA direct: %.3f ms\n", direct_ms);
            std::fprintf(stderr, "  contiguous Turbo4 FA:   %.3f ms\n", contiguous_ms);
            std::fprintf(stderr, "  non-contiguous pack+FA: %.3f ms\n", packed_ms);
        }
        time_large_prefill_cases(backend, stream, timing_start, timing_stop);
    }
    cudaFree(second_wave_pages_device);

    cudaFree(split_page_state_device);
    cudaFree(split_state_device);

    cudaEventDestroy(timing_stop);
    cudaEventDestroy(timing_start);
    cudaFree(page_mass_device);
    cudaFree(partial_state_device);
    cudaFree(output_device);
    cudaFree(query_position_device);
    cudaFree(native_mask_device);
    cudaFree(native_positions_device);
    cudaFree(pages_device);
    cudaFree(v_device);
    cudaFree(k_device);
    cudaFree(q_device);
    ggml_backend_free(backend);
    return 0;
}
