#include "ggml-cuda/fattn-paged-turbo4.cuh"
#include "ggml-backend-impl.h"

#include "ggml-cuda.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <vector>

static void cuda_check(cudaError_t status, const char * what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::abort();
    }
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

int main() {
    assert(ggml_cuda_fattn_turbo4_page_table_valid(nullptr, 0, 0) == false);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    assert(backend != nullptr);

    constexpr uint32_t n_head_q = 4;
    constexpr uint32_t n_head_kv = 1;
    constexpr uint32_t n_pages = 3;
    constexpr uint32_t n_rows = 529;
    constexpr uint32_t max_query_tokens = 3;
    constexpr size_t row_bytes = 2 * sizeof(block_turbo4_0);
    constexpr size_t page_stride = 256 * row_bytes;
    constexpr uint32_t n_physical_pages = 8;

    // The selected order is logical 3, 0, 1 while physical slots are 5, 1, 7.
    // Logical page 2 is absent, the final selected page is a 17-row tail, and
    // native positions are supplied in compact order rather than physical order.
    const ggml_cuda_fattn_turbo4_page pages[n_pages] = {
        { 3, 5,   0,  17, 768 },
        { 0, 1,  17, 256,   0 },
        { 1, 7, 273, 256, 256 },
    };
    assert(ggml_cuda_fattn_turbo4_page_table_valid(pages, n_pages, n_rows));

    std::vector<uint8_t> k_host(n_physical_pages * page_stride, 0);
    std::vector<uint8_t> v_host(n_physical_pages * page_stride, 0);
    fill_turbo4_page(k_host, 5 * page_stride, 8);
    fill_turbo4_page(k_host, 1 * page_stride, 8);
    fill_turbo4_page(k_host, 7 * page_stride, 8);
    fill_turbo4_page(v_host, 5 * page_stride, 8);
    fill_turbo4_page(v_host, 1 * page_stride, 9);
    fill_turbo4_page(v_host, 7 * page_stride, 10);

    std::vector<float> q_host(max_query_tokens * n_head_q * 256, 0.0f);
    std::vector<int64_t> native_positions;
    std::vector<uint8_t> native_mask(n_rows, 1);
    native_positions.reserve(n_rows);
    for (uint32_t row = 0; row < 17; ++row) native_positions.push_back(768 + row);
    for (uint32_t row = 0; row < 256; ++row) native_positions.push_back(row);
    for (uint32_t row = 0; row < 256; ++row) native_positions.push_back(256 + row);
    native_positions.back() = 1200; // causal rejection must use native metadata.
    native_mask[17] = 0;            // mask one compact row in the permuted page.
    const int64_t query_positions_host[max_query_tokens] = { 1000, 1100, 1200 };

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
    cuda_check(cudaMalloc(&q_device, q_host.size() * sizeof(float)), "q allocation");
    cuda_check(cudaMalloc(&k_device, k_host.size()), "k allocation");
    cuda_check(cudaMalloc(&v_device, v_host.size()), "v allocation");
    cuda_check(cudaMalloc(&pages_device, sizeof(pages)), "page table allocation");
    cuda_check(cudaMalloc(&native_positions_device, native_positions.size() * sizeof(int64_t)), "positions allocation");
    cuda_check(cudaMalloc(&native_mask_device, native_mask.size()), "mask allocation");
    cuda_check(cudaMalloc(&query_position_device, sizeof(query_positions_host)), "query position allocation");
    cuda_check(cudaMalloc(&output_device, q_host.size() * sizeof(float)), "output allocation");
    cuda_check(cudaMalloc(&page_mass_device, max_query_tokens * n_head_q * 4 * sizeof(float)), "mass allocation");
    cuda_check(cudaMalloc(&partial_state_device, max_query_tokens * n_head_q * (2 + 256) * sizeof(float)), "partial state allocation");
    cuda_check(cudaMemcpy(q_device, q_host.data(), q_host.size() * sizeof(float), cudaMemcpyHostToDevice), "q copy");
    cuda_check(cudaMemcpy(k_device, k_host.data(), k_host.size(), cudaMemcpyHostToDevice), "k copy");
    cuda_check(cudaMemcpy(v_device, v_host.data(), v_host.size(), cudaMemcpyHostToDevice), "v copy");
    cuda_check(cudaMemcpy(pages_device, pages, sizeof(pages), cudaMemcpyHostToDevice), "page table copy");
    cuda_check(cudaMemcpy(native_positions_device, native_positions.data(), native_positions.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "positions copy");
    cuda_check(cudaMemcpy(native_mask_device, native_mask.data(), native_mask.size(), cudaMemcpyHostToDevice), "mask copy");
    cuda_check(cudaMemcpy(query_position_device, query_positions_host, sizeof(query_positions_host), cudaMemcpyHostToDevice), "query position copy");

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
    cuda_check(cudaEventRecord(timing_start, stream), "timing start record");
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
    cuda_check(cudaEventRecord(timing_stop, stream), "timing stop record");
    cuda_check(cudaEventSynchronize(timing_stop), "timing stop synchronize");
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, timing_start, timing_stop), "timing readback");
    std::fprintf(stderr, "paged Turbo4 query tile: %.3f ms (four Q heads, 529 selected rows)\n", elapsed_ms);
    cuda_check(cudaDeviceSynchronize(), "direct page attention");
    std::vector<float> output_without_mass(q_host.size());
    cuda_check(cudaMemcpy(output_without_mass.data(), output_device, output_without_mass.size() * sizeof(float), cudaMemcpyDeviceToHost), "output readback");

    constexpr float c8 = 0.011353f;
    constexpr float c9 = 0.034311f;
    constexpr float c10 = 0.058069f;
    const float expected_527 = (17.0f * c8 + 255.0f * c9 + 255.0f * c10) / 527.0f;
    const float expected_528 = (17.0f * c8 + 255.0f * c9 + 256.0f * c10) / 528.0f;

    // Multiquery verification uses one CTA per head/query tile. Verify each
    // query's causal position and guard the following output query with a
    // canary so adjacent query results cannot alias.
    const std::vector<float> output_canary(q_host.size(), -12345.0f);
    for (uint32_t query_count = 1; query_count <= max_query_tokens; ++query_count) {
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
                        const float expected = query == 2 ? expected_528 : expected_527;
                        assert(std::fabs(multiquery_output[index] - expected) < 2.0e-6f);
                    } else {
                        assert(multiquery_output[index] == -12345.0f);
                    }
                }
            }
        }
    }
    params.n_query_tokens = 4;
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::unsupported_shape);
    params.n_query_tokens = 1;

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
    params.n_pages = n_pages;
    params.n_rows = n_rows;
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
        assert(output_without_mass[i] == output_with_mass[i]);
    }
    for (uint32_t query = 0; query < max_query_tokens; ++query) {
        const float expected = query == 2 ? expected_528 : expected_527;
        const float denominator = query == 2 ? 528.0f : 527.0f;
        for (uint32_t head = 0; head < n_head_q; ++head) {
            for (uint32_t d = 0; d < 256; ++d) {
                const size_t index = (size_t(query) * n_head_q + head) * 256 + d;
                assert(std::fabs(output_with_mass[index] - expected) < 2.0e-6f);
            }
            const size_t mass_base = (size_t(query) * n_head_q + head) * 4;
            assert(std::fabs(page_mass[mass_base + 0] - 255.0f / denominator) < 1.0e-5f);
            const float expected_page1 = query == 2 ? 256.0f : 255.0f;
            assert(std::fabs(page_mass[mass_base + 1] - expected_page1 / denominator) < 1.0e-5f);
            assert(std::fabs(page_mass[mass_base + 2] - 0.0f) < 1.0e-6f);
            assert(std::fabs(page_mass[mass_base + 3] - 17.0f / denominator) < 1.0e-5f);
        }
    }

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
            assert(std::fabs(state[1] - (query == 2 ? 528.0f : 527.0f)) < 1.0e-4f);
            for (uint32_t d = 0; d < 256; ++d) assert(std::isfinite(state[2 + d]));
        }
    }

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
