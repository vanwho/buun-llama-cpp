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
    constexpr uint32_t n_pages = 4;
    constexpr uint32_t n_rows = 530;
    constexpr uint32_t max_query_tokens = 3;
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

    std::vector<uint8_t> k_host(n_physical_pages * page_stride, 0);
    std::vector<uint8_t> v_host(n_physical_pages * page_stride, 0);
    fill_turbo4_page(k_host, 5 * page_stride, 8);
    fill_turbo4_page(k_host, 1 * page_stride, 8);
    fill_turbo4_page(k_host, 7 * page_stride, 8);
    fill_turbo4_page(k_host, 3 * page_stride, 8);
    fill_turbo4_page(v_host, 5 * page_stride, 8);
    fill_turbo4_page(v_host, 1 * page_stride, 9);
    fill_turbo4_page(v_host, 7 * page_stride, 10);
    fill_turbo4_page(v_host, 3 * page_stride, 11);

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
    std::fprintf(stderr, "paged Turbo4 query tile: %.3f ms (four Q heads, 530 selected rows)\n", elapsed_ms);
    cuda_check(cudaDeviceSynchronize(), "direct page attention");
    std::vector<float> output_without_mass(q_host.size());
    cuda_check(cudaMemcpy(output_without_mass.data(), output_device, output_without_mass.size() * sizeof(float), cudaMemcpyDeviceToHost), "output readback");

    constexpr float c8 = 0.011353f;
    constexpr float c9 = 0.034311f;
    constexpr float c10 = 0.058069f;
    constexpr float c11 = 0.083365f;
    const float expected_528 = (17.0f * c8 + 255.0f * c9 + 255.0f * c10 + c11) / 528.0f;
    const float expected_529 = (17.0f * c8 + 255.0f * c9 + 256.0f * c10 + c11) / 529.0f;

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
                        const float expected = query == 2 ? expected_529 : expected_528;
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
        const float expected = query == 2 ? expected_529 : expected_528;
        const float denominator = query == 2 ? 529.0f : 528.0f;
        for (uint32_t head = 0; head < n_head_q; ++head) {
            for (uint32_t d = 0; d < 256; ++d) {
                const size_t index = (size_t(query) * n_head_q + head) * 256 + d;
                assert(std::fabs(output_with_mass[index] - expected) < 2.0e-6f);
            }
            const size_t mass_base = (size_t(query) * n_head_q + head) * 4;
            assert(std::fabs(page_mass[mass_base + 0] - 255.0f / denominator) < 1.0e-5f);
            const float expected_page1 = query == 2 ? 256.0f : 255.0f;
            assert(std::fabs(page_mass[mass_base + 1] - expected_page1 / denominator) < 1.0e-5f);
            assert(std::fabs(page_mass[mass_base + 2] - 1.0f / denominator) < 1.0e-5f);
            assert(std::fabs(page_mass[mass_base + 3] - 17.0f / denominator) < 1.0e-5f);
        }
    }

    // The split-KV path uses the same [m,l,o] contract as exact page waves.
    // Exercise the serial fallback, two partitions, and the shape-selected
    // three-partition case against the serial result, including global page
    // mass after the merge.
    constexpr uint32_t split_capacity = 3;
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
            assert(std::fabs(split_mass[i] - serial_mass[i]) < 1.0e-5f);
        }
    }

    // Capture a same-shape serial control after CUDA module warm-up.  The
    // first timing above includes first-use compilation on some drivers, so
    // this pair is the useful kernel-only comparison for the receipt.
    params.split_kv_scratch = nullptr;
    params.split_kv_partition_stride_bytes = 0;
    params.split_kv_page_state = nullptr;
    params.split_kv_page_state_head_stride_bytes = 0;
    params.split_kv_page_state_query_stride_bytes = 0;
    params.split_kv_page_state_partition_stride_bytes = 0;
    params.split_kv_partition_capacity = 0;
    params.split_kv_page_count = 0;
    cuda_check(cudaEventRecord(timing_start, stream), "serial comparison timing start record");
    assert(ggml_cuda_flash_attn_ext_paged_turbo4(backend, params) == ggml_cuda_fattn_turbo4_paged_status::ok);
    cuda_check(cudaEventRecord(timing_stop, stream), "serial comparison timing stop record");
    cuda_check(cudaEventSynchronize(timing_stop), "serial comparison timing stop synchronize");
    float serial_comparison_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&serial_comparison_ms, timing_start, timing_stop), "serial comparison timing readback");
    std::fprintf(stderr, "paged Turbo4 serial control: %.3f ms (three Q tokens, 530 selected rows)\n",
        serial_comparison_ms);

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
    std::fprintf(stderr, "paged Turbo4 split control: %.3f ms (three Q tokens, 530 selected rows, capacity 3)\n",
        split_comparison_ms);

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
            assert(std::fabs(state[1] - (query == 2 ? 529.0f : 528.0f)) < 1.0e-4f);
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
        const float expected = query == 2 ? merged_expected_528 : merged_expected_527;
        for (uint32_t head = 0; head < n_head_q; ++head) {
            const float * state = partial_state.data() +
                (size_t(query) * n_head_q + head) * (2 + 256);
            assert(std::fabs(state[1] - (query == 2 ? 528.0f : 527.0f)) < 1.0e-4f);
            for (uint32_t d = 0; d < 256; ++d) {
                const size_t index = (size_t(query) * n_head_q + head) * 256 + d;
                assert(std::fabs(merged_output[index] - expected) < 2.0e-6f);
                assert(std::isfinite(state[2 + d]));
            }
        }
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
