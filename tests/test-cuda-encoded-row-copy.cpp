#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static double time_graph(ggml_backend_t backend, ggml_cgraph * graph) {
    for (int i = 0; i < 2; ++i) assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    std::vector<double> samples;
    for (int sample = 0; sample < 5; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 8; ++i) assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        const auto stop = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
        samples.push_back(elapsed_ms / 8.0);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

static void run_operation_timing(ggml_backend_t backend) {
    constexpr int64_t width = 256;
    constexpr int64_t hot_rows = 4096;
    constexpr int64_t update_rows = 256;
    ggml_context * ctx = ggml_init({ 64u * 1024u * 1024u, nullptr, true });
    assert(ctx != nullptr);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, update_rows);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, update_rows);
    ggml_tensor * canonical = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO4_0, width, hot_rows);
    ggml_tensor * duplicate = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO4_0, width, hot_rows);
    ggml_tensor * copied = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO4_0, width, hot_rows);
    ggml_set_name(canonical, "timing_cache_k");
    ggml_set_name(duplicate, "timing_cache_k_duplicate");
    ggml_set_name(copied, "timing_cache_k_copy");
    ggml_tensor * canonical_write = ggml_set_rows(ctx, canonical, input, indices);
    ggml_tensor * duplicate_write = ggml_set_rows(ctx, duplicate, input, indices);
    ggml_tensor * copied_write = ggml_set_rows_from_rows(ctx, copied, canonical_write, indices, indices);
    ggml_cgraph * duplicate_graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(duplicate_graph, canonical_write);
    ggml_build_forward_expand(duplicate_graph, duplicate_write);
    ggml_cgraph * copy_graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(copy_graph, canonical_write);
    ggml_build_forward_expand(copy_graph, copied_write);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);
    std::vector<float> values(size_t(width * update_rows));
    std::vector<int64_t> rows(static_cast<size_t>(update_rows));
    for (size_t i = 0; i < values.size(); ++i) values[i] = float(int(i % 97) - 48) * 0.02f;
    for (int64_t i = 0; i < update_rows; ++i) rows[size_t(i)] = i * 13 % hot_rows;
    ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(indices, rows.data(), 0, ggml_nbytes(indices));
    const double duplicate_ms = time_graph(backend, duplicate_graph);
    const double copy_ms = time_graph(backend, copy_graph);
    std::fprintf(stderr,
        "encoded_append_timing: H=%lld U=%lld Q=3 codec=Turbo4 duplicate_encode_ms=%.6f encoded_copy_ms=%.6f\n",
        (long long) hot_rows, (long long) update_rows, duplicate_ms, copy_ms);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) return 77;

    constexpr int64_t width = 256;
    constexpr int64_t physical_rows = 8;
    constexpr int64_t batch = 3;
    ggml_context * ctx = ggml_init({ 16u * 1024u * 1024u, nullptr, true });
    assert(ctx != nullptr);

    ggml_tensor * current = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, batch);
    ggml_tensor * current_v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, batch);
    ggml_tensor * source_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, batch);
    ggml_tensor * destination_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, batch);
    ggml_tensor * attention_source_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch);
    ggml_tensor * attention_destination_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch);
    ggml_tensor * canonical = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO4_0, width, physical_rows);
    ggml_tensor * packed = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO4_0, width, physical_rows);
    ggml_tensor * canonical_v = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO4_0, width, physical_rows);
    ggml_tensor * packed_v = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO4_0, width, physical_rows);
    ggml_tensor * q3_source = ggml_new_tensor_2d(ctx, GGML_TYPE_Q3_K, 256, physical_rows);
    ggml_tensor * q3_destination = ggml_new_tensor_2d(ctx, GGML_TYPE_Q3_K, 256, physical_rows);
    ggml_set_name(canonical, "encoded_cache_k");
    ggml_set_name(packed, "encoded_cache_k_packed");
    ggml_set_name(canonical_v, "encoded_cache_v");
    ggml_set_name(packed_v, "encoded_cache_v_packed");
    ggml_tensor * canonical_write = ggml_set_rows(ctx, canonical, current, source_indices);
    ggml_tensor * canonical_v_write = ggml_set_rows(ctx, canonical_v, current_v, source_indices);
    ggml_tensor * packed_write = ggml_set_rows_from_rows(ctx, packed, canonical_write,
        destination_indices, source_indices);
    ggml_tensor * packed_v_write = ggml_set_rows_from_rows(ctx, packed_v, canonical_v_write,
        destination_indices, source_indices);
    ggml_tensor * q3_copy = ggml_set_rows_from_rows(ctx, q3_destination, q3_source,
        destination_indices, source_indices);
    ggml_tensor * current_rows_k = ggml_get_rows(ctx, canonical_write, attention_source_indices);
    ggml_tensor * packed_rows_k = ggml_get_rows(ctx, packed_write, attention_destination_indices);
    ggml_tensor * current_rows_v = ggml_get_rows(ctx, canonical_v_write, attention_source_indices);
    ggml_tensor * packed_rows_v = ggml_get_rows(ctx, packed_v_write, attention_destination_indices);
    current_rows_k = ggml_reshape_3d(ctx, current_rows_k, width, batch, 1);
    packed_rows_k = ggml_reshape_3d(ctx, packed_rows_k, width, batch, 1);
    current_rows_v = ggml_reshape_3d(ctx, current_rows_v, width, batch, 1);
    packed_rows_v = ggml_reshape_3d(ctx, packed_rows_v, width, batch, 1);
    ggml_tensor * query = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, batch, 1);
    ggml_tensor * canonical_attention = ggml_flash_attn_ext(ctx, query,
        current_rows_k, current_rows_v, nullptr, 1.0f / std::sqrt(float(width)), 0.0f, 0.0f);
    ggml_tensor * packed_attention = ggml_flash_attn_ext(ctx, query,
        packed_rows_k, packed_rows_v, nullptr, 1.0f / std::sqrt(float(width)), 0.0f, 0.0f);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, packed_write);
    ggml_build_forward_expand(graph, packed_v_write);
    ggml_build_forward_expand(graph, q3_copy);
    ggml_build_forward_expand(graph, canonical_attention);
    ggml_build_forward_expand(graph, packed_attention);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);

    std::vector<float> input(size_t(width * batch));
    std::vector<float> input_v(size_t(width * batch));
    std::vector<float> q(static_cast<size_t>(width * batch), 0.0f);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = float(int(i % 73) - 36) * 0.03125f;
        input_v[i] = float(int(i % 47) - 23) * 0.046875f;
    }
    for (size_t i = 0; i < q.size(); ++i) q[i] = float(int(i % 31) - 15) * 0.025f;
    const int32_t attention_source[batch] = { 6, 1, 4 };
    const int32_t attention_destination[batch] = { 2, 0, 5 };
    int64_t physical[batch] = { 6, -1, -1 };
    int64_t compact[batch] = { 2, -1, -1 };
    std::vector<uint8_t> q3_source_bytes(ggml_nbytes(q3_source));
    std::vector<uint8_t> q3_canary(ggml_nbytes(q3_destination), 0xA5);
    for (size_t i = 0; i < q3_source_bytes.size(); ++i) q3_source_bytes[i] = uint8_t(i * 37 + 11);
    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(current_v, input_v.data(), 0, ggml_nbytes(current_v));
    ggml_backend_tensor_set(query, q.data(), 0, ggml_nbytes(query));
    ggml_backend_tensor_set(attention_source_indices, attention_source, 0, sizeof(attention_source));
    ggml_backend_tensor_set(attention_destination_indices, attention_destination, 0, sizeof(attention_destination));
    ggml_backend_tensor_set(source_indices, physical, 0, sizeof(physical));
    ggml_backend_tensor_set(destination_indices, compact, 0, sizeof(compact));
    ggml_backend_tensor_set(q3_source, q3_source_bytes.data(), 0, q3_source_bytes.size());
    ggml_backend_tensor_set(q3_destination, q3_canary.data(), 0, q3_canary.size());
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    const size_t row_bytes = ggml_row_size(GGML_TYPE_TURBO4_0, width);
    std::vector<uint8_t> canonical_bytes(ggml_nbytes(canonical));
    std::vector<uint8_t> packed_bytes(ggml_nbytes(packed));
    ggml_backend_tensor_get(canonical, canonical_bytes.data(), 0, canonical_bytes.size());
    ggml_backend_tensor_get(packed, packed_bytes.data(), 0, packed_bytes.size());
    assert(std::memcmp(canonical_bytes.data() + size_t(physical[0]) * row_bytes,
                       packed_bytes.data() + size_t(compact[0]) * row_bytes, row_bytes) == 0);

    // Acceptance growth from one row to two rows, followed by a rejected
    // provisional row. Negative indices preserve prior committed bytes.
    physical[1] = 1;
    compact[1] = 0;
    ggml_backend_tensor_set(source_indices, physical, 0, sizeof(physical));
    ggml_backend_tensor_set(destination_indices, compact, 0, sizeof(compact));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(canonical, canonical_bytes.data(), 0, canonical_bytes.size());
    ggml_backend_tensor_get(packed, packed_bytes.data(), 0, packed_bytes.size());
    assert(std::memcmp(canonical_bytes.data() + size_t(physical[0]) * row_bytes,
                       packed_bytes.data() + size_t(compact[0]) * row_bytes, row_bytes) == 0);
    assert(std::memcmp(canonical_bytes.data() + size_t(physical[1]) * row_bytes,
                       packed_bytes.data() + size_t(compact[1]) * row_bytes, row_bytes) == 0);
    const std::vector<uint8_t> committed_row(packed_bytes.begin() + size_t(compact[1]) * row_bytes,
                                              packed_bytes.begin() + size_t(compact[1] + 1) * row_bytes);
    physical[1] = -1;
    compact[1] = -1;
    physical[2] = 4;
    compact[2] = 5;
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(packed, packed_bytes.data(), 0, packed_bytes.size());
    assert(std::memcmp(committed_row.data(), packed_bytes.data() + size_t(compact[1] + 1) * row_bytes,
                       row_bytes) == 0);

    std::vector<uint8_t> q3_output(ggml_nbytes(q3_destination));
    ggml_backend_tensor_get(q3_destination, q3_output.data(), 0, q3_output.size());
    const size_t q3_row_bytes = ggml_row_size(GGML_TYPE_Q3_K, 256);
    assert(std::memcmp(q3_source_bytes.data() + size_t(physical[0]) * q3_row_bytes,
                       q3_output.data() + size_t(compact[0]) * q3_row_bytes,
                       q3_row_bytes) == 0);

    std::vector<float> canonical_attention_values(ggml_nelements(canonical_attention));
    std::vector<float> packed_attention_values(ggml_nelements(packed_attention));
    ggml_backend_tensor_get(canonical_attention, canonical_attention_values.data(), 0,
        ggml_nbytes(canonical_attention));
    ggml_backend_tensor_get(packed_attention, packed_attention_values.data(), 0,
        ggml_nbytes(packed_attention));
    for (size_t i = 0; i < canonical_attention_values.size(); ++i) {
        assert(std::abs(canonical_attention_values[i] - packed_attention_values[i]) <= 1e-6f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    run_operation_timing(backend);
    ggml_backend_free(backend);

    // The CPU path is deliberately a typed byte-storage check; it does not
    // execute TurboQuant inference or encode the Q3_K source rows.
    ggml_backend_t cpu = ggml_backend_cpu_init();
    assert(cpu != nullptr);
    ggml_context * cpu_ctx = ggml_init({ 4u * 1024u * 1024u, nullptr, true });
    assert(cpu_ctx != nullptr);
    ggml_tensor * cpu_src = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_Q3_K, 256, physical_rows);
    ggml_tensor * cpu_dst = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_Q3_K, 256, physical_rows);
    ggml_tensor * cpu_src_ids = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I64, batch);
    ggml_tensor * cpu_dst_ids = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I64, batch);
    ggml_tensor * cpu_copy = ggml_set_rows_from_rows(cpu_ctx, cpu_dst, cpu_src, cpu_dst_ids, cpu_src_ids);
    ggml_cgraph * cpu_graph = ggml_new_graph_custom(cpu_ctx, 16, false);
    ggml_build_forward_expand(cpu_graph, cpu_copy);
    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
    assert(cpu_buffer != nullptr);
    std::vector<uint8_t> cpu_source_bytes(ggml_nbytes(cpu_src));
    std::vector<uint8_t> cpu_output_bytes(ggml_nbytes(cpu_dst), 0x5A);
    for (size_t i = 0; i < cpu_source_bytes.size(); ++i) cpu_source_bytes[i] = uint8_t(i * 19 + 3);
    const int64_t cpu_source_rows[batch] = { 6, 1, 4 };
    const int64_t cpu_destination_rows[batch] = { 2, 0, 5 };
    ggml_backend_tensor_set(cpu_src, cpu_source_bytes.data(), 0, cpu_source_bytes.size());
    ggml_backend_tensor_set(cpu_dst, cpu_output_bytes.data(), 0, cpu_output_bytes.size());
    ggml_backend_tensor_set(cpu_src_ids, cpu_source_rows, 0, sizeof(cpu_source_rows));
    ggml_backend_tensor_set(cpu_dst_ids, cpu_destination_rows, 0, sizeof(cpu_destination_rows));
    assert(ggml_backend_graph_compute(cpu, cpu_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(cpu_dst, cpu_output_bytes.data(), 0, cpu_output_bytes.size());
    const size_t cpu_row_bytes = ggml_row_size(GGML_TYPE_Q3_K, 256);
    for (int64_t i = 0; i < batch; ++i) {
        assert(std::memcmp(cpu_source_bytes.data() + size_t(cpu_source_rows[i]) * cpu_row_bytes,
                           cpu_output_bytes.data() + size_t(cpu_destination_rows[i]) * cpu_row_bytes,
                           cpu_row_bytes) == 0);
    }
    ggml_backend_buffer_free(cpu_buffer);
    ggml_free(cpu_ctx);
    ggml_backend_free(cpu);

    std::fprintf(stderr, "cuda_encoded_row_copy: passed (fragmented rows, acceptance growth, rejected row, Q3 bytes, attention parity; CPU byte storage passed)\n");
    return 0;
}
