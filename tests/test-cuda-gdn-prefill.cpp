#include "models/models.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

struct run_result {
    std::vector<float> output;
    std::vector<float> snapshots;
    double setup_ms = 0.0;
    double run_ms = 0.0;
    double copy_ms = 0.0;
};

static std::vector<float> values(size_t count, uint32_t seed, float scale) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        seed = seed * 1664525u + 1013904223u;
        result[i] = (float(int((seed >> 8) % 2001) - 1000) / 1000.0f) * scale;
    }
    return result;
}

static run_result run_gdn(
        ggml_backend_t backend,
        int tokens,
        int heads_k,
        int heads_v,
        int snapshots,
        bool split,
        const std::vector<float> * initial_state = nullptr) {
    constexpr int d = 128;
    constexpr int sequences = 1;
    llm_graph_result graph_result(16384);
    llm_graph_params params{};
    params.gtype = LLM_GRAPH_TYPE_DEFAULT;
    params.hparams.n_layer_all = 1;
    params.res = &graph_result;
    params.cparams.n_rs_seq = snapshots - 1;
    llm_build_delta_net_base builder(params);
    ggml_context * ctx = graph_result.get_ctx();
    ggml_cgraph * graph = graph_result.get_gf();

    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads_k, tokens, sequences);
    auto * k = ggml_dup_tensor(ctx, q);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads_v, tokens, sequences);
    auto * gate = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, heads_v, tokens, sequences);
    auto * beta = ggml_dup_tensor(ctx, gate);
    auto * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, d, heads_v, sequences);

    const auto q_data = values(ggml_nelements(q), 1, 0.02f);
    const auto k_data = values(ggml_nelements(k), 2, 0.02f);
    const auto v_data = values(ggml_nelements(v), 3, 0.02f);
    const auto gate_data = values(ggml_nelements(gate), 4, 0.01f);
    const auto beta_data = values(ggml_nelements(beta), 5, 0.2f);
    const auto state_data = initial_state != nullptr ? *initial_state : values(ggml_nelements(state), 6, 0.001f);

    const auto t0 = std::chrono::steady_clock::now();
    ggml_tensor * result = nullptr;
    ggml_tensor * tail_result = nullptr;
    ggml_tensor * output = nullptr;
    int tail_tokens = 0;
    if (split && tokens >= 128 && tokens > snapshots) {
        tail_tokens = snapshots;
        const int prefix_tokens = tokens - tail_tokens;
        ggml_tensor * q_split = q;
        ggml_tensor * k_split = k;
        if (heads_k != heads_v) {
            q_split = ggml_repeat_4d(ctx, q, d, heads_v, tokens, sequences);
            k_split = ggml_repeat_4d(ctx, k, d, heads_v, tokens, sequences);
        }
        const auto time_view = [&](ggml_tensor * tensor, int count, int start) {
            return ggml_view_4d(ctx, tensor,
                    tensor->ne[0], tensor->ne[1], count, tensor->ne[3],
                    tensor->nb[1], tensor->nb[2], tensor->nb[3],
                    (size_t) start * tensor->nb[2]);
        };
        auto prefix = builder.build_delta_net_chunking(
                time_view(q_split, prefix_tokens, 0), time_view(k_split, prefix_tokens, 0),
                time_view(v, prefix_tokens, 0), time_view(gate, prefix_tokens, 0),
                time_view(beta, prefix_tokens, 0), state, 0);
        tail_result = ggml_gated_delta_net(ctx,
                time_view(q_split, tail_tokens, prefix_tokens),
                time_view(k_split, tail_tokens, prefix_tokens),
                time_view(v, tail_tokens, prefix_tokens),
                time_view(gate, tail_tokens, prefix_tokens),
                time_view(beta, tail_tokens, prefix_tokens), prefix.second, snapshots);
        auto * tail_output = ggml_view_4d(ctx, tail_result, d, heads_v, tail_tokens, sequences,
                ggml_row_size(tail_result->type, d), ggml_row_size(tail_result->type, d * heads_v),
                ggml_row_size(tail_result->type, d * heads_v * tail_tokens), 0);
        auto * prefix_output = prefix.first;
        output = ggml_concat(ctx, prefix_output, tail_output, 2);
        result = tail_result;
        ggml_build_forward_expand(graph, output);
        ggml_build_forward_expand(graph, tail_result);
    } else {
        tail_tokens = tokens;
        result = ggml_gated_delta_net(ctx, q, k, v, gate, beta, state, snapshots);
        output = ggml_view_4d(ctx, result, d, heads_v, tokens, sequences,
                ggml_row_size(result->type, d), ggml_row_size(result->type, d * heads_v),
                ggml_row_size(result->type, d * heads_v * tokens), 0);
        ggml_build_forward_expand(graph, output);
        ggml_build_forward_expand(graph, result);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);
    ggml_backend_buffer_clear(buffer, 0);
    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, k_data.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, v_data.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(gate, gate_data.data(), 0, ggml_nbytes(gate));
    ggml_backend_tensor_set(beta, beta_data.data(), 0, ggml_nbytes(beta));
    ggml_backend_tensor_set(state, state_data.data(), 0, ggml_nbytes(state));
    const auto t1 = std::chrono::steady_clock::now();

    const auto t2 = std::chrono::steady_clock::now();
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    const auto t3 = std::chrono::steady_clock::now();

    run_result output_result;
    output_result.output.resize(ggml_nelements(output));
    ggml_backend_tensor_get(output, output_result.output.data(), 0, ggml_nbytes(output));
    const size_t state_count = size_t(d) * d * heads_v;
    output_result.snapshots.resize(state_count * snapshots);
    ggml_tensor * state_view = ggml_view_3d(ctx, result,
            state_count, 1, snapshots,
            state_count * sizeof(float), state_count * sizeof(float),
            size_t(d) * heads_v * tail_tokens * sizeof(float));
    ggml_backend_tensor_get(state_view, output_result.snapshots.data(), 0, ggml_nbytes(state_view));
    const auto t4 = std::chrono::steady_clock::now();
    output_result.setup_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    output_result.run_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    output_result.copy_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    ggml_backend_buffer_free(buffer);
    return output_result;
}

static void compare(const run_result & reference, const run_result & actual,
        ggml_backend_t backend, int tokens, int heads_k, int heads_v, int snapshots) {
    GGML_ASSERT(reference.output.size() == actual.output.size());
    GGML_ASSERT(reference.snapshots.size() == actual.snapshots.size());
    float max_output_error = 0.0f;
    float max_state_error = 0.0f;
    for (size_t i = 0; i < reference.output.size(); ++i) {
        max_output_error = std::max(max_output_error, std::abs(reference.output[i] - actual.output[i]));
    }
    for (size_t i = 0; i < reference.snapshots.size(); ++i) {
        max_state_error = std::max(max_state_error, std::abs(reference.snapshots[i] - actual.snapshots[i]));
    }
    GGML_ASSERT(max_output_error <= 4e-3f);
    GGML_ASSERT(max_state_error <= 4e-3f);
    if (tokens >= 128 && snapshots > 1) {
        const size_t state_count = size_t(128) * 128 * heads_v;
        float max_next_error = 0.0f;
        for (int accepted = 0; accepted < snapshots; ++accepted) {
            std::vector<float> serial_state(reference.snapshots.begin() + accepted * state_count,
                    reference.snapshots.begin() + (accepted + 1) * state_count);
            std::vector<float> split_state(actual.snapshots.begin() + accepted * state_count,
                    actual.snapshots.begin() + (accepted + 1) * state_count);
            auto serial_next = run_gdn(backend, 1, heads_k, heads_v, 1, false, &serial_state);
            auto split_next = run_gdn(backend, 1, heads_k, heads_v, 1, false, &split_state);
            for (size_t i = 0; i < serial_next.output.size(); ++i) {
                max_next_error = std::max(max_next_error,
                        std::abs(serial_next.output[i] - split_next.output[i]));
            }
            for (size_t i = 0; i < serial_next.snapshots.size(); ++i) {
                max_next_error = std::max(max_next_error,
                        std::abs(serial_next.snapshots[i] - split_next.snapshots[i]));
            }
        }
        GGML_ASSERT(max_next_error <= 4e-3f);
        std::printf("PASS partial-acceptance next-decode N=%d K=%d Hk=%d Hv=%d max_error=%g\n",
                tokens, snapshots, heads_k, heads_v, max_next_error);
    }
    std::printf("PASS gdn-prefill N=%d K=%d Hk=%d Hv=%d max_output=%g max_snapshot=%g\n",
            tokens, snapshots, heads_k, heads_v, max_output_error, max_state_error);
}

int main() {
    if (!ggml_backend_cuda_get_device_count()) {
        return 77;
    }
    llm_graph_params graph_key_a{};
    llm_graph_params graph_key_b{};
    GGML_ASSERT(graph_key_a.allow_reuse(graph_key_b));
    graph_key_b.cparams.n_rs_seq = 1;
    GGML_ASSERT(!graph_key_a.allow_reuse(graph_key_b));
    graph_key_b = {};
    graph_key_b.gtype = LLM_GRAPH_TYPE_DECODER_MTP;
    GGML_ASSERT(!graph_key_a.allow_reuse(graph_key_b));

    auto * backend = ggml_backend_cuda_init(0);
    GGML_ASSERT(backend != nullptr);

    for (int heads_v : {4, 6}) {
        const int heads_k = heads_v == 6 ? 2 : 4;
        for (int snapshots : {1, 3}) {
            for (int tokens : {1, 3, 64, 128, 256, 257}) {
                auto serial = run_gdn(backend, tokens, heads_k, heads_v, snapshots, false);
                auto split = run_gdn(backend, tokens, heads_k, heads_v, snapshots, true);
                compare(serial, split, backend, tokens, heads_k, heads_v, snapshots);
            }
        }
    }

    // Model geometry uses H_v/H_k=3; time the full graph build and execution
    // separately so scratch/graph overhead is visible beside device work.
    std::vector<double> serial_setup, serial_graph, serial_copy, split_setup, split_graph, split_copy;
    for (int repeat = 0; repeat < 3; ++repeat) {
        auto serial = run_gdn(backend, 256, 16, 48, 3, false);
        auto split = run_gdn(backend, 256, 16, 48, 3, true);
        compare(serial, split, backend, 256, 16, 48, 3);
        serial_setup.push_back(serial.setup_ms);
        serial_graph.push_back(serial.run_ms);
        serial_copy.push_back(serial.copy_ms);
        split_setup.push_back(split.setup_ms);
        split_graph.push_back(split.run_ms);
        split_copy.push_back(split.copy_ms);
    }
    const auto median = [](std::vector<double> samples) {
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    };
    const double serial_total = median(serial_setup) + median(serial_graph) + median(serial_copy);
    const double split_total = median(split_setup) + median(split_graph) + median(split_copy);
    std::printf("TIMING GDN Q256 K3 Hk=16 Hv=48 repeats=3 serial_setup_ms=%.3f serial_graph_ms=%.3f serial_copy_ms=%.3f split_setup_ms=%.3f split_graph_ms=%.3f split_copy_ms=%.3f serial_total_ms=%.3f split_total_ms=%.3f\n",
            median(serial_setup), median(serial_graph), median(serial_copy), median(split_setup), median(split_graph), median(split_copy),
            serial_total, split_total);

    ggml_backend_free(backend);
    return 0;
}
