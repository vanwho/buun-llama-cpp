#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "llama-kv-prefetch.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <vector>

struct selector_event_state {
    bool complete = false;
};

static llama_kv_prefetch_mailbox_poll selector_event_poll(
        void * context, uint64_t) noexcept {
    return static_cast<selector_event_state *>(context)->complete
        ? llama_kv_prefetch_mailbox_poll::completed
        : llama_kv_prefetch_mailbox_poll::pending;
}

static void selector_event_release(void *, uint64_t) noexcept {}

static llama_kv_page_id fixture_page(uint32_t logical_page) {
    llama_kv_page_id id;
    id.session_generation = 1;
    id.sequence_id = 0;
    id.sequence_generation = 1;
    id.logical_page = logical_page;
    id.page_generation = logical_page + 1;
    id.representation_epoch = 1;
    id.model_identity = 1;
    id.topology_identity = 1;
    id.codec_digest = 1;
    id.codebook_digest = 1;
    id.rotation_digest = 1;
    id.meansub_digest = 1;
    id.position_begin = llama_pos(logical_page * 256);
    id.position_end = id.position_begin + 256;
    return id;
}

static void test_live_selector_mailbox(const std::vector<int32_t> & output) {
    // This is the graph-boundary half of the production chain: compact IDs
    // produced by the real selector are copied into the owner mailbox, then
    // consumed as authenticated page identities rather than raw positions.
    llama_kv_prefetch_mailbox mailbox({ 2, 8 });
    uint32_t slot = UINT32_MAX;
    llama_kv_prefetch_candidate * records = nullptr;
    assert(mailbox.acquire(slot, records) == llama_kv_prefetch_mailbox_status::ok);
    uint32_t written = 0;
    for (size_t rank = 0; rank < output.size(); ++rank) {
        if (output[rank] < 0 || output[rank] >= 5) continue;
        auto & candidate = records[written++];
        candidate.identity = fixture_page(uint32_t(output[rank]));
        candidate.attention_layer = 3;
        candidate.generation = 7;
        candidate.table_epoch = 11;
        candidate.score = 1.0f / (1.0f + float(rank));
        candidate.requested_bytes = 4096;
        candidate.content_version = 1;
        candidate.summary_version = 1;
        candidate.speculation_generation = 1;
        candidate.selector_rank = uint32_t(rank);
        candidate.query_position = 257;
        candidate.cold = rank >= 2;
        candidate.rollback_generation = candidate.identity.page_generation;
    }
    assert(written != 0);
    assert(mailbox.publish_ready(slot, written, 7) ==
           llama_kv_prefetch_mailbox_status::ok);
    std::vector<llama_kv_prefetch_candidate> ready;
    assert(mailbox.take_ready(ready) == written);
    for (const auto & candidate : ready) {
        assert(candidate.generation == 7 && candidate.table_epoch == 11 &&
               candidate.attention_layer == 3);
    }

    // A refresh cannot consume a third owned slot while two results are
    // outstanding; the caller retains its last valid selection and retries.
    for (uint32_t busy = 0; busy < 2; ++busy) {
        assert(mailbox.acquire(slot, records) == llama_kv_prefetch_mailbox_status::ok);
        records[0] = { fixture_page(uint32_t(busy)), 3, 7, 11, 0.0f,
                       4096, 0, 0, false };
        assert(mailbox.publish_ready(slot, 1, 7) ==
               llama_kv_prefetch_mailbox_status::ok);
    }
    assert(mailbox.acquire(slot, records) == llama_kv_prefetch_mailbox_status::full);
    mailbox.cancel();

    // A device-to-host enqueue is not readiness. The fixed slot remains
    // owned until its event completes, and a harmless table epoch change is
    // not used as a global-generation rejection at this boundary.
    selector_event_state event;
    llama_kv_prefetch_mailbox pending({ 2, 2 });
    pending.set_backend({ &event, selector_event_poll, nullptr, selector_event_release });
    assert(pending.acquire(slot, records) == llama_kv_prefetch_mailbox_status::ok);
    records[0].identity = fixture_page(4);
    records[0].attention_layer = 3;
    records[0].generation = 9;
    records[0].table_epoch = 12;
    records[0].score = 1.0f;
    records[0].requested_bytes = 4096;
    records[0].content_version = 1;
    records[0].summary_version = 1;
    records[0].speculation_generation = 1;
    records[0].selector_rank = 0;
    records[0].query_position = 257;
    records[0].rollback_generation = records[0].identity.page_generation;
    assert(pending.publish_pending(slot, 1, 9, 41) ==
           llama_kv_prefetch_mailbox_status::ok);
    assert(pending.pending_slots() == 1 && pending.ready_slots() == 0);
    assert(pending.poll(9, 0) == llama_kv_prefetch_mailbox_status::ok);
    assert(pending.pending_slots() == 1);
    event.complete = true;
    assert(pending.poll(9, 0) == llama_kv_prefetch_mailbox_status::ok);
    assert(pending.pending_slots() == 0 && pending.ready_slots() == 1);
    std::vector<llama_kv_prefetch_candidate> completed;
    assert(pending.take_ready(completed) == 1);
    assert(completed[0].identity == fixture_page(4) &&
           completed[0].content_version == 1 &&
           completed[0].query_position == 257);

    // A result from a reused query generation is discarded before it can be
    // observed by policy.
    event.complete = true;
    assert(pending.acquire(slot, records) == llama_kv_prefetch_mailbox_status::ok);
    records[0].identity = fixture_page(1);
    records[0].attention_layer = 3;
    records[0].generation = 8;
    records[0].score = 1.0f;
    records[0].requested_bytes = 4096;
    records[0].content_version = 1;
    records[0].summary_version = 1;
    records[0].speculation_generation = 1;
    records[0].query_position = 257;
    records[0].rollback_generation = records[0].identity.page_generation;
    assert(pending.publish_pending(slot, 1, 8, 42) ==
           llama_kv_prefetch_mailbox_status::ok);
    assert(pending.poll(9, 0) == llama_kv_prefetch_mailbox_status::stale_generation);
    assert(pending.ready_slots() == 0 && pending.pending_slots() == 0);
}

static void test_user_span_accumulator(ggml_backend_t backend) {
    ggml_init_params init = { 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(init);
    assert(ctx != nullptr);
    constexpr int64_t d = 3, heads = 2;
    ggml_tensor * sum_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, heads);
    ggml_tensor * count_a = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 2);
    ggml_tensor * sum_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, heads);
    ggml_tensor * count_b = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 2);
    auto make_step = [&](int rows) {
        struct step { ggml_tensor * q; ggml_tensor * p; ggml_tensor * c; } s;
        s.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, heads, rows);
        s.p = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, rows);
        s.c = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 3);
        return s;
    };
    auto first = make_step(2);
    auto second = make_step(2);
    auto final = make_step(1);
    auto other_layer = make_step(1);
    auto next_turn = make_step(2);
    ggml_tensor * a1 = ggml_kv_query_accumulate(ctx, first.q, first.p, sum_a, count_a, first.c);
    ggml_tensor * a2 = ggml_kv_query_accumulate(ctx, second.q, second.p, sum_a, count_a, second.c);
    ggml_tensor * a3 = ggml_kv_query_accumulate(ctx, final.q, final.p, sum_a, count_a, final.c);
    ggml_tensor * b3 = ggml_kv_query_accumulate(ctx, other_layer.q, other_layer.p, sum_b, count_b, other_layer.c);
    ggml_tensor * a4 = ggml_kv_query_accumulate(ctx, next_turn.q, next_turn.p, sum_a, count_a, next_turn.c);
    ggml_set_output(a3);
    ggml_set_output(b3);
    ggml_set_output(a4);
    ggml_cgraph * graph1 = ggml_new_graph_custom(ctx, 16, false);
    ggml_cgraph * graph2 = ggml_new_graph_custom(ctx, 16, false);
    ggml_cgraph * graph3 = ggml_new_graph_custom(ctx, 16, false);
    ggml_cgraph * graph4 = ggml_new_graph_custom(ctx, 16, false);
    ggml_cgraph * graph5 = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph1, a1);
    ggml_build_forward_expand(graph2, a2);
    ggml_build_forward_expand(graph3, a3);
    ggml_build_forward_expand(graph4, b3);
    ggml_build_forward_expand(graph5, a4);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);
    const auto fill = [&](const decltype(first) & s, int64_t turn, int64_t start,
                          int64_t end, const std::vector<int64_t> & positions,
                          float base) {
        std::vector<float> q(size_t(d * heads * positions.size()));
        for (int64_t row = 0; row < int64_t(positions.size()); ++row) {
            for (int64_t head = 0; head < heads; ++head) {
                for (int64_t coord = 0; coord < d; ++coord) {
                    q[size_t(coord + d * (head + heads * row))] = base + 100.0f * float(row) + float(head);
                }
            }
        }
        const int64_t control[3] = { turn, start, end };
        ggml_backend_tensor_set(s.q, q.data(), 0, ggml_nbytes(s.q));
        ggml_backend_tensor_set(s.p, positions.data(), 0, ggml_nbytes(s.p));
        ggml_backend_tensor_set(s.c, control, 0, sizeof(control));
    };
    const int64_t sentinel[2] = { 0, INT64_MIN };
    const float zeros[d * heads] = {};
    ggml_backend_tensor_set(sum_a, zeros, 0, ggml_nbytes(sum_a));
    ggml_backend_tensor_set(sum_b, zeros, 0, ggml_nbytes(sum_b));
    ggml_backend_tensor_set(count_a, sentinel, 0, sizeof(sentinel));
    ggml_backend_tensor_set(count_b, sentinel, 0, sizeof(sentinel));
    fill(first, 1, 10, 13, { 9, 10 }, -103.0f);   // exclude old prefix; include -3
    fill(second, 1, 10, 13, { 11, 13 }, 9.0f);    // include 9; exclude trailer
    fill(final, 1, 10, 13, { 12 }, -3.0f);        // include -3 in a partial final ubatch
    fill(other_layer, 1, 10, 13, { 12 }, 20.0f);
    fill(next_turn, 2, 30, 31, { 30, 31 }, 9.0f); // stale turn resets old sum
    assert(ggml_backend_graph_compute(backend, graph1) == GGML_STATUS_SUCCESS);
    assert(ggml_backend_graph_compute(backend, graph2) == GGML_STATUS_SUCCESS);
    assert(ggml_backend_graph_compute(backend, graph3) == GGML_STATUS_SUCCESS);
    std::vector<float> mean(size_t(d * heads));
    float max_abs_error = 0.0f;
    ggml_backend_tensor_get(a3, mean.data(), 0, ggml_nbytes(a3));
    for (int64_t head = 0; head < heads; ++head) {
        for (int64_t coord = 0; coord < d; ++coord) {
            max_abs_error = std::max(max_abs_error,
                    std::abs(mean[size_t(coord + d * head)] - (1.0f + head)));
            assert(max_abs_error < 1e-6f);
        }
    }
    assert(ggml_backend_graph_compute(backend, graph4) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(b3, mean.data(), 0, ggml_nbytes(b3));
    for (int64_t head = 0; head < heads; ++head) {
        for (int64_t coord = 0; coord < d; ++coord) {
            max_abs_error = std::max(max_abs_error,
                    std::abs(mean[size_t(coord + d * head)] - (20.0f + head)));
            assert(max_abs_error < 1e-6f);
        }
    }
    assert(ggml_backend_graph_compute(backend, graph5) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(a4, mean.data(), 0, ggml_nbytes(a4));
    for (int64_t head = 0; head < heads; ++head) {
        for (int64_t coord = 0; coord < d; ++coord) {
            max_abs_error = std::max(max_abs_error,
                    std::abs(mean[size_t(coord + d * head)] - (9.0f + head)));
            assert(max_abs_error < 1e-6f);
        }
    }
    std::fprintf(stderr, "user-span accumulator FP32 oracle max_abs_error=%.9g tolerance=1e-6\n",
            max_abs_error);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t backend = nullptr;
    ggml_backend_reg_t cuda_reg = ggml_backend_reg_by_name(GGML_CUDA_NAME);
    if (cuda_reg != nullptr && ggml_backend_reg_dev_count(cuda_reg) > 0) {
        backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(cuda_reg, 0), nullptr);
    }
    if (backend == nullptr) backend = ggml_backend_cpu_init();
    assert(backend != nullptr);
    std::fprintf(stderr, "kv page selector test backend=%s\n", ggml_backend_name(backend));
    test_user_span_accumulator(backend);

    constexpr int64_t d = 4;
    constexpr int64_t n_q_heads = 4;
    constexpr int64_t n_kv_heads = 2;
    constexpr int64_t n_q = 3;
    constexpr int64_t n_pages = 5;
    ggml_init_params init = { 2 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(init);
    assert(ctx != nullptr);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, n_q_heads, n_q);
    ggml_tensor * bounds = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, 2, n_kv_heads, n_pages);
    ggml_tensor * metadata = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 8, n_pages);
    ggml_tensor * membership = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_pages);
    ggml_tensor * query = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 4);
    ggml_tensor * selected = ggml_kv_page_select(ctx, q, bounds, metadata, membership, query,
                                                  2, 2, 4, 1, 0);
    ggml_tensor * selected_row_zero = ggml_kv_page_select(ctx, q, bounds, metadata,
            membership, query, 2, 2, 4, 0, 0);
    ggml_tensor * selected_mean = ggml_kv_page_select(ctx, q, bounds, metadata,
            membership, query, 2, 2, 4, -1, 0);
    ggml_tensor * q_decode = ggml_view_3d(ctx, q, d, n_q_heads, 1,
            q->nb[1], q->nb[2], 2 * q->nb[2]);
    ggml_tensor * selected_decode = ggml_kv_page_select(ctx, q_decode, bounds,
            metadata, membership, query, 2, 2, 4, 0, 0);
    // The production pager reads this compact result after the scheduler
    // fence.  Keep the selector output alive for that graph-result boundary.
    ggml_set_output(selected);
    ggml_set_output(selected_row_zero);
    ggml_set_output(selected_mean);
    ggml_set_output(selected_decode);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, selected);
    ggml_build_forward_expand(graph, selected_row_zero);
    ggml_build_forward_expand(graph, selected_mean);
    ggml_build_forward_expand(graph, selected_decode);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);

    std::vector<float> q_data(size_t(d * n_q_heads * n_q), 0.0f);
    for (int64_t row = 0; row < n_q; ++row) {
        for (int64_t head = 0; head < n_q_heads; ++head) {
            const size_t base = size_t((row * n_q_heads + head) * d);
            // First/final rows favor resident distractor 1. The relevant
            // middle row favors cold page 2 and must determine the mean.
            const float value = row == 1 ? -3.0f : 1.0f;
            q_data[base + 0] = value;
            q_data[base + 1] = value;
            q_data[base + 2] = value;
            q_data[base + 3] = value;
        }
    }
    std::vector<ggml_fp16_t> bound_data(size_t(d * 2 * n_kv_heads * n_pages));
    for (int64_t page = 0; page < n_pages; ++page) {
        for (int64_t head = 0; head < n_kv_heads; ++head) {
            for (int64_t coord = 0; coord < d; ++coord) {
                const size_t base = size_t(coord + d * (2 * (head + n_kv_heads * page)));
                const float low = page == 2 ? -10.0f : (page == 1 ? 0.0f : -1.0f);
                const float high = page == 1 ? 10.0f : (page == 2 ? 0.0f : 1.0f);
                bound_data[base] = ggml_fp32_to_fp16(low);
                bound_data[base + d] = ggml_fp32_to_fp16(high);
            }
        }
    }
    std::vector<int64_t> page_data = {
        0, 4, 1, 1, -1, -1, 1, 1,
        4, 4, 1, 2, -1, -1, 1, 1,
        8, 4, 1, 1, -1, -1, 1, 1,
        12, 4, 1, 1, -1, -1, 1, 1,
        8, 4, 2, 1, -1, -1, 1, 1,
    };
    const std::vector<int32_t> member = { 1, 1, 0, 0, 0 };
    // The final row maps to token position 11, so its causal query position is 12.
    const std::vector<int64_t> query_data = { 12, 1, 2, 1 };
    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(bounds, bound_data.data(), 0, ggml_nbytes(bounds));
    ggml_backend_tensor_set(metadata, page_data.data(), 0, ggml_nbytes(metadata));
    ggml_backend_tensor_set(membership, member.data(), 0, ggml_nbytes(membership));
    ggml_backend_tensor_set(query, query_data.data(), 0, ggml_nbytes(query));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<int32_t> output(4, -2);
    ggml_backend_tensor_get(selected, output.data(), 0, ggml_nbytes(selected));
    assert(output[2] == 2 && output[3] == -1);
    std::vector<int32_t> row_zero_output(4, -2);
    ggml_backend_tensor_get(selected_row_zero, row_zero_output.data(), 0,
            ggml_nbytes(selected_row_zero));
    assert(row_zero_output[0] == 1);
    std::vector<int32_t> mean_output(4, -2);
    ggml_backend_tensor_get(selected_mean, mean_output.data(), 0,
            ggml_nbytes(selected_mean));
    assert(mean_output[2] == 2 && mean_output[3] == -1);
    const auto first_output = output;
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(selected, output.data(), 0, ggml_nbytes(selected));
    assert(output == first_output);
    test_live_selector_mailbox(output);

    const int64_t earlier_query_position = 9;
    ggml_backend_tensor_set(query, &earlier_query_position, 0, sizeof(earlier_query_position));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(selected, output.data(), 0, ggml_nbytes(selected));
    assert(output[2] == -1);
    std::vector<int32_t> decode_output(4, -2);
    ggml_backend_tensor_get(selected_decode, decode_output.data(), 0,
            ggml_nbytes(selected_decode));
    assert(decode_output[0] == 1);
    const int64_t final_query_position = 12;
    ggml_backend_tensor_set(query, &final_query_position, 0, sizeof(final_query_position));

    // Non-finite query coordinates are not valid scores; no page may be
    // selected when every candidate score is invalid.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (float & value : q_data) value = nan;
    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(selected, output.data(), 0, ggml_nbytes(selected));
    for (const int32_t value : output) assert(value == -1);

    // A refreshed query must reject a page whose summary sideband is stale,
    // before its bounds participate in scoring.
    std::fill(q_data.begin(), q_data.end(), 1.0f);
    for (int64_t page = 0; page < n_pages; ++page) page_data[8 * page + 6] = 0;
    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(metadata, page_data.data(), 0, ggml_nbytes(metadata));
    const int64_t refresh = 1;
    ggml_backend_tensor_set(query, &refresh, 3 * sizeof(int64_t), sizeof(refresh));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(selected, output.data(), 0, ggml_nbytes(selected));
    assert(output[0] == -1);

    // Equal negative scores still use the ascending logical-page tie break.
    std::fill(q_data.begin(), q_data.end(), -1.0f);
    for (int64_t page = 0; page < n_pages; ++page) {
        page_data[8 * page + 6] = 1;
        for (int64_t head = 0; head < n_kv_heads; ++head) {
            for (int64_t coord = 0; coord < d; ++coord) {
                const size_t base = size_t(coord + d * (2 * (head + n_kv_heads * page)));
                bound_data[base] = ggml_fp32_to_fp16(1.0f);
                bound_data[base + d] = ggml_fp32_to_fp16(2.0f);
            }
        }
    }
    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(bounds, bound_data.data(), 0, ggml_nbytes(bounds));
    ggml_backend_tensor_set(metadata, page_data.data(), 0, ggml_nbytes(metadata));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(selected, output.data(), 0, ggml_nbytes(selected));
    assert(output[0] == 0 && output[1] == 1);
    assert(output[2] == 2 && output[3] == -1);

    // A disabled refresh and a stale snapshot publish only padding, never a
    // candidate from the previous generation.
    const int64_t disabled = 0;
    ggml_backend_tensor_set(query, &disabled, 3 * sizeof(int64_t), sizeof(disabled));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(selected, output.data(), 0, ggml_nbytes(selected));
    for (int32_t value : output) assert(value == -1);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
