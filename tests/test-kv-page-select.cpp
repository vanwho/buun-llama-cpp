#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-kv-query-probes.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "llama-kv-prefetch.h"
#include "llama-batch.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

static void test_final_user_query_capture(ggml_backend_t backend) {
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
    auto same_slot_next_request = make_step(1);
    auto next_turn = make_step(2);
    ggml_tensor * a1 = ggml_kv_query_accumulate(ctx, first.q, first.p, sum_a, count_a, first.c);
    ggml_tensor * a2 = ggml_kv_query_accumulate(ctx, second.q, second.p, sum_a, count_a, second.c);
    ggml_tensor * a3 = ggml_kv_query_accumulate(ctx, final.q, final.p, sum_a, count_a, final.c);
    ggml_tensor * b3 = ggml_kv_query_accumulate(ctx, other_layer.q, other_layer.p, sum_b, count_b, other_layer.c);
    ggml_tensor * a_request2 = ggml_kv_query_accumulate(ctx, same_slot_next_request.q,
            same_slot_next_request.p, sum_a, count_a, same_slot_next_request.c);
    ggml_tensor * a4 = ggml_kv_query_accumulate(ctx, next_turn.q, next_turn.p, sum_a, count_a, next_turn.c);
    ggml_set_output(a3);
    ggml_set_output(b3);
    ggml_set_output(a_request2);
    ggml_set_output(a4);
    ggml_cgraph * graph1 = ggml_new_graph_custom(ctx, 16, false);
    ggml_cgraph * graph2 = ggml_new_graph_custom(ctx, 16, false);
    ggml_cgraph * graph3 = ggml_new_graph_custom(ctx, 16, false);
    ggml_cgraph * graph4 = ggml_new_graph_custom(ctx, 16, false);
    ggml_cgraph * graph5 = ggml_new_graph_custom(ctx, 16, false);
    ggml_cgraph * graph6 = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph1, a1);
    ggml_build_forward_expand(graph2, a2);
    ggml_build_forward_expand(graph3, a3);
    ggml_build_forward_expand(graph4, b3);
    ggml_build_forward_expand(graph5, a4);
    ggml_build_forward_expand(graph6, a_request2);
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
    fill(first, 1, 10, 13, { 9, 10 }, -99.0f); // preceding filler is outside the query span
    fill(second, 1, 10, 13, { 11, 13 }, 3.0f); // query row spans ubatches; trailer is excluded
    fill(final, 1, 10, 13, { 12 }, 5.0f);       // last query row is in a partial final ubatch
    fill(other_layer, 1, 10, 13, { 12 }, 20.0f);
    fill(same_slot_next_request, 2, 30, 31, { 30 }, 7.0f); // new query generation resets same slot turn
    fill(next_turn, 3, 40, 41, { 40, 41 }, 9.0f); // a distinct pager turn also resets
    assert(ggml_backend_graph_compute(backend, graph1) == GGML_STATUS_SUCCESS);
    assert(ggml_backend_graph_compute(backend, graph2) == GGML_STATUS_SUCCESS);
    assert(ggml_backend_graph_compute(backend, graph3) == GGML_STATUS_SUCCESS);
    std::vector<float> mean(size_t(d * heads));
    float max_abs_error = 0.0f;
    ggml_backend_tensor_get(a3, mean.data(), 0, ggml_nbytes(a3));
    for (int64_t head = 0; head < heads; ++head) {
        for (int64_t coord = 0; coord < d; ++coord) {
            max_abs_error = std::max(max_abs_error,
                    std::abs(mean[size_t(coord + d * head)] - (3.0f + head)));
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
    assert(ggml_backend_graph_compute(backend, graph6) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(a_request2, mean.data(), 0, ggml_nbytes(a_request2));
    for (int64_t head = 0; head < heads; ++head) {
        for (int64_t coord = 0; coord < d; ++coord) {
            max_abs_error = std::max(max_abs_error,
                    std::abs(mean[size_t(coord + d * head)] - (7.0f + head)));
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
    std::fprintf(stderr, "final-user-query capture FP32 oracle max_abs_error=%.9g tolerance=1e-6\n",
            max_abs_error);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void test_split_query_probe_capture(ggml_backend_t backend) {
    ggml_init_params init = { 2 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(init);
    assert(ctx != nullptr);
    constexpr int64_t d = 1, heads = 1;
    auto * probes = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, heads, 4);
    auto * validity = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 9);
    auto make_capture = [&](int rows, int64_t generation, int64_t start, int64_t end) {
        struct capture { ggml_tensor * q; ggml_tensor * positions; ggml_tensor * result; } c;
        c.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, heads, rows);
        c.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, rows);
        c.result = ggml_kv_query_probes(ctx, c.q, c.positions, probes, validity,
                generation, start, end);
        return c;
    };
    auto first = make_capture(256, 11, 0, 293);
    auto final = make_capture(37, 11, 0, 293);
    auto next_turn = make_capture(1, 12, 0, 293);
    auto next_final = make_capture(1, 13, 293, 294);
    assert(probes && validity && first.result && final.result && next_turn.result && next_final.result);
    ggml_set_output(first.result);
    ggml_set_output(final.result);
    ggml_set_output(next_turn.result);
    ggml_set_output(next_final.result);
    auto * graph_first = ggml_new_graph_custom(ctx, 16, false);
    auto * graph_final = ggml_new_graph_custom(ctx, 16, false);
    auto * graph_next = ggml_new_graph_custom(ctx, 16, false);
    auto * graph_next_final = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph_first, first.result);
    ggml_build_forward_expand(graph_final, final.result);
    ggml_build_forward_expand(graph_next, next_turn.result);
    ggml_build_forward_expand(graph_next_final, next_final.result);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);
    std::vector<float> q_first(256, 1.0f), q_final(37, 2.0f), q_next(1, 3.0f);
    std::vector<int64_t> pos_first(256), pos_final(37);
    for (int64_t i = 0; i < 256; ++i) pos_first[size_t(i)] = i;
    // Keep the first graph's sampled probe at the split boundary while the
    // last graph supplies the remaining tail probes for this same capture.
    pos_first.back() = 285;
    for (int64_t i = 0; i < 37; ++i) pos_final[size_t(i)] = 256 + i;
    const int64_t initial_validity[9] = { INT64_MIN, -1, -1, -1, -1, 0, 0, 0, 0 };
    ggml_backend_tensor_set(validity, initial_validity, 0, sizeof(initial_validity));
    ggml_backend_tensor_set(first.q, q_first.data(), 0, ggml_nbytes(first.q));
    ggml_backend_tensor_set(first.positions, pos_first.data(), 0, ggml_nbytes(first.positions));
    assert(ggml_backend_graph_compute(backend, graph_first) == GGML_STATUS_SUCCESS);
    int64_t captured[9] = {};
    ggml_backend_tensor_get(validity, captured, 0, sizeof(captured));
    assert(captured[0] == 11 && captured[4] == 285 && captured[8] == 1);
    assert(captured[5] == 0 && captured[6] == 0 && captured[7] == 0);
    ggml_backend_tensor_set(final.q, q_final.data(), 0, ggml_nbytes(final.q));
    ggml_backend_tensor_set(final.positions, pos_final.data(), 0, ggml_nbytes(final.positions));
    assert(ggml_backend_graph_compute(backend, graph_final) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(validity, captured, 0, sizeof(captured));
    assert(captured[0] == 11 && captured[5] == 1 && captured[6] == 1 &&
            captured[7] == 1 && captured[8] == 1);
    ggml_backend_tensor_set(next_turn.q, q_next.data(), 0, ggml_nbytes(next_turn.q));
    const int64_t pos_next = 0;
    ggml_backend_tensor_set(next_turn.positions, &pos_next, 0, sizeof(pos_next));
    assert(ggml_backend_graph_compute(backend, graph_next) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(validity, captured, 0, sizeof(captured));
    assert(captured[0] == 12 && captured[5] == 0 && captured[6] == 0 &&
            captured[7] == 0 && captured[8] == 0);
    // Reuse this graph within its generation, then use a fresh graph for the
    // next generation. Integrated scheduler reuse must reject the old graph
    // because its immutable op parameters belong to generation 12.
    const int64_t pos_reused = 292;
    const float q_reused = 4.0f;
    ggml_backend_tensor_set(next_turn.q, &q_reused, 0, sizeof(q_reused));
    ggml_backend_tensor_set(next_turn.positions, &pos_reused, 0, sizeof(pos_reused));
    assert(ggml_backend_graph_compute(backend, graph_next) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(validity, captured, 0, sizeof(captured));
    assert(captured[0] == 12 && captured[1] == 292 && captured[5] == 1 &&
            captured[6] == 0 && captured[7] == 0 && captured[8] == 0);
    const int64_t pos_final_next = 293;
    ggml_backend_tensor_set(next_final.q, &q_reused, 0, sizeof(q_reused));
    ggml_backend_tensor_set(next_final.positions, &pos_final_next, 0, sizeof(pos_final_next));
    assert(ggml_backend_graph_compute(backend, graph_next_final) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(validity, captured, 0, sizeof(captured));
    assert(captured[0] == 13 && captured[1] == 293 && captured[5] == 1 &&
            captured[6] == 0 && captured[7] == 0 && captured[8] == 0);
    std::fprintf(stderr, "split query probe capture U=256/U=37 and turn reset passed\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

// Exercise the allocator/scheduler used by a real model, rebuilding transient
// graphs while the sequence/layer capture remains persistent. Direct backend
// fixtures alone do not cover scheduler input storage or graph replacement.
static void test_indexed_probe_scheduler(ggml_backend_t backend) {
    assert(ggml_kv_query_probe_target(100, 101, 0, GGML_KV_QUERY_PROBES_SPREAD) == 100);
    for (int slot = 1; slot < 4; ++slot) {
        assert(ggml_kv_query_probe_target(100, 101, slot, GGML_KV_QUERY_PROBES_SPREAD) == -1);
    }
    ggml_init_params owner_init = { 4096, nullptr, true };
    auto * owner = ggml_init(owner_init);
    assert(owner != nullptr);
    auto * probes = ggml_new_tensor_3d(owner, GGML_TYPE_F32, 2, 1, 4);
    auto * validity = ggml_new_tensor_1d(owner, GGML_TYPE_I64, 9);
    auto * buffer = ggml_backend_alloc_ctx_tensors(owner, backend);
    assert(buffer != nullptr);
    const int64_t invalid[9] = { INT64_MIN, -1, -1, -1, -1, 0, 0, 0, 0 };
    ggml_backend_tensor_set(validity, invalid, 0, sizeof(invalid));
    // The real model scheduler has a CPU fallback. Unallocated graph inputs
    // start there and are copied to CUDA splits; cover that storage boundary,
    // not just a graph whose inputs were allocated directly on CUDA.
    auto * cpu = ggml_backend_cpu_init();
    assert(cpu != nullptr);
    const bool gpu = std::strncmp(ggml_backend_name(backend), "CUDA", 4) == 0;
    ggml_backend_t backends[] = { backend, cpu };
    auto * sched = ggml_backend_sched_new(backends, nullptr, gpu ? 2 : 1, 128, false, true);
    assert(sched != nullptr);
    const auto run = [&](int rows, int64_t generation, int64_t end,
                         std::array<int32_t, 4> indices, float value) {
        ggml_init_params graph_init = { 1024 * 1024, nullptr, true };
        auto * ctx = ggml_init(graph_init);
        assert(ctx != nullptr);
        auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 1, rows);
        ggml_set_input(q);
        auto * capture = ggml_kv_query_probes_spread(ctx, q, probes, validity,
                generation, 0, end, indices.data());
        assert(capture->src[1] == nullptr);
        ggml_set_output(capture);
        auto * graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(graph, capture);
        assert(ggml_backend_sched_alloc_graph(sched, graph));
        std::vector<float> input(size_t(rows) * 2, value);
        ggml_backend_tensor_set(q, input.data(), 0, ggml_nbytes(q));
        // Repeat enough times to execute a captured CUDA graph, not just its
        // initial uncaptured warmup. Then discard this graph and its inputs.
        for (int repeat = 0; repeat < 4; ++repeat) {
            assert(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_sched_synchronize(sched);
            int64_t identity[9] = {};
            float output[8] = {};
            ggml_backend_tensor_get(validity, identity, 0, sizeof(identity));
            ggml_backend_tensor_get(probes, output, 0, sizeof(output));
            assert(identity[0] == generation);
            for (size_t slot = 0; slot < indices.size(); ++slot) {
                if (indices[slot] >= 0) {
                    assert(identity[5 + slot] == 1);
                    assert(identity[1 + slot] == ggml_kv_query_probe_target(
                            0, end, int(slot), GGML_KV_QUERY_PROBES_SPREAD));
                    assert(output[2 * slot] == value && output[2 * slot + 1] == value);
                }
            }
        }
        ggml_backend_sched_reset(sched);
        ggml_free(ctx);
    };
    run(256, 11, 293, { -1, 73, 146, 219 }, 1.0f);
    run(37, 11, 293, { 36, -1, -1, -1 }, 2.0f);
    run(1, 12, 4609, { 0, -1, -1, -1 }, 3.0f);
    run(1, 13, 4610, { 0, -1, -1, -1 }, 4.0f);
    int64_t identity[9] = {};
    ggml_backend_tensor_get(validity, identity, 0, sizeof(identity));
    assert(identity[0] == 13 && identity[5] == 1 && identity[6] == 0 && identity[7] == 0 && identity[8] == 0);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_backend_buffer_free(buffer);
    ggml_free(owner);
    std::fprintf(stderr, "indexed query probe scheduler: split/rebuild/CUDA replay/generation reset passed\n");
}

static void test_independent_query_probes(ggml_backend_t backend) {
    assert(backend != nullptr);
    ggml_init_params init = { 2 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(init);
    assert(ctx != nullptr);
    constexpr int64_t d = 2, heads = 1;
    ggml_tensor * probes = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, heads, 4);
    ggml_tensor * validity = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 9);
    struct step {
        ggml_tensor * q;
        ggml_tensor * positions;
        ggml_tensor * output;
        ggml_cgraph * graph;
    };
    auto make_step = [&](int rows, int64_t generation, int64_t start, int64_t end) {
        step s{};
        s.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, heads, rows);
        s.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, rows);
        s.output = ggml_kv_query_probes(ctx, s.q, s.positions, probes, validity,
                generation, start, end);
        s.graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_set_output(s.output);
        ggml_build_forward_expand(s.graph, s.output);
        return s;
    };
    std::vector<step> steps;
    for (const auto & spec : std::vector<std::array<int64_t, 4>> {
            { 2, 1, 100, 104 }, { 2, 1, 100, 104 }, { 2, 2, 200, 202 },
            { 1, 3, 300, 301 }, { 3, 4, 400, 403 }, { 8, 5, 500, 508 },
            { 1, 6, 600, 608 }, { 1, 6, 600, 608 }, { 1, 7, 700, 701 },
            { 2, 8, 800, 802 } }) {
        steps.push_back(make_step(int(spec[0]), spec[1], spec[2], spec[3]));
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);
    const int64_t invalid[9] = { INT64_MIN, -1, -1, -1, -1, 0, 0, 0, 0 };
    ggml_backend_tensor_set(validity, invalid, 0, sizeof(invalid));
    std::vector<float> q(size_t(d * heads * 8));
    std::vector<float> probe_values(size_t(d * heads * 4));
    std::vector<int64_t> meta(9);
    auto submit = [&](size_t which, int64_t generation, int64_t start, int64_t end,
                      const std::vector<int64_t> & positions,
                      const std::vector<float> & row_values) {
        step & s = steps[which];
        assert(positions.size() == size_t(s.q->ne[2]));
        assert(row_values.size() == positions.size() * size_t(d));
        int64_t owner[3] = {};
        std::memcpy(owner, s.output->op_params, sizeof(owner));
        assert(owner[0] == generation && owner[1] == start && owner[2] == end);
        ggml_backend_tensor_set(s.q, row_values.data(), 0, row_values.size() * sizeof(float));
        ggml_backend_tensor_set(s.positions, positions.data(), 0, positions.size() * sizeof(int64_t));
        assert(ggml_backend_graph_compute(backend, s.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(probes, probe_values.data(), 0, ggml_nbytes(probes));
        ggml_backend_tensor_get(validity, meta.data(), 0, ggml_nbytes(validity));
    };
    submit(0, 1, 100, 104, { 100, 101 }, { 1, 2, 3, 4 });
    assert(meta[0] == 1 && meta[5] == 0);
    submit(1, 1, 100, 104, { 102, 103 }, { 5, 6, 7, 8 });
    assert(meta[5] == 1 && meta[6] == 1 && meta[7] == 1 && meta[8] == 0);
    assert(probe_values[0] == 7 && probe_values[1] == 8); // end - 1
    assert(probe_values[2] == 5 && probe_values[3] == 6); // end - 2
    assert(probe_values[4] == 1 && probe_values[5] == 2); // end - 4

    // The same two-row tensor shape with another absolute span refreshes the
    // coordinate identity. A one-row span captures its final row. A three-row span captures only
    // end-1/end-2, and an eight-row span captures all four unique positions.
    submit(2, 2, 200, 202, { 200, 201 }, { 11, 12, 13, 14 });
    assert(meta[5] == 1 && meta[6] == 1 && meta[7] == 0 && meta[8] == 0);
    submit(3, 3, 300, 301, { 300 }, { 15, 16 });
    assert(meta[5] == 1 && meta[6] == 0 && meta[7] == 0 && meta[8] == 0);
    submit(4, 4, 400, 403, { 400, 401, 402 }, { 21, 22, 23, 24, 25, 26 });
    assert(meta[5] == 1 && meta[6] == 1 && meta[7] == 0 && meta[8] == 0);
    submit(5, 5, 500, 508, { 500, 501, 502, 503, 504, 505, 506, 507 },
            { 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45 });
    assert(meta[5] == 1 && meta[6] == 1 && meta[7] == 1 && meta[8] == 1);
    assert(probe_values[0] == 44 && probe_values[1] == 45); // 407
    assert(probe_values[2] == 42 && probe_values[3] == 43); // 406
    assert(probe_values[4] == 38 && probe_values[5] == 39); // 404
    assert(probe_values[6] == 30 && probe_values[7] == 31); // 400

    // Replaying a provisional row overwrites its slot; it does not duplicate
    // or add the value. A new generation clears validity before fresh capture.
    submit(6, 6, 600, 608, { 607 }, { 51, 52 });
    submit(7, 6, 600, 608, { 607 }, { 61, 62 });
    assert(probe_values[0] == 61 && probe_values[1] == 62);
    submit(8, 7, 700, 701, { 700 }, { 0, 0 });
    assert(meta[5] == 1 && meta[6] == 0 && meta[7] == 0 && meta[8] == 0);

    // The two-query mean cancels the needle direction, while an independent
    // retained row preserves it for the later reranker.
    submit(9, 8, 800, 802, { 800, 801 }, { 1, 0, -1, 0 });
    assert(probe_values[0] == -1 && probe_values[1] == 0);
    assert((1.0f + -1.0f) / 2.0f == 0.0f);
    std::fprintf(stderr, "independent query probes CPU oracle: split/short/duplicate/reset/cancellation cases passed\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void test_query_score_rank(ggml_backend_t backend) {
    constexpr int64_t d = 1, heads = 2, probes_n = 4, pages = 6;
    ggml_init_params init = { 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(init);
    assert(ctx && backend);
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, heads, 1);
    auto * bounds = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, 3, 1, pages);
    auto * metadata = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 9, pages);
    auto * membership = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, pages);
    auto * query = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 4);
    auto * probes = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, heads, probes_n);
    auto * validity = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 9);
    auto * ranked = ggml_kv_page_rank(ctx, q, bounds, metadata, membership, query,
            probes, validity, 2, 4, 4, -1, 0xf, 1.0f);
    assert(ranked && ranked->type == GGML_TYPE_I64 && ranked->ne[0] == 2);
    ggml_set_output(ranked);
    auto * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, ranked);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer);
    const float q_zero[heads] = {0, 0};
    ggml_backend_tensor_set(q, q_zero, 0, ggml_nbytes(q));
    std::vector<ggml_fp16_t> bounds_data(size_t(3 * pages));
    int64_t page_data[9 * pages] = {};
    int32_t resident[pages] = {1, 1, 0, 0, 0, 0};
    for (int64_t page = 0; page < pages; ++page) {
        page_data[page * 9 + 0] = page * 10;
        page_data[page * 9 + 1] = 4;
        page_data[page * 9 + 2] = 1;
        page_data[page * 9 + 3] = page + 1;
        page_data[page * 9 + 6] = 1;
        const float center = float(page);
        bounds_data[size_t(page * 3 + 0)] = ggml_fp32_to_fp16(center - 0.25f);
        bounds_data[size_t(page * 3 + 1)] = ggml_fp32_to_fp16(center + 0.25f);
        bounds_data[size_t(page * 3 + 2)] = ggml_fp32_to_fp16(center);
    }
    const int64_t query_data[4] = {100, 1, 20, 1};
    const int64_t validity_data[9] = {7, 96, 97, 98, 99, 1, 1, 1, 1};
    const float probe_data[heads * probes_n] = {1, 2, 3, 4, 1, 1, 1, 1};
    ggml_backend_tensor_set(bounds, bounds_data.data(), 0, ggml_nbytes(bounds));
    ggml_backend_tensor_set(metadata, page_data, 0, ggml_nbytes(metadata));
    ggml_backend_tensor_set(membership, resident, 0, ggml_nbytes(membership));
    ggml_backend_tensor_set(query, query_data, 0, sizeof(query_data));
    ggml_backend_tensor_set(probes, probe_data, 0, ggml_nbytes(probes));
    ggml_backend_tensor_set(validity, validity_data, 0, sizeof(validity_data));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_kv_page_rank_record output[6]{};
    ggml_backend_tensor_get(ranked, output, 0, ggml_nbytes(ranked));
    assert(output[0].logical_page >= 0 && output[0].peak_probability > output[0].mean_probability);
    for (const auto & record : output) {
        if (record.logical_page >= 0) {
            assert(record.validity_flags == 1 && std::isfinite(record.peak_probability) &&
                   std::isfinite(record.mean_probability));
        }
    }
    const float infinite_probes[heads * probes_n] = {
        INFINITY, INFINITY, INFINITY, INFINITY,
        INFINITY, INFINITY, INFINITY, INFINITY,
    };
    ggml_backend_tensor_set(probes, infinite_probes, 0, ggml_nbytes(probes));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(ranked, output, 0, ggml_nbytes(ranked));
    for (const auto & record : output) assert(record.logical_page == -1);
    const float probe_data_finite[heads * probes_n] = {1, 2, 3, 4, 1, 1, 1, 1};
    ggml_backend_tensor_set(probes, probe_data_finite, 0, ggml_nbytes(probes));
    bounds_data[size_t(2 * 3 + 1)] = ggml_fp32_to_fp16(std::numeric_limits<float>::infinity());
    ggml_backend_tensor_set(bounds, bounds_data.data(), 0, ggml_nbytes(bounds));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(ranked, output, 0, ggml_nbytes(ranked));
    bool saw_infinite_bound_winner = false;
    for (const auto & record : output) {
        if (record.logical_page == 2) {
            saw_infinite_bound_winner = record.peak_probability == 1.0f;
        }
    }
    assert(saw_infinite_bound_winner);

    int64_t invalid_probes[9] = {7, 96, 97, 98, 99, 0, 0, 0, 0};
    ggml_backend_tensor_set(validity, invalid_probes, 0, sizeof(invalid_probes));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(ranked, output, 0, ggml_nbytes(ranked));
    for (const auto & record : output) assert(record.logical_page == -1);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static std::vector<ggml_kv_page_rank_record> run_large_page_rank(
        ggml_backend_t backend, int64_t heads, int64_t pages) {
    constexpr int64_t dim = 8, probes_n = 4;
    constexpr int64_t resident_width = 16, cold_width = 32;
    ggml_init_params init = { 64 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(init);
    assert(ctx && backend);
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, heads, 1);
    auto * bounds = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dim, 3, heads, pages);
    auto * metadata = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 9, pages);
    auto * membership = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, pages);
    auto * query = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 4);
    auto * probes = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, heads, probes_n);
    auto * validity = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 9);
    auto * ranked = ggml_kv_page_rank(ctx, q, bounds, metadata, membership, query,
            probes, validity, resident_width, cold_width, 4, -1, 0xf, 1.0f);
    assert(ranked && ranked->ne[1] == resident_width + cold_width);
    ggml_set_output(ranked);
    auto * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, ranked);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer);

    std::vector<float> q_data(size_t(dim * heads), 0.0f);
    std::vector<float> probe_data(size_t(dim * heads * probes_n), 0.0f);
    for (int64_t probe = 0; probe < probes_n; ++probe) {
        const float direction = probe % 2 == 0 ? 1.0f : -1.0f;
        for (int64_t head = 0; head < heads; ++head) {
            probe_data[size_t(dim * (head + heads * probe))] = direction;
        }
    }
    std::vector<ggml_fp16_t> bound_data(size_t(dim * 3 * heads * pages));
    for (int64_t page = 0; page < pages; ++page) {
        for (int64_t head = 0; head < heads; ++head) {
            for (int64_t d = 0; d < dim; ++d) {
                const float low = page == 23 && d == 0 ? -20.0f :
                    page == 25 && d == 0 ? -50.0f : -0.25f;
                const float high = page == 22 && d == 0 ? 20.0f :
                    page == 24 && d == 0 ? 50.0f : 0.25f;
                const float mean = page == 20 && d == 0 ? 10.0f :
                    page == 21 && d == 0 ? -10.0f : 0.0f;
                const size_t index = size_t(d + dim * (3 * (head + heads * page)));
                bound_data[index] = ggml_fp32_to_fp16(low);
                bound_data[index + dim] = ggml_fp32_to_fp16(high);
                bound_data[index + 2 * dim] = ggml_fp32_to_fp16(mean);
            }
        }
    }
    std::vector<int64_t> page_data(size_t(9 * pages), 0);
    std::vector<int32_t> resident(size_t(pages), 0);
    for (int64_t page = 0; page < pages; ++page) {
        page_data[size_t(page * 9 + 0)] = page * 4;
        page_data[size_t(page * 9 + 1)] = 4;
        page_data[size_t(page * 9 + 2)] = page == 24 ? 2 : 1;
        page_data[size_t(page * 9 + 3)] = page + 1;
        page_data[size_t(page * 9 + 6)] = 1;
        if (page == 25) page_data[size_t(page * 9 + 1)] = 0;
        resident[size_t(page)] = page < resident_width ? 1 : 0;
    }
    const int64_t query_position = pages * 4 + 4;
    const int64_t query_data[4] = {query_position, 1, pages + 10, 1};
    const int64_t validity_data[9] = {7, query_position - 1, query_position - 2,
        query_position - 4, query_position - 8, 1, 1, 1, 1};
    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(bounds, bound_data.data(), 0, ggml_nbytes(bounds));
    ggml_backend_tensor_set(metadata, page_data.data(), 0, ggml_nbytes(metadata));
    ggml_backend_tensor_set(membership, resident.data(), 0, ggml_nbytes(membership));
    ggml_backend_tensor_set(query, query_data, 0, sizeof(query_data));
    ggml_backend_tensor_set(probes, probe_data.data(), 0, ggml_nbytes(probes));
    ggml_backend_tensor_set(validity, validity_data, 0, sizeof(validity_data));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<ggml_kv_page_rank_record> result(size_t(resident_width + cold_width));
    ggml_backend_tensor_get(ranked, result.data(), 0, ggml_nbytes(ranked));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

static void test_large_page_rank_cuda(ggml_backend_t cuda_backend) {
    constexpr size_t resident_width = 16;
    assert(std::strncmp(ggml_backend_name(cuda_backend), "CUDA", 4) == 0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    assert(cpu);
    for (const int64_t pages : {64, 256, 1024}) {
        for (const int64_t heads : {16, 200}) {
            const auto expected = run_large_page_rank(cpu, heads, pages);
            const auto actual = run_large_page_rank(cuda_backend, heads, pages);
            for (size_t i = 0; i < expected.size(); ++i) {
                if (actual[i].logical_page != expected[i].logical_page) {
                    std::fprintf(stderr, "rank mismatch heads=%" PRId64 " pages=%" PRId64
                            " slot=%zu CPU={%d,%.8f,%.8f} CUDA={%d,%.8f,%.8f}\n",
                            heads, pages, i, expected[i].logical_page, expected[i].peak_probability,
                            expected[i].mean_probability, actual[i].logical_page,
                            actual[i].peak_probability, actual[i].mean_probability);
                    if (i == resident_width + 2) {
                        for (size_t rank = resident_width; rank < actual.size(); ++rank) {
                            std::fprintf(stderr, "cold rank=%zu CPU=%d(%.4f,%.4f) CUDA=%d(%.4f,%.4f)\n",
                                    rank - resident_width, expected[rank].logical_page,
                                    expected[rank].peak_probability, expected[rank].mean_probability,
                                    actual[rank].logical_page, actual[rank].peak_probability,
                                    actual[rank].mean_probability);
                        }
                    }
                }
                assert(actual[i].logical_page == expected[i].logical_page);
                assert(actual[i].validity_flags == expected[i].validity_flags);
                assert(std::fabs(actual[i].peak_probability - expected[i].peak_probability) < 0.02f);
                assert(std::fabs(actual[i].mean_probability - expected[i].mean_probability) < 0.02f);
            }
            for (int32_t id : {22, 23}) {
                assert(std::any_of(actual.begin() + resident_width, actual.end(), [id](const auto & record) {
                    return record.logical_page == id && record.peak_probability > 0.99f;
                }));
            }
            std::fprintf(stderr, "CUDA PAGE_RANK executed heads=%" PRId64 " pages=%" PRId64
                    " cold_winners=22,23\n", heads, pages);
        }
    }
    ggml_backend_free(cpu);
}


static void test_query_probes(ggml_backend_t backend) {
    ggml_init_params init = { 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(init);
    assert(ctx != nullptr);
    constexpr int d = 3, heads = 2;
    ggml_tensor * sum = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, heads, 4);
    ggml_tensor * count = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 2);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, heads, 2);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 2);
    ggml_tensor * control = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 3);
    ggml_tensor * probes = ggml_kv_query_accumulate(ctx, q, positions, sum, count, control);
    ggml_set_output(probes);
    ggml_cgraph * capture = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(capture, probes);
    ggml_tensor * bounds = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, 3, 1, 2);
    ggml_tensor * metadata = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 8, 2);
    ggml_tensor * membership = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    ggml_tensor * query = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 4);
    // Use a separate selector graph so it cannot accumulate the final batch twice.
    ggml_tensor * frozen = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, heads, 4);
    ggml_tensor * selected = ggml_kv_page_select(ctx, frozen, bounds, metadata,
            membership, query, 0, 1, 4, -1, 0, 2);
    ggml_tensor * averaged = ggml_kv_page_select(ctx, frozen, bounds, metadata,
            membership, query, 0, 1, 4, 0, 0, 1);
    ggml_set_output(selected);
    ggml_set_output(averaged);
    ggml_cgraph * rank = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(rank, selected);
    ggml_build_forward_expand(rank, averaged);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);
    const int64_t sentinel[2] = { 0, INT64_MIN };
    ggml_backend_tensor_set(count, sentinel, 0, sizeof(sentinel));
    const auto step = [&](int64_t generation, int64_t start, int64_t end,
                          int64_t position, float first, float second) {
        const int64_t controls[3] = { generation, start, end };
        const int64_t pos[2] = { position, position + 1 };
        std::vector<float> values(d * heads * 2);
        std::fill(values.begin(), values.begin() + d * heads, first);
        std::fill(values.begin() + d * heads, values.end(), second);
        ggml_backend_tensor_set(q, values.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(positions, pos, 0, sizeof(pos));
        ggml_backend_tensor_set(control, controls, 0, sizeof(controls));
        assert(ggml_backend_graph_compute(backend, capture) == GGML_STATUS_SUCCESS);
    };
    step(1, 10, 14, 10, -3, -1);
    step(1, 10, 14, 12, 1, 3);
    std::vector<float> values(d * heads * 4);
    ggml_backend_tensor_get(probes, values.data(), 0, ggml_nbytes(probes));
    const float expected[4] = { 0, -3, -1, 3 };
    for (int p = 0; p < 4; ++p) {
        for (int i = 0; i < d * heads; ++i) assert(values[p * d * heads + i] == expected[p]);
    }
    ggml_backend_tensor_set(frozen, values.data(), 0, ggml_nbytes(frozen));
    std::vector<ggml_fp16_t> catalogue(d * 3 * 2, ggml_fp32_to_fp16(0));
    for (int page = 0; page < 2; ++page) {
        for (int i = 0; i < d; ++i) catalogue[page * d * 3 + 2 * d + i] =
                ggml_fp32_to_fp16(page == 0 ? -1.0f : 2.0f);
    }
    const int64_t page_metadata[16] = { 0, 4, 1, 1, -1, 0, 1, 0, 4, 4, 1, 1, -1, 0, 1, 0 };
    const int32_t memberships[2] = { 0, 0 };
    const int64_t query_metadata[4] = { 20, 1, 1, 1 };
    ggml_backend_tensor_set(bounds, catalogue.data(), 0, ggml_nbytes(bounds));
    ggml_backend_tensor_set(metadata, page_metadata, 0, sizeof(page_metadata));
    ggml_backend_tensor_set(membership, memberships, 0, sizeof(memberships));
    ggml_backend_tensor_set(query, query_metadata, 0, sizeof(query_metadata));
    assert(ggml_backend_graph_compute(backend, rank) == GGML_STATUS_SUCCESS);
    int32_t chosen = -1, old_choice = -1;
    ggml_backend_tensor_get(selected, &chosen, 0, sizeof(chosen));
    ggml_backend_tensor_get(averaged, &old_choice, 0, sizeof(old_choice));
    assert(chosen == 1 && old_choice == 0); // mean Q cancels and ties; real Q does not
    std::fill(values.begin(), values.end(), std::numeric_limits<float>::quiet_NaN());
    ggml_backend_tensor_set(frozen, values.data(), 0, ggml_nbytes(frozen));
    assert(ggml_backend_graph_compute(backend, rank) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(selected, &chosen, 0, sizeof(chosen));
    assert(chosen == -1); // invalid Q is never an authenticated score
    step(2, 20, 22, 20, 2, 4);
    ggml_backend_tensor_get(probes, values.data(), 0, ggml_nbytes(probes));
    const float next_expected[4] = { 3, 2, 2, 4 };
    for (int p = 0; p < 4; ++p) {
        for (int i = 0; i < d * heads; ++i) assert(values[p * d * heads + i] == next_expected[p]);
    }
    step(3, 0, 40, 38, 2, 4); // cached prefix: only final probe was observed
    ggml_backend_tensor_get(probes, values.data(), 0, ggml_nbytes(probes));
    assert(values[0] == 3 && std::isnan(values[d * heads]) &&
            std::isnan(values[2 * d * heads]) && values[3 * d * heads] == 4);
    step(4, 50, 60, 0, 2, 4); // no user row: no measured zero-valued probe
    ggml_backend_tensor_get(probes, values.data(), 0, ggml_nbytes(probes));
    for (const float value : values) assert(std::isnan(value));
    std::fprintf(stderr, "query-probe oracle: split/reset/cancellation/missing/nonfinite passed\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void test_section_major_native_positions() {
    // Match llama_batch_allocr: M-RoPE coordinates are separate planes.
    constexpr uint32_t rows = 9;
    std::array<llama_pos, rows * 4> positions{};
    for (uint32_t section = 0; section < 4; ++section) {
        for (uint32_t row = 0; row < rows; ++row) {
            positions[section * rows + row] = llama_pos(1000 * section + 200 + row);
        }
    }
    llama_ubatch ubatch{};
    ubatch.n_tokens = rows;
    ubatch.n_pos = 4;
    ubatch.pos = positions.data();
    for (uint32_t row = 0; row < rows; ++row) assert(ubatch.pos0(row) == llama_pos(200 + row));
    assert(ubatch.pos0(rows - 1) == 208);
    assert(positions[(rows - 1) * ubatch.n_pos] != ubatch.pos0(rows - 1));
}

int main() {
    test_section_major_native_positions();
    ggml_backend_load_all();
    ggml_backend_t backend = nullptr;
    const char * cuda_test = std::getenv("LLAMA_TEST_KV_PAGE_SELECT_CUDA");
    const char * cuda_tiny_test = std::getenv("LLAMA_TEST_KV_PAGE_SELECT_CUDA_TINY");
    const bool tiny_cuda = cuda_tiny_test != nullptr && std::strcmp(cuda_tiny_test, "1") == 0;
    if ((cuda_test != nullptr && std::strcmp(cuda_test, "1") == 0) || tiny_cuda) {
        ggml_backend_reg_t cuda_reg = ggml_backend_reg_by_name(GGML_CUDA_NAME);
        if (cuda_reg != nullptr && ggml_backend_reg_dev_count(cuda_reg) > 0) {
            backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(cuda_reg, 0), nullptr);
        }
    }
    if (backend == nullptr) backend = ggml_backend_cpu_init();
    assert(backend != nullptr);
    std::fprintf(stderr, "kv page selector test backend=%s\n", ggml_backend_name(backend));
    test_query_score_rank(backend);
    if (!tiny_cuda && std::strncmp(ggml_backend_name(backend), "CUDA", 4) == 0) {
        test_large_page_rank_cuda(backend);
    }
    test_independent_query_probes(backend);
    test_split_query_probe_capture(backend);
    test_indexed_probe_scheduler(backend);
    test_final_user_query_capture(backend);
    test_query_probes(backend);

    constexpr int64_t d = 128;
    constexpr int64_t n_q_heads = 4;
    constexpr int64_t n_kv_heads = 2;
    constexpr int64_t n_q = 3;
    constexpr int64_t n_pages = 5;
    ggml_init_params init = { 2 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(init);
    assert(ctx != nullptr);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, n_q_heads, n_q);
    ggml_tensor * bounds = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, 3, n_kv_heads, n_pages);
    ggml_tensor * metadata = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 8, n_pages);
    ggml_tensor * membership = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_pages);
    ggml_tensor * query = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 4);
    ggml_tensor * selected = ggml_kv_page_select(ctx, q, bounds, metadata, membership, query,
                                                  2, 2, 4, 1, 0, 0);
    ggml_tensor * selected_row_zero = ggml_kv_page_select(ctx, q, bounds, metadata,
            membership, query, 2, 2, 4, 0, 0, 0);
    ggml_tensor * selected_mean = ggml_kv_page_select(ctx, q, bounds, metadata,
            membership, query, 2, 2, 4, -1, 0, 0);
    ggml_tensor * selected_mean_k = ggml_kv_page_select(ctx, q, bounds, metadata,
            membership, query, 2, 2, 4, -1, 0, 1);
    ggml_tensor * current_cold_top1 = ggml_kv_page_select(ctx, q, bounds, metadata,
            membership, query, 2, 1, 4, -1, 0, 0);
    ggml_tensor * mean_k_cold_top1 = ggml_kv_page_select(ctx, q, bounds, metadata,
            membership, query, 2, 1, 4, -1, 0, 1);
    ggml_tensor * q_decode = ggml_view_3d(ctx, q, d, n_q_heads, 1,
            q->nb[1], q->nb[2], 2 * q->nb[2]);
    ggml_tensor * selected_decode = ggml_kv_page_select(ctx, q_decode, bounds,
            metadata, membership, query, 2, 2, 4, 0, 0, 0);
    // The production pager reads this compact result after the scheduler
    // fence.  Keep the selector output alive for that graph-result boundary.
    ggml_set_output(selected);
    ggml_set_output(selected_row_zero);
    ggml_set_output(selected_mean);
    ggml_set_output(selected_mean_k);
    ggml_set_output(current_cold_top1);
    ggml_set_output(mean_k_cold_top1);
    ggml_set_output(selected_decode);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, selected);
    ggml_build_forward_expand(graph, selected_row_zero);
    ggml_build_forward_expand(graph, selected_mean);
    ggml_build_forward_expand(graph, selected_mean_k);
    ggml_build_forward_expand(graph, current_cold_top1);
    ggml_build_forward_expand(graph, mean_k_cold_top1);
    ggml_build_forward_expand(graph, selected_decode);
    ggml_cgraph * graph_current = ggml_new_graph_custom(ctx, 16, false);
    ggml_cgraph * graph_mean_k = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph_current, current_cold_top1);
    ggml_build_forward_expand(graph_mean_k, mean_k_cold_top1);
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
    std::vector<ggml_fp16_t> bound_data(size_t(d * 3 * n_kv_heads * n_pages));
    for (int64_t page = 0; page < n_pages; ++page) {
        for (int64_t head = 0; head < n_kv_heads; ++head) {
            for (int64_t coord = 0; coord < d; ++coord) {
                const size_t base = size_t(coord + d * (3 * (head + n_kv_heads * page)));
                const float low = page == 2 ? -10.0f : (page == 1 ? 0.0f : -1.0f);
                const float high = page == 1 ? 10.0f : (page == 2 ? 0.0f : 1.0f);
                bound_data[base] = ggml_fp32_to_fp16(low);
                bound_data[base + d] = ggml_fp32_to_fp16(high);
                bound_data[base + 2 * d] = ggml_fp32_to_fp16(page == 2 ? 0.0f : page == 3 ? -20.0f : 0.0f);
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
    // The final row maps to token position 15, so its causal query position is 16.
    const std::vector<int64_t> query_data = { 16, 1, 2, 1 };
    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(bounds, bound_data.data(), 0, ggml_nbytes(bounds));
    ggml_backend_tensor_set(metadata, page_data.data(), 0, ggml_nbytes(metadata));
    ggml_backend_tensor_set(membership, member.data(), 0, ggml_nbytes(membership));
    ggml_backend_tensor_set(query, query_data.data(), 0, ggml_nbytes(query));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    constexpr int timing_iterations = 20;
    const auto current_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < timing_iterations; ++i) {
        assert(ggml_backend_graph_compute(backend, graph_current) == GGML_STATUS_SUCCESS);
    }
    ggml_backend_synchronize(backend);
    const double current_selector_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - current_begin).count() / timing_iterations;
    const auto mean_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < timing_iterations; ++i) {
        assert(ggml_backend_graph_compute(backend, graph_mean_k) == GGML_STATUS_SUCCESS);
    }
    ggml_backend_synchronize(backend);
    const double mean_selector_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - mean_begin).count() / timing_iterations;
    std::vector<int32_t> output(4, -2);
    ggml_backend_tensor_get(selected, output.data(), 0, ggml_nbytes(selected));
    assert(output[2] == 2 && output[3] == 3);
    std::vector<int32_t> row_zero_output(4, -2);
    ggml_backend_tensor_get(selected_row_zero, row_zero_output.data(), 0,
            ggml_nbytes(selected_row_zero));
    assert(row_zero_output[0] == 1);
    std::vector<int32_t> mean_output(4, -2);
    ggml_backend_tensor_get(selected_mean, mean_output.data(), 0,
            ggml_nbytes(selected_mean));
    assert(mean_output[2] == 2 && mean_output[3] == 3);
    std::vector<int32_t> mean_k_output(4, -2);
    ggml_backend_tensor_get(selected_mean_k, mean_k_output.data(), 0,
            ggml_nbytes(selected_mean_k));
    assert(mean_k_output[2] == 3 && mean_k_output[3] == 2);
    int32_t current_top1 = -1, mean_k_top1 = -1;
    ggml_backend_tensor_get(current_cold_top1, &current_top1, 2 * sizeof(int32_t), sizeof(int32_t));
    ggml_backend_tensor_get(mean_k_cold_top1, &mean_k_top1, 2 * sizeof(int32_t), sizeof(int32_t));
    assert(current_top1 == 2 && mean_k_top1 == 3);
    std::fprintf(stderr,
            "scorer_comparison backend=%s required_page=3 current_recall=0 mean_k_recall=1 "
            "current_false_promotions=1 mean_k_false_promotions=0 top1_overlap=0 "
            "current_selector_us=%.3f mean_k_selector_us=%.3f catalogue_bytes=%zu\n",
            ggml_backend_name(backend), current_selector_us, mean_selector_us,
            size_t(ggml_nbytes(bounds)));
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
    const int64_t final_query_position = 16;
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
    ggml_backend_tensor_get(mean_k_cold_top1, &mean_k_top1,
            2 * sizeof(int32_t), sizeof(int32_t));
    assert(mean_k_top1 == -1);

    // Equal negative scores still use the ascending logical-page tie break.
    std::fill(q_data.begin(), q_data.end(), -1.0f);
    for (int64_t page = 0; page < n_pages; ++page) {
        page_data[8 * page + 6] = 1;
        for (int64_t head = 0; head < n_kv_heads; ++head) {
            for (int64_t coord = 0; coord < d; ++coord) {
                const size_t base = size_t(coord + d * (3 * (head + n_kv_heads * page)));
                bound_data[base] = ggml_fp32_to_fp16(1.0f);
                bound_data[base + d] = ggml_fp32_to_fp16(2.0f);
                bound_data[base + 2 * d] = ggml_fp32_to_fp16(1.0f);
            }
        }
    }
    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(bounds, bound_data.data(), 0, ggml_nbytes(bounds));
    ggml_backend_tensor_set(metadata, page_data.data(), 0, ggml_nbytes(metadata));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(selected, output.data(), 0, ggml_nbytes(selected));
    assert(output[0] == 0 && output[1] == 1);
    assert(output[2] == 2 && output[3] == 3);
    ggml_backend_tensor_get(selected_mean_k, mean_k_output.data(), 0,
            ggml_nbytes(selected_mean_k));
    assert(mean_k_output[0] == 0 && mean_k_output[1] == 1);
    assert(mean_k_output[2] == 2 && mean_k_output[3] == 3);

    // A disabled refresh and a stale snapshot publish only padding, never a
    // candidate from the previous generation.
    const int64_t disabled = 0;
    ggml_backend_tensor_set(query, &disabled, 3 * sizeof(int64_t), sizeof(disabled));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(selected, output.data(), 0, ggml_nbytes(selected));
    for (int32_t value : output) assert(value == -1);
    ggml_backend_tensor_get(selected_mean_k, mean_k_output.data(), 0,
            ggml_nbytes(selected_mean_k));
    for (int32_t value : mean_k_output) assert(value == -1);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
