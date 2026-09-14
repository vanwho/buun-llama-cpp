#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama-kv-prefetch.h"

#include <cassert>
#include <cstdint>
#include <vector>

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
        candidate.score = -float(rank);
        candidate.requested_bytes = 4096;
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
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t cuda_device = ggml_backend_dev_by_name("CUDA0");
    ggml_backend_t backend = cuda_device != nullptr
        ? ggml_backend_dev_init(cuda_device, nullptr) : ggml_backend_cpu_init();
    assert(backend != nullptr);

    constexpr int64_t d = 4;
    constexpr int64_t n_q_heads = 4;
    constexpr int64_t n_kv_heads = 2;
    constexpr int64_t n_pages = 5;
    ggml_init_params init = { 2 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(init);
    assert(ctx != nullptr);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, n_q_heads, 1);
    ggml_tensor * bounds = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, 2, n_kv_heads, n_pages);
    ggml_tensor * metadata = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 4, n_pages);
    ggml_tensor * membership = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_pages);
    ggml_tensor * query = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 4);
    ggml_tensor * selected = ggml_kv_page_select(ctx, q, bounds, metadata, membership, query,
                                                  2, 2, 4, 0);
    // The production pager reads this compact result after the scheduler
    // fence.  Keep the selector output alive for that graph-result boundary.
    ggml_set_output(selected);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, selected);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);

    std::vector<float> q_data(size_t(d * n_q_heads), 0.0f);
    for (int64_t head = 0; head < n_q_heads; ++head) {
        q_data[size_t(head * d) + 0] = 1.0f;
        q_data[size_t(head * d) + 1] = -2.0f;
        q_data[size_t(head * d) + 2] = 3.0f;
        q_data[size_t(head * d) + 3] = -4.0f;
    }
    std::vector<ggml_fp16_t> bound_data(size_t(d * 2 * n_kv_heads * n_pages));
    for (int64_t page = 0; page < n_pages; ++page) {
        for (int64_t head = 0; head < n_kv_heads; ++head) {
            const float high = page == 1 ? 2.0f : (page == 2 ? 1.5f : 1.0f);
            for (int64_t coord = 0; coord < d; ++coord) {
                const size_t base = size_t(coord + d * (2 * (head + n_kv_heads * page)));
                bound_data[base] = ggml_fp32_to_fp16(-1.0f);
                bound_data[base + d] = ggml_fp32_to_fp16(high);
            }
        }
    }
    std::vector<int64_t> page_data = {
        0, 4, 1, 1,
        4, 4, 1, 2,
        8, 4, 1, 1,
        12, 4, 1, 1,
        8, 4, 2, 1,
    };
    const std::vector<int32_t> member = { 1, 1, 0, 0, 0 };
    const std::vector<int64_t> query_data = { 12, 1, 2, 1 };
    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(bounds, bound_data.data(), 0, ggml_nbytes(bounds));
    ggml_backend_tensor_set(metadata, page_data.data(), 0, ggml_nbytes(metadata));
    ggml_backend_tensor_set(membership, member.data(), 0, ggml_nbytes(membership));
    ggml_backend_tensor_set(query, query_data.data(), 0, ggml_nbytes(query));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<int32_t> output(4, -2);
    ggml_backend_tensor_get(selected, output.data(), 0, ggml_nbytes(selected));
    assert(output[0] == 1 && output[1] == 0);
    assert(output[2] == 2 && output[3] == -1);
    const auto first_output = output;
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(selected, output.data(), 0, ggml_nbytes(selected));
    assert(output == first_output);
    test_live_selector_mailbox(output);

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
