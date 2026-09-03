#include "llama-kv-attention-view.h"
#include "llama-kv-attention-op.h"
#include "llama-graph.h"

#include <cassert>
#include <vector>

static llama_kv_page_record page(uint32_t logical, uint32_t slot, llama_pos end) {
    llama_kv_page_record result;
    result.id.sequence_id = 0;
    result.id.logical_page = logical;
    result.id.position_begin = llama_pos(logical * VBR_GENERATION_PAGE_CELLS);
    result.id.position_end = end;
    result.physical_slot = slot;
    result.state = llama_kv_page_state::gpu_host_clean;
    result.host_valid = true;
    return result;
}

static llama_kv_residency_snapshot make_snapshot() {
    llama_kv_residency_table table(8);
    auto tx = table.begin();
    assert(table.replace(tx, page(0, 5, 256)) == llama_kv_residency_status::ok);
    assert(table.replace(tx, page(1, 1, 512)) == llama_kv_residency_status::ok);
    assert(table.replace(tx, page(2, 7, 768)) == llama_kv_residency_status::ok);
    assert(table.replace(tx, page(3, 3, 900)) == llama_kv_residency_status::ok);
    assert(table.publish(tx) == llama_kv_residency_status::ok);
    return table.snapshot();
}

static void test_sequence_scoped_lookup() {
    llama_kv_residency_table table(8);
    auto tx = table.begin();
    const auto seq0 = page(0, 1, 256);
    auto seq1 = page(0, 6, 256);
    seq1.id.sequence_id = 1;
    const auto first = table.replace(tx, seq0);
    const auto second = table.replace(tx, seq1);
    const auto published = table.publish(tx);
    assert(first == llama_kv_residency_status::ok);
    assert(second == llama_kv_residency_status::ok);
    assert(published == llama_kv_residency_status::ok);
    (void) first;
    (void) second;
    (void) published;

    const auto snapshot = table.snapshot();
    const auto filtered = snapshot.for_sequence(1);
    assert(filtered.epoch() == snapshot.epoch());
    assert(filtered.slot_capacity() == snapshot.slot_capacity());
    assert(filtered.pages().size() == 1);
    assert(filtered.pages()[0].id.sequence_id == 1);
    assert(filtered.pages()[0].physical_slot == 6);

    llama_kv_attention_view_status status;
    const auto view = llama_kv_attention_view::build(snapshot, { 0 }, 1, status);
    assert(status == llama_kv_attention_view_status::ok);
    assert(view.valid() && view.pages()[0].source_physical_slot == 6);
}

static void test_compact_permuted_gapped_view() {
    llama_kv_attention_view_status status;
    const auto view = llama_kv_attention_view::build(make_snapshot(), { 2, 0 }, status);
    assert(status == llama_kv_attention_view_status::ok);
    assert(view.valid() && view.graph_epoch() == 1);
    assert(view.get_n_kv() == 512);
    assert(view.pages().size() == 2);
    assert(view.pages()[0].logical_page == 2);
    assert(view.pages()[0].source_physical_slot == 7);
    assert(view.pages()[0].compact_row_begin == 0);
    assert(view.pages()[1].compact_row_begin == 256);
    assert(view.native_positions()[0] == 512);
    assert(view.native_positions()[255] == 767);
    assert(view.native_positions()[256] == 0);
    assert(view.native_positions().back() == 255);
    assert(view.native_mask().size() == view.get_n_kv());
    assert(view.native_mask().back() == 1);

    auto fence = view.acquire_graph_fence();
    assert(fence.active());
    fence.release();
    assert(!fence.active());
}

static void test_tail_and_rejections() {
    llama_kv_attention_view_status status;
    const auto tail = llama_kv_attention_view::build(make_snapshot(), { 3 }, status);
    assert(status == llama_kv_attention_view_status::ok && tail.get_n_kv() == 132);

    const auto duplicate = llama_kv_attention_view::build(make_snapshot(), { 0, 0 }, status);
    assert(!duplicate.valid() && status == llama_kv_attention_view_status::duplicate_page);
    const auto absent = llama_kv_attention_view::build(make_snapshot(), { 3 }, status);
    assert(!absent.valid() && status == llama_kv_attention_view_status::not_resident);
}

static void test_operator_contract() {
    llama_kv_attention_view_status view_status;
    const auto view = llama_kv_attention_view::build(make_snapshot(), { 2, 0 }, view_status);
    assert(view_status == llama_kv_attention_view_status::ok);

    llama_kv_attention_operator_params params;
    params.mode = llama_kv_attention_operator_mode::selective;
    params.type_k = GGML_TYPE_TURBO4_0;
    params.type_v = GGML_TYPE_TURBO4_0;
    params.head_dim_k = 256;
    params.head_dim_v = 256;
    params.n_head_q = 16;
    params.n_head_kv = 4;
    params.n_query_tokens = 2;
    params.n_batch = 2;
    params.query_positions = { 700, 701, 500, 501 };

    llama_kv_attention_operator_status status;
    const auto metadata = llama_kv_attention_operator_metadata::build(view, params, status);
    assert(status == llama_kv_attention_operator_status::ok);
    assert(metadata.valid() && metadata.enabled());
    assert(metadata.table_epoch() == view.graph_epoch());
    assert(metadata.graph_reuse_key() == 1);
    assert(metadata.get_n_kv() == view.get_n_kv());
    assert(metadata.n_head_q() / metadata.n_head_kv() == 4);
    assert(metadata.query_positions().size() == 4);
    assert(llama_kv_attention_operator_check_backend(
                GGML_BACKEND_DEVICE_TYPE_CPU, metadata) ==
           llama_kv_attention_backend_status::supported_reference);
    assert(llama_kv_attention_operator_check_backend(
                GGML_BACKEND_DEVICE_TYPE_GPU, metadata) ==
           llama_kv_attention_backend_status::unsupported_backend);
    assert(llama_kv_attention_operator_check_backend(
                static_cast<ggml_backend_t>(nullptr), metadata) ==
           llama_kv_attention_backend_status::unsupported_backend);

    auto wrong_type = params;
    wrong_type.type_v = GGML_TYPE_F16;
    assert(!llama_kv_attention_operator_metadata::build(view, wrong_type, status).valid());
    assert(status == llama_kv_attention_operator_status::invalid_type);

    auto wrong_gqa = params;
    wrong_gqa.n_head_q = 15;
    assert(!llama_kv_attention_operator_metadata::build(view, wrong_gqa, status).valid());
    assert(status == llama_kv_attention_operator_status::invalid_shape);

    auto non_causal = params;
    non_causal.causal = false;
    assert(!llama_kv_attention_operator_metadata::build(view, non_causal, status).valid());
    assert(status == llama_kv_attention_operator_status::non_causal);

    params.mode = llama_kv_attention_operator_mode::off;
    assert(!llama_kv_attention_operator_metadata::build(view, params, status).valid());
    assert(status == llama_kv_attention_operator_status::disabled);

    llm_graph_params graph_a = {};
    graph_a.kv_attention_table_epoch = view.graph_epoch();
    auto graph_b = graph_a;
    assert(graph_a.allow_reuse(graph_a));
    graph_b.kv_attention_table_epoch++;
    assert(!graph_a.allow_reuse(graph_b));
}

int main() {
    test_compact_permuted_gapped_view();
    test_tail_and_rejections();
    test_sequence_scoped_lookup();
    test_operator_contract();
    return 0;
}
