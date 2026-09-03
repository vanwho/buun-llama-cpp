#include "llama-kv-attention-view.h"

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

int main() {
    test_compact_permuted_gapped_view();
    test_tail_and_rejections();
    return 0;
}
