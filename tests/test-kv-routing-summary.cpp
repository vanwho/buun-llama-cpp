#include "llama-kv-routing-summary.h"

#include <cassert>

static llama_kv_page_record make_page(uint32_t logical, uint32_t slot, llama_pos end,
                                      uint32_t generation = 1) {
    llama_kv_page_record page;
    page.id.sequence_id = 0;
    page.id.logical_page = logical;
    page.id.page_generation = generation;
    page.id.position_begin = llama_pos(logical * VBR_GENERATION_PAGE_CELLS);
    page.id.position_end = end;
    page.physical_slot = slot;
    page.state = uint32_t(end - page.id.position_begin) < VBR_GENERATION_PAGE_CELLS
        ? llama_kv_page_state::filling_gpu : llama_kv_page_state::gpu_host_clean;
    page.host_valid = true;
    return page;
}

static llama_kv_residency_snapshot snapshot() {
    llama_kv_residency_table table(8);
    auto tx = table.begin();
    assert(table.replace(tx, make_page(0, 4, 256)) == llama_kv_residency_status::ok);
    assert(table.replace(tx, make_page(1, 1, 512)) == llama_kv_residency_status::ok);
    assert(table.replace(tx, make_page(2, 7, 700)) == llama_kv_residency_status::ok);
    assert(table.publish(tx) == llama_kv_residency_status::ok);
    return table.snapshot();
}

static llama_kv_routing_page_input input(const llama_kv_page_record & page, float needle) {
    llama_kv_routing_page_input result;
    result.id = page.id;
    const size_t rows = size_t(page.id.position_end - page.id.position_begin);
    result.rotated_k_rows.assign(rows * 4, 0.0f);
    // Put the needle in the first and last sampled rows, exercising the
    // deterministic representative selection without depending on row order.
    result.rotated_k_rows[0] = needle;
    result.rotated_k_rows[(rows - 1) * 4] = needle;
    return result;
}

int main() {
    const auto snap = snapshot();
    std::vector<llama_kv_routing_page_input> inputs;
    for (const auto & page : snap.pages()) inputs.push_back(input(page, float(page.id.logical_page + 1)));

    llama_kv_routing_summary_config config;
    config.vector_dim = 4;
    llama_kv_routing_summary_status status;
    const auto store = llama_kv_routing_summary_store::build(snap, inputs, config, status);
    assert(status == llama_kv_routing_summary_status::ok && store.valid());
    assert(store.version() == LLAMA_KV_ROUTING_SUMMARY_VERSION);
    assert(store.accounting().page_count == 3);
    assert(store.accounting().representative_count == 4);
    assert(store.accounting().payload_bytes == 3 * 4 * 4 * sizeof(float));
    assert(store.accounting().charged_bytes == store.accounting().logical_bytes);

    const auto ranked = store.score(snap, { 1, 0, 0, 0 }, 2);
    assert(ranked.status == llama_kv_routing_summary_status::ok);
    assert(ranked.pages_scored == 3 && ranked.top_pages.size() == 2);
    assert(ranked.top_pages[0].logical_page == 2);
    assert(ranked.comparisons == 12);
    assert(store.score(snap, { 0, 1, 0 }, 1).status ==
           llama_kv_routing_summary_status::invalid_argument);

    // Equal scores use logical page as the stable tie breaker.
    std::vector<float> zero(4, 0.0f);
    const auto ties = store.score(snap, zero, 3);
    assert(ties.top_pages[0].logical_page == 0 && ties.top_pages[1].logical_page == 1);

    llama_kv_routing_summary_config too_small = config;
    too_small.byte_budget = store.accounting().charged_bytes - 1;
    assert(!llama_kv_routing_summary_store::build(snap, inputs, too_small, status).valid());
    assert(status == llama_kv_routing_summary_status::insufficient_budget);

    auto missing = inputs;
    missing.pop_back();
    assert(!llama_kv_routing_summary_store::build(snap, missing, config, status).valid());
    assert(status == llama_kv_routing_summary_status::missing_page);

    llama_kv_residency_table changed(8);
    auto tx = changed.begin();
    assert(changed.replace(tx, make_page(0, 4, 256, 2)) == llama_kv_residency_status::ok);
    assert(changed.replace(tx, make_page(1, 1, 512)) == llama_kv_residency_status::ok);
    assert(changed.replace(tx, make_page(2, 7, 700)) == llama_kv_residency_status::ok);
    assert(changed.publish(tx) == llama_kv_residency_status::ok);
    assert(store.score(changed.snapshot(), { 1, 0, 0, 0 }, 1).status ==
           llama_kv_routing_summary_status::stale_summary);
    return 0;
}
