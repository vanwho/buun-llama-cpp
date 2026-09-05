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

static llama_kv_routing_page_input sparse_input(const llama_kv_page_record & page, float needle) {
    llama_kv_routing_page_input result;
    result.id = page.id;
    const uint32_t rows = uint32_t(page.id.position_end - page.id.position_begin);
    result.row_indices = { 0, rows / 3, (2 * rows) / 3, rows - 1 };
    result.rotated_k_rows.assign(4 * 4, 0.0f);
    for (size_t i = 0; i < result.row_indices.size(); ++i) {
        result.rotated_k_rows[i * 4] = needle;
    }
    result.source_bytes = 4 * sizeof(float) * 4;
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

    // A cold host record is part of the logical index even though it is not in
    // the residency snapshot. Its summary survives the residency-only view.
    auto complete_inventory = snap.pages();
    auto cold_page = make_page(3, UINT32_MAX, 900);
    cold_page.state = llama_kv_page_state::host_clean;
    cold_page.host_valid = true;
    complete_inventory.push_back(cold_page);
    auto complete_inputs = inputs;
    complete_inputs.push_back(input(cold_page, 9.0f));
    const auto complete = llama_kv_routing_summary_store::build(
            snap, complete_inventory, complete_inputs, config, status);
    assert(status == llama_kv_routing_summary_status::ok && complete.valid());
    assert(complete.accounting().page_count == 4);
    const auto complete_ranked = complete.score(
            snap, complete_inventory, { 1, 0, 0, 0 }, 4);
    assert(complete_ranked.status == llama_kv_routing_summary_status::ok);
    assert(complete_ranked.pages_scored == 4);
    assert(complete.contains(3));
    auto retained = complete.reconcile(snap, complete_inventory, status);
    assert(status == llama_kv_routing_summary_status::ok && retained.contains(3));

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

    // Seal-time producers may provide only the bounded sampled rows. The
    // store must not require a full-page source image in that form.
    std::vector<llama_kv_routing_page_input> sparse_inputs;
    for (const auto & page : snap.pages()) sparse_inputs.push_back(sparse_input(page, float(page.id.logical_page + 1)));
    const auto sparse = llama_kv_routing_summary_store::build(snap, sparse_inputs, config, status);
    assert(status == llama_kv_routing_summary_status::ok && sparse.valid());
    assert(sparse.accounting().source_rows == 12);
    assert(sparse.accounting().source_bytes == 12 * sizeof(float) * 4);

    auto centroid_config = config;
    centroid_config.form = llama_kv_routing_summary_form::centroid_upper_bound;
    const auto centroid = llama_kv_routing_summary_store::build(
            snap, sparse_inputs, centroid_config, status);
    assert(status == llama_kv_routing_summary_status::ok && centroid.valid());
    assert(centroid.accounting().payload_bytes == 3 * (4 + 1) * sizeof(float));
    const auto bounded = centroid.score(snap, { 1, 0, 0, 0 }, 3);
    assert(bounded.status == llama_kv_routing_summary_status::ok);
    assert(bounded.top_pages[0].upper_bound);

    // Page seals update one summary without rereading the other pages.
    llama_kv_residency_table incremental_table(8);
    auto incremental_tx = incremental_table.begin();
    assert(incremental_table.replace(incremental_tx, make_page(0, 4, 256)) == llama_kv_residency_status::ok);
    assert(incremental_table.publish(incremental_tx) == llama_kv_residency_status::ok);
    auto incremental = llama_kv_routing_summary_store{};
    incremental = incremental.update_page(incremental_table.snapshot(),
            sparse_input(incremental_table.snapshot().pages()[0], 5.0f), config, status);
    assert(status == llama_kv_routing_summary_status::ok && incremental.contains(0));
    const uint64_t first_hash = incremental.content_hash();
    auto add_tx = incremental_table.begin();
    assert(incremental_table.replace(add_tx, make_page(1, 1, 512)) == llama_kv_residency_status::ok);
    assert(incremental_table.publish(add_tx) == llama_kv_residency_status::ok);
    const auto page_one = incremental_table.snapshot().pages()[1];
    incremental = incremental.update_page(incremental_table.snapshot(),
            sparse_input(page_one, 6.0f), config, status);
    assert(status == llama_kv_routing_summary_status::ok && incremental.contains(0) && incremental.contains(1));
    assert(incremental.accounting().build_count == 2 && incremental.content_hash() != first_hash);

    auto changed_tx = incremental_table.begin();
    auto changed_page = changed_tx.pages()[0];
    changed_page.id.page_generation++;
    assert(incremental_table.update(changed_tx, changed_page) == llama_kv_residency_status::ok);
    assert(incremental_table.publish(changed_tx) == llama_kv_residency_status::ok);
    auto reconciled = incremental.reconcile(incremental_table.snapshot(), status);
    assert(status == llama_kv_routing_summary_status::ok && !reconciled.contains(0) && reconciled.contains(1));
    assert(reconciled.accounting().invalidation_count == 1);
    return 0;
}
