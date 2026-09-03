#include "llama-kv-routing-retrieval.h"

#include <algorithm>
#include <cassert>
#include <iostream>

static llama_kv_page_record make_page(uint32_t logical, llama_kv_page_state state,
                                      uint32_t generation = 1) {
    llama_kv_page_record page;
    page.id.session_generation = 7;
    page.id.sequence_id = 0;
    page.id.sequence_generation = 9;
    page.id.logical_page = logical;
    page.id.page_generation = generation;
    page.id.representation_epoch = 11;
    page.id.model_identity = 13;
    page.id.topology_identity = 17;
    page.id.codec_digest = 19;
    page.id.codebook_digest = 23;
    page.id.rotation_digest = 29;
    page.id.meansub_digest = 31;
    page.id.position_begin = llama_pos(logical * VBR_GENERATION_PAGE_CELLS);
    page.id.position_end = logical == 5
        ? page.id.position_begin + 80
        : page.id.position_begin + VBR_GENERATION_PAGE_CELLS;
    page.physical_slot = logical;
    page.state = state;
    page.host_valid = true;
    page.pin_count = logical == 1 ? 1 : 0;
    return page;
}

static llama_kv_residency_snapshot make_snapshot() {
    llama_kv_residency_table table(8);
    auto tx = table.begin();
    for (uint32_t logical = 0; logical < 6; ++logical) {
        const auto state = logical == 5 ? llama_kv_page_state::filling_gpu
                                        : llama_kv_page_state::gpu_host_clean;
        assert(table.replace(tx, make_page(logical, state)) == llama_kv_residency_status::ok);
    }
    assert(table.publish(tx) == llama_kv_residency_status::ok);
    return table.snapshot();
}

static llama_kv_routing_page_input input(const llama_kv_page_record & page, float value) {
    llama_kv_routing_page_input result;
    result.id = page.id;
    const uint32_t rows = uint32_t(page.id.position_end - page.id.position_begin);
    result.row_indices = { 0, rows / 3, (2 * rows) / 3, rows - 1 };
    result.rotated_k_rows.assign(4 * 4, 0.0f);
    for (size_t i = 0; i < result.row_indices.size(); ++i) {
        result.rotated_k_rows[i * 4] = value;
    }
    result.source_bytes = result.rotated_k_rows.size() * sizeof(float);
    return result;
}

static bool contains(const llama_kv_routing_retrieval_result & result, uint32_t logical,
                     llama_kv_routing_retrieval_reason reason) {
    return std::find_if(result.selected.begin(), result.selected.end(),
            [&](const auto & entry) {
                return entry.id.logical_page == logical && entry.reason == reason;
            }) != result.selected.end();
}

int main() {
    const auto snapshot = make_snapshot();
    llama_kv_routing_summary_config summary_config;
    summary_config.vector_dim = 4;
    summary_config.layer_index = 2;
    summary_config.head_index = 3;
    std::vector<llama_kv_routing_page_input> inputs;
    for (const auto & page : snapshot.pages()) {
        inputs.push_back(input(page, page.id.logical_page == 3 ? 10.0f : 1.0f));
    }
    llama_kv_routing_summary_status summary_status;
    const auto summaries = llama_kv_routing_summary_store::build(
            snapshot, inputs, summary_config, summary_status);
    assert(summary_status == llama_kv_routing_summary_status::ok && summaries.valid());

    llama_kv_routing_query query;
    query.values = { 1.0f, 0.0f, 0.0f, 0.0f };
    query.query_generation = 41;
    query.token_index = 900;
    query.table_epoch = snapshot.epoch();
    query.model_identity = 13;
    query.topology_identity = 17;
    query.representation_epoch = 11;
    query.session_generation = 7;
    query.sequence_generation = 9;
    query.sequence_id = 0;
    query.position = 5 * VBR_GENERATION_PAGE_CELLS + 20;
    query.layer_index = 2;
    query.head_index = 3;

    std::vector<llama_kv_routing_page_attributes> attributes(snapshot.pages().size());
    for (size_t i = 0; i < attributes.size(); ++i) attributes[i].id = snapshot.pages()[i].id;
    attributes[0].structural = true;
    attributes[4].recent = true;
    attributes[5].current = true;
    // The residency pin is authoritative even when the application attribute
    // provider does not label the page as in-flight.

    llama_kv_routing_retrieval_config config;
    config.capacity_pages = 6;
    config.summary_top_k = 1;
    config.exploration_pages = 1;
    config.exploration_seed = 0;
    config.exploration_turn = 0;
    const auto selected = llama_kv_routing_retrieve(
            snapshot, summaries, query, config, attributes);
    assert(selected.status == llama_kv_routing_retrieval_status::ok);
    assert(selected.selected.size() == 6);
    assert(selected.metrics.valid_pages == 6);
    assert(selected.metrics.summary_pages_scored == 6);
    assert(selected.metrics.summary_comparisons == 24);
    assert(contains(selected, 0, llama_kv_routing_retrieval_reason::structural));
    assert(contains(selected, 1, llama_kv_routing_retrieval_reason::mandatory));
    assert(contains(selected, 4, llama_kv_routing_retrieval_reason::recent));
    assert(contains(selected, 3, llama_kv_routing_retrieval_reason::summary));
    assert(contains(selected, 2, llama_kv_routing_retrieval_reason::exploration));
    assert(!contains(selected, 5, llama_kv_routing_retrieval_reason::summary));
    assert(selected.selected[0].id.logical_page == 5); // current/write page first
    assert(selected.selected[0].page_distance == 0);

    // A capacity that admits the full logical inventory is dense-equivalent,
    // even when the explicit candidate budgets are zero.
    config.capacity_pages = 6;
    config.summary_top_k = 0;
    config.exploration_pages = 0;
    const auto all_pages = llama_kv_routing_retrieve(
            snapshot, summaries, query, config, attributes);
    assert(all_pages.status == llama_kv_routing_retrieval_status::ok);
    assert(all_pages.selected.size() == 6);

    // Missing summaries are unavailable evidence, not zero-scored pages. The
    // previous target is used only as an explicit safe fallback.
    config.capacity_pages = 5;
    config.summary_top_k = 1;
    const auto unavailable = llama_kv_routing_retrieve(
            snapshot, llama_kv_routing_summary_store{}, query, config, attributes, {
                snapshot.pages()[2].id });
    assert(unavailable.status == llama_kv_routing_retrieval_status::unavailable_summary);
    assert(unavailable.metrics.fallback_used && !unavailable.metrics.summary_complete);
    assert(contains(unavailable, 2, llama_kv_routing_retrieval_reason::fallback));
    for (const auto & entry : unavailable.selected) assert(!entry.score_available);

    config.capacity_pages = 2;
    const auto overflow = llama_kv_routing_retrieve(
            snapshot, summaries, query, config, attributes);
    assert(overflow.status == llama_kv_routing_retrieval_status::mandatory_overflow);
    assert(overflow.selected.empty());

    auto stale_query = query;
    stale_query.table_epoch = snapshot.epoch() + 1;
    assert(llama_kv_routing_retrieve(snapshot, summaries, stale_query, config, attributes).status ==
           llama_kv_routing_retrieval_status::stale_query);
    auto invalid_query = query;
    invalid_query.values = { 1.0f };
    assert(llama_kv_routing_retrieve(snapshot, summaries, invalid_query, config, attributes).status ==
           llama_kv_routing_retrieval_status::invalid_argument);

    // Exploration rotates over the same deterministic logical order.
    std::vector<llama_kv_routing_page_attributes> no_attributes;
    config.capacity_pages = 3;
    config.summary_top_k = 0;
    config.exploration_pages = 2;
    config.exploration_turn = 0;
    const auto exploration_zero = llama_kv_routing_retrieve(
            snapshot, summaries, query, config, no_attributes);
    config.exploration_turn = 1;
    const auto exploration_one = llama_kv_routing_retrieve(
            snapshot, summaries, query, config, no_attributes);
    assert(exploration_zero.status == llama_kv_routing_retrieval_status::ok);
    assert(exploration_one.status == llama_kv_routing_retrieval_status::ok);
    assert(exploration_zero.selected[0].id.logical_page == 1); // residency pin
    assert(exploration_zero.selected[1].id.logical_page == 0);
    assert(exploration_one.selected[0].id.logical_page == 1);
    assert(exploration_one.selected[1].id.logical_page == 2);

    std::cout << "kv routing retrieval checks passed\n";
}
