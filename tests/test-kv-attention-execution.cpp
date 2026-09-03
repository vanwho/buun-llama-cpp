#include "llama-kv-attention-execution.h"
#include "llama-graph.h"

#include <cassert>

static llama_kv_page_record page(uint32_t logical, uint32_t slot, llama_pos end) {
    llama_kv_page_record result;
    result.id.sequence_id = 0;
    result.id.logical_page = logical;
    result.id.position_begin = llama_pos(logical * VBR_GENERATION_PAGE_CELLS);
    result.id.position_end = end;
    result.physical_slot = slot;
    result.state = uint32_t(end - result.id.position_begin) < VBR_GENERATION_PAGE_CELLS
        ? llama_kv_page_state::filling_gpu : llama_kv_page_state::gpu_host_clean;
    result.host_valid = true;
    return result;
}

static llama_kv_residency_snapshot snapshot_slots(
        uint32_t slot0, uint32_t slot1, uint32_t slot2, uint32_t last_end = 700) {
    llama_kv_residency_table table(8);
    auto tx = table.begin();
    assert(table.replace(tx, page(0, slot0, 256)) == llama_kv_residency_status::ok);
    assert(table.replace(tx, page(1, slot1, 512)) == llama_kv_residency_status::ok);
    assert(table.replace(tx, page(2, slot2, last_end)) == llama_kv_residency_status::ok);
    assert(table.publish(tx) == llama_kv_residency_status::ok);
    if (last_end != 700) {
        auto next = table.begin();
        assert(table.publish(next) == llama_kv_residency_status::ok);
    }
    return table.snapshot();
}

static llama_kv_residency_snapshot snapshot(uint32_t last_end = 700) {
    return snapshot_slots(5, 1, 7, last_end);
}

static llama_kv_attention_operator_metadata metadata(
        const llama_kv_residency_snapshot & snap, uint32_t n_query, uint32_t n_batch,
        const std::vector<uint32_t> & selected_pages = { 2, 0 },
        llama_pos query_position = 600) {
    llama_kv_attention_view_status view_status;
    const auto view = llama_kv_attention_view::build(snap, selected_pages, view_status);
    assert(view_status == llama_kv_attention_view_status::ok);

    llama_kv_attention_operator_params params;
    params.mode = llama_kv_attention_operator_mode::selective;
    params.type_k = GGML_TYPE_TURBO4_0;
    params.type_v = GGML_TYPE_TURBO4_0;
    params.head_dim_k = 256;
    params.head_dim_v = 256;
    params.n_head_q = 16;
    params.n_head_kv = 4;
    params.n_query_tokens = n_query;
    params.n_batch = n_batch;
    params.query_positions.resize(size_t(n_query) * n_batch, query_position);

    llama_kv_attention_operator_status status;
    auto result = llama_kv_attention_operator_metadata::build(view, params, status);
    assert(status == llama_kv_attention_operator_status::ok);
    return result;
}

static void test_prefill_admission() {
    llama_kv_attention_prefill_admission admission;
    assert(admission.append(0, 128) == llama_kv_attention_execution_status::ok);
    assert(admission.append(0, 128) == llama_kv_attention_execution_status::ok);
    assert(admission.append(1, 44) == llama_kv_attention_execution_status::ok);
    assert(admission.append(2, 1) == llama_kv_attention_execution_status::invalid_prefill_transition);
    assert(admission.finish_tail() == llama_kv_attention_execution_status::ok);
    assert(admission.begin_decode() == llama_kv_attention_execution_status::ok);
    assert(admission.decode_ready() && admission.phase() == llama_kv_attention_execution_phase::decode);
    assert(admission.page_count() == 2 && admission.resident_rows() == 300);
}

static void test_routes_epochs_and_fences() {
    const auto selected_prefill = metadata(snapshot(), 2, 1);
    const auto selected_decode = metadata(snapshot(), 1, 1);
    llama_kv_attention_scratch_request scratch;
    scratch.resident_rows = selected_prefill.get_n_kv();
    scratch.transfer_rows = 16;
    scratch.router_rows = 8;
    scratch.bytes_per_row = 4;
    assert(scratch.required_rows() == selected_prefill.get_n_kv() + 24);
    assert(scratch.required_bytes() == scratch.required_rows() * 4);

    llama_kv_attention_execution execution(llama_kv_attention_execution_mode::selective);
    auto first = execution.prepare(selected_prefill, llama_kv_attention_execution_phase::prefill,
            3, 7, true, scratch);
    assert(first.status == llama_kv_attention_execution_status::ok);
    assert(first.route == llama_kv_attention_execution_route::selected_reference);
    assert(first.graph_rebuild && execution.in_flight_graphs() == 1);
    execution.complete_one_graph();

    auto reused = execution.prepare(selected_prefill, llama_kv_attention_execution_phase::prefill,
            3, 7, true, scratch);
    assert(!reused.graph_rebuild && execution.in_flight_graphs() == 1);
    execution.complete_one_graph();

    auto representation = execution.prepare(selected_prefill, llama_kv_attention_execution_phase::prefill,
            4, 7, true, scratch);
    assert(representation.graph_rebuild);
    execution.complete_one_graph();

    auto shape = execution.prepare(selected_prefill, llama_kv_attention_execution_phase::prefill,
            4, 8, true, scratch);
    assert(shape.graph_rebuild);
    execution.complete_one_graph();

    auto direct = execution.prepare(selected_decode, llama_kv_attention_execution_phase::decode,
            4, 8, true, scratch);
    assert(direct.route == llama_kv_attention_execution_route::selected_direct);
    assert(direct.graph_rebuild && execution.in_flight_graphs() == 1);

    // A second table epoch must coexist with the old graph until both complete.
    auto changed_table = execution.prepare(metadata(snapshot(701), 1, 1),
            llama_kv_attention_execution_phase::decode, 4, 8, true, scratch);
    assert(changed_table.graph_rebuild && execution.in_flight_graphs() == 2);
    execution.complete_one_graph();
    execution.complete_one_graph();
    assert(execution.in_flight_graphs() == 0);
}

static void test_fallbacks_and_graph_key() {
    const auto selected = metadata(snapshot(), 1, 1);
    llama_kv_attention_scratch_request scratch;
    llama_kv_attention_execution execution(llama_kv_attention_execution_mode::selective);

    auto reference = execution.prepare(selected, llama_kv_attention_execution_phase::decode,
            1, 1, false, scratch);
    assert(reference.route == llama_kv_attention_execution_route::selected_reference);
    execution.complete_one_graph();

    auto prompt_shape = metadata(snapshot(), 2, 1);
    auto prompt_reference = execution.prepare(prompt_shape,
            llama_kv_attention_execution_phase::decode, 1, 1, true, scratch);
    assert(prompt_reference.route == llama_kv_attention_execution_route::selected_reference);
    execution.complete_one_graph();

    llama_kv_attention_execution observing(llama_kv_attention_execution_mode::observe);
    auto observe = observing.prepare({}, llama_kv_attention_execution_phase::decode,
            0, 0, true, scratch);
    assert(observe.route == llama_kv_attention_execution_route::observe);
    observing.complete_one_graph();

    llama_kv_attention_execution disabled(llama_kv_attention_execution_mode::off);
    auto off = disabled.prepare(selected, llama_kv_attention_execution_phase::decode,
            0, 0, true, scratch);
    assert(off.status == llama_kv_attention_execution_status::disabled);
    assert(off.route == llama_kv_attention_execution_route::dense);
    assert(disabled.in_flight_graphs() == 0);

    auto refusal = execution.prepare({}, llama_kv_attention_execution_phase::decode,
            1, 1, true, scratch);
    assert(refusal.status == llama_kv_attention_execution_status::invalid_metadata);
    assert(refusal.route == llama_kv_attention_execution_route::refusal);

    llama_kv_attention_scratch_request overflow;
    overflow.resident_rows = UINT64_MAX;
    auto overflow_result = execution.prepare(selected,
            llama_kv_attention_execution_phase::decode, 1, 1, true, overflow);
    assert(overflow_result.status == llama_kv_attention_execution_status::overflow);
    assert(overflow_result.route == llama_kv_attention_execution_route::refusal);

    llm_graph_params a = {};
    a.kv_attention_table_epoch = 1;
    a.kv_attention_content_key = 11;
    a.kv_attention_representation_epoch = 2;
    a.kv_attention_shape_epoch = 3;
    auto b = a;
    llama_kv_attention_execution_metrics metrics_a;
    llama_kv_attention_execution_metrics metrics_b;
    a.kv_attention_metrics = &metrics_a;
    b.kv_attention_metrics = &metrics_b;
    // Metrics ownership is controller state, not graph topology.
    assert(a.allow_reuse(b));
    ++b.kv_attention_representation_epoch;
    assert(!a.allow_reuse(b));
    b = a;
    ++b.kv_attention_content_key;
    assert(!a.allow_reuse(b));
    execution.complete_one_graph();
}

static void test_epoch_matrix_and_lifetime_metrics() {
    const auto base = metadata(snapshot(), 1, 1);
    const auto reordered = metadata(snapshot(), 1, 1, { 0, 2 });
    const auto remapped = metadata(snapshot_slots(6, 1, 7), 1, 1);
    const auto grown = metadata(snapshot(701), 1, 1);
    const auto query_changed = metadata(snapshot(), 1, 1, { 2, 0 }, 601);
    const auto shape_changed = metadata(snapshot(), 2, 1);

    assert(base.graph_content_key() != reordered.graph_content_key());
    assert(base.graph_content_key() != remapped.graph_content_key());
    assert(base.table_epoch() != grown.table_epoch());
    assert(base.graph_content_key() != grown.graph_content_key());
    assert(base.graph_content_key() != query_changed.graph_content_key());
    assert(base.graph_content_key() != shape_changed.graph_content_key());

    llama_kv_attention_scratch_request scratch;
    scratch.resident_rows = base.get_n_kv();
    scratch.bytes_per_row = 64;
    llama_kv_attention_execution execution(llama_kv_attention_execution_mode::selective);
    execution.reset_metrics();

    const auto first = execution.prepare(base, llama_kv_attention_execution_phase::decode,
            11, 22, true, scratch);
    assert(first.route == llama_kv_attention_execution_route::selected_direct);
    assert(first.graph_rebuild);

    auto irrelevant_scratch = scratch;
    irrelevant_scratch.router_rows = 99;
    const auto reused = execution.prepare(base, llama_kv_attention_execution_phase::decode,
            11, 22, true, irrelevant_scratch);
    assert(!reused.graph_rebuild);

    const auto reference = execution.prepare(base, llama_kv_attention_execution_phase::decode,
            11, 22, false, scratch);
    assert(reference.route == llama_kv_attention_execution_route::selected_reference);
    assert(reference.graph_rebuild);

    const auto order_change = execution.prepare(reordered,
            llama_kv_attention_execution_phase::decode, 11, 22, false, scratch);
    assert(order_change.graph_rebuild);
    const auto slot_change = execution.prepare(remapped,
            llama_kv_attention_execution_phase::decode, 11, 22, false, scratch);
    assert(slot_change.graph_rebuild);
    const auto representation_change = execution.prepare(base,
            llama_kv_attention_execution_phase::decode, 12, 22, false, scratch);
    assert(representation_change.graph_rebuild);
    const auto query_change = execution.prepare(query_changed,
            llama_kv_attention_execution_phase::decode, 12, 22, false, scratch);
    assert(query_change.graph_rebuild);
    const auto tail_growth = execution.prepare(grown,
            llama_kv_attention_execution_phase::decode, 12, 22, false, scratch);
    assert(tail_growth.graph_rebuild);
    const auto shape_change = execution.prepare(shape_changed,
            llama_kv_attention_execution_phase::prefill, 12, 23, false, scratch);
    assert(shape_change.graph_rebuild);

    const auto & counters = execution.metrics();
    assert(counters.graph_capture_count == 8);
    assert(counters.graph_replay_count == 1);
    assert(counters.graph_rebuild_count == counters.graph_capture_count);
    assert(counters.graph_submission_count == 9);
    assert(counters.table_upload_bytes == 2 * (4 * sizeof(uint32_t) + sizeof(int64_t)) * 2);
    assert(counters.scratch_high_water_rows == irrelevant_scratch.required_rows());
    assert(counters.scratch_high_water_bytes == irrelevant_scratch.required_bytes());

    // Clearing the current key must leave the old immutable view leased. The
    // replacement is visible to later submissions while both leases coexist.
    llama_kv_attention_execution fenced(llama_kv_attention_execution_mode::selective);
    assert(fenced.prepare(base, llama_kv_attention_execution_phase::decode,
                1, 1, true, scratch).graph_rebuild);
    assert(fenced.in_flight_graphs() == 1);
    fenced.clear();
    assert(!fenced.has_graph() && fenced.in_flight_graphs() == 1);
    assert(fenced.prepare(grown, llama_kv_attention_execution_phase::decode,
                1, 1, true, scratch).graph_rebuild);
    assert(fenced.in_flight_graphs() == 2);
    fenced.complete_one_graph();
    assert(fenced.in_flight_graphs() == 1);
    fenced.complete_one_graph();
    assert(fenced.in_flight_graphs() == 0);
}

int main() {
    test_prefill_admission();
    test_routes_epochs_and_fences();
    test_fallbacks_and_graph_key();
    test_epoch_matrix_and_lifetime_metrics();
    return 0;
}
