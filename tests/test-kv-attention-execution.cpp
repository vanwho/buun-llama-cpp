#include "llama-kv-attention-execution.h"
#include "llama-graph.h"
#include "llama-context.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

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

static llama_kv_residency_snapshot snapshot_high_logical_positions() {
    llama_kv_residency_table table(8);
    auto tx = table.begin();
    assert(table.replace(tx, page(1024, 5, 1024 * VBR_GENERATION_PAGE_CELLS + 256)) ==
           llama_kv_residency_status::ok);
    assert(table.replace(tx, page(1025, 1, 1025 * VBR_GENERATION_PAGE_CELLS + 256)) ==
           llama_kv_residency_status::ok);
    assert(table.replace(tx, page(1026, 7, 1026 * VBR_GENERATION_PAGE_CELLS + 188)) ==
           llama_kv_residency_status::ok);
    assert(table.publish(tx) == llama_kv_residency_status::ok);
    return table.snapshot();
}

static llama_kv_attention_operator_metadata metadata(
        const llama_kv_residency_snapshot & snap, uint32_t n_query, uint32_t n_batch,
        const std::vector<uint32_t> & selected_pages = { 2, 0 },
        llama_pos query_position = 600, uint32_t n_head_q = 16,
        uint32_t n_head_kv = 4, ggml_type type_k = GGML_TYPE_TURBO4_0,
        ggml_type type_v = GGML_TYPE_TURBO4_0) {
    llama_kv_attention_view_status view_status;
    const auto view = llama_kv_attention_view::build(snap, selected_pages, view_status);
    assert(view_status == llama_kv_attention_view_status::ok);

    llama_kv_attention_operator_params params;
    params.mode = llama_kv_attention_operator_mode::selective;
    params.type_k = type_k;
    params.type_v = type_v;
    const bool turbo_kv = ggml_is_turbo_kv_type(type_k) || ggml_is_turbo_kv_type(type_v);
    params.domain_k = turbo_kv
        ? llama_kv_attention_representation_domain::turbo_rotated
        : llama_kv_attention_representation_domain::original;
    params.domain_v = params.domain_k;
    params.head_dim_k = 256;
    params.head_dim_v = 256;
    params.n_head_q = n_head_q;
    params.n_head_kv = n_head_kv;
    params.n_query_tokens = n_query;
    params.n_batch = n_batch;
    params.query_positions.resize(size_t(n_query) * n_batch, query_position);

    llama_kv_attention_operator_status status;
    auto result = llama_kv_attention_operator_metadata::build(view, params, status);
    assert(status == llama_kv_attention_operator_status::ok);
    return result;
}

static void test_prefill_admission() {
    assert(!llama_kv_attention_hotpath_profile_enabled(nullptr));
    assert(!llama_kv_attention_hotpath_profile_enabled("0"));
    assert(!llama_kv_attention_hotpath_profile_enabled("true"));
    assert(llama_kv_attention_hotpath_profile_enabled("1"));
    assert(LLAMA_KV_ATTENTION_PREFILL_QUERY_TILE == 64);
    llama_kv_attention_prefill_admission admission;
    assert(admission.append(0, 128) == llama_kv_attention_execution_status::ok);
    assert(admission.append(0, 128) == llama_kv_attention_execution_status::ok);
    assert(admission.append(1, 44) == llama_kv_attention_execution_status::ok);
    assert(admission.append(2, 1) == llama_kv_attention_execution_status::invalid_prefill_transition);
    assert(admission.finish_tail() == llama_kv_attention_execution_status::ok);
    assert(admission.begin_decode() == llama_kv_attention_execution_status::ok);
    assert(admission.decode_ready() && admission.phase() == llama_kv_attention_execution_phase::decode);
    assert(admission.page_count() == 2 && admission.resident_rows() == 300);

    // The live prefill scheduler uses the physical window, not the logical
    // context, as its upper bound. A zero window preserves the unbounded
    // feature-off/observe convention.
    assert(llama_kv_attention_prefill_chunk_size(4096, 2) == 512);
    assert(llama_kv_attention_prefill_chunk_size(4096, 0) == 4096);
    assert(llama_kv_attention_prefill_chunk_size(4096, 2, 128) == 256);
    assert(llama_kv_attention_prefill_chunk_size(4096, 2, 256, 16) == 16);
    assert(llama_kv_attention_prefill_chunk_size(8, 2, 256, 16) == 8);
    assert(llama_kv_attention_prefill_chunk_size(4096, 2, 256, 32) == 32);
    assert(llama_kv_attention_prefill_chunk_size(0, 2) == 0);

    const auto batch_256 = llama_kv_attention_prefill_batch_plan_make(256, 2);
    assert(batch_256.requested_batch == 256 &&
           batch_256.physical_write_capacity == 512 &&
           batch_256.effective_batch == 256 && batch_256.subbatch_count == 1);
    const auto batch_512 = llama_kv_attention_prefill_batch_plan_make(512, 2);
    assert(batch_512.effective_batch == 512 && batch_512.subbatch_count == 1);
    const auto batch_split = llama_kv_attention_prefill_batch_plan_make(1024, 2);
    assert(batch_split.effective_batch == 512 && batch_split.subbatch_count == 2);
    const auto batch_h_small = llama_kv_attention_prefill_batch_plan_make(512, 1);
    assert(batch_h_small.physical_write_capacity == 256 &&
           batch_h_small.effective_batch == 256 && batch_h_small.subbatch_count == 2);

    // The model batch may cross the CUDA tile boundary; the direct backend
    // subdivides it in grid.z instead of refusing the whole operator.
    const auto selected_large = metadata(snapshot(), 65, 1);
    llama_kv_attention_execution large_execution(llama_kv_attention_execution_mode::selective);
    large_execution.set_route_override("packed");
    const auto large_decision = large_execution.prepare(
            selected_large, llama_kv_attention_execution_phase::prefill,
            1, 1, true, {}, {}, false, true);
    assert(large_decision.status == llama_kv_attention_execution_status::ok);
    assert(large_decision.route == llama_kv_attention_execution_route::selected_packed);
}

static void test_graph_thread_config_cache() {
    llama_context_graph_thread_config_cache cache;
    auto pool_a = reinterpret_cast<ggml_threadpool_t>(uintptr_t(1));
    auto pool_b = reinterpret_cast<ggml_threadpool_t>(uintptr_t(2));

    assert(cache.needs_update(8, pool_a));
    cache.record(8, pool_a);
    assert(!cache.needs_update(8, pool_a));
    assert(cache.needs_update(4, pool_a));
    assert(cache.needs_update(8, pool_b));
}

static void test_query_pages_cover_cross_page_ubatch() {
    std::vector<llama_pos> positions;
    positions.reserve(256);
    for (llama_pos position = 4982; position < 5238; ++position) {
        positions.push_back(position);
    }
    std::vector<uint32_t> pages;
    assert(llama_kv_attention_query_page_ids(positions, 256, pages));
    assert(pages.size() == 2 && pages[0] == 19 && pages[1] == 20);
    assert(llama_kv_attention_query_page_ids({ 0, 1, 255, 256, 257 }, 256, pages));
    assert(pages.size() == 2 && pages[0] == 0 && pages[1] == 1);
    assert(llama_kv_attention_query_page_ids({ 0, 256, 0 }, 256, pages));
    assert(pages.size() == 2 && pages[0] == 0 && pages[1] == 1);
    std::vector<uint32_t> refreshed;
    assert(llama_kv_attention_refresh_page_ids({ 0, 1 }, { 7, 5 }, 2, refreshed));
    assert(refreshed.size() == 2 && refreshed[0] == 0 && refreshed[1] == 1);
    assert(!llama_kv_attention_refresh_page_ids({ 0, 1, 2 }, { 7 }, 2, refreshed));
    assert(refreshed.empty());
    assert(!llama_kv_attention_query_page_ids({ 0, -1 }, 256, pages));
    assert(pages.empty());
    std::fprintf(stdout, "mandatory_query_pages_survive_routing_refresh=pass "
        "nonmonotonic_positions=0,256,0 unique_pages=2\n");
}

static void test_routes_epochs_and_fences() {
    const auto selected_prefill = metadata(snapshot(), 2, 1);
    const auto selected_decode = metadata(snapshot(), 1, 1);
    const auto selected_single_page = metadata(snapshot(), 1, 1, { 0 }, 255);
    llama_kv_attention_scratch_request scratch;
    scratch.resident_rows = selected_prefill.get_n_kv();
    scratch.transfer_rows = 16;
    scratch.router_rows = 8;
    scratch.bytes_per_row = 4;
    assert(scratch.required_rows() == selected_prefill.get_n_kv() + 24);
    assert(scratch.required_bytes() == scratch.required_rows() * 4);
    scratch.packed_bytes = 64;
    assert(scratch.required_bytes() == scratch.required_rows() * 4 + 64);
    scratch.packed_bytes = 0;

    llama_kv_attention_execution execution(llama_kv_attention_execution_mode::selective);
    execution.set_route_override("packed");
    auto first = execution.prepare(selected_prefill, llama_kv_attention_execution_phase::prefill,
            3, 7, true, scratch, {}, false, true);
    assert(first.status == llama_kv_attention_execution_status::ok);
    assert(first.route == llama_kv_attention_execution_route::selected_packed);
    assert(first.graph_rebuild && execution.in_flight_graphs() == 1);
    execution.complete_one_graph();

    auto reused = execution.prepare(selected_prefill, llama_kv_attention_execution_phase::prefill,
            3, 7, true, scratch, {}, false, true);
    assert(!reused.graph_rebuild && execution.in_flight_graphs() == 1);
    execution.complete_one_graph();

    auto representation = execution.prepare(selected_prefill, llama_kv_attention_execution_phase::prefill,
            4, 7, true, scratch, {}, false, true);
    assert(representation.graph_rebuild);
    execution.complete_one_graph();

    auto shape = execution.prepare(selected_prefill, llama_kv_attention_execution_phase::prefill,
            4, 8, true, scratch, {}, false, true);
    assert(shape.graph_rebuild);
    execution.complete_one_graph();

    auto direct = execution.prepare(selected_decode, llama_kv_attention_execution_phase::decode,
            4, 8, true, scratch, {}, false, true);
    assert(direct.route == llama_kv_attention_execution_route::selected_packed);
    assert(direct.graph_rebuild && execution.in_flight_graphs() == 1);

    // A tail-only publication keeps the physical source views and replays
    // the graph while the mutable current-row descriptor is refreshed.
    auto changed_table = execution.prepare(metadata(snapshot(701), 1, 1),
            llama_kv_attention_execution_phase::decode, 4, 8, true, scratch, {}, false, true);
    assert(!changed_table.graph_rebuild && execution.in_flight_graphs() == 2);
    execution.complete_one_graph();
    execution.complete_one_graph();
    assert(execution.in_flight_graphs() == 0);

    const auto & route_metrics = execution.metrics();
    assert(route_metrics.prefill_routes.selected_packed == 4);
    assert(route_metrics.decode_routes.selected_packed == 2);
    assert(route_metrics.selected_page_count == 2);
    assert(route_metrics.selected_page_ids.size() == 2 &&
           route_metrics.selected_page_ids[0] == 0 &&
           route_metrics.selected_page_ids[1] == 2);
    const auto selected_mtp = metadata(snapshot(), 2, 1);
    const auto mtp = execution.prepare(selected_mtp,
            llama_kv_attention_execution_phase::mtp_verify, 4, 8, true, scratch, {}, false, true);
    assert(mtp.route == llama_kv_attention_execution_route::selected_packed);
    assert(execution.metrics().mtp_verify_routes.selected_packed == 1);
    execution.complete_one_graph();
    assert(llama_kv_attention_execution_phase_name(
            llama_kv_attention_execution_phase::mtp_verify) == std::string("mtp_verify"));

    execution.set_route_override("auto");
    const auto dense = execution.prepare(selected_prefill,
            llama_kv_attention_execution_phase::prefill, 5, 9, false, scratch,
            {}, true, false);
    assert(dense.route == llama_kv_attention_execution_route::selected_dense);
    execution.complete_one_graph();

    execution.set_route_override("packed");
    const auto packed = execution.prepare(selected_prefill,
            llama_kv_attention_execution_phase::prefill, 6, 10, false, scratch,
            {}, false, true);
    assert(packed.route == llama_kv_attention_execution_route::selected_packed);
    execution.complete_one_graph();
    execution.record_pack(128, 7);
    assert(execution.metrics().pack_bytes == 128 &&
           execution.metrics().pack_time_us == 7 &&
           execution.metrics().pack_epochs == 1);

    // Automatic noncontiguous Turbo4 selection uses the retained packed view.
    llama_kv_attention_execution automatic_packed(
            llama_kv_attention_execution_mode::selective);
    const auto packed_default = automatic_packed.prepare(selected_prefill,
            llama_kv_attention_execution_phase::prefill, 7, 11, true, scratch,
            {}, false, true);
    assert(packed_default.route == llama_kv_attention_execution_route::selected_packed);
    assert(automatic_packed.metrics().selected_page_ids.size() == 2 &&
           automatic_packed.metrics().selected_page_ids[0] == 0 &&
           automatic_packed.metrics().selected_page_ids[1] == 2);
    assert(automatic_packed.metrics().pack_bytes == 0);
    assert(automatic_packed.metrics().pack_epochs == 0);
    automatic_packed.complete_one_graph();

    // A table/tail publication updates the mutable descriptor inputs while
    // preserving the packed graph topology and version-keyed packed owner.
    const auto packed_replay = automatic_packed.prepare(metadata(snapshot(701), 2, 1),
            llama_kv_attention_execution_phase::prefill, 7, 11, true, scratch,
            {}, false, true);
    assert(!packed_replay.graph_rebuild);
    assert(automatic_packed.metrics().selected_page_ids.size() == 2 &&
           automatic_packed.metrics().selected_page_ids[0] == 0 &&
           automatic_packed.metrics().selected_page_ids[1] == 2);
    assert(automatic_packed.metrics().pack_bytes == 0);
    assert(automatic_packed.metrics().pack_epochs == 0);
    automatic_packed.complete_one_graph();

    // Small prefill verification shares the measured packed route.
    const auto packed_prefill = metadata(snapshot(), 2, 1);
    const auto packed_policy = automatic_packed.prepare(packed_prefill,
            llama_kv_attention_execution_phase::prefill, 8, 12, true, scratch,
            {}, false, true);
    assert(packed_policy.route == llama_kv_attention_execution_route::selected_packed);
    assert(automatic_packed.metrics().pack_bytes == 0);
    assert(automatic_packed.metrics().pack_epochs == 0);
    automatic_packed.complete_one_graph();

    llama_kv_attention_execution explicit_reference(
            llama_kv_attention_execution_mode::selective);
    explicit_reference.set_route_override("reference");
    const auto reference_control = explicit_reference.prepare(selected_prefill,
            llama_kv_attention_execution_phase::prefill, 7, 11, true, scratch,
            {}, false, true);
    assert(reference_control.route == llama_kv_attention_execution_route::selected_reference);
    explicit_reference.complete_one_graph();

    // Explicit direct remains available for diagnostics and uses the same
    // selection as automatic packed dispatch.
    llama_kv_attention_execution explicit_direct(
            llama_kv_attention_execution_mode::selective);
    explicit_direct.set_route_override("direct");
    const auto explicit_direct_decision = explicit_direct.prepare(selected_single_page,
            llama_kv_attention_execution_phase::decode, 7, 11, true, scratch,
            {}, false, true);
    assert(explicit_direct_decision.route == llama_kv_attention_execution_route::selected_direct);
    const auto explicit_page_ids = explicit_direct.metrics().selected_page_ids;
    explicit_direct.complete_one_graph();

    llama_kv_attention_execution automatic_selected(
            llama_kv_attention_execution_mode::selective);
    const auto automatic_selected_decision = automatic_selected.prepare(selected_single_page,
            llama_kv_attention_execution_phase::decode, 7, 11, true, scratch,
            {}, false, true);
    assert(automatic_selected_decision.route == llama_kv_attention_execution_route::selected_packed);
    assert(automatic_selected.metrics().selected_page_ids == explicit_page_ids);
    assert(automatic_selected.metrics().pack_bytes == 0);
    assert(automatic_selected.metrics().pack_epochs == 0);
    automatic_selected.complete_one_graph();

    // Forced routes compare the same metadata and fail closed when their
    // capability contract is absent. This is the diagnostic seam used by the
    // live dispatch measurements, not a normal pressure fallback.
    llama_kv_attention_execution forced(llama_kv_attention_execution_mode::selective);
    forced.set_route_override("packed");
    const auto forced_packed = forced.prepare(selected_prefill,
            llama_kv_attention_execution_phase::prefill, 9, 13, true, scratch,
            {}, true, true);
    assert(forced_packed.status == llama_kv_attention_execution_status::ok);
    assert(forced_packed.route == llama_kv_attention_execution_route::selected_packed);
    forced.complete_one_graph();
    llama_kv_attention_execution diagnostic_direct(
            llama_kv_attention_execution_mode::selective);
    diagnostic_direct.set_route_override("direct");
    const auto forced_direct = diagnostic_direct.prepare(selected_single_page,
            llama_kv_attention_execution_phase::decode, 9, 13, true, scratch,
            {}, false, true);
    assert(forced_direct.status == llama_kv_attention_execution_status::ok);
    assert(forced_direct.route == llama_kv_attention_execution_route::selected_direct);
    diagnostic_direct.complete_one_graph();
    forced.set_route_override("dense");
    const auto refused_dense = forced.prepare(selected_prefill,
            llama_kv_attention_execution_phase::prefill, 10, 14, true, scratch,
            {}, false, true);
    assert(refused_dense.status == llama_kv_attention_execution_status::not_configured);
    assert(refused_dense.route == llama_kv_attention_execution_route::refusal);
    assert(refused_dense.reason.find("route override 'dense'") != std::string::npos);
    forced.set_route_override("invalid-route");
    assert(forced.planned_route(selected_prefill,
            llama_kv_attention_execution_phase::decode, true, true, true) ==
           llama_kv_attention_execution_route::refusal);
    assert(forced.metrics().route_override_accepted == 1);
    assert(forced.metrics().route_override_refused == 1);

    execution.record_wait_time_us(7);
    execution.record_copy_time_us(11);
    execution.record_queue_time_us(13);
    execution.metrics_mutable().record_hotpath_time(
            execution.metrics_mutable().graph_build_us, 17);
    execution.metrics_mutable().record_hotpath_time(
            execution.metrics_mutable().route_decision_us, 19);
    execution.metrics_mutable().record_graph_rebuild_reason(
            llama_kv_attention_graph_reason(true, false, false, false, false));
    assert(execution.metrics().wait_time_us == 7);
    assert(execution.metrics().copy_time_us == 11);
    assert(execution.metrics().queue_time_us == 13);
    assert(execution.metrics().graph_build_us == 17);
    assert(execution.metrics().route_decision_us == 19);
    assert(execution.metrics().rebuild_reason_row_capacity == 1);
    assert(llama_kv_attention_graph_reason(false, true, false, false, false) ==
           llama_kv_attention_graph_rebuild_reason::physical_key);
    assert(llama_kv_attention_graph_reason(false, false, true, false, false) ==
           llama_kv_attention_graph_rebuild_reason::content_key);
    assert(llama_kv_attention_graph_reason(false, false, false, true, false) ==
           llama_kv_attention_graph_rebuild_reason::ubatch_shape);
    assert(llama_kv_attention_graph_reason(false, false, false, false, true) ==
           llama_kv_attention_graph_rebuild_reason::source_lifetime);
    const auto reset_epoch = execution.metrics_reset_epoch();
    execution.reset_metrics();
    assert(execution.metrics_reset_epoch() == reset_epoch + 1);
    assert(execution.metrics().wait_time_us == 0);
    assert(execution.metrics().graph_build_us == 0);
    assert(execution.metrics().route_decision_us == 0);
    assert(execution.metrics().rebuild_reason_row_capacity == 0);
}

static void test_packed_cache_identity_and_versions() {
    assert(llama_kv_attention_packed_page_action_make(91, 91, 256, 256) ==
           llama_kv_attention_packed_page_action::reuse);
    assert(llama_kv_attention_packed_page_action_make(92, 92, 200, 188) ==
           llama_kv_attention_packed_page_action::direct_write);
    assert(llama_kv_attention_packed_page_action_make(92, 93, 188, 200) ==
           llama_kv_attention_packed_page_action::copy);

    const auto snap = snapshot();
    llama_kv_attention_view_status view_status;
    const auto view = llama_kv_attention_view::build(snap, { 2, 0 }, view_status);
    assert(view_status == llama_kv_attention_view_status::ok);

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);
    llama_kv_attention_packed_cache cache;
    const ggml_init_params params = { 2 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * context = ggml_init(params);
    assert(context != nullptr);
    ggml_tensor * source_k = ggml_new_tensor_4d(context, GGML_TYPE_TURBO4_0, 256, 4, 2048, 1);
    ggml_tensor * source_v = ggml_new_tensor_4d(context, GGML_TYPE_TURBO4_0, 256, 4, 2048, 1);
    assert(source_k != nullptr && source_v != nullptr);
    ggml_set_name(source_k, "cache_k_l3_ms7");
    ggml_set_name(source_v, "cache_v_l3_ms7");

    const auto selected_metadata = metadata(snap, 1, 1);
    assert(selected_metadata.get_n_kv() == 444);
    assert(llama_kv_attention_packed_row_capacity(selected_metadata) == 512);
    const auto earlier_prefill_view = metadata(snap, 1, 1, { 0 }, 255);
    assert(llama_kv_attention_packed_row_capacity(earlier_prefill_view) == 256);
    // A same-physical-key prefill update that adds another selected page
    // needs a larger packed owner and must force graph reconstruction.
    assert(llama_kv_attention_packed_row_capacity(earlier_prefill_view) !=
           llama_kv_attention_packed_row_capacity(selected_metadata));
    const size_t k_row_bytes = ggml_row_size(source_k->type, source_k->ne[0] * source_k->ne[1]);
    const size_t v_row_bytes = ggml_row_size(source_v->type, source_v->ne[0] * source_v->ne[1]);
    assert(llama_kv_attention_packed_allocation_bytes(512, k_row_bytes, v_row_bytes) ==
           size_t(512) * (k_row_bytes + v_row_bytes));

    constexpr uint32_t row_capacity = 1024;
    auto * first = cache.find_or_create(3, 0, 11, 17, view.pages(), source_k, source_v,
            backend, row_capacity);
    assert(first != nullptr);
    assert(std::strcmp(first->k->name, source_k->name) == 0);
    assert(std::strcmp(first->v->name, source_v->name) == 0);
    assert(first->k->ne[2] == row_capacity && first->v->ne[2] == row_capacity);
    assert(cache.content_version(first, 0) == UINT64_MAX);
    cache.set_content_version(first, 0, 91);
    assert(cache.content_version(first, 0) == 91);

    auto * reused = cache.find_or_create(3, 0, 11, 17, view.pages(), source_k, source_v,
            backend, row_capacity);
    assert(reused == first && cache.content_version(reused, 0) == 91);

    // A graph replay must retain the same owner and its encoded-domain names
    // through submission; submitting duplicate references must still create
    // one lease for this graph.
    cache.begin_graph_build();
    auto * replay = cache.find_or_create(3, 0, 11, 17, view.pages(), source_k, source_v,
            backend, row_capacity);
    assert(replay == first);
    assert(cache.submit_graph({ replay, replay }));
    cache.complete_one_graph();
    cache.release_completed();

    auto * representation_refresh = cache.find_or_create(
            3, 0, 12, 17, view.pages(), source_k, source_v, backend, row_capacity);
    assert(representation_refresh == first);

    auto reordered = llama_kv_attention_view::build(snap, { 0, 2 }, view_status);
    assert(view_status == llama_kv_attention_view_status::ok);
    auto * reordered_entry = cache.find_or_create(
            3, 0, 11, 17, reordered.pages(), source_k, source_v, backend, row_capacity);
    assert(reordered_entry == first);
    assert(reordered_entry->k == first->k && reordered_entry->v == first->v);

    // Owner sizing follows the compact extent, not the order in which a
    // caller presents the selected pages.  This is the small first-request
    // boundary that previously let the last page under-size the destination.
    auto unsorted_pages = view.pages();
    assert(unsorted_pages.size() >= 2);
    std::swap(unsorted_pages.front(), unsorted_pages.back());
    unsorted_pages.back().compact_row_begin = 0;
    unsorted_pages.front().compact_row_begin = row_capacity - unsorted_pages.front().row_count;
    auto * unsorted_entry = cache.find_or_create(
            4, 1, 11, 17, unsorted_pages, source_k, source_v, backend, 0);
    assert(unsorted_entry != nullptr);
    assert(unsorted_entry->k->ne[2] == row_capacity);
    cache.clear_sequence(1);

    const auto tail_snapshot = snapshot(600);
    auto tail = llama_kv_attention_view::build(tail_snapshot, { 2, 0 }, view_status);
    assert(view_status == llama_kv_attention_view_status::ok);
    auto * tail_entry = cache.find_or_create(
            3, 0, 99, 17, tail.pages(), source_k, source_v, backend, row_capacity);
    assert(tail_entry == first && tail_entry->k->ne[2] == row_capacity);

    // An immutable partial-page prefix keeps its packed content version when
    // one row is appended. Only the new tail is supplied by the current graph.
    auto before_append_pages = view.pages();
    assert(before_append_pages[0].row_count > 1);
    --before_append_pages[0].row_count;
    --before_append_pages[0].native_position_end;
    auto * append_entry = cache.find_or_create(6, 6, 99, 17, before_append_pages,
            source_k, source_v, backend, row_capacity);
    assert(append_entry != nullptr);
    for (uint32_t i = 0; i < before_append_pages.size(); ++i) {
        cache.set_content_version(append_entry, i, before_append_pages[i].page_generation);
    }
    assert(llama_kv_attention_packed_page_action_make(
            cache.content_version(append_entry, 1), before_append_pages[1].page_generation,
            before_append_pages[1].row_count, append_entry->slots[1].valid_rows) ==
            llama_kv_attention_packed_page_action::reuse);

    auto * append_refresh = cache.find_or_create(6, 6, 100, 17, view.pages(),
            source_k, source_v, backend, row_capacity);
    assert(append_refresh == append_entry);
    assert(append_refresh->slots[0].valid_rows == before_append_pages[0].row_count + 1);
    assert(cache.content_version(append_refresh, 0) == before_append_pages[0].page_generation);
    assert(cache.content_version(append_refresh, 1) == before_append_pages[1].page_generation);
    assert(llama_kv_attention_packed_page_action_make(
            cache.content_version(append_refresh, 0), view.pages()[0].page_generation,
            view.pages()[0].row_count, before_append_pages[0].row_count) ==
            llama_kv_attention_packed_page_action::direct_write);
    assert(llama_kv_attention_packed_page_action_make(
            cache.content_version(append_refresh, 1), view.pages()[1].page_generation,
            view.pages()[1].row_count, before_append_pages[1].row_count) ==
            llama_kv_attention_packed_page_action::reuse);
    assert(append_refresh->slots[0].valid_rows - before_append_pages[0].row_count == 1);

    auto rollback_pages = view.pages();
    ++rollback_pages[0].page_generation;
    auto * rollback_refresh = cache.find_or_create(6, 6, 101, 17, rollback_pages,
            source_k, source_v, backend, row_capacity);
    assert(rollback_refresh == append_entry);
    assert(cache.content_version(rollback_refresh, 0) == UINT64_MAX);
    assert(cache.content_version(rollback_refresh, 1) == view.pages()[1].page_generation);
    assert(llama_kv_attention_packed_page_action_make(
            cache.content_version(rollback_refresh, 0), rollback_pages[0].page_generation,
            rollback_pages[0].row_count, append_refresh->slots[0].valid_rows) ==
            llama_kv_attention_packed_page_action::copy);
    assert(llama_kv_attention_packed_page_action_make(
            cache.content_version(rollback_refresh, 1), rollback_pages[1].page_generation,
            rollback_pages[1].row_count, append_refresh->slots[1].valid_rows) ==
            llama_kv_attention_packed_page_action::reuse);
    cache.clear_sequence(6);

    auto changed_pages = tail.pages();
    changed_pages[0].page_generation++;
    auto * generation_refresh = cache.find_or_create(
            3, 0, 100, 17, changed_pages, source_k, source_v, backend, row_capacity);
    assert(generation_refresh == first);
    assert(cache.content_version(generation_refresh, 0) == UINT64_MAX);
    assert(generation_refresh->slots[0].page_generation == changed_pages[0].page_generation);
    std::fprintf(stdout, "packed_delta_copy_contract=pass frozen_copy_rows=0 "
        "append_direct_write_rows=1 rollback_invalidated_pages=1 "
        "rollback_generation=%u\n", rollback_pages[0].page_generation);

    // The old owner remains live while its simulated graph consumer is in
    // flight. A structural replacement is allowed, but switching to a third
    // owner in the same domain is refused until that consumer completes.
    assert(cache.submit_graph({ first }));

    auto * new_lifetime = cache.find_or_create(
            3, 0, 11, 18, view.pages(), source_k, source_v, backend, row_capacity);
    assert(new_lifetime != nullptr && new_lifetime != first);
    assert(cache.content_version(new_lifetime, 0) == UINT64_MAX);
    assert(cache.size() == 2);

    cache.begin_graph_build();
    cache.release_completed();
    assert(cache.size() == 2);
    assert(cache.find_or_create(3, 0, 13, 17, view.pages(), source_k, source_v,
            backend, row_capacity) == nullptr);
    cache.complete_one_graph();
    cache.release_completed();
    assert(cache.size() == 1);

    // A capacity change is structural, while a representation epoch change
    // above was deliberately not. Clearing the sequence retires the active
    // owner once no consumer remains.
    auto * larger = cache.find_or_create(
            3, 0, 14, 18, view.pages(), source_k, source_v, backend, 1536);
    assert(larger != nullptr && larger != new_lifetime);
    assert(larger->k->ne[2] == 1536);
    assert(cache.size() == 2);
    cache.clear_sequence(0);
    assert(cache.size() == 0);

    // Owners created during a graph build are provisional until the graph
    // lease is submitted. A failed allocation/bind must reclaim that owner
    // without touching already completed cache state.
    cache.begin_graph_build();
    auto * provisional = cache.find_or_create(
            3, 0, 15, 19, view.pages(), source_k, source_v, backend, 512);
    assert(provisional != nullptr && cache.size() == 1);
    cache.abort_graph_build();
    assert(cache.size() == 0);

    ggml_free(context);
    ggml_backend_free(backend);
}

static void test_packed_view_copy_intervals() {
    const auto snap = snapshot();
    llama_kv_attention_view_status view_status;
    const auto view = llama_kv_attention_view::build(snap, { 2, 0 }, view_status);
    assert(view_status == llama_kv_attention_view_status::ok);
    const auto & pages = view.pages();
    assert(pages.size() == 2);
    assert(pages[0].native_position_begin == 0 && pages[0].row_count == 256);
    assert(pages[1].native_position_begin == 512 && pages[1].row_count == 188);

    // Metadata query positions can be duplicated and nonmonotonic. The
    // optimized plan must preserve membership semantics across both selected
    // pages and retain the same current/history span boundaries.
    const std::vector<llama_pos> queries = { 600, 254, 512, 255, 600, 699 };
    std::vector<llama_kv_attention_view_copy_interval> intervals;
    uint64_t query_positions_examined = 0;
    assert(llama_kv_attention_view_copy_intervals(
            pages, queries, intervals, &query_positions_examined));
    const auto expect = [&](size_t index, uint32_t page_index, uint32_t row_begin,
            uint32_t row_count, bool current_rows) {
        assert(index < intervals.size());
        const auto & interval = intervals[index];
        assert(interval.page_index == page_index);
        assert(interval.row_begin == row_begin);
        assert(interval.row_count == row_count);
        assert(interval.current_rows == current_rows);
    };
    assert(intervals.size() == 7);
    expect(0, 0, 0, 254, false);
    expect(1, 0, 254, 2, true);
    expect(2, 1, 0, 1, true);
    expect(3, 1, 1, 87, false);
    expect(4, 1, 88, 1, true);
    expect(5, 1, 89, 98, false);
    expect(6, 1, 187, 1, true);
    assert(query_positions_examined == 5);

    // Count the old row-by-query membership comparisons on this exact shape.
    // The new plan visits only the unique query positions that fall in a page.
    uint64_t old_membership_comparisons = 0;
    for (const auto & page : pages) {
        for (uint32_t row = 0; row < page.row_count; ++row) {
            const llama_pos position = page.native_position_begin + row;
            for (const llama_pos query : queries) {
                ++old_membership_comparisons;
                if (query == position) {
                    break;
                }
            }
        }
    }

    // Appending the current tail row changes only that page's final interval;
    // the other frozen page remains one unchanged historical span.
    auto before_append_pages = pages;
    assert(before_append_pages[0].row_count > 1);
    --before_append_pages[0].row_count;
    --before_append_pages[0].native_position_end;
    std::vector<llama_kv_attention_view_copy_interval> before_append;
    assert(llama_kv_attention_view_copy_intervals(
            before_append_pages, { 255 }, before_append));
    assert(before_append.size() == 2);
    assert(before_append[0].page_index == 0 && before_append[0].row_count == 255 &&
            !before_append[0].current_rows);
    assert(before_append[1].page_index == 1 && before_append[1].row_count == 188 &&
            !before_append[1].current_rows);

    std::vector<llama_kv_attention_view_copy_interval> after_append;
    assert(llama_kv_attention_view_copy_intervals(pages, { 255 }, after_append));
    assert(after_append.size() == 3);
    assert(after_append[0].page_index == 0 && after_append[0].row_count == 255 &&
            !after_append[0].current_rows);
    assert(after_append[1].page_index == 0 && after_append[1].row_begin == 255 &&
            after_append[1].row_count == 1 && after_append[1].current_rows);
    assert(after_append[2].page_index == 1 && after_append[2].row_count == 188 &&
            !after_append[2].current_rows);
    auto rollback_pages = pages;
    ++rollback_pages[0].page_generation;
    assert(rollback_pages[1].page_generation == pages[1].page_generation);
    std::vector<llama_kv_attention_view_copy_interval> after_rollback;
    assert(llama_kv_attention_view_copy_intervals(rollback_pages, { 255 }, after_rollback));
    assert(after_rollback.size() == after_append.size());
    for (size_t i = 0; i < after_append.size(); ++i) {
        assert(after_rollback[i].page_index == after_append[i].page_index);
        assert(after_rollback[i].row_begin == after_append[i].row_begin);
        assert(after_rollback[i].row_count == after_append[i].row_count);
        assert(after_rollback[i].current_rows == after_append[i].current_rows);
    }
    assert(llama_kv_attention_packed_page_action_make(
            pages[1].page_generation, rollback_pages[1].page_generation,
            pages[1].row_count, pages[1].row_count) ==
            llama_kv_attention_packed_page_action::reuse);
    assert(llama_kv_attention_packed_page_action_make(
            pages[0].page_generation, rollback_pages[0].page_generation,
            pages[0].row_count, pages[0].row_count) ==
            llama_kv_attention_packed_page_action::copy);

    std::fprintf(stdout, "packed_view_copy_intervals=pass "
        "old_row_membership_comparisons=%llu new_query_positions_examined=%llu "
        "cross_page_intervals=%zu append_direct_write_rows=1 "
        "frozen_page_reused=1 rollback_invalidated_pages=1\n",
        (unsigned long long) old_membership_comparisons,
        (unsigned long long) query_positions_examined, intervals.size());
}

static void test_view_sized_scratch_contract() {
    const auto selected = metadata(snapshot(), 2, 1);
    const auto high_selected = metadata(snapshot_high_logical_positions(), 2, 1,
            { 1026, 1024 }, 1026 * VBR_GENERATION_PAGE_CELLS + 100);
    assert(high_selected.get_n_kv() == selected.get_n_kv());
    llama_kv_attention_execution execution(llama_kv_attention_execution_mode::selective);

    assert(execution.planned_route(selected,
            llama_kv_attention_execution_phase::decode, true, false, true) ==
           llama_kv_attention_execution_route::selected_packed);
    const auto single_page = metadata(snapshot(), 1, 1, { 0 }, 255);
    assert(execution.planned_route(single_page,
            llama_kv_attention_execution_phase::decode, true, false, true) ==
           llama_kv_attention_execution_route::selected_packed);
    assert(execution.planned_route(selected,
            llama_kv_attention_execution_phase::mtp_verify, true, false, false) ==
           llama_kv_attention_execution_route::selected_direct);
    llama_kv_attention_execution native_mtp_execution(
            llama_kv_attention_execution_mode::selective);
    native_mtp_execution.set_native_mtp_enabled(true);
    assert(native_mtp_execution.planned_route(selected,
            llama_kv_attention_execution_phase::prefill, true, false, true) ==
           llama_kv_attention_execution_route::selected_packed);
    assert(native_mtp_execution.planned_route(selected,
            llama_kv_attention_execution_phase::decode, true, false, true) ==
           llama_kv_attention_execution_route::selected_packed);
    assert(native_mtp_execution.planned_route(selected,
            llama_kv_attention_execution_phase::mtp_verify, true, false, true) ==
           llama_kv_attention_execution_route::selected_packed);
    assert(execution.planned_route(selected,
            llama_kv_attention_execution_phase::prefill, true, false, true) ==
           llama_kv_attention_execution_route::selected_packed);
    for (const auto phase : { llama_kv_attention_execution_phase::prefill,
                              llama_kv_attention_execution_phase::decode,
                              llama_kv_attention_execution_phase::mtp_verify }) {
        assert(execution.planned_route(selected, phase, true, false, true) ==
               llama_kv_attention_execution_route::selected_packed);
        assert(execution.planned_route(selected, phase, true, false, false) ==
               llama_kv_attention_execution_route::selected_direct);
    }
    assert(execution.planned_route(selected,
            llama_kv_attention_execution_phase::prefill, false, true, false) ==
           llama_kv_attention_execution_route::selected_dense);
    assert(execution.planned_route(selected,
            llama_kv_attention_execution_phase::prefill, false, false, true) ==
           llama_kv_attention_execution_route::selected_packed);
    assert(execution.planned_route(selected,
            llama_kv_attention_execution_phase::prefill, false, false, false) ==
           llama_kv_attention_execution_route::refusal);
    // The CUDA paged primitive selects a larger launch tile for multi-query
    // prefill; route admission must not cap it at decode-sized batches.
    assert(execution.planned_route(metadata(snapshot(), 3, 1),
            llama_kv_attention_execution_phase::prefill, true, false, false) ==
           llama_kv_attention_execution_route::selected_direct);
    // A genuinely contiguous selected view uses mature dense FA before the
    // paged consumer, even when both capability contracts are available.
    assert(execution.planned_route(metadata(snapshot(), 1, 1, { 0 }, 255),
            llama_kv_attention_execution_phase::decode, true, true, true) ==
           llama_kv_attention_execution_route::selected_dense);
    const auto full_history_snapshot = snapshot_slots(5, 1, 7, 601);
    const auto complete_mtp_history = metadata(full_history_snapshot,
            3, 1, { 2, 1, 0 }, 600, 24, 4);
    const auto bounded_mtp_history = metadata(full_history_snapshot,
            3, 1, { 2, 0 }, 600, 24, 4);
    assert(complete_mtp_history.get_n_kv() == 601);
    assert(bounded_mtp_history.get_n_kv() < 601);
    assert(execution.planned_route(complete_mtp_history,
            llama_kv_attention_execution_phase::mtp_verify, true, true, true) ==
           llama_kv_attention_execution_route::selected_dense);
    assert(execution.planned_route(bounded_mtp_history,
            llama_kv_attention_execution_phase::mtp_verify, true, true, true) ==
           llama_kv_attention_execution_route::selected_packed);
    std::fprintf(stdout, "mtp_sparse_history_route=pass complete=dense bounded=packed\n");
    const auto standard_dense = metadata(snapshot(), 1, 1, { 0 }, 255, 16, 4,
            GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
    const auto standard_dense_view = llama_kv_attention_dense_view_check(standard_dense, 8);
    assert(standard_dense_view.eligible);
    assert(execution.planned_route(standard_dense,
            llama_kv_attention_execution_phase::decode, false, true, false) ==
           llama_kv_attention_execution_route::selected_dense);

    const auto batched_small = metadata(snapshot(), 32, 1);
    const auto batched_large_q = metadata(snapshot(), 256, 1);
    assert(execution.planned_route(batched_small,
            llama_kv_attention_execution_phase::prefill, true, false, true) ==
           llama_kv_attention_execution_route::selected_direct);
    assert(execution.planned_route(batched_large_q,
            llama_kv_attention_execution_phase::prefill, true, false, true) ==
           llama_kv_attention_execution_route::selected_direct);
    assert(execution.planned_route(batched_small,
            llama_kv_attention_execution_phase::prefill, true, true, true) ==
           llama_kv_attention_execution_route::selected_dense);
    assert(execution.planned_route(batched_large_q,
            llama_kv_attention_execution_phase::prefill, true, true, true) ==
           llama_kv_attention_execution_route::selected_direct);

    llama_kv_residency_table large_table(16);
    auto large_tx = large_table.begin();
    std::vector<uint32_t> all_pages;
    for (uint32_t i = 0; i < 16; ++i) {
        assert(large_table.replace(large_tx, page(i, i, llama_pos((i + 1) * 256))) ==
               llama_kv_residency_status::ok);
        all_pages.push_back(i);
    }
    assert(large_table.publish(large_tx) == llama_kv_residency_status::ok);
    const auto batched_4k = metadata(large_table.snapshot(), 32, 1,
            all_pages, 4095, 16, 4);
    assert(batched_4k.get_n_kv() == 4096);
    assert(execution.planned_route(batched_4k,
            llama_kv_attention_execution_phase::prefill, true, false, true) ==
           llama_kv_attention_execution_route::selected_packed);
    assert(execution.planned_route(batched_4k,
            llama_kv_attention_execution_phase::prefill, true, true, true) ==
           llama_kv_attention_execution_route::selected_dense);
    assert(execution.planned_route(metadata(large_table.snapshot(), 256, 1,
                all_pages, 4351, 16, 4),
            llama_kv_attention_execution_phase::prefill, true, false, true) ==
           llama_kv_attention_execution_route::selected_direct);

    llama_kv_attention_execution packed_execution(llama_kv_attention_execution_mode::selective);
    packed_execution.set_route_override("packed");
    assert(packed_execution.planned_route(selected,
            llama_kv_attention_execution_phase::prefill, false, false, true) ==
           llama_kv_attention_execution_route::selected_packed);
    for (const uint32_t query_count : { 1u, 2u, 3u, 64u, 128u }) {
        assert(packed_execution.planned_route(metadata(snapshot(), query_count, 1),
                llama_kv_attention_execution_phase::prefill, true, false, true) ==
               llama_kv_attention_execution_route::selected_packed);
    }

    llama_kv_attention_scratch_request view;
    view.route = llama_kv_attention_execution_route::selected_reference;
    view.phase = llama_kv_attention_execution_phase::mtp_verify;
    view.context_role = llama_kv_attention_scratch_context_role::draft;
    view.materialized_k_rows = 17;
    view.materialized_v_rows = 9;
    view.materialized_k_bytes_per_row = 6;
    view.materialized_v_bytes_per_row = 10;
    view.packed_bytes = 7;
    view.resident_rows = 3;
    assert(view.materialized_rows() == 17);
    assert(view.required_rows() == 17);
    assert(view.required_bytes() == 17 * 6 + 9 * 10 + 7);
    assert(std::string(llama_kv_attention_scratch_context_role_name(
            view.context_role)) == "draft");
    view.context_role = llama_kv_attention_scratch_context_role::target;
    assert(std::string(llama_kv_attention_scratch_context_role_name(
            view.context_role)) == "target");

    llama_kv_attention_scratch_request direct = view;
    direct.route = llama_kv_attention_execution_route::selected_direct;
    direct.materialized_k_rows = 0;
    direct.materialized_v_rows = 0;
    direct.materialized_k_bytes_per_row = 0;
    direct.materialized_v_bytes_per_row = 0;
    direct.packed_bytes = 0;
    direct.resident_rows = 0;
    assert(direct.required_rows() == 0);
    assert(direct.required_bytes() == 0);

    llama_kv_attention_scratch_request overflow = view;
    overflow.materialized_k_rows = std::numeric_limits<uint64_t>::max();
    overflow.materialized_k_bytes_per_row = 2;
    assert(overflow.required_bytes() == std::numeric_limits<size_t>::max());

    const auto max_query = metadata(snapshot(), 65, 1);
    assert(packed_execution.planned_route(max_query,
            llama_kv_attention_execution_phase::mtp_verify, true, false, true) ==
           llama_kv_attention_execution_route::selected_packed);
}

static void test_fallbacks_and_graph_key() {
    const auto selected = metadata(snapshot(), 1, 1);
    llama_kv_attention_scratch_request scratch;
    llama_kv_attention_execution execution(llama_kv_attention_execution_mode::selective);

    auto reference = execution.prepare(selected, llama_kv_attention_execution_phase::decode,
            1, 1, false, scratch);
    assert(reference.status == llama_kv_attention_execution_status::not_configured);
    assert(reference.route == llama_kv_attention_execution_route::refusal);
    assert(reference.reason.find("automatic selective route refused") != std::string::npos);
    assert(reference.reason.find("phase=decode") != std::string::npos);
    assert(reference.reason.find("query_tile=1") != std::string::npos);
    assert(reference.reason.find("selected_reference_requires_explicit_override") !=
           std::string::npos);
    assert(execution.metrics().automatic_reference_prevented == 1);

    execution.set_route_override("packed");
    auto prompt_shape = metadata(snapshot(), 65, 1);
    auto prompt_reference = execution.prepare(prompt_shape,
            llama_kv_attention_execution_phase::decode, 1, 1, true, scratch, {}, false, true);
    assert(prompt_reference.route == llama_kv_attention_execution_route::selected_packed);
    execution.complete_one_graph();

    auto tile64 = execution.prepare(metadata(snapshot(), 64, 1),
            llama_kv_attention_execution_phase::prefill, 1, 1, true, scratch,
            {}, false, true);
    assert(tile64.route == llama_kv_attention_execution_route::selected_packed);
    execution.complete_one_graph();

    auto tile3 = execution.prepare(metadata(snapshot(), 3, 1),
            llama_kv_attention_execution_phase::prefill, 1, 1, true, scratch, {}, false, true);
    assert(tile3.route == llama_kv_attention_execution_route::selected_packed);
    execution.complete_one_graph();

    llama_kv_attention_execution multi_query_direct(
            llama_kv_attention_execution_mode::selective);
    const auto multi_query = multi_query_direct.prepare(metadata(snapshot(), 3, 1),
            llama_kv_attention_execution_phase::prefill, 1, 2, true, scratch,
            {}, false, false);
    assert(multi_query.status == llama_kv_attention_execution_status::ok);
    assert(multi_query.route == llama_kv_attention_execution_route::selected_direct);
    assert(multi_query.reason.find("qualified Turbo4 selective prefill query tile") !=
           std::string::npos);
    multi_query_direct.complete_one_graph();

    auto tile65 = execution.prepare(metadata(snapshot(), 65, 1),
            llama_kv_attention_execution_phase::prefill, 1, 1, true, scratch,
            {}, false, true);
    assert(tile65.route == llama_kv_attention_execution_route::selected_packed);
    execution.complete_one_graph();

    // Qwen3.5 uses 24 query heads and 4 KV heads (GQA ratio 6). The mature
    // packed FA path must preserve that real-model grouping.
    llama_kv_attention_execution qwen_gqa(
            llama_kv_attention_execution_mode::selective);
    qwen_gqa.set_route_override("packed");
    const auto qwen_prefill = metadata(snapshot(), 3, 1, { 2, 0 }, 600, 24, 4);
    const auto qwen_direct = qwen_gqa.prepare(qwen_prefill,
            llama_kv_attention_execution_phase::prefill, 1, 1, true, scratch, {}, false, true);
    assert(qwen_direct.route == llama_kv_attention_execution_route::selected_packed);
    qwen_gqa.complete_one_graph();

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

    llama_kv_attention_execution direct_guard(llama_kv_attention_execution_mode::selective);
    direct_guard.set_route_override("direct");
    const auto invalid_direct = direct_guard.prepare({},
            llama_kv_attention_execution_phase::decode, 1, 1, true, scratch);
    assert(invalid_direct.status == llama_kv_attention_execution_status::invalid_metadata);
    assert(invalid_direct.route == llama_kv_attention_execution_route::refusal);

    llama_kv_attention_scratch_request overflow;
    overflow.resident_rows = UINT64_MAX;
    auto overflow_result = execution.prepare(selected,
            llama_kv_attention_execution_phase::decode, 1, 1, true, overflow);
    assert(overflow_result.status == llama_kv_attention_execution_status::overflow);
    assert(overflow_result.route == llama_kv_attention_execution_route::refusal);

    llm_graph_params a = {};
    a.kv_attention_table_epoch = 1;
    a.kv_attention_content_key = 11;
    a.kv_attention_layout_key = 12;
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
    assert(a.allow_reuse(b));
    b = a;
    ++b.kv_attention_table_epoch;
    assert(a.allow_reuse(b));
    b = a;
    ++b.kv_attention_layout_key;
    assert(!a.allow_reuse(b));
    execution.complete_one_graph();

    llama_kv_attention_execution exact(llama_kv_attention_execution_mode::exact);
    const auto exact_route = exact.prepare({},
            llama_kv_attention_execution_phase::prefill, 0, 1, false, scratch);
    assert(exact_route.status == llama_kv_attention_execution_status::ok);
    assert(exact_route.route == llama_kv_attention_execution_route::exact_reference);
    exact.complete_one_graph();

    llama_kv_attention_execution exact_direct(llama_kv_attention_execution_mode::exact);
    const auto exact_direct_route = exact_direct.prepare(metadata(snapshot(), 1, 1),
            llama_kv_attention_execution_phase::decode, 0, 1, true, scratch);
    assert(exact_direct_route.status == llama_kv_attention_execution_status::ok);
    assert(exact_direct_route.route == llama_kv_attention_execution_route::exact_direct);
    exact_direct.complete_one_graph();
}

static void test_epoch_matrix_and_lifetime_metrics() {
    const auto base = metadata(snapshot(), 1, 1);
    const auto reordered = metadata(snapshot(), 1, 1, { 0, 2 });
    const auto remapped = metadata(snapshot_slots(6, 1, 7), 1, 1);
    const auto grown = metadata(snapshot(701), 1, 1);
    const auto query_changed = metadata(snapshot(), 1, 1, { 2, 0 }, 601);
    const auto shape_changed = metadata(snapshot(), 2, 1);

    // Reordering the same selected set is canonicalized before graph keys are
    // formed, so it does not force a new compact layout or repack.
    assert(base.graph_content_key() == reordered.graph_content_key());
    assert(base.graph_content_key() != remapped.graph_content_key());
    assert(base.table_epoch() != grown.table_epoch());
    assert(base.graph_content_key() != grown.graph_content_key());
    assert(base.graph_content_key() == query_changed.graph_content_key());
    assert(base.graph_content_key() != shape_changed.graph_content_key());
    // Page slots, logical order, and query positions are mutable descriptor
    // data. Only a dimension/configuration change changes graph layout.
    assert(base.graph_layout_key() == reordered.graph_layout_key());
    assert(base.graph_layout_key() == remapped.graph_layout_key());
    assert(base.graph_layout_key() == query_changed.graph_layout_key());
    assert(base.graph_layout_key() == grown.graph_layout_key());
    assert(base.graph_layout_key() != shape_changed.graph_layout_key());

    llama_kv_attention_scratch_request scratch;
    scratch.resident_rows = base.get_n_kv();
    scratch.bytes_per_row = 64;
    llama_kv_attention_execution execution(llama_kv_attention_execution_mode::selective);
    execution.set_route_override("packed");
    execution.reset_metrics();

    const auto first = execution.prepare(base, llama_kv_attention_execution_phase::decode,
            11, 22, true, scratch, {}, false, true);
    assert(first.route == llama_kv_attention_execution_route::selected_packed);
    assert(first.graph_rebuild);

    auto irrelevant_scratch = scratch;
    irrelevant_scratch.router_rows = 99;
    const auto reused = execution.prepare(base, llama_kv_attention_execution_phase::decode,
            11, 22, true, irrelevant_scratch, {}, false, true);
    assert(!reused.graph_rebuild);

    execution.set_route_override("reference");
    const auto reference = execution.prepare(base, llama_kv_attention_execution_phase::decode,
            11, 22, false, scratch);
    assert(reference.route == llama_kv_attention_execution_route::selected_reference);
    assert(reference.graph_rebuild);

    const auto order_change = execution.prepare(reordered,
            llama_kv_attention_execution_phase::decode, 11, 22, false, scratch);
    assert(!order_change.graph_rebuild);
    const auto slot_change = execution.prepare(remapped,
            llama_kv_attention_execution_phase::decode, 11, 22, false, scratch);
    assert(!slot_change.graph_rebuild);
    const auto representation_change = execution.prepare(base,
            llama_kv_attention_execution_phase::decode, 12, 22, false, scratch);
    assert(representation_change.graph_rebuild);
    const auto query_change = execution.prepare(query_changed,
            llama_kv_attention_execution_phase::decode, 12, 22, false, scratch);
    assert(!query_change.graph_rebuild);
    const auto tail_growth = execution.prepare(grown,
            llama_kv_attention_execution_phase::decode, 12, 22, false, scratch);
    assert(tail_growth.graph_rebuild);
    const auto shape_change = execution.prepare(shape_changed,
            llama_kv_attention_execution_phase::prefill, 12, 23, false, scratch);
    assert(shape_change.graph_rebuild);

    const auto & counters = execution.metrics();
    assert(counters.graph_capture_count == 5);
    assert(counters.graph_replay_count == 4);
    assert(counters.graph_rebuild_count == counters.graph_capture_count);
    assert(counters.graph_submission_count == 9);
    // Packed FA has no direct page-table upload. Its persistent owner and
    // typed page copies carry the selected rows instead.
    assert(counters.table_upload_bytes == 0);
    assert(counters.scratch_high_water_rows == irrelevant_scratch.required_rows());
    assert(counters.scratch_high_water_bytes == irrelevant_scratch.required_bytes());
    std::printf("kv-attention-profile submissions=%llu captures=%llu replays=%llu "
                "rebuilds=%llu table_upload_bytes=%llu epoch_changes=%llu\n",
            (unsigned long long) counters.graph_submission_count,
            (unsigned long long) counters.graph_capture_count,
            (unsigned long long) counters.graph_replay_count,
            (unsigned long long) counters.graph_rebuild_count,
            (unsigned long long) counters.table_upload_bytes,
            (unsigned long long) counters.table_epoch_changes);

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

static void test_no_change_decode_replay() {
    const auto stable = metadata(snapshot(), 1, 1);
    const auto tail_advanced = metadata(snapshot(701), 1, 1);
    assert(stable.graph_layout_key() == tail_advanced.graph_layout_key());
    assert(stable.graph_physical_key() == tail_advanced.graph_physical_key());
    assert(stable.table_epoch() != tail_advanced.table_epoch());

    llama_kv_attention_scratch_request scratch;
    scratch.resident_rows = stable.get_n_kv();
    scratch.bytes_per_row = 64;
    llama_kv_attention_execution execution(llama_kv_attention_execution_mode::selective);
    execution.set_route_override("packed");

    const auto first = execution.prepare(stable,
            llama_kv_attention_execution_phase::decode, 3, 11, true, scratch, {}, false, true);
    assert(first.graph_rebuild);
    const auto no_change = execution.prepare(tail_advanced,
            llama_kv_attention_execution_phase::decode, 3, 11, true, scratch, {}, false, true);
    assert(!no_change.graph_rebuild);
    assert(execution.in_flight_graphs() == 2);
    assert(execution.metrics().graph_capture_count == 1);
    assert(execution.metrics().graph_replay_count == 1);
    assert(execution.metrics().table_epoch_changes == 1);

    // Replaying the shape does not drop either immutable view lease while the
    // mutable descriptor input is refreshed for the next submission.
    execution.complete_one_graph();
    execution.complete_one_graph();
    assert(execution.in_flight_graphs() == 0);
}

static void test_committed_layer_union() {
    const std::vector<std::vector<uint32_t>> layers = {
        { 4, 8, 4 },
        { 8, 12 },
        { 21 },
    };
    const std::vector<uint32_t> committed = { 4, 8, 12, 21, 25 };
    std::vector<uint32_t> selected;
    assert(llama_kv_attention_committed_pages(layers, committed, selected));
    assert(selected.size() == committed.size());
    for (const uint32_t page : committed) {
        assert(std::find(selected.begin(), selected.end(), page) != selected.end());
    }
    assert(std::find(layers[0].begin(), layers[0].end(), 21) == layers[0].end());
    assert(std::find(selected.begin(), selected.end(), 21) != selected.end());
    assert(std::find(selected.begin(), selected.end(), 25) != selected.end());
}

int main() {
    test_prefill_admission();
    test_graph_thread_config_cache();
    test_query_pages_cover_cross_page_ubatch();
    test_routes_epochs_and_fences();
    test_packed_view_copy_intervals();
    test_packed_cache_identity_and_versions();
    test_view_sized_scratch_contract();
    test_fallbacks_and_graph_key();
    test_epoch_matrix_and_lifetime_metrics();
    test_no_change_decode_replay();
    test_committed_layer_union();
    return 0;
}
