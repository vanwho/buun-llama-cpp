#include "llama-kv-pager.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#undef NDEBUG
#include <cassert>

static llama_kv_pager_resources resources(uint64_t capacity, uint64_t page_bytes) {
    llama_kv_pager_resources result;
    result.admission.capacity_bytes = capacity;
    result.admission.target_page_bytes = page_bytes;
    result.admission.turbo4_scratch_bytes = 64;
    result.admission.mtp_present = false;
    result.host_budget_known = true;
    result.host_budget_bytes = 1u << 20;
    result.allocator_granularity = 64;
    return result;
}

static llama_kv_pager_geometry geometry(uint64_t context) {
    llama_kv_pager_geometry result;
    result.context_tokens = context;
    result.page_tokens = 256;
    result.attention_layers = 16;
    result.kv_heads = 4;
    result.key_length = 128;
    result.value_length = 128;
    result.page_bytes = 128;
    return result;
}

static void test_turn_epoch_state_and_geometry() {
    llama_kv_pager_turn_geometry derived;
    assert(llama_kv_pager_derive_turn_geometry(5, 256, 300, 1, 0, derived));
    assert(derived.hot_pages == 5 && derived.generation_pages == 2);
    assert(derived.generation_tokens == 512 && derived.retrieval_pages == 3);
    assert(uint64_t(derived.retrieval_pages) + derived.generation_pages <= derived.hot_pages);
    // With only one of the three retrieval pages selected, generation can
    // borrow the two unoccupied historical slots while preserving G.
    const uint32_t borrowed_generation_pages =
        llama_kv_pager_generation_available_pages(derived, 1);
    assert(borrowed_generation_pages == 4);
    assert(borrowed_generation_pages >= derived.generation_pages);
    assert(llama_kv_pager_generation_available_pages(derived, 4) == 0);
    assert(!llama_kv_pager_derive_turn_geometry(2, 256, 513, 0, 0, derived));
    assert(!llama_kv_pager_derive_turn_geometry(4, 256, 256, 4, 0, derived));
    assert(llama_kv_pager_derive_turn_geometry(5, 256, 256, 1, 0, derived, 2));
    assert(derived.retrieval_pages == 2 && derived.generation_pages == 1);
    assert(llama_kv_pager_derive_turn_geometry(5, 256, 256, 1, 1, derived));
    assert(derived.generation_pages == 2 && derived.retrieval_pages == 3);
    assert(!llama_kv_pager_derive_turn_geometry(5, 256, 256, 0, 0, derived, 5));
    assert(!llama_kv_pager_derive_turn_geometry(UINT32_MAX, UINT32_MAX,
        UINT32_MAX, 0, UINT32_MAX, derived));

    llama_kv_retrieval_policy policy;
    assert(llama_kv_pager_parse_retrieval_policy("turn", policy));
    assert(policy == llama_kv_retrieval_policy::turn);
    assert(llama_kv_pager_parse_retrieval_policy("cadence", policy));
    assert(policy == llama_kv_retrieval_policy::cadence);
    assert(!llama_kv_pager_parse_retrieval_policy("accepted-token", policy));

    llama_kv_pager_config config;
    config.mode = llama_kv_pager_mode::selective;
    config.hot_pages.automatic = false;
    config.hot_pages.value = 4;
    config.retrieval_pages.automatic = false;
    config.retrieval_pages.value = 3;
    llama_kv_pager_backend backend;
    backend.allocate = [](uint64_t bytes, llama_kv_pager_allocation & allocation) {
        allocation.handle = reinterpret_cast<void *>(uintptr_t(1));
        allocation.requested_bytes = bytes;
        allocation.realized_bytes = bytes;
        return true;
    };
    backend.release = [](llama_kv_pager_allocation & allocation) { allocation = {}; };
    llama_kv_pager_status status;
    auto pager = llama_kv_pager::create(
        config, geometry(1024), resources(640, 128), backend, status);
    assert(pager && status == llama_kv_pager_status::ok);
    assert(pager->snapshot().retrieval_pages == 3);
    assert(pager->snapshot().generation_pages == 1);

    llama_kv_page_id frontier;
    frontier.sequence_id = 0;
    frontier.sequence_generation = 7;
    frontier.logical_page = 2;
    frontier.position_begin = 512;
    frontier.position_end = 767;
    assert(pager->transition_turn(0, 1, 0,
        llama_kv_pager_turn_phase::query_provisional, 700, 704, frontier, 0) ==
        llama_kv_pager_turn_status::ok);
    const auto provisional = pager->turn_state(0);
    assert(provisional.turn_id == 1 && provisional.retrieval_epoch == 0);
    assert(provisional.query_start == 700 && provisional.query_end == 704);

    assert(pager->transition_turn(0, 2, 0,
        llama_kv_pager_turn_phase::retrieval_commit, 700, 704, frontier, 33) ==
        llama_kv_pager_turn_status::stale_turn);
    assert(pager->turn_state(0).phase == provisional.phase);
    assert(pager->transition_turn(0, 1, 0,
        llama_kv_pager_turn_phase::retrieval_commit, 701, 704, frontier, 33) ==
        llama_kv_pager_turn_status::invalid_query);
    assert(pager->turn_state(0).query_start == provisional.query_start);
    assert(pager->transition_turn(0, 1, 0,
        llama_kv_pager_turn_phase::retrieval_commit, 700, 704, frontier, 33,
        { { frontier, 9 } }) == llama_kv_pager_turn_status::ok);
    const auto committed = pager->turn_state(0);
    assert(committed.retrieval_epoch == 1 && committed.frozen_history_generation == 33);
    assert(committed.selected_history.size() == 1);
    assert(committed.selected_history[0].identity == frontier);
    assert(committed.selected_history[0].content_version == 9);
    assert(llama_kv_pager_history_selection_equal(
        committed.selected_history, { { frontier, 9 } }));
    // Table epochs can advance for a query-tail mutation without changing the
    // logical historical view; content-version changes do require replay.
    assert(!llama_kv_pager_history_selection_equal(
        committed.selected_history, { { frontier, 10 } }));
    llama_kv_page_id other_history = frontier;
    other_history.logical_page++;
    assert(!llama_kv_pager_history_selection_equal(
        committed.selected_history, { { other_history, 9 } }));
    assert(llama_kv_pager_page_is_before_query(512, 256, 768));
    assert(!llama_kv_pager_page_is_before_query(512, 256, 640));
    assert(!llama_kv_pager_page_is_before_query(512, 1, 512));
    // Query rollback edits only the provisional attention tail. It must not
    // reinstall the selection that existed before retrieval publication.
    const uint64_t query_rollback_epoch = pager->residency().epoch();
    assert(pager->mutate({ llama_kv_pager_mutation_kind::remove, 0, -1,
        700, -1, 0, 0, query_rollback_epoch }) ==
        llama_kv_pager_write_status::ok);
    const auto after_query_rollback = pager->turn_state(0);
    assert(after_query_rollback.phase == llama_kv_pager_turn_phase::retrieval_commit);
    assert(after_query_rollback.selected_history.size() == 1 &&
        after_query_rollback.selected_history[0].identity == frontier &&
        after_query_rollback.selected_history[0].content_version == 9);
    std::cout << "query_checkpoint_restore_and_mapping=pass history_map=retained "
                 "query_tail_mutation=pass\n";
    assert(pager->snapshot().mutable_page_table_epoch !=
        pager->snapshot().frozen_history_generation);
    assert(pager->transition_turn(0, 1, 1,
        llama_kv_pager_turn_phase::query_replay, 700, 704, frontier, 33) ==
        llama_kv_pager_turn_status::ok);
    assert(pager->transition_turn(0, 1, 1,
        llama_kv_pager_turn_phase::generating, 700, 704, frontier, 33) ==
        llama_kv_pager_turn_status::ok);
    const auto frozen = pager->turn_state(0);
    assert(frozen.phase == llama_kv_pager_turn_phase::generating);
    assert(frozen.turn_id == 1 && frozen.frozen_history_generation == 33);
    assert(frozen.selected_history.size() == 1 &&
        frozen.selected_history[0].identity == frontier &&
        frozen.selected_history[0].content_version == 9);
    assert(pager->clear_turn_state(0, 1, 1) == llama_kv_pager_turn_status::ok);
    const auto cleared = pager->turn_state(0);
    assert(cleared.phase == llama_kv_pager_turn_phase::idle);
    assert(cleared.turn_id == 1 && cleared.retrieval_epoch == 1);
    assert(cleared.selected_history.empty() && !cleared.has_committed_frontier);
    assert(pager->transition_turn(0, 2, 1,
        llama_kv_pager_turn_phase::query_provisional, 800, 802, frontier, 0) ==
        llama_kv_pager_turn_status::ok);

    auto frontier_config = config;
    llama_kv_pager_backend frontier_backend;
    frontier_backend.allocate = [](uint64_t bytes, llama_kv_pager_allocation & allocation) {
        allocation.handle = reinterpret_cast<void *>(uintptr_t(2));
        allocation.requested_bytes = bytes;
        allocation.realized_bytes = bytes;
        return true;
    };
    frontier_backend.release = [](llama_kv_pager_allocation & allocation) { allocation = {}; };
    auto frontier_pager = llama_kv_pager::create(frontier_config, geometry(1024),
            resources(640, 128), frontier_backend, status);
    assert(frontier_pager && status == llama_kv_pager_status::ok);
    llama_kv_pager_write_ticket frontier_ticket;
    for (llama_pos position = 256; position < 330; ++position) {
        assert(frontier_pager->begin_write(0, 11, position, frontier_ticket) ==
                llama_kv_pager_write_status::ok);
        assert(frontier_pager->complete_write(frontier_ticket, 1, true) ==
                llama_kv_pager_write_status::ok);
    }
    const auto mixed_frontier = frontier_pager->residency().pages().front();
    assert(mixed_frontier.id.position_begin == 256 &&
            mixed_frontier.id.position_end == 330 && mixed_frontier.pin_count == 1);
    assert(frontier_pager->transition_turn(0, 44, 0,
            llama_kv_pager_turn_phase::query_provisional, 300, 330,
            mixed_frontier.id, 0) == llama_kv_pager_turn_status::ok);
    assert(frontier_pager->transition_turn(0, 44, 0,
            llama_kv_pager_turn_phase::retrieval_commit, 300, 330,
            mixed_frontier.id, 45, {}) == llama_kv_pager_turn_status::ok);
    assert(frontier_pager->transition_turn(0, 44, 1,
            llama_kv_pager_turn_phase::query_replay, 300, 330,
            mixed_frontier.id, 45) == llama_kv_pager_turn_status::ok);
    assert(frontier_pager->mutate({ llama_kv_pager_mutation_kind::remove, 0, -1,
            300, 330, 0, 11 }) == llama_kv_pager_write_status::ok);
    const auto rewound_frontier = frontier_pager->residency().pages();
    assert(rewound_frontier.size() == 1 && rewound_frontier[0].pin_count == 1 &&
            rewound_frontier[0].id.logical_page == mixed_frontier.id.logical_page &&
            rewound_frontier[0].id.page_generation == mixed_frontier.id.page_generation &&
            rewound_frontier[0].id.position_end == 300);
    uint32_t frontier_row = UINT32_MAX;
    assert(frontier_pager->physical_row(0, 299, frontier_row));
    assert(!frontier_pager->physical_row(0, 300, frontier_row));
    assert(frontier_pager->turn_state(0).phase ==
            llama_kv_pager_turn_phase::query_replay);
    std::cout << "query_replay_mixed_frontier=pass prefix=retained query_tail=removed writer_pin=retained\n";
}

static void test_layer_slot_geometry() {
    llama_kv_pager_geometry layered;
    layered.context_tokens = 512;
    layered.page_tokens = 256;
    layered.attention_layers = 2;
    layered.kv_heads = 4;
    layered.key_length = 128;
    layered.value_length = 128;
    layered.page_bytes = 128;
    layered.layer_k_offsets = { 0, 64 };
    layered.layer_v_offsets = { 32, 96 };
    layered.layer_k_page_bytes = { 32, 32 };
    layered.layer_v_page_bytes = { 32, 32 };
    layered.model_layer_ids = { 4, 9 };

    llama_kv_pager_config config;
    config.mode = llama_kv_pager_mode::selective;
    config.hot_pages.automatic = false;
    config.hot_pages.value = 2;
    llama_kv_pager_snapshot snapshot;
    llama_kv_pager_status status;
    assert(llama_kv_pager_plan(
            config, layered, resources(512, 128), snapshot, status));
    assert(status == llama_kv_pager_status::ok);
    assert(snapshot.physical_page_count == 2);
    assert(snapshot.geometry.layer_slot_counts == std::vector<uint32_t>({ 2, 2 }));
    assert(snapshot.geometry.layer_slot_bases == std::vector<uint32_t>({ 0, 2 }));
    assert(snapshot.physical_layer_slot_count == 4);
    assert(snapshot.physical_bytes == 256);
}

static void test_dynamic_live_geometry_and_capture() {
    // Live geometry is not tied to the legacy 16-layer VBR snapshot width.
    for (const uint32_t layers : { 1u, 3u, 17u, 31u }) {
        llama_kv_pager_geometry layered;
        layered.context_tokens = 512;
        layered.page_tokens = 256;
        layered.attention_layers = layers;
        layered.kv_heads = 2;
        layered.key_length = 128;
        layered.value_length = 128;
        layered.page_bytes = uint64_t(layers) * 2 * 64;
        layered.layer_k_offsets.resize(layers);
        layered.layer_v_offsets.resize(layers);
        layered.layer_k_page_bytes.assign(layers, 64);
        layered.layer_v_page_bytes.assign(layers, 64);
        layered.model_layer_ids.reserve(layers);
        layered.unit_descriptors.reserve(size_t(layers) * 2);
        uint64_t offset = 0;
        for (uint32_t layer = 0; layer < layers; ++layer) {
            layered.model_layer_ids.push_back(layer * 2 + 1);
            layered.layer_k_offsets[layer] = offset;
            layered.layer_v_offsets[layer] = offset + 64;
            layered.unit_descriptors.push_back({
                layer * 2, layer, layer * 2 + 1, 0, GGML_TYPE_TURBO4_0,
                2, 128, 64, 64, offset,
            });
            layered.unit_descriptors.push_back({
                layer * 2 + 1, layer, layer * 2 + 1, 1, GGML_TYPE_TURBO4_0,
                2, 128, 64, 64, offset + 64,
            });
            offset += 128;
        }
        llama_kv_pager_config config;
        config.mode = llama_kv_pager_mode::selective;
        config.hot_pages.automatic = false;
        config.hot_pages.value = 2;
        llama_kv_pager_snapshot snapshot;
        llama_kv_pager_status status;
        assert(llama_kv_pager_plan(
                config, layered, resources(1u << 20, layered.page_bytes),
                snapshot, status));
        assert(snapshot.geometry.unit_descriptors.size() == size_t(layers) * 2);
        assert(snapshot.geometry.model_layer_ids.back() == (layers - 1) * 2 + 1);
        assert(snapshot.geometry.layer_slot_bases.back() ==
                (layers - 1) * snapshot.physical_page_count);
    }

    llama_kv_page_id page;
    page.session_generation = 1;
    page.sequence_id = 0;
    page.sequence_generation = 1;
    page.page_generation = 1;
    page.representation_epoch = 1;
    page.model_identity = 1;
    page.topology_identity = 1;
    page.codec_digest = page.codebook_digest = page.rotation_digest = page.meansub_digest = 1;
    page.position_begin = 0;
    page.position_end = VBR_GENERATION_PAGE_CELLS;
    vbr_selected_page_capture_request request;
    request.source_namespace = 1;
    request.child_id = 0;
    request.stream_index = 0;
    request.unit_count = 6;
    request.expected_unit_generations.resize(request.unit_count);
    vbr_selected_page_range range;
    range.identity = page;
    range.positions.resize(VBR_GENERATION_PAGE_CELLS);
    range.physical_cells.resize(VBR_GENERATION_PAGE_CELLS);
    for (uint32_t row = 0; row < VBR_GENERATION_PAGE_CELLS; ++row) {
        range.positions[row] = llama_pos(row);
        range.physical_cells[row] = row;
    }
    request.pages.push_back(std::move(range));
    std::vector<vbr_selected_page_unit_source> sources;
    for (uint32_t unit = 0; unit < request.unit_count; ++unit) {
        request.required_unit_ids.push_back(unit);
        vbr_selected_page_unit_source source;
        source.logical_unit_id = unit;
        source.row_count = VBR_GENERATION_PAGE_CELLS;
        source.row_bytes = 1;
        source.source_identity = unit + 1;
        source.source.size = VBR_GENERATION_PAGE_CELLS;
        sources.push_back(source);
    }
    vbr_selected_page_capture_limits limits;
    limits.max_units = request.unit_count;
    vbr_selected_page_capture_quote quote;
    assert(vbr_selected_page_capture_project(request, sources, limits, quote) ==
            vbr_selected_page_capture_status::ok);
    assert(quote.unit_count == request.unit_count);
    assert(quote.payload_bytes == uint64_t(request.unit_count) * VBR_GENERATION_PAGE_CELLS);
}

static void test_full_256k_capacity_plan() {
    llama_kv_pager_config config;
    config.mode = llama_kv_pager_mode::selective;
    config.hot_pages.automatic = false;
    config.hot_pages.value = 4;

    auto full_resources = resources(2u << 20, 128);
    full_resources.physical_page_cap = 4;
    full_resources.admission.mtp_present = true;
    full_resources.admission.mtp_tokens = 262144;
    full_resources.admission.mtp_k_row_bytes = 1;
    full_resources.admission.mtp_v_row_bytes = 1;
    full_resources.admission.mtp_is_turbo4 = true;

    llama_kv_pager_snapshot snapshot;
    llama_kv_pager_status status;
    assert(llama_kv_pager_plan(
            config, geometry(262144), full_resources, snapshot, status));
    assert(status == llama_kv_pager_status::ok);

    // The complete logical target is addressable through host metadata while
    // the physical device slab remains bounded to the configured hot set.
    assert(snapshot.geometry.context_tokens == 262144);
    assert(snapshot.logical_page_count == 1024);
    assert(snapshot.host_metadata_bytes == 1024 * sizeof(llama_kv_page_id));
    assert(snapshot.physical_page_count == 4);
    assert(snapshot.physical_rows == 4 * 256);
    assert(snapshot.physical_bytes == 4 * 128);
    assert(snapshot.physical_page_count < snapshot.logical_page_count);

    // Native Turbo4 MTP retains the full resolved target row count independently
    // of the bounded attention hot pages.
    assert(snapshot.mtp_rows == 262144);
    assert(snapshot.admission.mtp_bytes == 262144 * 2);
    assert(snapshot.admission.accepted_target_tokens == 4 * 256);
    assert(snapshot.admission.accepted_target_tokens < snapshot.mtp_rows);
}

static uint64_t routing_provider_calls = 0;
static uint64_t routing_provider_source_bytes = 0;

static bool build_routing_summary(
        void *, const llama_kv_page_record & page,
        const llama_kv_routing_summary_config & config,
        llama_kv_routing_page_input & output) noexcept {
    ++routing_provider_calls;
    output = {};
    output.id = page.id;
    const uint32_t rows = uint32_t(page.id.position_end - page.id.position_begin);
    output.row_indices = { 0, rows / 3, (2 * rows) / 3, rows - 1 };
    output.rotated_k_rows.assign(output.row_indices.size() * config.vector_dim, 0.0f);
    for (size_t i = 0; i < output.row_indices.size(); ++i) {
        output.rotated_k_rows[i * config.vector_dim] = float(page.id.logical_page + 1);
    }
    output.source_bytes = output.rotated_k_rows.size() * sizeof(float);
    routing_provider_source_bytes += output.source_bytes;
    return config.representative_count == output.row_indices.size();
}

struct host_page_fixture {
    static constexpr uint64_t source_namespace = 0x9911;
    static constexpr uint32_t row_count = 512;
    static constexpr uint64_t row_bytes = 2;

    std::vector<std::vector<uint8_t>> storage;
    std::vector<vbr_selected_page_unit_source> sources;
    vbr_selected_page_capture_snapshot snapshot;
    std::vector<ggml_tensor *> device_tensors;
    std::thread::id prepare_thread;
    std::thread::id owner_thread;
    bool reject_recheck = false;
    bool use_requested_page_identity = false;

    static bool read(
            const void * context, uint64_t offset,
            uint8_t * destination, size_t size) noexcept {
        const auto * bytes = static_cast<const std::vector<uint8_t> *>(context);
        if (bytes == nullptr || offset > bytes->size() ||
            size > bytes->size() - offset) return false;
        std::memcpy(destination, bytes->data() + offset, size);
        return true;
    }

    static bool acquire(
            void * context,
            const vbr_selected_page_capture_request &,
            vbr_selected_page_capture_snapshot & output) noexcept {
        output = static_cast<host_page_fixture *>(context)->snapshot;
        return true;
    }

    static bool recheck(
            void * context,
            const vbr_selected_page_capture_snapshot & expected) noexcept {
        const auto & self = *static_cast<host_page_fixture *>(context);
        if (self.reject_recheck) return false;
        const auto & current = self.snapshot;
        if (current.pages != expected.pages ||
                current.units.size() != expected.units.size() ||
                current.unit_descriptors.size() != expected.unit_descriptors.size()) {
            return false;
        }
        for (size_t i = 0; i < current.units.size(); ++i) {
            if (current.units[i].generation.repr_gen !=
                    expected.units[i].generation.repr_gen ||
                    current.units[i].generation.publish_seq !=
                    expected.units[i].generation.publish_seq ||
                    current.unit_descriptors[i].repr_gen !=
                    expected.unit_descriptors[i].repr_gen) {
                return false;
            }
        }
        return true;
    }

    static void release(
            void *, const vbr_selected_page_capture_snapshot &) noexcept {}

    static bool prepare(
            void * context, const llama_kv_page_record & page,
            vbr_selected_page_capture_request & request,
            std::vector<vbr_selected_page_unit_source> & output_sources,
            vbr_selected_page_capture_snapshot_provider & snapshots) noexcept {
        auto & self = *static_cast<host_page_fixture *>(context);
        self.prepare_thread = std::this_thread::get_id();
        if (self.use_requested_page_identity) self.snapshot.pages[0] = page.id;
        request = {};
        request.source_namespace = source_namespace;
        request.child_id = 0;
        request.stream_index = 0;
        request.expected_unit_generations.resize(
                VBR_SELECTED_PAGE_REQUIRED_UNITS);
        for (uint32_t unit = 0; unit < VBR_SELECTED_PAGE_REQUIRED_UNITS; ++unit) {
            request.required_unit_ids.push_back(unit);
            request.expected_unit_generations[unit] =
                    self.snapshot.units[unit].generation;
        }
        vbr_selected_page_range range;
        range.identity = page.id;
        const uint32_t count = uint32_t(page.id.position_end - page.id.position_begin);
        range.tail = count != VBR_GENERATION_PAGE_CELLS;
        range.positions.resize(count);
        range.physical_cells.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            range.positions[i] = page.id.position_begin + llama_pos(i);
            range.physical_cells[i] =
                    page.physical_slot * VBR_GENERATION_PAGE_CELLS + i;
        }
        request.pages.push_back(std::move(range));
        output_sources = self.sources;
        snapshots = { &self, acquire, recheck, release };
        return true;
    }

    void initialize(uint32_t rows = row_count) {
        storage.resize(VBR_SELECTED_PAGE_REQUIRED_UNITS);
        sources.reserve(VBR_SELECTED_PAGE_REQUIRED_UNITS);
        snapshot.source_namespace = source_namespace;
        snapshot.child_id = 0;
        snapshot.stream_index = 0;
        llama_kv_page_id page;
        page.session_generation = 1;
        page.sequence_id = 1;
        page.sequence_generation = 1;
        page.logical_page = 0;
        page.page_generation = 3;
        page.representation_epoch = 4;
        page.model_identity = 5;
        page.topology_identity = 6;
        page.codec_digest = 7;
        page.codebook_digest = 8;
        page.rotation_digest = 9;
        page.meansub_digest = 10;
        page.position_begin = 0;
        page.position_end = VBR_GENERATION_PAGE_CELLS;
        snapshot.pages.push_back(page);
        for (uint32_t unit = 0; unit < VBR_SELECTED_PAGE_REQUIRED_UNITS; ++unit) {
            storage[unit].resize(rows * row_bytes);
            for (size_t i = 0; i < storage[unit].size(); ++i) {
                storage[unit][i] = uint8_t(unit + i);
            }
            vbr_selected_page_unit_source source;
            source.logical_unit_id = unit;
            source.row_count = rows;
            source.row_bytes = row_bytes;
            source.source_identity = 0x1000 + unit;
            source.source.size = storage[unit].size();
            source.source.context = &storage[unit];
            source.source.read = read;
            sources.push_back(source);

            vbr_capture_projected_shard_source projected;
            projected.shard_index = 0;
            projected.row_count = rows;
            projected.row_bytes = row_bytes;
            projected.source_identity = source.source_identity;
            projected.source = source.source;
            vbr_capture_unit_snapshot unit_snapshot;
            unit_snapshot.source_namespace = source_namespace;
            unit_snapshot.child_id = 0;
            unit_snapshot.logical_unit_id = unit;
            unit_snapshot.lineage_uuid = { 11, 12 };
            unit_snapshot.controller_generation = 13;
            unit_snapshot.generation.repr_gen = 14;
            unit_snapshot.generation.current_type = GGML_TYPE_TURBO4_0;
            unit_snapshot.generation.last_source_type = GGML_TYPE_TURBO4_0;
            unit_snapshot.generation.domain = vbr_repr_domain::full;
            assert(vbr_capture_projected_shard_topology(
                    { projected }, unit_snapshot.shard_count,
                    unit_snapshot.shard_topology_digest));
            snapshot.units.push_back(unit_snapshot);

            vbr_artifact_unit_descriptor descriptor;
            descriptor.child_id = 0;
            descriptor.logical_unit_id = unit;
            descriptor.lineage_uuid = unit_snapshot.lineage_uuid;
            descriptor.repr_gen = unit_snapshot.generation.repr_gen;
            descriptor.current_type = GGML_TYPE_TURBO4_0;
            descriptor.last_source_type = GGML_TYPE_TURBO4_0;
            descriptor.representation.kind =
                    vbr_artifact_representation_kind::approximate;
            descriptor.representation.codec_id = 4;
            descriptor.representation.codec_version = 1;
            descriptor.representation.reference_digest.fill(1);
            descriptor.side = (unit & 1u)
                    ? vbr_artifact_side::value : vbr_artifact_side::key;
            descriptor.layout = vbr_artifact_layout::row_major;
            descriptor.n_stream = 1;
            descriptor.wm_cells = rows;
            descriptor.codebook_digest.fill(2);
            descriptor.rotation_digest.fill(3);
            descriptor.meansub_digest.fill(4);
            descriptor.row_codec_version = 1;
            vbr_artifact_shard_descriptor shard;
            shard.row_count = rows;
            shard.column_count = 1;
            shard.row_bytes = row_bytes;
            shard.payload_bytes = storage[unit].size();
            descriptor.shards.push_back(shard);
            snapshot.unit_descriptors.push_back(std::move(descriptor));
        }
    }

    bool bind_cuda(ggml_backend_t backend, ggml_backend_dev_t device) {
        ggml_context * context = ggml_init({
            VBR_SELECTED_PAGE_REQUIRED_UNITS * ggml_tensor_overhead(),
            nullptr, true,
        });
        if (!context) return false;
        device_tensors.reserve(VBR_SELECTED_PAGE_REQUIRED_UNITS);
        for (uint32_t unit = 0; unit < VBR_SELECTED_PAGE_REQUIRED_UNITS; ++unit) {
            device_tensors.push_back(ggml_new_tensor_1d(
                    context, GGML_TYPE_I8, storage[unit].size()));
        }
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(
                context, backend);
        if (!buffer) {
            ggml_free(context);
            device_tensors.clear();
            return false;
        }
        for (uint32_t unit = 0; unit < VBR_SELECTED_PAGE_REQUIRED_UNITS; ++unit) {
            ggml_backend_tensor_set(device_tensors[unit], storage[unit].data(),
                    0, storage[unit].size());
            auto & source = sources[unit];
            source.source.context = nullptr;
            source.source.read = nullptr;
            source.source.backend = backend;
            source.source.device = device;
            source.source.tensor = device_tensors[unit];
            vbr_capture_projected_shard_source projected;
            projected.shard_index = 0;
            projected.row_count = source.row_count;
            projected.row_bytes = source.row_bytes;
            projected.source_identity = source.source_identity;
            projected.source = source.source;
            assert(vbr_capture_projected_shard_topology(
                    { projected }, snapshot.units[unit].shard_count,
                    snapshot.units[unit].shard_topology_digest));
        }
        // Keep the backend-owned tensors alive for the caller.  The test frees
        // the context/buffer only after the host worker has drained.
        cuda_context = context;
        cuda_buffer = buffer;
        return true;
    }

    void release_cuda(ggml_backend_t backend) {
        if (cuda_buffer) ggml_backend_buffer_free(cuda_buffer);
        cuda_buffer = nullptr;
        if (cuda_context) ggml_free(cuda_context);
        cuda_context = nullptr;
        if (backend) ggml_backend_free(backend);
        device_tensors.clear();
    }

    ggml_context * cuda_context = nullptr;
    ggml_backend_buffer_t cuda_buffer = nullptr;
};

static void test_host_seal_boundary() {
    host_page_fixture fixture;
    fixture.initialize();
    fixture.owner_thread = std::this_thread::get_id();
    auto host_resources = resources(1u << 20, 128);
    host_resources.host_capture_enabled = true;
    host_resources.host_source_namespace = host_page_fixture::source_namespace;
    host_resources.host_child_id = 0;
    host_resources.host_stream_index = 0;
    host_resources.host_lanes = { { nullptr, nullptr, true } };
    host_resources.host_ring_bytes = 128;
    host_resources.host_chunk_bytes = 64;
    host_resources.host_budget.host.pageable_cap = 1u << 20;
    host_resources.host_budget.host.pageable_state =
            llama_cache_budget_capacity_state::known;
    host_resources.host_budget.host.pinned_cap = 128;
    host_resources.host_budget.host.pinned_state =
            llama_cache_budget_capacity_state::known;
    host_resources.host_budget.host.total_cap = 1u << 20;
    host_resources.host_budget.host.total_state =
            llama_cache_budget_capacity_state::known;
    llama_kv_pager_host_status host_status;
    auto host = llama_kv_pager_host::create(
            host_resources, { &fixture, host_page_fixture::prepare }, host_status);
    assert(host && host_status == llama_kv_pager_host_status::ok);
    llama_kv_page_record page;
    page.id = fixture.snapshot.pages[0];
    page.physical_slot = 0;
    page.state = llama_kv_page_state::gpu_dirty;
    auto result = host->seal(page);
    assert(result.status == llama_kv_pager_host_status::ok);
    assert(host->snapshot().live_pages == 1);
    const auto live_pages = host->pages();
    assert(live_pages.size() == 1);
    assert(live_pages[0].page.identity == page.id);
    assert(!live_pages[0].obsolete);
    assert(host->invalidate(page.id));
    assert(host->snapshot().live_pages == 0);
    assert(host->snapshot().obsolete_pages == 1);
    assert(host->pages().empty());
}

static void test_query_longer_than_generation_tail() {
    host_page_fixture fixture;
    fixture.initialize();
    fixture.owner_thread = std::this_thread::get_id();

    llama_kv_pager_config config;
    config.mode = llama_kv_pager_mode::selective;
    config.hot_pages.automatic = false;
    config.hot_pages.value = 2;
    auto host_resources = resources(320, 128);
    host_resources.host_capture_enabled = true;
    host_resources.host_source_namespace = host_page_fixture::source_namespace;
    host_resources.host_child_id = 0;
    host_resources.host_stream_index = 0;
    host_resources.host_lanes = { { nullptr, nullptr, true } };
    host_resources.host_ring_bytes = 128;
    host_resources.host_chunk_bytes = 64;
    host_resources.host_budget.host.pageable_cap = 1u << 20;
    host_resources.host_budget.host.pageable_state =
            llama_cache_budget_capacity_state::known;
    host_resources.host_budget.host.pinned_cap = 128;
    host_resources.host_budget.host.pinned_state =
            llama_cache_budget_capacity_state::known;
    host_resources.host_budget.host.total_cap = 1u << 20;
    host_resources.host_budget.host.total_state =
            llama_cache_budget_capacity_state::known;

    llama_kv_pager_backend backend;
    backend.allocate = [](uint64_t bytes, llama_kv_pager_allocation & allocation) {
        allocation.handle = reinterpret_cast<void *>(uintptr_t(3));
        allocation.requested_bytes = bytes;
        allocation.realized_bytes = bytes;
        return true;
    };
    backend.release = [](llama_kv_pager_allocation & allocation) { allocation = {}; };
    llama_kv_pager_status status;
    auto pager = llama_kv_pager::create(config, geometry(1024), host_resources,
            backend, status);
    assert(pager && status == llama_kv_pager_status::ok);
    assert(pager->snapshot().physical_page_count == 2);
    pager->bind_representation_identity(5, 6, 7, 8, 9, 10, 4);
    pager->set_host_provider({ &fixture, host_page_fixture::prepare });
    const auto generation_tail_tokens = uint64_t(pager->snapshot().generation_pages) *
        pager->snapshot().geometry.page_tokens;
    assert(generation_tail_tokens < 768);

    llama_kv_page_id frontier;
    frontier.sequence_id = 0;
    frontier.sequence_generation = 11;
    frontier.position_begin = 0;
    frontier.position_end = 1;
    assert(pager->transition_turn(0, 55, 0,
            llama_kv_pager_turn_phase::query_provisional, 0, 768,
            frontier, 0) == llama_kv_pager_turn_status::ok);

    llama_kv_pager_write_ticket ticket;
    for (llama_pos position = 0; position < 768; ++position) {
        if (position != 0 && position % 256 == 0) {
            const auto previous = pager->residency().pages();
            const auto old_frontier = std::find_if(previous.begin(), previous.end(),
                    [&](const auto & page) {
                return page.id.position_begin == position - 256;
            });
            assert(old_frontier != previous.end());
            fixture.snapshot.pages.assign(1, old_frontier->id);
        }
        assert(pager->begin_write(0, 11, position, ticket) ==
                llama_kv_pager_write_status::ok);
        assert(pager->complete_write(ticket, 32, true) ==
                llama_kv_pager_write_status::ok);
    }

    const auto records = pager->exact_page_records(0);
    assert(records.size() == 3);
    assert(std::count_if(records.begin(), records.end(), [](const auto & page) {
        return page.physical_slot != UINT32_MAX;
    }) == 2);
    const auto host_pages = pager->host_catalog()->pages();
    assert(host_pages.size() >= 1);
    assert(pager->turn_state(0).phase == llama_kv_pager_turn_phase::query_provisional);
    std::cout << "query_longer_than_generation_tail=pass query_tokens=768 generation_tail_tokens="
              << generation_tail_tokens << " physical_pages=2 host_pages="
              << host_pages.size() << "\n";
}

static void test_cuda_async_host_publication() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * candidate = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(candidate) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            device = candidate;
            break;
        }
    }
    if (!device) return;
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    assert(backend != nullptr);
    if (!backend) return;

    host_page_fixture fixture;
    fixture.initialize();
    fixture.owner_thread = std::this_thread::get_id();
    if (!fixture.bind_cuda(backend, device)) {
        ggml_backend_free(backend);
        return;
    }
    auto host_resources = resources(1u << 20, 128);
    host_resources.host_capture_enabled = true;
    host_resources.host_source_namespace = host_page_fixture::source_namespace;
    host_resources.host_child_id = 0;
    host_resources.host_stream_index = 0;
    host_resources.host_backend = backend;
    host_resources.host_lanes = { { device, backend, false } };
    // The complete page is much larger than this pinned staging ring. The
    // capture must stream through the fixed slab and publish pageable bytes.
    host_resources.host_ring_bytes = 2u * 64u;
    host_resources.host_chunk_bytes = 64u;
    host_resources.host_budget.host.pageable_cap = 1u << 20;
    host_resources.host_budget.host.pageable_state =
            llama_cache_budget_capacity_state::known;
    host_resources.host_budget.host.pinned_cap = host_resources.host_ring_bytes;
    host_resources.host_budget.host.pinned_state =
            llama_cache_budget_capacity_state::known;
    host_resources.host_budget.host.total_cap = 1u << 20;
    host_resources.host_budget.host.total_state =
            llama_cache_budget_capacity_state::known;
    llama_kv_pager_host_status host_status;
    auto host = llama_kv_pager_host::create(
            host_resources, { &fixture, host_page_fixture::prepare }, host_status);
    assert(host && host_status == llama_kv_pager_host_status::ok);
    assert(host && host->async_enabled());
    if (host) {
        llama_kv_page_record page;
        page.id = fixture.snapshot.pages[0];
        page.physical_slot = 0;
        page.state = llama_kv_page_state::gpu_dirty;
        const auto queued = host->enqueue(page, 7);
        assert(queued.status == llama_kv_pager_host_status::ok);
        assert(queued.queued);
        assert(host->snapshot().live_pages == 0);
        assert(fixture.prepare_thread == fixture.owner_thread);

        auto next_page = page;
        ++next_page.id.logical_page;
        ++next_page.id.page_generation;
        next_page.id.position_begin += VBR_GENERATION_PAGE_CELLS;
        next_page.id.position_end += VBR_GENERATION_PAGE_CELLS;
        const auto queued_next = host->enqueue(next_page, 7);
        assert(queued_next.status == llama_kv_pager_host_status::ok);
        assert(queued_next.queued);
        auto over_capacity_page = next_page;
        ++over_capacity_page.id.logical_page;
        ++over_capacity_page.id.page_generation;
        over_capacity_page.id.position_begin += VBR_GENERATION_PAGE_CELLS;
        over_capacity_page.id.position_end += VBR_GENERATION_PAGE_CELLS;
        const auto over_capacity = host->enqueue(over_capacity_page, 7);
        assert(over_capacity.status == llama_kv_pager_host_status::ring_unavailable);

        // The worker owns only the immutable snapshot acquired by enqueue.
        // Mutating the live generation before the owner drains completion must
        // invalidate the old bytes instead of making them canonical.
        ++fixture.snapshot.units[0].generation.repr_gen;
        ++fixture.snapshot.unit_descriptors[0].repr_gen;

        std::vector<llama_kv_pager_host_completion> completed;
        assert(host->wait() >= 2);
        host->drain(completed);
        assert(completed.size() == 2);
        uint64_t stale_actual_d2h_bytes = 0;
        if (completed.size() == 2) {
            assert(completed[0].content_version == 7);
            assert(completed[0].result.status != llama_kv_pager_host_status::ok);
            assert(completed[1].content_version == 7);
            assert(completed[1].result.status != llama_kv_pager_host_status::ok);
            stale_actual_d2h_bytes = completed[0].result.transfer.bytes +
                completed[1].result.transfer.bytes;
            assert(host->snapshot().live_pages == 0);
            std::cout << "query_checkpoint_stale_d2h=discarded old_content_version=7 "
                         "multi_page_captures=2\n";

            // A later owner-prepared generation remains publishable, proving
            // that stale completion did not poison or pin the slot.
            const auto retry = host->enqueue(page, 8);
            assert(retry.status == llama_kv_pager_host_status::ok);
            completed.clear();
            assert(host->wait() >= 1);
            host->drain(completed);
            assert(completed.size() == 1);
            assert(completed[0].result.status == llama_kv_pager_host_status::ok);
            assert(completed[0].result.queued == false);
            assert(completed[0].result.transfer.event_completions > 0);
            assert(completed[0].result.transfer.bytes > host_resources.host_ring_bytes);
            assert(completed[0].result.pageable_bytes >= completed[0].result.transfer.bytes);
            assert(completed[0].result.pinned_bytes <= host_resources.host_ring_bytes);
            assert(stale_actual_d2h_bytes > 0);
            fprintf(stderr,
                    "inclusive_host_and_page_seal_topology=pass async useful_d2h=%llu "
                    "actual_d2h=%llu "
                    "pageable=%llu metadata=%llu pinned_payload=%llu peak_pinned=%llu "
                    "submitted_chunks=%llu copy_backpressure_waits=%llu events=%llu "
                    "queue_bound_pages=2 capacity_rejected=1 async_captures=3\n",
                    (unsigned long long) completed[0].result.transfer.bytes,
                    (unsigned long long) (stale_actual_d2h_bytes +
                        completed[0].result.transfer.bytes),
                    (unsigned long long) completed[0].result.pageable_bytes,
                    (unsigned long long) completed[0].result.metadata_bytes,
                    (unsigned long long) completed[0].result.pinned_bytes,
                    (unsigned long long) host_resources.host_ring_bytes,
                    (unsigned long long) completed[0].result.transfer.submitted_chunks,
                    (unsigned long long) completed[0].result.transfer.backpressure_waits,
                    (unsigned long long) completed[0].result.transfer.event_completions);
            assert(host->snapshot().live_pages == 1);
            const auto pages = host->pages();
            assert(pages.size() == 1 && pages[0].page.units.size() ==
                    VBR_SELECTED_PAGE_REQUIRED_UNITS);
            assert(pages[0].page.positions.size() == VBR_GENERATION_PAGE_CELLS);
            for (uint32_t unit = 0; unit < VBR_SELECTED_PAGE_REQUIRED_UNITS; ++unit) {
                const auto & host_unit = pages[0].page.units[unit];
                const size_t expected_bytes =
                    VBR_GENERATION_PAGE_CELLS * host_page_fixture::row_bytes;
                assert(host_unit.valid_rows == VBR_GENERATION_PAGE_CELLS);
                assert(host_unit.bytes && host_unit.bytes->size() == expected_bytes);
                std::vector<uint8_t> published(expected_bytes);
                assert(host_unit.bytes->read(0, published.data(), published.size()));
                assert(std::equal(published.begin(), published.end(),
                    fixture.storage[unit].begin()));
            }

            const auto cancelled_enqueue = host->enqueue(page, 9);
            assert(cancelled_enqueue.status == llama_kv_pager_host_status::ok);
            assert(host->invalidate(page.id));
            completed.clear();
            assert(host->wait() >= 1);
            host->drain(completed);
            assert(completed.size() == 1);
            assert(completed[0].result.status != llama_kv_pager_host_status::ok);
            assert(host->snapshot().live_pages == 0);
        }
    }
    host.reset();
    fixture.release_cuda(backend);
}

static void test_compact_checkpoint_page_identity() {
    llama_kv_pager_config config;
    config.mode = llama_kv_pager_mode::selective;
    config.hot_pages.automatic = false;
    config.hot_pages.value = 2;

    llama_kv_pager_backend backend;
    backend.allocate = [](uint64_t bytes, llama_kv_pager_allocation & allocation) {
        allocation.handle = reinterpret_cast<void *>(uintptr_t(7));
        allocation.requested_bytes = bytes;
        allocation.realized_bytes = bytes;
        return true;
    };
    backend.release = [](llama_kv_pager_allocation & allocation) { allocation = {}; };
    auto pager_resources = resources(4096, 128);
    pager_resources.physical_page_cap = 2;
    llama_kv_pager_status status;
    auto pager = llama_kv_pager::create(
            config, geometry(1025), pager_resources, backend, status);
    assert(pager && status == llama_kv_pager_status::ok);
    assert(pager->snapshot().physical_page_count == 2);

    llama_kv_page_id first;
    first.session_generation = 9;
    first.sequence_id = 3;
    first.sequence_generation = 11;
    first.logical_page = 0;
    first.page_generation = 101;
    first.position_begin = 0;
    first.position_end = 256;
    llama_kv_pager_write_ticket ticket;
    assert(pager->begin_restore_page(first, 3, ticket) == llama_kv_pager_write_status::ok);
    assert(pager->complete_write(ticket, 32, true) == llama_kv_pager_write_status::ok);
    assert(pager->begin_restore_page(first, 17, ticket) == llama_kv_pager_write_status::ok);
    assert(pager->complete_write(ticket, 32, true) == llama_kv_pager_write_status::ok);
    assert(pager->seal_ready_pages() == 0);

    llama_kv_page_id tail = first;
    tail.logical_page = 2;
    tail.page_generation = 102;
    tail.position_begin = 512;
    tail.position_end = 520;
    assert(pager->begin_restore_page(tail, 515, ticket) == llama_kv_pager_write_status::ok);
    assert(pager->complete_write(ticket, 32, true) == llama_kv_pager_write_status::ok);
    assert(pager->seal_ready_pages() == 0);

    uint32_t physical = UINT32_MAX;
    assert(pager->physical_row(3, 3, physical));
    assert(pager->physical_row(3, 515, physical));
    const auto records = pager->exact_page_records(3);
    assert(records.size() == 2);
    assert(std::any_of(records.begin(), records.end(), [&](const auto & record) {
        return record.id == first && record.physical_slot != UINT32_MAX;
    }));
    assert(std::any_of(records.begin(), records.end(), [&](const auto & record) {
        return record.id == tail && record.physical_slot != UINT32_MAX;
    }));

    // A failed graph must not leave a partially restored row published.
    assert(pager->begin_restore_page(tail, 519, ticket) == llama_kv_pager_write_status::ok);
    assert(pager->complete_write(ticket, 32, false) == llama_kv_pager_write_status::ok);
    assert(!pager->physical_row(3, 519, physical));
    const auto no_victim = pager->begin_restore_page(
            llama_kv_page_id{ 9, 3, 11, 3, 103, 0, 0, 0, 0, 0, 0, 0, 768, 770 },
            769, ticket);
    assert(no_victim == llama_kv_pager_write_status::no_victim ||
        no_victim == llama_kv_pager_write_status::all_pinned);
}

static void test_mode_lifecycle_matrix() {
    llama_kv_pager_config config;
    config.page_size = 256;

    int allocations = 0;
    int releases = 0;
    llama_kv_pager_backend backend;
    backend.allocate = [&](uint64_t bytes, llama_kv_pager_allocation & allocation) {
        ++allocations;
        allocation.handle = reinterpret_cast<void *>(uintptr_t(0x44));
        allocation.requested_bytes = bytes;
        allocation.realized_bytes = bytes;
        return true;
    };
    backend.release = [&](llama_kv_pager_allocation & allocation) {
        ++releases;
        allocation = {};
    };

    for (const auto mode : {
            llama_kv_pager_mode::observe,
            llama_kv_pager_mode::selective,
            llama_kv_pager_mode::exact }) {
        config.mode = mode;
        llama_kv_pager_status status;
        {
            auto pager = llama_kv_pager::create(
                    config, geometry(1025), resources(1024, 128), backend, status);
            assert(pager && status == llama_kv_pager_status::ok);
            assert(pager->snapshot().initialized);
            assert(pager->snapshot().logical_page_count == 5);
            if (mode == llama_kv_pager_mode::observe) {
                assert(pager->snapshot().physical_page_count == 0);
                assert(pager->residency().slot_capacity() == 0);
            } else {
                assert(pager->snapshot().physical_page_count == 5);
                assert(pager->residency().slot_capacity() == 5);
            }
        }
    }
    assert(allocations == 2 && releases == 2);

    // Feature-off is a valid ordinary configuration and never constructs a
    // pager owner or requests a backend allocation.
    config.mode = llama_kv_pager_mode::off;
    llama_kv_pager_snapshot off_snapshot;
    llama_kv_pager_status off_status;
    assert(llama_kv_pager_plan(
            config, geometry(1025), resources(1024, 128), off_snapshot, off_status));
    assert(off_status == llama_kv_pager_status::disabled);
    assert(!off_snapshot.initialized && off_snapshot.logical_page_count == 0);
}

static void test_pager_host_mutation() {
    host_page_fixture fixture;
    fixture.initialize();
    auto host_resources = resources(1u << 20, 128);
    host_resources.host_capture_enabled = true;
    host_resources.host_source_namespace = host_page_fixture::source_namespace;
    host_resources.host_child_id = 0;
    host_resources.host_stream_index = 0;
    host_resources.host_lanes = { { nullptr, nullptr, true } };
    host_resources.host_ring_bytes = 128;
    host_resources.host_chunk_bytes = 64;
    host_resources.host_budget.host.pageable_cap = 1u << 20;
    host_resources.host_budget.host.pageable_state =
            llama_cache_budget_capacity_state::known;
    host_resources.host_budget.host.pinned_cap = 128;
    host_resources.host_budget.host.pinned_state =
            llama_cache_budget_capacity_state::known;
    host_resources.host_budget.host.total_cap = 1u << 20;
    host_resources.host_budget.host.total_state =
            llama_cache_budget_capacity_state::known;

    llama_kv_pager_config config;
    config.mode = llama_kv_pager_mode::selective;
    llama_kv_pager_status status;
    llama_kv_pager_backend backend;
    backend.allocate = [](uint64_t bytes, llama_kv_pager_allocation & allocation) {
        allocation.handle = reinterpret_cast<void *>(uintptr_t(0x55));
        allocation.requested_bytes = bytes;
        allocation.realized_bytes = bytes;
        return true;
    };
    backend.release = [](llama_kv_pager_allocation & allocation) { allocation = {}; };
    auto pager = llama_kv_pager::create(
            config, geometry(512), host_resources, backend, status);
    assert(pager && status == llama_kv_pager_status::ok);
    pager->bind_representation_identity(5, 6, 7, 8, 9, 10, 4);
    pager->set_host_provider({ &fixture, host_page_fixture::prepare });

    llama_kv_pager_write_ticket ticket;
    for (llama_pos position = 0; position < 256; ++position) {
        assert(pager->begin_write(0, 1, position, ticket) == llama_kv_pager_write_status::ok);
        assert(pager->complete_write(ticket, 32, true) == llama_kv_pager_write_status::ok);
    }
    assert(pager->residency().pages().size() == 1);
    assert(pager->residency().pages()[0].pin_count == 1);
    uint32_t current_page = UINT32_MAX;
    uint32_t current_slot = UINT32_MAX;
    assert(pager->current_page(0, current_page, current_slot));
    assert(current_page == 0 && current_slot == pager->residency().pages()[0].physical_slot);
    assert(!pager->current_page(1, current_page, current_slot));
    fixture.snapshot.pages[0] = pager->residency().pages()[0].id;
    assert(pager->seal_ready_pages() == 1);
    assert(pager->residency().pages()[0].pin_count == 0);
    assert(pager->host_catalog()->snapshot().live_pages == 1);
    assert(pager->exact_page_records(0).size() == 1);

    // Freeze the canonical resident page as history. A generation write may
    // append into the next page, but cannot overwrite the selected identity.
    const auto frozen_record = pager->residency().pages()[0];
    const uint64_t query_refreshes_before_freeze = pager->query_refresh_count();
    assert(pager->transition_turn(0, 44, 0,
            llama_kv_pager_turn_phase::query_provisional, 32, 36,
            frozen_record.id, 0) == llama_kv_pager_turn_status::ok);
    assert(pager->transition_turn(0, 44, 0,
            llama_kv_pager_turn_phase::retrieval_commit, 32, 36,
            frozen_record.id, 45,
            { { frozen_record.id, frozen_record.content_version } }) ==
            llama_kv_pager_turn_status::ok);
    assert(pager->transition_turn(0, 44, 1,
            llama_kv_pager_turn_phase::generating, 32, 36,
            frozen_record.id, 45) == llama_kv_pager_turn_status::ok);
    assert(pager->begin_write(0, 1, 0, ticket) ==
            llama_kv_pager_write_status::all_pinned);
    assert(pager->begin_write(0, 1, 256, ticket) ==
            llama_kv_pager_write_status::ok);
    assert(pager->current_page(0, current_page, current_slot));
    assert(current_page == 1 && current_slot == ticket.physical_slot);
    assert(pager->cancel_write(ticket) == llama_kv_pager_write_status::ok);
    assert(pager->query_refresh_count() == query_refreshes_before_freeze);
    assert(pager->clear_turn_state(0, 44, 1) == llama_kv_pager_turn_status::ok);

    uint8_t prior_byte = 0;
    assert(pager->host_catalog()->pages()[0].page.units[0].bytes->read(
            0, &prior_byte, 1));
    fixture.storage[0][0] ^= 0x5a;
    assert(fixture.storage[0][0] != prior_byte);
    assert(pager->begin_write(0, 1, 0, ticket) == llama_kv_pager_write_status::ok);
    assert(pager->current_page(0, current_page, current_slot));
    assert(current_page == 0 && current_slot == ticket.physical_slot);
    assert(pager->complete_write(ticket, 32, true) == llama_kv_pager_write_status::ok);
    assert(pager->seal_ready_pages() == 1);
    const auto rewritten_pages = pager->host_catalog()->pages();
    assert(rewritten_pages.size() == 1);
    uint8_t rewritten_byte = 0;
    assert(rewritten_pages[0].page.units[0].bytes->read(0, &rewritten_byte, 1));
    assert(rewritten_byte == fixture.storage[0][0]);

    // Batch admission must seal a completed write frontier before deciding
    // that the one-page hot window has no victim.  The old preflight rejected
    // this boundary before begin_write() got its normal maintenance chance.
    llama_kv_pager_config batch_config = config;
    batch_config.hot_pages.automatic = false;
    batch_config.hot_pages.value = 1;
    auto batch_pager = llama_kv_pager::create(
            batch_config, geometry(512), host_resources, backend, status);
    assert(batch_pager && status == llama_kv_pager_status::ok);
    batch_pager->bind_representation_identity(5, 6, 7, 8, 9, 10, 4);
    batch_pager->set_host_provider({ &fixture, host_page_fixture::prepare });
    for (llama_pos position = 0; position < 256; ++position) {
        assert(batch_pager->begin_write(0, 1, position, ticket) ==
                llama_kv_pager_write_status::ok);
        assert(batch_pager->complete_write(ticket, 32, true) ==
                llama_kv_pager_write_status::ok);
    }
    assert(batch_pager->residency().pages().size() == 1);
    fixture.snapshot.pages[0] = batch_pager->residency().pages()[0].id;
    std::vector<llama_kv_pager_write_ticket> batch_boundary_tickets;
    assert(batch_pager->begin_write_batch(0, 1, { 256 },
            batch_boundary_tickets) == llama_kv_pager_write_status::ok);
    assert(batch_boundary_tickets.size() == 1);
    assert(batch_boundary_tickets[0].logical_page == 1);
    assert(batch_pager->cancel_write(batch_boundary_tickets[0]) ==
            llama_kv_pager_write_status::ok);

    // Reproduce the H4096 admission edge after a slot erase. The cache starts
    // with a short warmup, clears that ownership, then fills every P256 hot
    // page before requesting the first row beyond H. The newly completed
    // frontier must receive both its host seal and routing summary before it
    // can be selected as an atomic batch victim.
    host_page_fixture h4096_fixture;
    h4096_fixture.initialize(4096);
    h4096_fixture.use_requested_page_identity = true;
    auto h4096_resources = resources(4096, 128);
    h4096_resources.host_capture_enabled = true;
    h4096_resources.host_source_namespace = host_page_fixture::source_namespace;
    h4096_resources.host_child_id = 0;
    h4096_resources.host_stream_index = 0;
    h4096_resources.host_lanes = { { nullptr, nullptr, true } };
    h4096_resources.host_ring_bytes = 128;
    h4096_resources.host_chunk_bytes = 64;
    h4096_resources.host_budget.host.pageable_cap = 1u << 20;
    h4096_resources.host_budget.host.pageable_state =
            llama_cache_budget_capacity_state::known;
    h4096_resources.host_budget.host.pinned_cap = 128;
    h4096_resources.host_budget.host.pinned_state =
            llama_cache_budget_capacity_state::known;
    h4096_resources.host_budget.host.total_cap = 1u << 20;
    h4096_resources.host_budget.host.total_state =
            llama_cache_budget_capacity_state::known;
    llama_kv_pager_config h4096_config = config;
    h4096_config.hot_pages.automatic = false;
    h4096_config.hot_pages.value = 16;
    auto h4096_pager = llama_kv_pager::create(
            h4096_config, geometry(4352), h4096_resources, backend, status);
    assert(h4096_pager && status == llama_kv_pager_status::ok);
    h4096_pager->bind_representation_identity(5, 6, 7, 8, 9, 10, 4);
    h4096_pager->set_host_provider(
            { &h4096_fixture, host_page_fixture::prepare });
    h4096_pager->set_routing_summary_provider(
            { nullptr, build_routing_summary });
    for (llama_pos position = 0; position < 40; ++position) {
        assert(h4096_pager->begin_write(0, 1, position, ticket) ==
                llama_kv_pager_write_status::ok);
        assert(h4096_pager->complete_write(ticket, 32, true) ==
                llama_kv_pager_write_status::ok);
    }
    h4096_pager->release_sequence_pins(0);
    llama_kv_pager_mutation erase_warmup;
    erase_warmup.kind = llama_kv_pager_mutation_kind::clear;
    erase_warmup.sequence_id = 0;
    assert(h4096_pager->mutate(erase_warmup) ==
            llama_kv_pager_write_status::ok);
    assert(h4096_pager->residency().pages().empty());

    for (llama_pos page_begin = 0; page_begin < 4096; page_begin += 256) {
        std::vector<llama_pos> page_positions;
        page_positions.reserve(256);
        for (llama_pos position = page_begin; position < page_begin + 256; ++position) {
            page_positions.push_back(position);
        }
        assert(h4096_pager->begin_write_batch(0, 1, page_positions,
                batch_boundary_tickets) == llama_kv_pager_write_status::ok);
        assert(batch_boundary_tickets.size() == page_positions.size());
        for (const auto & page_ticket : batch_boundary_tickets) {
            assert(page_ticket.logical_page == uint32_t(page_begin / 256));
            assert(h4096_pager->complete_write(page_ticket, 32, true) ==
                    llama_kv_pager_write_status::ok);
        }
    }
    std::vector<llama_pos> beyond_h_positions;
    for (llama_pos position = 4096; position < 4352; ++position) {
        beyond_h_positions.push_back(position);
    }
    assert(h4096_pager->begin_write_batch(0, 1, beyond_h_positions,
            batch_boundary_tickets) == llama_kv_pager_write_status::ok);
    assert(batch_boundary_tickets.size() == beyond_h_positions.size());
    assert(std::all_of(batch_boundary_tickets.begin(), batch_boundary_tickets.end(),
            [](const auto & page_ticket) { return page_ticket.logical_page == 16; }));
    const auto h4096_resident_pages = h4096_pager->residency().pages();
    assert(h4096_resident_pages.size() == 16);
    for (uint32_t logical = 1; logical <= 16; ++logical) {
        const auto resident = std::find_if(h4096_resident_pages.begin(),
                h4096_resident_pages.end(), [logical](const auto & page) {
                    return page.id.logical_page == logical;
                });
        assert(resident != h4096_resident_pages.end());
        if (logical < 16) assert(resident->host_valid);
    }
    const auto h4096_host_pages = h4096_pager->host_catalog()->pages();
    assert(h4096_host_pages.size() == 16);
    const auto evicted_page = std::find_if(h4096_host_pages.begin(),
            h4096_host_pages.end(), [](const auto & page) {
                return page.page.identity.logical_page == 0;
            });
    assert(evicted_page != h4096_host_pages.end());
    uint8_t retained_byte = 0;
    assert(evicted_page->page.units[0].bytes->read(0, &retained_byte, 1));
    assert(retained_byte == h4096_fixture.storage[0][0]);
    uint32_t h4096_physical_row = UINT32_MAX;
    for (llama_pos position = 4096; position < 4352; ++position) {
        assert(h4096_pager->physical_row(0, position, h4096_physical_row));
    }
    for (auto it = batch_boundary_tickets.rbegin();
            it != batch_boundary_tickets.rend(); ++it) {
        assert(h4096_pager->cancel_write(*it) ==
                llama_kv_pager_write_status::ok);
    }
    std::cout << "h4096_warmup_erase_reservation=pass P=256 H=4096 C=4352 "
                 "routing_summary=ready atomic_batch=256\n";

    // A completed tail is sealed with only its committed rows. It must be
    // readable from the canonical catalog without turning padding into valid
    // positions, and a clean replacement must leave that host page alive.
    llama_kv_pager_config tail_config = config;
    tail_config.hot_pages.automatic = false;
    tail_config.hot_pages.value = 1;
    auto tail_pager = llama_kv_pager::create(
            tail_config, geometry(512), host_resources, backend, status);
    assert(tail_pager && status == llama_kv_pager_status::ok);
    tail_pager->bind_representation_identity(5, 6, 7, 8, 9, 10, 4);
    tail_pager->set_host_provider({ &fixture, host_page_fixture::prepare });
    for (llama_pos position = 0; position < 17; ++position) {
        assert(tail_pager->begin_write(0, 1, position, ticket) ==
                llama_kv_pager_write_status::ok);
        assert(tail_pager->complete_write(ticket, 32, true) ==
                llama_kv_pager_write_status::ok);
    }
    fixture.snapshot.pages[0] = tail_pager->residency().pages()[0].id;
    // The active mutable tail is not canonicalized at a scheduler fence. It
    // must not trigger a host copy or routing-summary work per generated token.
    assert(tail_pager->seal_ready_pages() == 0);
    const uint64_t tail_seal_calls = tail_pager->seal_calls();
    assert(tail_pager->seal_ready_pages() == 0);
    assert(tail_pager->seal_calls() == tail_seal_calls);
    assert(tail_pager->host_catalog()->snapshot().live_pages == 0);

    // Once a later page takes over, the old tail is immutable and can be
    // published exactly once before its slot is reused.
    const auto tail_pages = tail_pager->host_catalog()->pages();
    assert(tail_pages.empty());
    const auto tail_before_eviction = tail_pager->residency().pages();
    assert(tail_before_eviction.size() == 1);
    const auto tail_content_version = tail_before_eviction[0].content_version;
    assert(tail_pager->begin_write(0, 1, 256, ticket) ==
            llama_kv_pager_write_status::ok);
    const auto retained_tail_pages = tail_pager->host_catalog()->pages();
    assert(retained_tail_pages.size() == 1);
    assert(retained_tail_pages[0].page.tail);
    assert(retained_tail_pages[0].page.positions.size() == 17);
    assert(retained_tail_pages[0].page.units.size() == VBR_SELECTED_PAGE_REQUIRED_UNITS);
    for (const auto & unit : retained_tail_pages[0].page.units) {
        assert(unit.valid_rows == 17);
        assert(unit.bytes && unit.bytes->size() == 17 * host_page_fixture::row_bytes);
    }
    const auto cold_tail_id = retained_tail_pages[0].page.identity;
    assert(retained_tail_pages[0].page.identity == cold_tail_id);
    assert(retained_tail_pages[0].page.tail);
    const auto cold_records = tail_pager->exact_page_records(0);
    const auto cold_record = std::find_if(cold_records.begin(), cold_records.end(),
            [&](const auto & record) { return record.id == cold_tail_id; });
    assert(cold_record != cold_records.end());
    assert(cold_record->valid_length == 17);
    assert(cold_record->content_version == tail_content_version);
    assert(cold_record->host_valid && !cold_record->dirty);
    assert(cold_record->physical_slot == UINT32_MAX);
    assert(tail_pager->cancel_write(ticket) == llama_kv_pager_write_status::ok);

    auto hole_pager = llama_kv_pager::create(
            tail_config, geometry(512), host_resources, backend, status);
    assert(hole_pager && status == llama_kv_pager_status::ok);
    hole_pager->bind_representation_identity(5, 6, 7, 8, 9, 10, 4);
    hole_pager->set_host_provider({ &fixture, host_page_fixture::prepare });
    llama_kv_pager_write_ticket hole_ticket;
    for (llama_pos position : { llama_pos(0), llama_pos(2) }) {
        assert(hole_pager->begin_write(0, 1, position, hole_ticket) ==
                llama_kv_pager_write_status::ok);
        assert(hole_pager->complete_write(hole_ticket, 32, true) ==
                llama_kv_pager_write_status::ok);
    }
    assert(hole_pager->seal_ready_pages() == 0);
    assert(hole_pager->host_catalog()->snapshot().live_pages == 0);

    const uint64_t stale_epoch = pager->residency().epoch();
    llama_kv_pager_write_ticket release_ticket;
    assert(pager->begin_write(0, 1, 0, release_ticket) ==
        llama_kv_pager_write_status::ok);
    assert(pager->residency().pages()[0].pin_count == 1);
    // A stale rollback must not release the frontier as a side effect. The
    // release is staged with a successful mutation instead.
    assert(pager->mutate({
            llama_kv_pager_mutation_kind::remove, 0, -1, 0, 1, 0, 1,
            stale_epoch - 1, true }) == llama_kv_pager_write_status::stale_generation);
    assert(pager->residency().pages()[0].pin_count == 1);
    assert(pager->cancel_write(release_ticket) == llama_kv_pager_write_status::ok);

    const uint64_t epoch = pager->residency().epoch();
    assert(pager->mutate({
            llama_kv_pager_mutation_kind::remove, 0, -1, 0, 256, 0, 1,
            epoch - 1 }) == llama_kv_pager_write_status::stale_generation);
    assert(pager->residency().epoch() == epoch);

    // A completion captured before a partial-tail edit must not publish after
    // that edit, even when the physical page is reused in place.
    llama_kv_pager_write_ticket stale_ticket;
    assert(pager->begin_write(0, 1, 0, stale_ticket) == llama_kv_pager_write_status::ok);
    const uint32_t old_page_generation = stale_ticket.page_generation;
    assert(pager->mutate({
            llama_kv_pager_mutation_kind::remove, 0, -1, 0, 1, 0, 1,
            pager->residency().epoch(), true }) == llama_kv_pager_write_status::ok);
    assert(pager->residency().pages()[0].id.page_generation != old_page_generation);
    assert(pager->complete_write(stale_ticket, 32, true) ==
        llama_kv_pager_write_status::stale_generation);
    assert(pager->mutate({
            llama_kv_pager_mutation_kind::remove, 0, -1, 0, 256, 0, 1,
            pager->residency().epoch() }) == llama_kv_pager_write_status::ok);
    assert(pager->host_catalog()->snapshot().live_pages == 0);
    assert(pager->exact_page_records(0).empty());
}

static void test_generation_ring_victim_and_history_pins() {
    host_page_fixture fixture;
    fixture.initialize();
    auto host_resources = resources(1u << 20, 128);
    host_resources.host_capture_enabled = true;
    host_resources.host_source_namespace = host_page_fixture::source_namespace;
    host_resources.host_child_id = 0;
    host_resources.host_stream_index = 0;
    host_resources.host_lanes = { { nullptr, nullptr, false } };
    host_resources.host_ring_bytes = 128;
    host_resources.host_chunk_bytes = 64;
    host_resources.host_budget.host.pageable_cap = 1u << 20;
    host_resources.host_budget.host.pageable_state = llama_cache_budget_capacity_state::known;
    host_resources.host_budget.host.pinned_cap = 128;
    host_resources.host_budget.host.pinned_state = llama_cache_budget_capacity_state::known;
    host_resources.host_budget.host.total_cap = 1u << 20;
    host_resources.host_budget.host.total_state = llama_cache_budget_capacity_state::known;

    llama_kv_pager_config config;
    config.mode = llama_kv_pager_mode::selective;
    config.hot_pages.automatic = false;
    config.hot_pages.value = 3;
    config.retrieval_pages.automatic = false;
    config.retrieval_pages.value = 2;
    llama_kv_pager_backend backend;
    backend.allocate = [](uint64_t bytes, llama_kv_pager_allocation & allocation) {
        allocation.handle = reinterpret_cast<void *>(uintptr_t(0x97));
        allocation.requested_bytes = bytes;
        allocation.realized_bytes = bytes;
        return true;
    };
    backend.release = [](llama_kv_pager_allocation & allocation) { allocation = {}; };
    llama_kv_pager_status status;
    auto pager = llama_kv_pager::create(
            config, geometry(2048), host_resources, backend, status);
    assert(pager && status == llama_kv_pager_status::ok);
    assert(pager->snapshot().physical_page_count == 3);
    assert(pager->snapshot().generation_pages == 1);
    pager->bind_representation_identity(5, 6, 7, 8, 9, 10, 4);
    pager->set_host_provider({ &fixture, host_page_fixture::prepare });

    llama_kv_pager_write_ticket ticket;
    for (llama_pos position = 0; position < 256; ++position) {
        assert(pager->begin_write(0, 1, position, ticket) == llama_kv_pager_write_status::ok);
        assert(pager->complete_write(ticket, 32, true) == llama_kv_pager_write_status::ok);
    }
    fixture.snapshot.pages[0] = pager->residency().pages()[0].id;
    assert(pager->seal_ready_pages() == 1);
    const auto history = pager->residency().pages()[0];
    assert(pager->transition_turn(0, 97, 0,
            llama_kv_pager_turn_phase::query_provisional, 250, 256,
            history.id, 0) == llama_kv_pager_turn_status::ok);
    assert(pager->transition_turn(0, 97, 0,
            llama_kv_pager_turn_phase::retrieval_commit, 250, 256,
            history.id, 1, { { history.id, history.content_version } }) ==
            llama_kv_pager_turn_status::ok);
    assert(pager->transition_turn(0, 97, 1,
            llama_kv_pager_turn_phase::generating, 250, 256,
            history.id, 1) == llama_kv_pager_turn_status::ok);

    // G is one page (256 tokens). Accept more than two G while keeping the
    // selected historical needle frozen in the physical target window.
    const uint64_t generation_page_tokens = pager->snapshot().geometry.page_tokens;
    const uint64_t accepted_tokens = 2 * generation_page_tokens + 17;
    for (llama_pos position = 256; position < 256 + llama_pos(accepted_tokens); ++position) {
        if (position > 256 && position % 256 == 0) {
            const uint32_t prior_logical = uint32_t(position / 256 - 1);
            const auto current = pager->residency().pages();
            fixture.snapshot.pages.clear();
            for (const auto & page : current) {
                if (page.id.logical_page >= 1 && page.id.logical_page <= prior_logical) {
                    fixture.snapshot.pages.push_back(page.id);
                }
            }
            assert(!fixture.snapshot.pages.empty());
        }
        if (position == 512) {
            (void) pager->seal_ready_pages();
            const auto indexed = pager->turn_state(0).completed_generation_pages;
            assert(indexed.size() == 1 && indexed[0].logical_page == 1);
        }
        assert(pager->begin_write(0, 1, position, ticket) == llama_kv_pager_write_status::ok);
        assert(pager->complete_write(ticket, 32, true) == llama_kv_pager_write_status::ok);
        if (position == 768) (void) pager->seal_ready_pages();
    }
    assert(accepted_tokens > 2 * uint64_t(pager->snapshot().generation_pages) *
            pager->snapshot().geometry.page_tokens);
    const auto pages = pager->residency().pages();
    bool retained_history = false;
    bool evicted_oldest_generation = true;
    bool retained_newer_generation = false;
    bool retained_mutable_tail = false;
    for (const auto & page : pages) {
        retained_history = retained_history || page.id == history.id;
        evicted_oldest_generation = evicted_oldest_generation && page.id.logical_page != 1;
        retained_newer_generation = retained_newer_generation || page.id.logical_page == 2;
        retained_mutable_tail = retained_mutable_tail || page.id.logical_page == 3;
    }
    assert(retained_history && evicted_oldest_generation && retained_newer_generation &&
            retained_mutable_tail && pages.size() == 3);
    const auto generation_state = pager->turn_state(0);
    assert(std::none_of(generation_state.completed_generation_pages.begin(),
            generation_state.completed_generation_pages.end(),
            [](const auto & id) { return id.logical_page == 1 || id.logical_page == 3; }));

    const auto tail = std::find_if(pages.begin(), pages.end(),
            [](const auto & page) { return page.id.logical_page == 3; });
    assert(tail != pages.end() && tail->id.position_end == 785 && tail->pin_count != 0);
    fixture.snapshot.pages[0] = tail->id;
    assert(pager->clear_turn_state(0, 97, 1) == llama_kv_pager_turn_status::ok);
    assert(pager->turn_state(0).completed_generation_pages.empty());
    // A rejected seal cannot enter the generation queue or displace frozen
    // history. The partial current write is made immutable at the boundary,
    // its attempted host capture is rejected, and admission reports no safe
    // victim while retaining both existing identities.
    host_page_fixture rejected_fixture;
    rejected_fixture.initialize();
    auto rejected_resources = host_resources;
    rejected_resources.host_source_namespace = host_page_fixture::source_namespace;
    llama_kv_pager_config rejected_config = config;
    rejected_config.hot_pages.value = 2;
    rejected_config.retrieval_pages.value = 1;
    auto rejected_pager = llama_kv_pager::create(
            rejected_config, geometry(1024), rejected_resources, backend, status);
    assert(rejected_pager && status == llama_kv_pager_status::ok);
    rejected_pager->bind_representation_identity(5, 6, 7, 8, 9, 10, 4);
    rejected_pager->set_host_provider(
            { &rejected_fixture, host_page_fixture::prepare });
    for (llama_pos position = 0; position < 256; ++position) {
        assert(rejected_pager->begin_write(0, 1, position, ticket) ==
                llama_kv_pager_write_status::ok);
        assert(rejected_pager->complete_write(ticket, 32, true) ==
                llama_kv_pager_write_status::ok);
    }
    rejected_fixture.snapshot.pages[0] = rejected_pager->residency().pages()[0].id;
    assert(rejected_pager->seal_ready_pages() == 1);
    const auto rejected_history = rejected_pager->residency().pages()[0];
    assert(rejected_pager->transition_turn(0, 98, 0,
            llama_kv_pager_turn_phase::query_provisional, 250, 256,
            rejected_history.id, 0) == llama_kv_pager_turn_status::ok);
    assert(rejected_pager->transition_turn(0, 98, 0,
            llama_kv_pager_turn_phase::retrieval_commit, 250, 256,
            rejected_history.id, 1,
            { { rejected_history.id, rejected_history.content_version } }) ==
            llama_kv_pager_turn_status::ok);
    assert(rejected_pager->transition_turn(0, 98, 1,
            llama_kv_pager_turn_phase::generating, 250, 256,
            rejected_history.id, 1) == llama_kv_pager_turn_status::ok);
    assert(rejected_pager->begin_write(0, 1, 256, ticket) ==
            llama_kv_pager_write_status::ok);
    assert(rejected_pager->complete_write(ticket, 32, true) ==
            llama_kv_pager_write_status::ok);
    const auto dirty_generation = rejected_pager->residency().pages();
    const auto dirty = std::find_if(dirty_generation.begin(), dirty_generation.end(),
            [](const auto & page) { return page.id.logical_page == 1; });
    assert(dirty != dirty_generation.end() && !dirty->host_valid);
    rejected_fixture.snapshot.pages[0] = dirty->id;
    rejected_fixture.reject_recheck = true;
    assert(rejected_pager->begin_write(0, 1, 512, ticket) ==
            llama_kv_pager_write_status::no_victim);
    const auto after_rejection = rejected_pager->residency().pages();
    assert(after_rejection.size() == 2);
    assert(std::any_of(after_rejection.begin(), after_rejection.end(),
            [&](const auto & page) { return page.id == rejected_history.id; }));
    assert(std::none_of(after_rejection.begin(), after_rejection.end(),
            [](const auto & page) { return page.id.logical_page == 2; }));

    // Before the generation FIFO contains a completed page, a full hot pool
    // may still need to reclaim an unselected prior page for the generation
    // tail. Keep the selected page and active query page stable while doing so.
    host_page_fixture borrow_fixture;
    borrow_fixture.initialize();
    auto borrow_config = config;
    auto borrow_resources = host_resources;
    borrow_resources.host_source_namespace = host_page_fixture::source_namespace;
    borrow_config.hot_pages.value = 3;
    borrow_config.retrieval_pages.value = 2;
    auto borrow_pager = llama_kv_pager::create(
            borrow_config, geometry(1024), borrow_resources, backend, status);
    assert(borrow_pager && status == llama_kv_pager_status::ok);
    borrow_pager->bind_representation_identity(5, 6, 7, 8, 9, 10, 4);
    borrow_pager->set_host_provider(
            { &borrow_fixture, host_page_fixture::prepare });
    for (uint32_t logical = 0; logical < 3; ++logical) {
        const llama_pos begin = llama_pos(logical * 256);
        for (llama_pos position = begin; position < begin + 256; ++position) {
            assert(borrow_pager->begin_write(0, 1, position, ticket) ==
                    llama_kv_pager_write_status::ok);
            assert(borrow_pager->complete_write(ticket, 32, true) ==
                    llama_kv_pager_write_status::ok);
        }
        const auto pages_before_seal = borrow_pager->residency().pages();
        const auto page = std::find_if(pages_before_seal.begin(), pages_before_seal.end(),
                [&](const auto & value) { return value.id.logical_page == logical; });
        assert(page != pages_before_seal.end());
        if (logical < 2) {
            borrow_fixture.snapshot.pages[0] = page->id;
            assert(borrow_pager->seal_ready_pages() == 1);
        }
    }
    const auto before_borrow = borrow_pager->residency().pages();
    const auto selected_page = *std::find_if(before_borrow.begin(), before_borrow.end(),
            [](const auto & page) { return page.id.logical_page == 0; });
    const auto query_page = *std::find_if(before_borrow.begin(), before_borrow.end(),
            [](const auto & page) { return page.id.logical_page == 2; });
    assert(borrow_pager->transition_turn(0, 99, 0,
            llama_kv_pager_turn_phase::query_provisional, 512, 768,
            query_page.id, 0) == llama_kv_pager_turn_status::ok);
    assert(borrow_pager->transition_turn(0, 99, 0,
            llama_kv_pager_turn_phase::retrieval_commit, 512, 768,
            query_page.id, 1,
            { { selected_page.id, selected_page.content_version } }) ==
            llama_kv_pager_turn_status::ok);
    assert(borrow_pager->transition_turn(0, 99, 1,
            llama_kv_pager_turn_phase::query_replay, 512, 768,
            query_page.id, 1) == llama_kv_pager_turn_status::ok);
    assert(borrow_pager->transition_turn(0, 99, 1,
            llama_kv_pager_turn_phase::generating, 512, 768,
            query_page.id, 1) == llama_kv_pager_turn_status::ok);
    std::vector<llama_kv_pager_write_ticket> borrow_tickets;
    assert(borrow_pager->begin_write_batch(0, 1, { 768, 769, 770 },
            borrow_tickets) == llama_kv_pager_write_status::ok);
    assert(borrow_tickets.size() == 3);
    for (const auto & borrow_ticket : borrow_tickets) {
        assert(borrow_pager->complete_write(borrow_ticket, 32, true) ==
                llama_kv_pager_write_status::ok);
    }
    const auto after_borrow = borrow_pager->residency().pages();
    assert(std::any_of(after_borrow.begin(), after_borrow.end(),
            [&](const auto & page) { return page.id == selected_page.id; }));
    assert(std::any_of(after_borrow.begin(), after_borrow.end(),
            [&](const auto & page) { return page.id == query_page.id; }));
    assert(std::none_of(after_borrow.begin(), after_borrow.end(),
            [](const auto & page) { return page.id.logical_page == 1; }));
    assert(std::any_of(after_borrow.begin(), after_borrow.end(),
            [](const auto & page) { return page.id.logical_page == 3; }));
    std::cout << "generation_ring_victim_and_history_pins=pass accepted_tokens="
              << accepted_tokens << " G=" << pager->snapshot().generation_pages
              << " H=" << pager->snapshot().physical_page_count
              << " oldest_generation_evicted=1 history_retained=1 mutable_tail_retained=1"
                 " rejected_seal_excluded=1\n";
}

int main() {
    test_turn_epoch_state_and_geometry();
    test_layer_slot_geometry();
    test_dynamic_live_geometry_and_capture();
    test_host_seal_boundary();
    test_query_longer_than_generation_tail();
    test_cuda_async_host_publication();
    test_compact_checkpoint_page_identity();
    test_mode_lifecycle_matrix();
    test_pager_host_mutation();
    test_generation_ring_victim_and_history_pins();
    test_full_256k_capacity_plan();
    llama_kv_pager_config off;
    llama_kv_pager_snapshot snapshot;
    llama_kv_pager_status status;
    assert(llama_kv_pager_plan(off, geometry(1024), resources(4096, 128), snapshot, status));
    assert(status == llama_kv_pager_status::disabled && !snapshot.initialized);

    llama_kv_pager_config config;
    config.mode = llama_kv_pager_mode::selective;
    auto plan_resources = resources(1024, 128);
    assert(llama_kv_pager_plan(config, geometry(1025), plan_resources, snapshot, status));
    assert(status == llama_kv_pager_status::ok);
    assert(snapshot.logical_page_count == 5 && snapshot.physical_page_count == 5);
    assert(snapshot.physical_rows == 5 * 256);
    assert(snapshot.host_metadata_bytes == 5 * sizeof(llama_kv_page_id));

    config.hot_pages.automatic = false;
    config.hot_pages.value = 2;
    plan_resources.physical_page_cap = 0;
    assert(llama_kv_pager_plan(config, geometry(1025), plan_resources, snapshot, status));
    assert(snapshot.physical_page_count == 2 && snapshot.physical_rows == 2 * 256);

    config.hot_pages.automatic = true;
    plan_resources.physical_page_cap = 1;
    assert(llama_kv_pager_plan(config, geometry(1025), plan_resources, snapshot, status));
    assert(snapshot.physical_page_count == 1);
    config.hot_pages.automatic = false;
    config.hot_pages.value = 2;
    plan_resources.physical_page_cap = 0;

    auto tiny = resources(128, 128);
    assert(!llama_kv_pager_plan(config, geometry(1024), tiny, snapshot, status));
    assert(status == llama_kv_pager_status::admission);

    auto no_host = resources(1024, 128);
    no_host.host_budget_known = false;
    assert(!llama_kv_pager_plan(config, geometry(1024), no_host, snapshot, status));
    assert(status == llama_kv_pager_status::host_budget);

    auto invalid = geometry(1024);
    invalid.page_tokens = 128;
    assert(!llama_kv_pager_plan(config, invalid, resources(1024, 128), snapshot, status));
    assert(status == llama_kv_pager_status::invalid_geometry);

    ggml_backend_t external_backend = ggml_backend_cpu_init();
    assert(external_backend != nullptr);
    ggml_backend_buffer_t external_buffer = ggml_backend_alloc_buffer(external_backend, 128);
    assert(external_buffer != nullptr);
    ggml_context * external_ctx = ggml_init({ 1024, nullptr, true });
    assert(external_ctx != nullptr);
    ggml_tensor * external_tensor = ggml_new_tensor_1d(external_ctx, GGML_TYPE_I8, 128);
    assert(external_tensor != nullptr);
    assert(ggml_backend_tensor_alloc(external_buffer, external_tensor,
            ggml_backend_buffer_get_base(external_buffer)) == GGML_STATUS_SUCCESS);
    auto external_config = config;
    external_config.hot_pages.automatic = false;
    external_config.hot_pages.value = 1;
    auto external_resources = resources(1024, 128);
    external_resources.external_storage_buffer = external_buffer;
    external_resources.external_storage_tensor = external_tensor;
    auto external_pager = llama_kv_pager::create(
            external_config, geometry(1024), external_resources, {}, status);
    assert(external_pager && status == llama_kv_pager_status::ok);
    assert(external_pager->snapshot().realized_bytes == 128);
    external_pager.reset();
    ggml_free(external_ctx);
    ggml_backend_buffer_free(external_buffer);
    ggml_backend_free(external_backend);

    int allocations = 0;
    int releases = 0;
    llama_kv_pager_backend backend;
    backend.allocate = [&](uint64_t bytes, llama_kv_pager_allocation & allocation) {
        ++allocations;
        allocation.handle = reinterpret_cast<void *>(uintptr_t(1));
        allocation.requested_bytes = bytes;
        allocation.realized_bytes = bytes;
        return true;
    };
    backend.release = [&](llama_kv_pager_allocation & allocation) {
        ++releases;
        allocation = {};
    };
    {
        auto pager = llama_kv_pager::create(config, geometry(1025), plan_resources, backend, status);
        assert(pager && status == llama_kv_pager_status::ok);
        assert(pager->snapshot().initialized);
        assert(pager->residency().slot_capacity() == 2);
    }
    assert(allocations == 1 && releases == 1);

    // An allocator OOM is retried only with a strictly smaller page-aligned
    // target pool. The fake fails the first two large candidates and succeeds
    // at two pages, giving a deterministic termination receipt.
    int retry_allocations = 0;
    llama_kv_pager_backend retry_backend;
    retry_backend.allocate = [&](uint64_t bytes, llama_kv_pager_allocation & allocation) {
        ++retry_allocations;
        const uint64_t page_bytes = 128;
        if (bytes > 2 * page_bytes) {
            return false;
        }
        allocation.handle = reinterpret_cast<void *>(uintptr_t(4));
        allocation.requested_bytes = bytes;
        allocation.realized_bytes = bytes;
        return true;
    };
    retry_backend.release = [](llama_kv_pager_allocation & allocation) { allocation = {}; };
    auto retry_config = config;
    retry_config.hot_pages.automatic = true;
    auto retry_resources = resources(2048, 128);
    auto retry_pager = llama_kv_pager::create(
            retry_config, geometry(1025), retry_resources, retry_backend, status);
    assert(retry_pager && status == llama_kv_pager_status::ok);
    assert(retry_pager->snapshot().physical_page_count == 2);
    assert(retry_pager->snapshot().admission_attempts.size() == 3);
    assert(retry_pager->snapshot().admission_attempts[0].page_cap >
            retry_pager->snapshot().admission_attempts[1].page_cap);
    assert(retry_pager->snapshot().admission_attempts[1].page_cap >
            retry_pager->snapshot().admission_attempts[2].page_cap);
    assert(retry_pager->snapshot().admission_attempts.back().allocation_succeeded);
    assert(retry_allocations == 3);
    retry_pager.reset();

    backend.allocate = [](uint64_t, llama_kv_pager_allocation &) { return false; };
    assert(!llama_kv_pager::create(config, geometry(1025), plan_resources, backend, status));
    assert(status == llama_kv_pager_status::allocation);

    backend.allocate = [](uint64_t bytes, llama_kv_pager_allocation & allocation) {
        allocation.handle = reinterpret_cast<void *>(uintptr_t(2));
        allocation.requested_bytes = bytes;
        allocation.realized_bytes = bytes + 64;
        return true;
    };
    assert(!llama_kv_pager::create(config, geometry(1025), plan_resources, backend, status));
    assert(status == llama_kv_pager_status::realized_mismatch);

    plan_resources.duplicate_representation_authority = true;
    assert(!llama_kv_pager_plan(config, geometry(1024), plan_resources, snapshot, status));
    assert(status == llama_kv_pager_status::unsupported_authority);

    int write_allocations = 0;
    llama_kv_pager_backend write_backend;
    write_backend.allocate = [&](uint64_t bytes, llama_kv_pager_allocation & allocation) {
        ++write_allocations;
        allocation.handle = reinterpret_cast<void *>(uintptr_t(3));
        allocation.requested_bytes = bytes;
        allocation.realized_bytes = bytes;
        return true;
    };
    write_backend.release = [](llama_kv_pager_allocation & allocation) { allocation = {}; };
    auto pager = llama_kv_pager::create(config, geometry(1025), resources(1024, 128), write_backend, status);
    assert(pager && write_allocations == 1 && status == llama_kv_pager_status::ok);
    pager->set_routing_summary_provider({ nullptr, build_routing_summary });

    // Runtime graph inputs carry model layer IDs, not compact geometry
    // ordinals.  Sparse IDs must still resolve to the same bounded physical
    // row; an unknown ID must not silently fall back to a logical row.
    auto sparse_geometry = geometry(1025);
    sparse_geometry.model_layer_ids = { 4, 9 };
    sparse_geometry.attention_layers = 2;
    auto sparse_pager = llama_kv_pager::create(
            config, sparse_geometry, resources(1024, 128), write_backend, status);
    assert(sparse_pager && status == llama_kv_pager_status::ok);
    llama_kv_pager_write_ticket sparse_ticket;
    assert(sparse_pager->begin_write(0, 1, 3, 9, sparse_ticket) == llama_kv_pager_write_status::ok);
    assert(sparse_ticket.attention_layer == 9);
    uint32_t sparse_row = UINT32_MAX;
    assert(sparse_pager->physical_row(0, 3, 4, sparse_row) && sparse_row == 3);
    assert(sparse_pager->physical_row(0, 3, 9, sparse_row) && sparse_row == 3);
    assert(!sparse_pager->physical_row(0, 3, 1, sparse_row));

    // A graph write must use the authenticated reservation, not only a
    // matching logical lookup. Exercise the K and V callers independently.
    uint32_t k_row = UINT32_MAX;
    uint32_t v_row = UINT32_MAX;
    assert(sparse_pager->physical_row(sparse_ticket, 4, k_row) && k_row == 3);
    assert(sparse_pager->physical_row(sparse_ticket, 9, v_row) && v_row == 3);
    auto stale_page = sparse_ticket;
    stale_page.page_generation++;
    assert(!sparse_pager->physical_row(stale_page, 9, v_row));
    auto stale_sequence = sparse_ticket;
    stale_sequence.sequence_generation++;
    assert(!sparse_pager->physical_row(stale_sequence, 9, v_row));
    auto overflow_ticket = sparse_ticket;
    overflow_ticket.physical_slot = sparse_pager->snapshot().physical_page_count;
    assert(!sparse_pager->physical_row(overflow_ticket, 9, v_row));
    auto unpublished = sparse_ticket;
    unpublished.logical_page = 99;
    assert(!sparse_pager->physical_row(unpublished, 9, v_row));
    assert(sparse_pager->cancel_write(sparse_ticket) == llama_kv_pager_write_status::ok);

    // The final valid row of a partial page is addressable, while its padding
    // remains refused and cannot become a write destination.
    auto tail_geometry = geometry(513);
    tail_geometry.model_layer_ids = { 4, 9 };
    tail_geometry.attention_layers = 2;
    auto tail_pager = llama_kv_pager::create(
            config, tail_geometry, resources(1024, 128), write_backend, status);
    assert(tail_pager && status == llama_kv_pager_status::ok);
    llama_kv_pager_write_ticket tail_ticket;
    assert(tail_pager->begin_write(0, 1, 512, 4, tail_ticket) ==
            llama_kv_pager_write_status::ok);
    assert(tail_pager->physical_row(tail_ticket, 4, v_row) &&
            v_row == tail_ticket.physical_row && v_row % 256 == 0);
    assert(tail_pager->complete_write(tail_ticket, 1, true) ==
            llama_kv_pager_write_status::ok);
    assert(!tail_pager->physical_row(0, 513, v_row));

    // A whole prefill batch reserves its write frontier before graph
    // submission. Crossing the physical H=2-page window must roll back every
    // earlier row rather than leaving a partially admitted prefix.
    std::vector<llama_pos> oversized_positions;
    for (llama_pos position = 0; position <= 1024; ++position) {
        oversized_positions.push_back(position);
    }
    std::vector<llama_kv_pager_write_ticket> batch_tickets;
    const auto oversized_status = pager->begin_write_batch(
            0, 11, oversized_positions, batch_tickets);
    assert(oversized_status == llama_kv_pager_write_status::no_victim ||
           oversized_status == llama_kv_pager_write_status::all_pinned);
    assert(batch_tickets.empty());
    assert(pager->residency().pages().empty());

    // Once the hot window is full, a failed crossing batch must not evict a
    // cleanly committed prefix or leave a partial new page behind.
    auto full_resources = resources(1024, 128);
    auto full_pager = llama_kv_pager::create(config, geometry(1025),
            full_resources, write_backend, status);
    assert(full_pager && status == llama_kv_pager_status::ok);
    llama_kv_pager_write_ticket full_ticket;
    for (llama_pos position : { llama_pos(0), llama_pos(256) }) {
        assert(full_pager->begin_write(0, 11, position, full_ticket) ==
                llama_kv_pager_write_status::ok);
        assert(full_pager->complete_write(full_ticket, 32, true) ==
                llama_kv_pager_write_status::ok);
    }
    const auto full_before = full_pager->residency();
    std::vector<llama_pos> crossing_positions;
    for (llama_pos position = 512; position <= 768; ++position) {
        crossing_positions.push_back(position);
    }
    assert(full_pager->begin_write_batch(0, 11, crossing_positions, batch_tickets) ==
            llama_kv_pager_write_status::no_victim);
    assert(batch_tickets.empty());
    const auto full_after = full_pager->residency();
    // Admission releases the completed prior frontier before evaluating
    // capacity. The rejected batch may publish that pin release, but it must
    // not evict or partially allocate any page.
    assert(full_after.pages().size() == full_before.pages().size());
    for (size_t i = 0; i < full_before.pages().size(); ++i) {
        assert(full_after.pages()[i].id == full_before.pages()[i].id);
        assert(full_after.pages()[i].physical_slot == full_before.pages()[i].physical_slot);
    }

    std::vector<llama_pos> batch_positions;
    for (llama_pos position = 0; position < 300; ++position) {
        batch_positions.push_back(position);
    }
    assert(pager->begin_write_batch(0, 11, batch_positions, batch_tickets) ==
        llama_kv_pager_write_status::ok);
    assert(batch_tickets.size() == batch_positions.size());
    for (auto it = batch_tickets.rbegin(); it != batch_tickets.rend(); ++it) {
        assert(pager->cancel_write(*it) == llama_kv_pager_write_status::ok);
    }
    assert(pager->residency().pages().empty());

    // Exercise one-token and three-token production packets, including a
    // single ubatch that crosses the logical page boundary. Reverse ticket
    // cancellation must remove both provisional pages as one rejected suffix.
    assert(pager->begin_write_batch(0, 11, { 510 }, batch_tickets) ==
            llama_kv_pager_write_status::ok);
    assert(batch_tickets.size() == 1 && batch_tickets[0].logical_page == 1);
    assert(pager->cancel_write(batch_tickets[0]) == llama_kv_pager_write_status::ok);
    assert(pager->begin_write_batch(0, 11, { 511, 512, 513 }, batch_tickets) ==
            llama_kv_pager_write_status::ok);
    assert(batch_tickets.size() == 3 && batch_tickets[0].logical_page == 1 &&
            batch_tickets[1].logical_page == 2 && batch_tickets[2].logical_page == 2);
    for (auto it = batch_tickets.rbegin(); it != batch_tickets.rend(); ++it) {
        assert(pager->cancel_write(*it) == llama_kv_pager_write_status::ok);
    }
    assert(pager->residency().pages().empty());
    std::cout << "batch_ring_admission_and_rollback=pass packets=1,3 "
        << "cross_page_batch=1 rejected_suffix=atomic\n";

    llama_kv_pager_write_ticket ticket;
    assert(pager->begin_write(0, 11, 3, ticket) == llama_kv_pager_write_status::ok);
    assert(ticket.logical_page == 0 && ticket.physical_row == 3);
    uint32_t physical_row = UINT32_MAX;
    assert(pager->physical_row(0, 3, physical_row) && physical_row == 3);
    assert(pager->complete_write(ticket, 1, true) == llama_kv_pager_write_status::ok);
    assert(pager->residency().pages().size() == 1 && pager->residency().pages()[0].pin_count == 1);
    assert(pager->mutate({ llama_kv_pager_mutation_kind::remove, 0, -1, 3, 4, 0, 11 }) ==
        llama_kv_pager_write_status::all_pinned);
    assert(pager->cancel_write(ticket) == llama_kv_pager_write_status::ok);
    assert(!pager->physical_row(0, 3, physical_row));

    // A failed graph with repeated positions must roll back in ticket order
    // opposite to reservation order, so the earlier ticket removes the row.
    llama_kv_pager_write_ticket first_ticket;
    llama_kv_pager_write_ticket duplicate_ticket;
    assert(pager->begin_write(0, 11, 7, first_ticket) == llama_kv_pager_write_status::ok);
    assert(pager->begin_write(0, 11, 7, duplicate_ticket) == llama_kv_pager_write_status::ok);
    assert(pager->complete_write(duplicate_ticket, 32, false) == llama_kv_pager_write_status::ok);
    assert(pager->complete_write(first_ticket, 32, false) == llama_kv_pager_write_status::ok);
    assert(!pager->physical_row(0, 7, physical_row));

    assert(pager->begin_write(0, 12, 9, ticket) == llama_kv_pager_write_status::ok);
    assert(pager->begin_write(0, 13, 9, duplicate_ticket) == llama_kv_pager_write_status::stale_generation);
    assert(pager->cancel_write(ticket) == llama_kv_pager_write_status::ok);

    assert(pager->begin_write(0, 11, 3, ticket) == llama_kv_pager_write_status::ok);
    assert(pager->complete_write(ticket, 32, true) == llama_kv_pager_write_status::ok);
    assert(pager->begin_write(0, 11, 259, ticket) == llama_kv_pager_write_status::ok);
    assert(pager->complete_write(ticket, 32, true) == llama_kv_pager_write_status::ok);
    // Advancing the write frontier releases the previous partial page. Cancel the
    // temporary frontier so the following metadata mutations have no pinned target.
    assert(pager->begin_write(0, 11, 3, ticket) == llama_kv_pager_write_status::ok);
    assert(pager->complete_write(ticket, 32, true) == llama_kv_pager_write_status::ok);
    assert(pager->cancel_write(ticket) == llama_kv_pager_write_status::ok);
    assert(pager->mutate({ llama_kv_pager_mutation_kind::remove, 0, -1, 0, 256, 0, 11 }) ==
        llama_kv_pager_write_status::ok);
    assert(!pager->physical_row(0, 3, physical_row));
    assert(pager->physical_row(0, 259, physical_row));
    assert(pager->mutate({ llama_kv_pager_mutation_kind::shift, 0, -1, 256, 1024, 256, 11 }) ==
        llama_kv_pager_write_status::ok);
    assert(pager->physical_row(0, 515, physical_row));

    assert(pager->mutate({ llama_kv_pager_mutation_kind::copy, 0, 1, 512, 768, 0, 11 }) ==
        llama_kv_pager_write_status::ok);
    assert(pager->physical_row(1, 515, physical_row));
    assert(pager->mutate({ llama_kv_pager_mutation_kind::keep, 0, -1, 0, 0, 0, 0 }) ==
        llama_kv_pager_write_status::ok);
    assert(!pager->physical_row(1, 515, physical_row));

    // The production owner can build a bounded summary after the post-graph
    // fence even when host backing is unavailable in this local fake.
    auto summary_config = config;
    summary_config.hot_pages.automatic = false;
    summary_config.hot_pages.value = 4;
    auto summary_pager = llama_kv_pager::create(
            summary_config, geometry(1024), resources(2048, 128), write_backend, status);
    assert(summary_pager && status == llama_kv_pager_status::ok);
    routing_provider_calls = 0;
    routing_provider_source_bytes = 0;
    summary_pager->set_routing_summary_provider({ nullptr, build_routing_summary });
    for (llama_pos position = 0; position <= 256; ++position) {
        assert(summary_pager->begin_write(0, 1, position, ticket) == llama_kv_pager_write_status::ok);
        assert(summary_pager->complete_write(ticket, 32, true) == llama_kv_pager_write_status::ok);
    }
    assert(summary_pager->seal_ready_pages() == 1);
    const uint64_t initial_seal_scan_count = summary_pager->seal_pages_scanned();
    assert(initial_seal_scan_count == 1);
    const uint64_t initial_summary_calls = routing_provider_calls;
    assert(initial_summary_calls == 64);
    assert(summary_pager->summary_build_calls() == 64);
    assert(summary_pager->summary_build_bytes() == routing_provider_source_bytes);
    assert(summary_pager->seal_ready_pages() == 0);
    assert(summary_pager->seal_pages_scanned() == initial_seal_scan_count);
    assert(routing_provider_calls == initial_summary_calls);
    assert(summary_pager->routing_summaries().valid());
    // Runtime retrieval keeps one independently tagged summary per layer/KV
    // head, even when a fixture provider supplies the same shape for each.
    assert(summary_pager->routing_summary_index().table_count() == 64);
    assert(summary_pager->routing_summary_accounting().source_rows == 4);
    std::vector<float> summary_query(256, 0.0f);
    summary_query[0] = 1.0f;
    const auto summary_scores = summary_pager->routing_summaries().score(
            summary_pager->residency(), summary_query, 1);
    assert(summary_scores.status == llama_kv_routing_summary_status::ok);

    // Extending the tail advances only that page's content version. The
    // already-clean page is neither sampled nor republished.
    for (llama_pos position = 257; position < 512; ++position) {
        assert(summary_pager->begin_write(0, 1, position, ticket) == llama_kv_pager_write_status::ok);
        assert(summary_pager->complete_write(ticket, 32, true) == llama_kv_pager_write_status::ok);
    }
    assert(summary_pager->seal_ready_pages() == 1);
    assert(summary_pager->seal_pages_scanned() == initial_seal_scan_count + 1);
    assert(routing_provider_calls == initial_summary_calls + 64);
    const uint64_t after_tail_calls = routing_provider_calls;
    assert(summary_pager->begin_write(0, 1, 512, ticket) == llama_kv_pager_write_status::ok);
    assert(summary_pager->complete_write(ticket, 32, true) == llama_kv_pager_write_status::ok);
    assert(summary_pager->seal_ready_pages() == 0);
    assert(routing_provider_calls == after_tail_calls);

    // A speculative overwrite is cancelled, but the next successful overwrite
    // must refresh the summary for that page rather than retaining stale rows.
    assert(summary_pager->begin_write(0, 1, 0, ticket) == llama_kv_pager_write_status::ok);
    assert(summary_pager->complete_write(ticket, 32, false) == llama_kv_pager_write_status::ok);
    const uint64_t after_rollback = routing_provider_calls;
    assert(summary_pager->begin_write(0, 1, 0, ticket) == llama_kv_pager_write_status::ok);
    assert(summary_pager->complete_write(ticket, 32, true) == llama_kv_pager_write_status::ok);
    const auto overwrite_sealed = summary_pager->seal_ready_pages();
    assert(overwrite_sealed == 2);
    assert(routing_provider_calls == after_rollback + 128);

    config.hot_pages.automatic = true;
    config.hot_pages.value = 0;
    auto constrained = llama_kv_pager::create(config, geometry(1536), resources(768, 128), write_backend, status);
    assert(constrained && constrained->snapshot().physical_page_count == 5);
    for (llama_pos position = 0; position < 5 * 256; position += 256) {
        assert(constrained->begin_write(0, 1, position, ticket) == llama_kv_pager_write_status::ok);
        assert(constrained->complete_write(ticket, 32, true) == llama_kv_pager_write_status::ok);
    }
    const auto no_slot = constrained->begin_write(0, 1, 5 * 256, ticket);
    assert(no_slot == llama_kv_pager_write_status::no_victim ||
        no_slot == llama_kv_pager_write_status::all_pinned);

    const auto supported = llama_kv_pager_evaluate_capability(
        config, true, true, true, true, true, true, true, true, true, true, false);
    assert(supported.supported && supported.reasons.size() == 0);
    const auto refused = llama_kv_pager_evaluate_capability(
        config, false, false, false, false, false, false, false, false, false, false, true);
    assert(!refused.supported && refused.reasons.size() == 11);
    for (const auto reason : refused.reasons) {
        assert(llama_kv_pager_capability_reason_name(reason) != nullptr);
    }
    std::cout << "frozen_history_and_mtp_epoch=pass\n";
    return 0;
}
