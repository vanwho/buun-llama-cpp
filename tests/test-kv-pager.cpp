#include "llama-kv-pager.h"

#include <array>
#include <cstdint>
#include <cstring>
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

static bool build_routing_summary(
        void *, const llama_kv_page_record & page,
        const llama_kv_routing_summary_config & config,
        llama_kv_routing_page_input & output) noexcept {
    output = {};
    output.id = page.id;
    const uint32_t rows = uint32_t(page.id.position_end - page.id.position_begin);
    output.row_indices = { 0, rows / 3, (2 * rows) / 3, rows - 1 };
    output.rotated_k_rows.assign(output.row_indices.size() * config.vector_dim, 0.0f);
    for (size_t i = 0; i < output.row_indices.size(); ++i) {
        output.rotated_k_rows[i * config.vector_dim] = float(page.id.logical_page + 1);
    }
    output.source_bytes = output.rotated_k_rows.size() * sizeof(float);
    return config.representative_count == output.row_indices.size();
}

struct host_page_fixture {
    static constexpr uint64_t source_namespace = 0x9911;
    static constexpr uint32_t row_count = 512;
    static constexpr uint64_t row_bytes = 2;

    std::vector<std::vector<uint8_t>> storage;
    std::vector<vbr_selected_page_unit_source> sources;
    vbr_selected_page_capture_snapshot snapshot;

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
            void *, const vbr_selected_page_capture_snapshot &) noexcept {
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
        range.positions.resize(VBR_GENERATION_PAGE_CELLS);
        range.physical_cells.resize(VBR_GENERATION_PAGE_CELLS);
        for (uint32_t i = 0; i < VBR_GENERATION_PAGE_CELLS; ++i) {
            range.positions[i] = page.id.position_begin + llama_pos(i);
            range.physical_cells[i] =
                    page.physical_slot * VBR_GENERATION_PAGE_CELLS + i;
        }
        request.pages.push_back(std::move(range));
        output_sources = self.sources;
        snapshots = { &self, acquire, recheck, release };
        return true;
    }

    void initialize() {
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
            storage[unit].resize(row_count * row_bytes);
            for (size_t i = 0; i < storage[unit].size(); ++i) {
                storage[unit][i] = uint8_t(unit + i);
            }
            vbr_selected_page_unit_source source;
            source.logical_unit_id = unit;
            source.row_count = row_count;
            source.row_bytes = row_bytes;
            source.source_identity = 0x1000 + unit;
            source.source.size = storage[unit].size();
            source.source.context = &storage[unit];
            source.source.read = read;
            sources.push_back(source);

            vbr_capture_projected_shard_source projected;
            projected.shard_index = 0;
            projected.row_count = row_count;
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
            descriptor.wm_cells = row_count;
            descriptor.codebook_digest.fill(2);
            descriptor.rotation_digest.fill(3);
            descriptor.meansub_digest.fill(4);
            descriptor.row_codec_version = 1;
            vbr_artifact_shard_descriptor shard;
            shard.row_count = row_count;
            shard.column_count = 1;
            shard.row_bytes = row_bytes;
            shard.payload_bytes = storage[unit].size();
            descriptor.shards.push_back(shard);
            snapshot.unit_descriptors.push_back(std::move(descriptor));
        }
    }
};

static void test_host_seal_boundary() {
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
    fixture.snapshot.pages[0] = pager->residency().pages()[0].id;
    assert(pager->seal_ready_pages() == 1);
    assert(pager->residency().pages()[0].pin_count == 0);
    assert(pager->host_catalog()->snapshot().live_pages == 1);
    assert(pager->exact_page_records(0).size() == 1);

    const uint64_t epoch = pager->residency().epoch();
    assert(pager->mutate({
            llama_kv_pager_mutation_kind::remove, 0, -1, 0, 256, 0, 1,
            epoch - 1 }) == llama_kv_pager_write_status::stale_generation);
    assert(pager->residency().epoch() == epoch);

    // A completion captured before a partial-tail edit must not publish after
    // that edit, even when the physical page is reused in place.
    llama_kv_pager_write_ticket stale_ticket;
    assert(pager->begin_write(0, 1, 0, stale_ticket) == llama_kv_pager_write_status::ok);
    pager->release_sequence_pins(0); // post-fence cancellation boundary
    const uint32_t old_page_generation = stale_ticket.page_generation;
    assert(pager->mutate({
            llama_kv_pager_mutation_kind::remove, 0, -1, 0, 1, 0, 1,
            pager->residency().epoch() }) == llama_kv_pager_write_status::ok);
    assert(pager->residency().pages()[0].id.page_generation != old_page_generation);
    assert(pager->complete_write(stale_ticket, 32, true) ==
        llama_kv_pager_write_status::stale_generation);
    assert(pager->mutate({
            llama_kv_pager_mutation_kind::remove, 0, -1, 0, 256, 0, 1,
            pager->residency().epoch() }) == llama_kv_pager_write_status::ok);
    assert(pager->host_catalog()->snapshot().live_pages == 0);
    assert(pager->exact_page_records(0).empty());
}

int main() {
    test_host_seal_boundary();
    test_mode_lifecycle_matrix();
    test_pager_host_mutation();
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
    assert(llama_kv_pager_plan(config, geometry(1025), plan_resources, snapshot, status));
    assert(snapshot.physical_page_count == 2 && snapshot.physical_rows == 2 * 256);

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
    auto summary_pager = llama_kv_pager::create(
            config, geometry(512), resources(1024, 128), write_backend, status);
    assert(summary_pager && status == llama_kv_pager_status::ok);
    summary_pager->set_routing_summary_provider({ nullptr, build_routing_summary });
    for (llama_pos position = 0; position <= 256; ++position) {
        assert(summary_pager->begin_write(0, 1, position, ticket) == llama_kv_pager_write_status::ok);
        assert(summary_pager->complete_write(ticket, 32, true) == llama_kv_pager_write_status::ok);
    }
    assert(summary_pager->seal_ready_pages() == 1);
    assert(summary_pager->routing_summaries().valid());
    assert(summary_pager->routing_summary_accounting().source_rows == 4);
    std::vector<float> summary_query(256, 0.0f);
    summary_query[0] = 1.0f;
    const auto summary_scores = summary_pager->routing_summaries().score(
            summary_pager->residency(), summary_query, 1);
    assert(summary_scores.status == llama_kv_routing_summary_status::ok);

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
    return 0;
}
