#include "llama-kv-pager.h"

#include <cstdint>

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

int main() {
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
