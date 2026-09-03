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
