#include "llama-kv-pager.h"

#include <limits>
#include <new>

namespace {
bool mul(uint64_t a, uint64_t b, uint64_t & out) noexcept {
    if (a && b > std::numeric_limits<uint64_t>::max() / a) return false;
    out = a * b; return true;
}

bool add(uint64_t a, uint64_t b, uint64_t & out) noexcept {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    out = a + b; return true;
}
}

const char * llama_kv_pager_status_name(llama_kv_pager_status status) noexcept {
    switch (status) {
        case llama_kv_pager_status::ok: return "ok";
        case llama_kv_pager_status::disabled: return "disabled";
        case llama_kv_pager_status::invalid_geometry: return "invalid_geometry";
        case llama_kv_pager_status::unsupported_authority: return "unsupported_authority";
        case llama_kv_pager_status::missing_backend: return "missing_backend";
        case llama_kv_pager_status::host_budget: return "host_budget";
        case llama_kv_pager_status::admission: return "admission";
        case llama_kv_pager_status::allocation: return "allocation";
        case llama_kv_pager_status::realized_mismatch: return "realized_mismatch";
        case llama_kv_pager_status::overflow: return "overflow";
    }
    return "invalid";
}

bool llama_kv_pager_plan(const llama_kv_pager_config & config,
        const llama_kv_pager_geometry & geometry, llama_kv_pager_resources resources,
        llama_kv_pager_snapshot & output, llama_kv_pager_status & status) noexcept {
    output = {};
    status = llama_kv_pager_status::invalid_geometry;
    if (!config.enabled()) { status = llama_kv_pager_status::disabled; return true; }
    std::string error;
    if (!config.validate(error) || geometry.context_tokens == 0 || geometry.page_tokens == 0 ||
        geometry.page_tokens != config.page_size || geometry.attention_layers == 0 ||
        geometry.kv_heads == 0 || geometry.key_length == 0 || geometry.value_length == 0 ||
        geometry.page_bytes == 0) return false;
    if (resources.duplicate_representation_authority) { status = llama_kv_pager_status::unsupported_authority; return false; }
    const uint64_t logical = (geometry.context_tokens - 1) / geometry.page_tokens + 1;
    if (logical > std::numeric_limits<uint32_t>::max()) {
        status = llama_kv_pager_status::overflow;
        return false;
    }
    if (!resources.host_budget_known || resources.host_budget_bytes == 0) { status = llama_kv_pager_status::host_budget; return false; }
    if (!mul(logical, sizeof(llama_kv_page_id), resources.host_metadata_bytes) ||
        resources.host_metadata_bytes > resources.host_budget_bytes) { status = llama_kv_pager_status::host_budget; return false; }
    auto & admission = resources.admission;
    admission.page_tokens = geometry.page_tokens;
    admission.logical_page_count = logical;
    admission.target_page_bytes = geometry.page_bytes;
    admission.user_page_cap = config.hot_pages.automatic ? 0 : config.hot_pages.value;
    admission.allocation_granularity = resources.allocator_granularity;
    const auto result = llama_cache_budget_admit(admission);
    if (result.refusal != llama_cache_budget_admission_refusal::none || result.admitted_pages == 0) {
        status = llama_kv_pager_status::admission; output.admission = result; return false;
    }
    uint64_t rows = 0, bytes = 0;
    if (!mul(result.admitted_pages, geometry.page_tokens, rows) ||
        !mul(result.admitted_pages, result.page_charge_bytes, bytes)) { status = llama_kv_pager_status::overflow; return false; }
    output.geometry = geometry;
    output.admission = result;
    output.logical_page_count = uint32_t(logical);
    output.physical_page_count = uint32_t(result.admitted_pages);
    output.physical_rows = rows;
    output.physical_bytes = bytes;
    output.host_metadata_bytes = resources.host_metadata_bytes;
    status = llama_kv_pager_status::ok;
    return true;
}

std::unique_ptr<llama_kv_pager> llama_kv_pager::create(
        const llama_kv_pager_config & config, const llama_kv_pager_geometry & geometry,
        llama_kv_pager_resources resources, llama_kv_pager_backend backend,
        llama_kv_pager_status & status) noexcept {
    status = llama_kv_pager_status::invalid_geometry;
    if (!config.enabled()) { status = llama_kv_pager_status::disabled; return nullptr; }
    if (!backend.allocate || !backend.release) { status = llama_kv_pager_status::missing_backend; return nullptr; }
    auto output = std::unique_ptr<llama_kv_pager>(new (std::nothrow) llama_kv_pager);
    if (!output) { status = llama_kv_pager_status::allocation; return nullptr; }
    output->backend_ = std::move(backend);
    try {
        if (!llama_kv_pager_plan(config, geometry, resources, output->snapshot_, status)) return nullptr;
        if (!output->backend_.allocate(output->snapshot_.physical_bytes, output->allocation_) ||
            output->allocation_.handle == nullptr || output->allocation_.realized_bytes != output->snapshot_.physical_bytes) {
            const bool allocated = output->allocation_.handle != nullptr;
            const bool mismatch = allocated && output->allocation_.realized_bytes != output->snapshot_.physical_bytes;
            if (allocated) output->backend_.release(output->allocation_);
            status = mismatch ? llama_kv_pager_status::realized_mismatch : llama_kv_pager_status::allocation;
            return nullptr;
        }
        output->snapshot_.realized_bytes = output->allocation_.realized_bytes;
        output->snapshot_.initialized = true;
        output->logical_pages_.resize(output->snapshot_.logical_page_count);
        output->residency_ = llama_kv_residency_table(output->snapshot_.physical_page_count);
        return output;
    } catch (...) {
        status = llama_kv_pager_status::allocation;
        return nullptr;
    }
}

llama_kv_pager::~llama_kv_pager() {
    if (allocation_.handle && backend_.release) backend_.release(allocation_);
}
