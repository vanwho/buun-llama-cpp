#pragma once

#include "llama-cache-budget.h"
#include "llama-kv-pager-config.h"
#include "llama-kv-residency.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

struct llama_kv_pager_geometry {
    uint64_t context_tokens = 0;
    uint32_t page_tokens = 256;
    uint32_t attention_layers = 0;
    uint32_t kv_heads = 0;
    uint32_t key_length = 0;
    uint32_t value_length = 0;
    uint64_t page_bytes = 0; // complete opaque K/V page across all attention layers
};

struct llama_kv_pager_resources {
    llama_cache_budget_admission_input admission;
    uint64_t host_budget_bytes = 0;
    uint64_t host_metadata_bytes = 0;
    uint64_t allocator_granularity = 1;
    bool host_budget_known = false;
    bool duplicate_representation_authority = false;
};

struct llama_kv_pager_allocation {
    void * handle = nullptr;
    uint64_t requested_bytes = 0;
    uint64_t realized_bytes = 0;
};

struct llama_kv_pager_backend {
    std::function<bool(uint64_t, llama_kv_pager_allocation &)> allocate;
    std::function<void(llama_kv_pager_allocation &)> release;
};

struct llama_kv_pager_snapshot {
    llama_kv_pager_geometry geometry;
    llama_cache_budget_admission_result admission;
    uint32_t logical_page_count = 0;
    uint32_t physical_page_count = 0;
    uint64_t physical_rows = 0;
    uint64_t physical_bytes = 0;
    uint64_t host_metadata_bytes = 0;
    uint64_t realized_bytes = 0;
    bool initialized = false;
};

enum class llama_kv_pager_status : uint8_t {
    ok = 0,
    disabled,
    invalid_geometry,
    unsupported_authority,
    missing_backend,
    host_budget,
    admission,
    allocation,
    realized_mismatch,
    overflow,
};

const char * llama_kv_pager_status_name(llama_kv_pager_status status) noexcept;

bool llama_kv_pager_plan(
        const llama_kv_pager_config & config,
        const llama_kv_pager_geometry & geometry,
        llama_kv_pager_resources resources,
        llama_kv_pager_snapshot & output,
        llama_kv_pager_status & status) noexcept;

class llama_kv_pager {
public:
    static std::unique_ptr<llama_kv_pager> create(
            const llama_kv_pager_config & config,
            const llama_kv_pager_geometry & geometry,
            llama_kv_pager_resources resources,
            llama_kv_pager_backend backend,
            llama_kv_pager_status & status) noexcept;

    ~llama_kv_pager();
    llama_kv_pager(const llama_kv_pager &) = delete;
    llama_kv_pager & operator=(const llama_kv_pager &) = delete;

    const llama_kv_pager_snapshot & snapshot() const noexcept { return snapshot_; }
    llama_kv_residency_snapshot residency() const noexcept { return residency_.snapshot(); }

private:
    llama_kv_pager() = default;
    llama_kv_pager_snapshot snapshot_;
    llama_kv_pager_backend backend_;
    llama_kv_pager_allocation allocation_;
    std::vector<llama_kv_page_id> logical_pages_;
    llama_kv_residency_table residency_{0};
};
