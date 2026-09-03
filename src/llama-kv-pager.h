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

enum class llama_kv_pager_write_status : uint8_t {
    ok = 0,
    disabled,
    invalid_position,
    no_victim,
    all_pinned,
    stale_generation,
    transaction,
    overflow,
};

const char * llama_kv_pager_write_status_name(llama_kv_pager_write_status status) noexcept;

struct llama_kv_pager_write_ticket {
    int32_t sequence_id = -1;
    uint64_t sequence_generation = 0;
    uint32_t logical_page = UINT32_MAX;
    uint32_t physical_slot = UINT32_MAX;
    uint32_t physical_row = UINT32_MAX;
    uint32_t page_generation = 0;
    llama_pos position = -1;
    bool page_created = false;
    bool row_was_valid = false;
};

enum class llama_kv_pager_mutation_kind : uint8_t {
    remove = 0,
    copy,
    keep,
    shift,
    rewind,
    clear,
};

struct llama_kv_pager_mutation {
    llama_kv_pager_mutation_kind kind = llama_kv_pager_mutation_kind::remove;
    int32_t sequence_id = -1;
    int32_t destination_sequence_id = -1;
    llama_pos position_begin = 0;
    llama_pos position_end = 0;
    llama_pos shift = 0;
    uint64_t sequence_generation = 0;
};

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

    // Reserve the physical row for one logical position before graph submission. The returned
    // row is an implementation detail; callers must continue to use the logical position for
    // RoPE and masking. A page is kept pinned while it is the current partial write page.
    llama_kv_pager_write_status begin_write(
            int32_t sequence_id, uint64_t sequence_generation, llama_pos position,
            llama_kv_pager_write_ticket & ticket) noexcept;
    llama_kv_pager_write_status complete_write(
            const llama_kv_pager_write_ticket & ticket, uint32_t completed_segments,
            bool graph_succeeded) noexcept;
    llama_kv_pager_write_status cancel_write(
            const llama_kv_pager_write_ticket & ticket) noexcept;
    bool physical_row(
            int32_t sequence_id, llama_pos position, uint32_t & row) const noexcept;

    // Apply a metadata mutation as one table publication. Payload movement is deliberately
    // deferred to the residency transfer owner; this method never publishes a half mutation.
    llama_kv_pager_write_status mutate(const llama_kv_pager_mutation & mutation) noexcept;

private:
    struct page_state {
        llama_kv_page_record record;
        std::vector<uint8_t> valid_rows;
        uint32_t completed_segments = 0;
        bool present = false;
    };

    llama_kv_pager_write_status publish_page(page_state & page) noexcept;
    llama_kv_pager_write_status erase_page(page_state & page) noexcept;
    page_state * find_page(int32_t sequence_id, uint32_t logical_page) noexcept;
    const page_state * find_page(int32_t sequence_id, uint32_t logical_page) const noexcept;
    page_state * find_slot(uint32_t slot) noexcept;
    void release_current_pin(page_state * except) noexcept;

    llama_kv_pager() = default;
    llama_kv_pager_snapshot snapshot_;
    llama_kv_pager_backend backend_;
    llama_kv_pager_allocation allocation_;
    std::vector<page_state> pages_;
    std::vector<int32_t> slot_pages_;
    uint32_t current_page_index_ = UINT32_MAX;
    uint64_t mutation_generation_ = 1;
    llama_kv_residency_table residency_{0};
};
