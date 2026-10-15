#pragma once

#include "llama-vbr-generation-types.h"

#include <cstdint>
#include <memory>
#include <vector>

struct llama_kv_fixed_window_geometry {
    uint32_t page_tokens = VBR_GENERATION_PAGE_CELLS;
    uint32_t logical_pages = 0;
    uint32_t target_layers = 16;
    uint32_t kv_heads = 4;
    uint32_t key_length = 256;
    uint32_t value_length = 256;
    uint32_t bits_per_value_numerator = 33;
    uint32_t bits_per_value_denominator = 8;
};

struct llama_kv_fixed_window_derived_geometry {
    uint64_t values_per_token = 0;
    uint64_t bytes_per_token = 0;
    uint64_t row_bytes = 0;
    uint64_t page_bytes = 0;
    uint64_t full_context_bytes = 0;
};

bool llama_kv_fixed_window_derive_geometry(
    const llama_kv_fixed_window_geometry & input,
    llama_kv_fixed_window_derived_geometry & output) noexcept;

struct llama_kv_fixed_window_selection {
    std::vector<uint32_t> logical_pages;
};

// Deterministic debug selector: choose the newest pages first, then add
// explicit page ids in caller order. No attention score or dynamic policy is
// consulted, and the output never exceeds resident_capacity.
bool llama_kv_fixed_window_select(
    uint32_t logical_page_count,
    uint32_t resident_capacity,
    uint32_t newest_page_count,
    const std::vector<uint32_t> & explicit_pages,
    llama_kv_fixed_window_selection & output) noexcept;

enum class llama_kv_fixed_window_status : uint8_t {
    ok = 0,
    invalid_argument,
    geometry_mismatch,
    host_budget,
    slot_unavailable,
    pinned,
    host_missing,
    checksum_mismatch,
    dirty,
    not_found,
    short_page,
    overflow,
    _count,
};

const char * llama_kv_fixed_window_status_name(
    llama_kv_fixed_window_status status) noexcept;

struct llama_kv_fixed_window_config {
    llama_kv_fixed_window_geometry geometry;
    uint32_t resident_slot_capacity = 0;
    uint64_t host_budget_bytes = 0;
    uint64_t staging_bytes = 0;
};

struct llama_kv_fixed_window_page {
    uint32_t logical_page = UINT32_MAX;
    uint32_t physical_slot = UINT32_MAX;
    uint32_t valid_tokens = 0;
    uint64_t host_payload_bytes = 0;
    uint64_t checksum = 0;
    bool host_valid = false;
    bool dirty = false;
    bool pinned = false;
};

struct llama_kv_fixed_window_ledger {
    uint64_t resident_pages = 0;
    uint64_t host_valid_pages = 0;
    uint64_t pinned_pages = 0;
    uint64_t resident_bytes = 0;
    uint64_t host_payload_bytes = 0;
    uint64_t staging_bytes = 0;
    uint64_t d2h_seal_useful_bytes = 0;
    uint64_t d2h_seal_aligned_bytes = 0;
    uint64_t d2h_eviction_useful_bytes = 0;
    uint64_t d2h_eviction_aligned_bytes = 0;
    uint64_t h2d_useful_bytes = 0;
    uint64_t h2d_aligned_bytes = 0;
    uint64_t checksum_failures = 0;
    uint64_t stale_rejects = 0;
    uint64_t evictions = 0;
};

class llama_kv_fixed_window {
public:
    static std::unique_ptr<llama_kv_fixed_window> create(
        const llama_kv_fixed_window_config & config,
        llama_kv_fixed_window_derived_geometry & derived,
        llama_kv_fixed_window_status & status) noexcept;
    ~llama_kv_fixed_window();

    llama_kv_fixed_window(const llama_kv_fixed_window &) = delete;
    llama_kv_fixed_window & operator=(const llama_kv_fixed_window &) = delete;

    const llama_kv_fixed_window_derived_geometry & geometry() const noexcept;
    const llama_kv_fixed_window_ledger & ledger() const noexcept;
    const llama_kv_fixed_window_page * find(uint32_t logical_page) const noexcept;

    // The current partial page is GPU-authoritative and remains pinned until
    // the caller seals or explicitly abandons it.
    llama_kv_fixed_window_status begin_fill(
        uint32_t logical_page, uint32_t physical_slot,
        uint32_t valid_tokens) noexcept;
    llama_kv_fixed_window_status seal(
        uint32_t logical_page, uint32_t valid_tokens,
        uint64_t checksum, uint64_t aligned_bytes) noexcept;
    llama_kv_fixed_window_status seal_host_only(
        uint32_t logical_page, uint32_t valid_tokens,
        uint64_t checksum, uint64_t aligned_bytes) noexcept;
    llama_kv_fixed_window_status mutate(uint32_t logical_page) noexcept;
    llama_kv_fixed_window_status pin(uint32_t logical_page) noexcept;
    llama_kv_fixed_window_status unpin(uint32_t logical_page) noexcept;

    // Promotion is legal only from a clean canonical host page. A checksum
    // mismatch refuses publication before the slot becomes resident.
    llama_kv_fixed_window_status promote(
        uint32_t logical_page, uint32_t physical_slot,
        uint64_t checksum, uint64_t aligned_bytes) noexcept;

    // Clean eviction only drops the mapping. It intentionally does not add
    // any D2H bytes or invoke a transfer path.
    llama_kv_fixed_window_status evict_clean(
        uint32_t logical_page) noexcept;

private:
    struct impl;
    llama_kv_fixed_window_status seal_impl(
        uint32_t logical_page, uint32_t valid_tokens,
        uint64_t checksum, uint64_t aligned_bytes,
        bool require_resident) noexcept;
    explicit llama_kv_fixed_window(std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;
};
