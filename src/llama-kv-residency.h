#pragma once

#include "llama-vbr-generation-types.h"

#include <cstdint>
#include <memory>
#include <vector>

enum class llama_kv_page_state : uint8_t {
    absent,
    filling_gpu,
    sealing_host,
    gpu_host_clean,
    host_clean,
    loading_gpu,
    gpu_dirty,
    resealing_host,
    invalid,
};

enum class llama_kv_residency_status : uint8_t {
    ok,
    invalid_argument,
    duplicate_logical_page,
    duplicate_physical_slot,
    out_of_range,
    invalid_position_range,
    stale_epoch,
    pinned_slot,
    not_found,
    transaction_closed,
};

struct llama_kv_page_id {
    uint64_t session_generation = 0;
    int32_t sequence_id = -1;
    uint64_t sequence_generation = 0;
    uint32_t logical_page = 0;
    uint32_t page_generation = 0;
    uint64_t representation_epoch = 0;
    uint64_t model_identity = 0;
    uint64_t topology_identity = 0;
    uint64_t codec_digest = 0;
    uint64_t codebook_digest = 0;
    uint64_t rotation_digest = 0;
    uint64_t meansub_digest = 0;
    llama_pos position_begin = -1;
    llama_pos position_end = -1;
};

bool operator==(const llama_kv_page_id & lhs, const llama_kv_page_id & rhs) noexcept;
bool operator!=(const llama_kv_page_id & lhs, const llama_kv_page_id & rhs) noexcept;

// `tail` is true only for the current partially-filled write page. Sealed pages
// must contain exactly VBR_GENERATION_PAGE_CELLS positions.
bool llama_kv_page_id_valid(const llama_kv_page_id & id, bool tail) noexcept;
uint32_t llama_kv_page_count(uint32_t logical_cells) noexcept;

struct llama_kv_page_record {
    llama_kv_page_id id;
    uint32_t physical_slot = UINT32_MAX;
    llama_kv_page_state state = llama_kv_page_state::absent;
    bool host_valid = false;
    bool dirty = false;
    uint32_t pin_count = 0;
};

class llama_kv_residency_snapshot {
public:
    llama_kv_residency_snapshot() = default;

    uint64_t epoch() const noexcept;
    uint32_t slot_capacity() const noexcept;
    const std::vector<llama_kv_page_record> & pages() const noexcept;

    // Return an immutable view containing only one sequence's pages.  The
    // table may contain the same logical page number for several sequences;
    // selected attention must never resolve such a page by logical number
    // alone.
    llama_kv_residency_snapshot for_sequence(int32_t sequence_id) const noexcept;

private:
    struct state;
    explicit llama_kv_residency_snapshot(std::shared_ptr<const state> value) noexcept;
    std::shared_ptr<const state> state_;

    friend class llama_kv_residency_table;
};

class llama_kv_residency_transaction {
public:
    llama_kv_residency_transaction() = default;

    uint64_t base_epoch() const noexcept;
    bool active() const noexcept;
    const std::vector<llama_kv_page_record> & pages() const noexcept;

private:
    uint64_t base_epoch_ = 0;
    uint32_t slot_capacity_ = 0;
    std::vector<llama_kv_page_record> pages_;
    bool active_ = false;

    friend class llama_kv_residency_table;
};

class llama_kv_residency_table {
public:
    explicit llama_kv_residency_table(uint32_t slot_capacity);

    llama_kv_residency_snapshot snapshot() const noexcept;
    llama_kv_residency_transaction begin() const noexcept;

    llama_kv_residency_status replace(
            llama_kv_residency_transaction & tx,
            const llama_kv_page_record & page) const noexcept;
    // Replace an existing logical page record without changing its physical slot. This is used
    // while a write page remains pinned and its valid-range/state advances.
    llama_kv_residency_status update(
            llama_kv_residency_transaction & tx,
            const llama_kv_page_record & page) const noexcept;
    llama_kv_residency_status erase(
        llama_kv_residency_transaction & tx,
        const llama_kv_page_id & id) const noexcept;
    llama_kv_residency_status publish(llama_kv_residency_transaction & tx) noexcept;
    void rollback(llama_kv_residency_transaction & tx) const noexcept;

private:
    std::shared_ptr<const llama_kv_residency_snapshot::state> state_;
};
