#pragma once

#include "llama-kv-residency.h"

#include <cstdint>
#include <memory>
#include <vector>

// Backend-neutral description of one compact physical page in the selected
// attention view. compact_row_begin is independent of logical_page and
// source_physical_slot; native positions must be used for RoPE and masking.
struct llama_kv_attention_view_page {
    uint32_t logical_page = UINT32_MAX;
    uint32_t source_physical_slot = UINT32_MAX;
    uint32_t compact_row_begin = 0;
    uint32_t row_count = 0;
    llama_pos native_position_begin = -1;
    llama_pos native_position_end = -1;
    uint32_t page_generation = 0;
};

// One contiguous source/destination copy span in a packed selected view.
// Current-query spans are written by the graph; history spans copy the
// existing native Turbo4 rows into the compact owner.
struct llama_kv_attention_view_copy_interval {
    uint32_t page_index = 0;
    uint32_t row_begin = 0;
    uint32_t row_count = 0;
    bool current_rows = false;
};

// Build the same per-page current/history spans as row-by-row membership
// checks, while sorting/deduplicating arbitrary query positions once and
// visiting only queries that fall inside selected pages. The optional count
// reports visited unique query positions for deterministic tests/diagnostics.
bool llama_kv_attention_view_copy_intervals(
        const std::vector<llama_kv_attention_view_page> & pages,
        const std::vector<llama_pos> & query_positions,
        std::vector<llama_kv_attention_view_copy_interval> & intervals,
        uint64_t * query_positions_examined = nullptr) noexcept;

enum class llama_kv_attention_view_status : uint8_t {
    ok = 0,
    invalid_argument,
    duplicate_page,
    not_resident,
    invalid_position_range,
    overflow,
};

const char * llama_kv_attention_view_status_name(
        llama_kv_attention_view_status status) noexcept;

// Return the unique logical pages touched by the current query rows, in query
// order. Selected attention must include every such page before historical
// pages consume the remaining bounded view.
bool llama_kv_attention_query_page_ids(
        const std::vector<llama_pos> & query_positions,
        uint32_t page_tokens,
        std::vector<uint32_t> & page_ids) noexcept;

// Rebuild a bounded selected set after an advisory route becomes stale.
// Mandatory query pages retain priority; optional pages fill remaining slots.
bool llama_kv_attention_refresh_page_ids(
        const std::vector<uint32_t> & query_pages,
        const std::vector<uint32_t> & optional_pages,
        uint32_t capacity,
        std::vector<uint32_t> & page_ids) noexcept;

// Keep only unique optional logical pages represented by the current resident
// snapshot. Routing refresh must not select host-only inventory entries.
bool llama_kv_attention_resident_page_ids(
        const std::vector<uint32_t> & candidate_pages,
        const std::vector<uint32_t> & resident_pages,
        std::vector<uint32_t> & page_ids) noexcept;

class llama_kv_attention_view {
public:
    llama_kv_attention_view() = default;

    // Build an immutable compact view from one residency snapshot. The
    // selected pages are canonicalized into native position order, while the
    // selected set itself is preserved, including arbitrary gaps. No KV
    // payload is duplicated here: the rows are a bounded copy plan from
    // source physical slots into compact slots.
    static llama_kv_attention_view build(
            const llama_kv_residency_snapshot & snapshot,
            const std::vector<uint32_t> & selected_pages,
            llama_kv_attention_view_status & status) noexcept;

    static llama_kv_attention_view build(
            const llama_kv_residency_snapshot & snapshot,
            const std::vector<uint32_t> & selected_pages,
            int32_t sequence_id,
            llama_kv_attention_view_status & status) noexcept;

    bool valid() const noexcept { return state_ != nullptr; }
    uint64_t graph_epoch() const noexcept;
    // Graph builders must use this value for the K/V attention dimension.
    uint32_t get_n_kv() const noexcept;
    uint32_t row_count() const noexcept { return get_n_kv(); }

    const std::vector<llama_kv_attention_view_page> & pages() const noexcept;
    const std::vector<llama_pos> & native_positions() const noexcept;
    // One for a valid row and zero for padding in the selected tail page.
    const std::vector<uint8_t> & native_mask() const noexcept;

    // A graph keeps this fence alive until execution has completed. It holds
    // the immutable view state (and therefore the residency snapshot) without
    // exposing backend-specific events to the reference path.
    class graph_fence {
    public:
        graph_fence() = default;
        bool active() const noexcept { return state_ != nullptr; }
        void release() noexcept { state_.reset(); }

    private:
        struct state;
        explicit graph_fence(std::shared_ptr<const state> state) noexcept;
        std::shared_ptr<const state> state_;
        friend class llama_kv_attention_view;
    };

    graph_fence acquire_graph_fence() const noexcept;

private:
    struct state;
    explicit llama_kv_attention_view(std::shared_ptr<const state> state) noexcept;
    std::shared_ptr<const state> state_;
};

// CPU metadata only. Completeness is a coverage proof, never a resident count.
bool llama_kv_attention_full_prefix(
        const llama_kv_residency_snapshot & snapshot, int32_t sequence_id,
        int64_t end, uint32_t page_capacity) noexcept;

// A bounded certificate of the historical rows actually submitted for a user
// span. Current-message rows use the same causal R+Q rule before/after commit.
// The commit caller must fence successful graph submissions before trusting it.
class llama_kv_attention_replay_certificate {
public:
    void reset(int32_t sequence_id, int64_t begin, int64_t end,
               uint32_t page_tokens = VBR_GENERATION_PAGE_CELLS,
               uint32_t attention_pages = UINT32_MAX) noexcept;
    void invalidate() noexcept { valid_ = false; }
    bool observed() const noexcept { return observed_; }
    void observe(const llama_kv_residency_snapshot & snapshot,
                 const std::vector<uint32_t> & selected,
                 int64_t first, int64_t last) noexcept;
    bool matches(const llama_kv_residency_snapshot & snapshot,
                 const std::vector<uint32_t> & final_selected) const noexcept;
private:
    struct entry {
        llama_kv_page_id id;
        uint64_t version = 0;
        bool append_seam = false;
    };
    bool history(const llama_kv_residency_snapshot & snapshot,
                 const std::vector<uint32_t> & selected,
                 std::vector<entry> & out) const;
    static bool equal(const std::vector<entry> & a, const std::vector<entry> & b,
                      int64_t begin, bool seam_rolled_out = false) noexcept;
    int64_t query_floor(int64_t end, uint32_t capacity, const std::vector<entry> & history) const noexcept;
    int32_t sequence_ = -1;
    int64_t begin_ = -1, end_ = -1, covered_ = -1;
    bool valid_ = false, observed_ = false;
    uint32_t page_tokens_ = 0, attention_pages_ = 0;
    std::vector<entry> history_;
};
