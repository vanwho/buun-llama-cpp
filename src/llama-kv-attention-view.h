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
};

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

class llama_kv_attention_view {
public:
    llama_kv_attention_view() = default;

    // Build an immutable compact view from one residency snapshot. The
    // selected order is preserved, including arbitrary permutation and gaps.
    // No KV payload is duplicated here: the rows are a bounded copy plan from
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
