#include "llama-kv-attention-view.h"

#include <limits>
#include <new>

struct llama_kv_attention_view::state {
    uint64_t epoch = 0;
    llama_kv_residency_snapshot snapshot;
    std::vector<llama_kv_attention_view_page> pages;
    std::vector<llama_pos> native_positions;
    std::vector<uint8_t> native_mask;
};

struct llama_kv_attention_view::graph_fence::state {
    std::shared_ptr<const llama_kv_attention_view::state> view;
};

const char * llama_kv_attention_view_status_name(
        llama_kv_attention_view_status status) noexcept {
    switch (status) {
        case llama_kv_attention_view_status::ok: return "ok";
        case llama_kv_attention_view_status::invalid_argument: return "invalid_argument";
        case llama_kv_attention_view_status::duplicate_page: return "duplicate_page";
        case llama_kv_attention_view_status::not_resident: return "not_resident";
        case llama_kv_attention_view_status::invalid_position_range: return "invalid_position_range";
        case llama_kv_attention_view_status::overflow: return "overflow";
    }
    return "invalid";
}

llama_kv_attention_view::llama_kv_attention_view(
        std::shared_ptr<const state> state) noexcept : state_(std::move(state)) {}

llama_kv_attention_view::graph_fence::graph_fence(
        std::shared_ptr<const state> state) noexcept : state_(std::move(state)) {}

llama_kv_attention_view llama_kv_attention_view::build(
        const llama_kv_residency_snapshot & snapshot,
        const std::vector<uint32_t> & selected_pages,
        llama_kv_attention_view_status & status) noexcept {
    return build(snapshot, selected_pages, -1, status);
}

llama_kv_attention_view llama_kv_attention_view::build(
        const llama_kv_residency_snapshot & snapshot,
        const std::vector<uint32_t> & selected_pages,
        int32_t sequence_id,
        llama_kv_attention_view_status & status) noexcept {
    status = llama_kv_attention_view_status::invalid_argument;
    if (snapshot.epoch() == 0 || selected_pages.empty()) {
        return {};
    }

    try {
        auto result = std::make_shared<llama_kv_attention_view::state>();
        result->epoch = snapshot.epoch();
        result->snapshot = snapshot;
        result->pages.reserve(selected_pages.size());

        uint64_t rows = 0;
        for (const uint32_t logical_page : selected_pages) {
            const llama_kv_page_record * found = nullptr;
            for (const auto & page : snapshot.pages()) {
                if (page.id.logical_page == logical_page &&
                    (sequence_id < 0 || page.id.sequence_id == sequence_id)) {
                    found = &page;
                    break;
                }
            }
            if (found == nullptr) {
                status = llama_kv_attention_view_status::not_resident;
                return {};
            }
            for (const auto & page : result->pages) {
                if (page.logical_page == logical_page) {
                    status = llama_kv_attention_view_status::duplicate_page;
                    return {};
                }
            }
            if (found->physical_slot == UINT32_MAX ||
                (found->state != llama_kv_page_state::filling_gpu &&
                 found->state != llama_kv_page_state::gpu_host_clean &&
                 found->state != llama_kv_page_state::gpu_dirty)) {
                status = llama_kv_attention_view_status::not_resident;
                return {};
            }

            const llama_pos begin = found->id.position_begin;
            const llama_pos end = found->id.position_end;
            const uint64_t count = end >= begin ? uint64_t(end - begin) : 0;
            if (begin < 0 || end <= begin || begin % VBR_GENERATION_PAGE_CELLS != 0 ||
                logical_page != uint32_t(begin / VBR_GENERATION_PAGE_CELLS) ||
                count > VBR_GENERATION_PAGE_CELLS) {
                status = llama_kv_attention_view_status::invalid_position_range;
                return {};
            }
            if (rows > std::numeric_limits<uint32_t>::max() - count) {
                status = llama_kv_attention_view_status::overflow;
                return {};
            }

            llama_kv_attention_view_page view_page;
            view_page.logical_page = logical_page;
            view_page.source_physical_slot = found->physical_slot;
            view_page.compact_row_begin = uint32_t(rows);
            view_page.row_count = uint32_t(count);
            view_page.native_position_begin = begin;
            view_page.native_position_end = end;
            result->pages.push_back(view_page);
            rows += count;
        }

        if (rows == 0 || rows > std::numeric_limits<size_t>::max()) {
            status = llama_kv_attention_view_status::overflow;
            return {};
        }
        result->native_positions.reserve(size_t(rows));
        result->native_mask.reserve(size_t(rows));
        for (const auto & page : result->pages) {
            for (uint32_t row = 0; row < page.row_count; ++row) {
                result->native_positions.push_back(page.native_position_begin + row);
                result->native_mask.push_back(1);
            }
        }
        status = llama_kv_attention_view_status::ok;
        return llama_kv_attention_view(std::move(result));
    } catch (const std::bad_alloc &) {
        status = llama_kv_attention_view_status::overflow;
        return {};
    }
}

uint64_t llama_kv_attention_view::graph_epoch() const noexcept {
    return state_ ? state_->epoch : 0;
}

uint32_t llama_kv_attention_view::get_n_kv() const noexcept {
    return state_ ? uint32_t(state_->native_positions.size()) : 0;
}

const std::vector<llama_kv_attention_view_page> & llama_kv_attention_view::pages() const noexcept {
    static const std::vector<llama_kv_attention_view_page> empty;
    return state_ ? state_->pages : empty;
}

const std::vector<llama_pos> & llama_kv_attention_view::native_positions() const noexcept {
    static const std::vector<llama_pos> empty;
    return state_ ? state_->native_positions : empty;
}

const std::vector<uint8_t> & llama_kv_attention_view::native_mask() const noexcept {
    static const std::vector<uint8_t> empty;
    return state_ ? state_->native_mask : empty;
}

llama_kv_attention_view::graph_fence llama_kv_attention_view::acquire_graph_fence() const noexcept {
    if (!state_) return {};
    auto fence = std::make_shared<graph_fence::state>();
    fence->view = state_;
    return graph_fence(std::move(fence));
}
