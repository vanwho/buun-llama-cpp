#include "llama-kv-attention-view.h"

#include <algorithm>
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

bool llama_kv_attention_view_copy_intervals(
        const std::vector<llama_kv_attention_view_page> & pages,
        const std::vector<llama_pos> & query_positions,
        std::vector<llama_kv_attention_view_copy_interval> & intervals,
        uint64_t * query_positions_examined) noexcept {
    intervals.clear();
    if (query_positions_examined != nullptr) {
        *query_positions_examined = 0;
    }

    try {
        std::vector<llama_pos> sorted_queries = query_positions;
        std::sort(sorted_queries.begin(), sorted_queries.end());
        sorted_queries.erase(std::unique(sorted_queries.begin(), sorted_queries.end()),
                sorted_queries.end());

        for (size_t page_index = 0; page_index < pages.size(); ++page_index) {
            const auto & page = pages[page_index];
            if (page.native_position_begin < 0 ||
                    uint64_t(page.native_position_begin) >
                        uint64_t(std::numeric_limits<llama_pos>::max()) - page.row_count) {
                intervals.clear();
                return false;
            }
            if (page.row_count == 0) {
                continue;
            }

            const llama_pos page_begin = page.native_position_begin;
            const llama_pos page_end = page_begin + page.row_count;
            auto query = std::lower_bound(sorted_queries.begin(), sorted_queries.end(), page_begin);
            uint32_t row_begin = 0;
            while (query != sorted_queries.end() && *query < page_end) {
                if (query_positions_examined != nullptr) {
                    ++*query_positions_examined;
                }
                const uint32_t current_row = uint32_t(*query - page_begin);
                if (current_row > row_begin) {
                    intervals.push_back({ uint32_t(page_index), row_begin,
                            current_row - row_begin, false });
                }

                uint32_t current_end = current_row + 1;
                ++query;
                while (query != sorted_queries.end() && *query < page_end &&
                        *query == page_begin + current_end) {
                    if (query_positions_examined != nullptr) {
                        ++*query_positions_examined;
                    }
                    ++current_end;
                    ++query;
                }
                intervals.push_back({ uint32_t(page_index), current_row,
                        current_end - current_row, true });
                row_begin = current_end;
            }
            if (row_begin < page.row_count) {
                intervals.push_back({ uint32_t(page_index), row_begin,
                        page.row_count - row_begin, false });
            }
        }
        return true;
    } catch (...) {
        intervals.clear();
        if (query_positions_examined != nullptr) {
            *query_positions_examined = 0;
        }
        return false;
    }
}

bool llama_kv_attention_query_page_ids(
        const std::vector<llama_pos> & query_positions,
        uint32_t page_tokens,
        std::vector<uint32_t> & page_ids) noexcept {
    page_ids.clear();
    if (page_tokens == 0) return false;
    try {
        for (const llama_pos position : query_positions) {
            if (position < 0) {
                page_ids.clear();
                return false;
            }
            const uint64_t logical = uint64_t(position) / page_tokens;
            if (logical > UINT32_MAX) {
                page_ids.clear();
                return false;
            }
            const uint32_t page = uint32_t(logical);
            if (std::find(page_ids.begin(), page_ids.end(), page) == page_ids.end()) {
                page_ids.push_back(page);
            }
        }
        return !page_ids.empty();
    } catch (...) {
        page_ids.clear();
        return false;
    }
}

bool llama_kv_attention_refresh_page_ids(
        const std::vector<uint32_t> & query_pages,
        const std::vector<uint32_t> & optional_pages,
        uint32_t capacity,
        std::vector<uint32_t> & page_ids) noexcept {
    page_ids.clear();
    try {
        for (const uint32_t page : query_pages) {
            if (std::find(page_ids.begin(), page_ids.end(), page) == page_ids.end()) {
                page_ids.push_back(page);
            }
        }
        if (page_ids.empty() || page_ids.size() > capacity) {
            page_ids.clear();
            return false;
        }
        for (const uint32_t page : optional_pages) {
            if (page_ids.size() >= capacity) break;
            if (std::find(page_ids.begin(), page_ids.end(), page) == page_ids.end()) {
                page_ids.push_back(page);
            }
        }
        return true;
    } catch (...) {
        page_ids.clear();
        return false;
    }
}

bool llama_kv_attention_resident_page_ids(
        const std::vector<uint32_t> & candidate_pages,
        const std::vector<uint32_t> & resident_pages,
        std::vector<uint32_t> & page_ids) noexcept {
    page_ids.clear();
    try {
        for (const uint32_t page : candidate_pages) {
            if (std::find(resident_pages.begin(), resident_pages.end(), page) ==
                    resident_pages.end() ||
                    std::find(page_ids.begin(), page_ids.end(), page) != page_ids.end()) {
                continue;
            }
            page_ids.push_back(page);
        }
        return true;
    } catch (...) {
        page_ids.clear();
        return false;
    }
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
        std::vector<const llama_kv_page_record *> selected_records;
        selected_records.reserve(selected_pages.size());

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
            for (const auto * page : selected_records) {
                if (page->id.logical_page == logical_page &&
                        (sequence_id < 0 || page->id.sequence_id == sequence_id)) {
                    status = llama_kv_attention_view_status::duplicate_page;
                    return {};
                }
            }
            if (found->physical_slot == UINT32_MAX ||
                (found->state != llama_kv_page_state::filling_gpu &&
                 found->state != llama_kv_page_state::sealing_host &&
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
            selected_records.push_back(found);
        }

        std::sort(selected_records.begin(), selected_records.end(),
                [](const auto * lhs, const auto * rhs) {
            if (lhs->id.position_begin != rhs->id.position_begin) {
                return lhs->id.position_begin < rhs->id.position_begin;
            }
            if (lhs->id.logical_page != rhs->id.logical_page) {
                return lhs->id.logical_page < rhs->id.logical_page;
            }
            return lhs->physical_slot < rhs->physical_slot;
        });

        result->pages.reserve(selected_records.size());
        uint64_t rows = 0;
        for (const auto * found : selected_records) {
            const uint64_t count = uint64_t(found->id.position_end - found->id.position_begin);
            if (rows > std::numeric_limits<uint32_t>::max() - count) {
                status = llama_kv_attention_view_status::overflow;
                return {};
            }

            llama_kv_attention_view_page view_page;
            view_page.logical_page = found->id.logical_page;
            view_page.source_physical_slot = found->physical_slot;
            view_page.compact_row_begin = uint32_t(rows);
            view_page.row_count = uint32_t(count);
            view_page.native_position_begin = found->id.position_begin;
            view_page.native_position_end = found->id.position_end;
            view_page.page_generation = found->id.page_generation;
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
