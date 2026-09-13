#include "llama-kv-attention-op.h"

#include <limits>
#include <new>
#include <utility>

struct llama_kv_attention_operator_metadata::state {
    llama_kv_attention_operator_params params;
    llama_kv_attention_view view;
    uint64_t content_key = 0;
    uint64_t layout_key = 0;
    uint64_t physical_key = 0;
};

static void attention_key_mix(uint64_t & key, uint64_t value) noexcept {
    key ^= value;
    key *= 1099511628211ull;
}

static uint64_t attention_content_key(
        const llama_kv_attention_view & view,
        const llama_kv_attention_operator_params & params) noexcept {
    uint64_t key = 1469598103934665603ull;
    attention_key_mix(key, uint64_t(params.type_k));
    attention_key_mix(key, uint64_t(params.type_v));
    attention_key_mix(key, uint64_t(params.domain_k));
    attention_key_mix(key, uint64_t(params.domain_v));
    attention_key_mix(key, params.head_dim_k);
    attention_key_mix(key, params.head_dim_v);
    attention_key_mix(key, params.n_head_q);
    attention_key_mix(key, params.n_head_kv);
    attention_key_mix(key, params.n_query_tokens);
    attention_key_mix(key, params.n_batch);
    attention_key_mix(key, params.causal ? 1 : 0);
    for (const auto & page : view.pages()) {
        attention_key_mix(key, page.logical_page);
        attention_key_mix(key, page.source_physical_slot);
        attention_key_mix(key, page.compact_row_begin);
        attention_key_mix(key, page.row_count);
        attention_key_mix(key, uint64_t(page.native_position_begin));
        attention_key_mix(key, uint64_t(page.native_position_end));
        attention_key_mix(key, page.page_generation);
    }
    for (const llama_pos position : view.native_positions()) {
        attention_key_mix(key, uint64_t(position));
    }
    for (const uint8_t valid : view.native_mask()) {
        attention_key_mix(key, valid);
    }
    for (const llama_pos position : params.query_positions) {
        attention_key_mix(key, uint64_t(position));
    }
    return key == 0 ? 1 : key;
}

static uint64_t attention_layout_key(
        const llama_kv_attention_view & view,
        const llama_kv_attention_operator_params & params) noexcept {
    (void) view;
    uint64_t key = 1469598103934665603ull;
    attention_key_mix(key, uint64_t(params.type_k));
    attention_key_mix(key, uint64_t(params.type_v));
    attention_key_mix(key, uint64_t(params.domain_k));
    attention_key_mix(key, uint64_t(params.domain_v));
    attention_key_mix(key, params.head_dim_k);
    attention_key_mix(key, params.head_dim_v);
    attention_key_mix(key, params.n_head_q);
    attention_key_mix(key, params.n_head_kv);
    attention_key_mix(key, params.n_query_tokens);
    attention_key_mix(key, params.n_batch);
    attention_key_mix(key, params.causal ? 1 : 0);
    // Page membership, tails and positions are mutable descriptor content.
    // The graph input reserves its capacity from the pager geometry, so none
    // of those values may change the reusable graph class.
    attention_key_mix(key, params.query_positions.size());
    return key == 0 ? 1 : key;
}

static uint64_t attention_physical_key(
        const llama_kv_attention_view & view) noexcept {
    uint64_t key = 1469598103934665603ull;
    for (const auto & page : view.pages()) {
        attention_key_mix(key, page.source_physical_slot);
        attention_key_mix(key, page.compact_row_begin);
        attention_key_mix(key, page.row_count);
    }
    return key == 0 ? 1 : key;
}

const char * llama_kv_attention_operator_status_name(
        llama_kv_attention_operator_status status) noexcept {
    switch (status) {
        case llama_kv_attention_operator_status::ok: return "ok";
        case llama_kv_attention_operator_status::disabled: return "disabled";
        case llama_kv_attention_operator_status::invalid_argument: return "invalid_argument";
        case llama_kv_attention_operator_status::empty_table: return "empty_table";
        case llama_kv_attention_operator_status::invalid_page_table: return "invalid_page_table";
        case llama_kv_attention_operator_status::invalid_shape: return "invalid_shape";
        case llama_kv_attention_operator_status::invalid_type: return "invalid_type";
        case llama_kv_attention_operator_status::non_causal: return "non_causal";
        case llama_kv_attention_operator_status::overflow: return "overflow";
    }
    return "invalid";
}

const char * llama_kv_attention_backend_status_name(
        llama_kv_attention_backend_status status) noexcept {
    switch (status) {
        case llama_kv_attention_backend_status::supported_reference: return "supported_reference";
        case llama_kv_attention_backend_status::disabled: return "disabled";
        case llama_kv_attention_backend_status::invalid_metadata: return "invalid_metadata";
        case llama_kv_attention_backend_status::unsupported_backend: return "unsupported_backend";
        case llama_kv_attention_backend_status::unsupported_kv_type: return "unsupported_kv_type";
    }
    return "invalid";
}

llama_kv_attention_operator_metadata::llama_kv_attention_operator_metadata(
        std::shared_ptr<const state> state) noexcept : state_(std::move(state)) {}

llama_kv_attention_operator_metadata llama_kv_attention_operator_metadata::build(
        const llama_kv_attention_view & view,
        const llama_kv_attention_operator_params & params,
        llama_kv_attention_operator_status & status) noexcept {
    status = llama_kv_attention_operator_status::invalid_argument;
    if (params.mode == llama_kv_attention_operator_mode::off) {
        status = llama_kv_attention_operator_status::disabled;
        return {};
    }
    if (params.mode != llama_kv_attention_operator_mode::selective || !view.valid()) {
        return {};
    }
    if (!params.causal) {
        status = llama_kv_attention_operator_status::non_causal;
        return {};
    }
    if (view.pages().empty() || view.get_n_kv() == 0) {
        status = llama_kv_attention_operator_status::empty_table;
        return {};
    }
    if (params.page_tokens != VBR_GENERATION_PAGE_CELLS) {
        status = llama_kv_attention_operator_status::invalid_page_table;
        return {};
    }
    if (params.type_k != GGML_TYPE_TURBO4_0 || params.type_v != GGML_TYPE_TURBO4_0) {
        status = llama_kv_attention_operator_status::invalid_type;
        return {};
    }
    if (params.head_dim_k == 0 || params.head_dim_v == 0 ||
        params.n_head_q == 0 || params.n_head_kv == 0 ||
        params.n_head_q < params.n_head_kv ||
        params.n_head_q % params.n_head_kv != 0 ||
        params.n_query_tokens == 0 || params.n_batch == 0) {
        status = llama_kv_attention_operator_status::invalid_shape;
        return {};
    }
    const uint64_t query_count = uint64_t(params.n_query_tokens) * params.n_batch;
    if (query_count > std::numeric_limits<size_t>::max() ||
        params.query_positions.size() != size_t(query_count)) {
        status = llama_kv_attention_operator_status::invalid_shape;
        return {};
    }
    for (const llama_pos position : params.query_positions) {
        if (position < 0) {
            status = llama_kv_attention_operator_status::invalid_shape;
            return {};
        }
    }
    for (size_t i = 0; i < view.pages().size(); ++i) {
        const auto & page = view.pages()[i];
        const uint64_t compact_end = uint64_t(page.compact_row_begin) + page.row_count;
        const uint64_t native_rows = page.native_position_end >= page.native_position_begin
            ? uint64_t(page.native_position_end - page.native_position_begin) : 0;
        if (page.source_physical_slot == UINT32_MAX || page.row_count == 0 ||
            page.row_count > VBR_GENERATION_PAGE_CELLS || page.native_position_begin < 0 ||
            page.native_position_begin % params.page_tokens != 0 ||
            page.logical_page != uint32_t(page.native_position_begin / params.page_tokens) ||
            page.native_position_end <= page.native_position_begin ||
            native_rows != page.row_count || compact_end > view.get_n_kv()) {
            status = llama_kv_attention_operator_status::invalid_page_table;
            return {};
        }
        if (i != 0 && page.compact_row_begin !=
                view.pages()[i - 1].compact_row_begin + view.pages()[i - 1].row_count) {
            status = llama_kv_attention_operator_status::invalid_page_table;
            return {};
        }
    }

    try {
        auto state = std::make_shared<llama_kv_attention_operator_metadata::state>();
        state->params = params;
        state->view = view;
        state->content_key = attention_content_key(view, params);
        state->layout_key = attention_layout_key(view, params);
        state->physical_key = attention_physical_key(view);
        status = llama_kv_attention_operator_status::ok;
        return llama_kv_attention_operator_metadata(std::move(state));
    } catch (const std::bad_alloc &) {
        status = llama_kv_attention_operator_status::overflow;
        return {};
    }
}

bool llama_kv_attention_operator_metadata::enabled() const noexcept {
    return state_ != nullptr && state_->params.mode == llama_kv_attention_operator_mode::selective;
}

llama_kv_attention_operator_mode llama_kv_attention_operator_metadata::mode() const noexcept {
    return state_ ? state_->params.mode : llama_kv_attention_operator_mode::off;
}

uint64_t llama_kv_attention_operator_metadata::table_epoch() const noexcept {
    return state_ ? state_->view.graph_epoch() : 0;
}

uint64_t llama_kv_attention_operator_metadata::graph_content_key() const noexcept {
    return state_ ? state_->content_key : 0;
}

uint64_t llama_kv_attention_operator_metadata::graph_physical_key() const noexcept {
    return state_ ? state_->physical_key : 0;
}

uint64_t llama_kv_attention_operator_metadata::graph_layout_key() const noexcept {
    return state_ ? state_->layout_key : 0;
}

uint32_t llama_kv_attention_operator_metadata::get_n_kv() const noexcept {
    return state_ ? state_->view.get_n_kv() : 0;
}

ggml_type llama_kv_attention_operator_metadata::type_k() const noexcept {
    return state_ ? state_->params.type_k : GGML_TYPE_COUNT;
}

ggml_type llama_kv_attention_operator_metadata::type_v() const noexcept {
    return state_ ? state_->params.type_v : GGML_TYPE_COUNT;
}

llama_kv_attention_representation_domain llama_kv_attention_operator_metadata::domain_k() const noexcept {
    return state_ ? state_->params.domain_k : llama_kv_attention_representation_domain::original;
}

llama_kv_attention_representation_domain llama_kv_attention_operator_metadata::domain_v() const noexcept {
    return state_ ? state_->params.domain_v : llama_kv_attention_representation_domain::original;
}

uint32_t llama_kv_attention_operator_metadata::head_dim_k() const noexcept { return state_ ? state_->params.head_dim_k : 0; }
uint32_t llama_kv_attention_operator_metadata::head_dim_v() const noexcept { return state_ ? state_->params.head_dim_v : 0; }
uint32_t llama_kv_attention_operator_metadata::n_head_q() const noexcept { return state_ ? state_->params.n_head_q : 0; }
uint32_t llama_kv_attention_operator_metadata::n_head_kv() const noexcept { return state_ ? state_->params.n_head_kv : 0; }
uint32_t llama_kv_attention_operator_metadata::n_query_tokens() const noexcept { return state_ ? state_->params.n_query_tokens : 0; }
uint32_t llama_kv_attention_operator_metadata::n_batch() const noexcept { return state_ ? state_->params.n_batch : 0; }
bool llama_kv_attention_operator_metadata::causal() const noexcept { return state_ && state_->params.causal; }

const std::vector<llama_kv_attention_view_page> & llama_kv_attention_operator_metadata::page_table() const noexcept {
    static const std::vector<llama_kv_attention_view_page> empty;
    return state_ ? state_->view.pages() : empty;
}

const std::vector<llama_pos> & llama_kv_attention_operator_metadata::native_positions() const noexcept {
    static const std::vector<llama_pos> empty;
    return state_ ? state_->view.native_positions() : empty;
}

const std::vector<uint8_t> & llama_kv_attention_operator_metadata::native_mask() const noexcept {
    static const std::vector<uint8_t> empty;
    return state_ ? state_->view.native_mask() : empty;
}

const std::vector<llama_pos> & llama_kv_attention_operator_metadata::query_positions() const noexcept {
    static const std::vector<llama_pos> empty;
    return state_ ? state_->params.query_positions : empty;
}

llama_kv_attention_dense_view_eligibility llama_kv_attention_dense_view_check(
        const llama_kv_attention_operator_metadata & metadata,
        uint32_t physical_page_count) noexcept {
    llama_kv_attention_dense_view_eligibility result;
    if (!metadata.valid() || !metadata.enabled()) {
        result.reason = "metadata is disabled";
        return result;
    }
    if (!metadata.causal() || metadata.type_k() != GGML_TYPE_TURBO4_0 ||
            metadata.type_v() != GGML_TYPE_TURBO4_0 ||
            metadata.domain_k() != llama_kv_attention_representation_domain::turbo_rotated ||
            metadata.domain_v() != llama_kv_attention_representation_domain::turbo_rotated) {
        result.reason = "Turbo4 causal representation contract is not satisfied";
        return result;
    }
    if (physical_page_count == 0 || metadata.page_table().empty()) {
        result.reason = "physical page capacity or page table is empty";
        return result;
    }

    const auto & pages = metadata.page_table();
    const auto & queries = metadata.query_positions();
    if (queries.empty()) {
        result.reason = "query position list is empty";
        return result;
    }
    for (size_t i = 1; i < queries.size(); ++i) {
        if (queries[i] < queries[i - 1]) {
            result.reason = "query positions are not in native causal order";
            return result;
        }
    }

    uint64_t physical_row_begin = 0;
    uint64_t rows = 0;
    llama_pos previous_native_end = -1;
    uint32_t previous_logical_page = UINT32_MAX;
    uint32_t previous_physical_slot = UINT32_MAX;
    for (size_t i = 0; i < pages.size(); ++i) {
        const auto & page = pages[i];
        if (page.row_count == 0 || page.row_count > VBR_GENERATION_PAGE_CELLS ||
                page.source_physical_slot >= physical_page_count ||
                page.native_position_begin < 0 ||
                page.native_position_end != page.native_position_begin + llama_pos(page.row_count) ||
                page.native_position_begin % llama_pos(VBR_GENERATION_PAGE_CELLS) != 0) {
            result.reason = "page has an invalid codec, tail, or native range";
            return result;
        }
        if (i != 0) {
            if (page.logical_page != previous_logical_page + 1 ||
                    page.source_physical_slot != previous_physical_slot + 1 ||
                    page.native_position_begin != previous_native_end) {
                result.reason = "selected pages are not contiguous in native and physical order";
                return result;
            }
            // A partial page is a valid tail only at the end of the selected
            // prefix.  This prevents poisoned padding from entering dense FA.
            if (pages[i - 1].row_count != VBR_GENERATION_PAGE_CELLS) {
                result.reason = "a partial page precedes the selected tail";
                return result;
            }
        }
        if (i + 1 != pages.size() && page.row_count != VBR_GENERATION_PAGE_CELLS) {
            result.reason = "a partial page is not the selected tail";
            return result;
        }
        if (i == 0) {
            physical_row_begin = uint64_t(page.source_physical_slot) * VBR_GENERATION_PAGE_CELLS;
        }
        previous_native_end = page.native_position_end;
        previous_logical_page = page.logical_page;
        previous_physical_slot = page.source_physical_slot;
        rows += page.row_count;
    }

    if (physical_row_begin > UINT32_MAX || rows == 0 || rows != metadata.get_n_kv() ||
            physical_row_begin + rows > uint64_t(physical_page_count) * VBR_GENERATION_PAGE_CELLS) {
        result.reason = "compact row span exceeds the physical pager window";
        return result;
    }

    result.eligible = true;
    result.source_row_begin = uint32_t(physical_row_begin);
    result.row_count = uint32_t(rows);
    result.reason = "contiguous native and physical Turbo4 rows";
    return result;
}

llama_kv_attention_view::graph_fence llama_kv_attention_operator_metadata::acquire_graph_fence() const noexcept {
    return state_ ? state_->view.acquire_graph_fence() : llama_kv_attention_view::graph_fence{};
}

llama_kv_attention_backend_status llama_kv_attention_operator_check_backend(
        enum ggml_backend_dev_type device_type,
        const llama_kv_attention_operator_metadata & metadata) noexcept {
    if (!metadata.valid()) {
        return llama_kv_attention_backend_status::disabled;
    }
    if (!metadata.enabled()) {
        return llama_kv_attention_backend_status::disabled;
    }
    if (metadata.type_k() != GGML_TYPE_TURBO4_0 || metadata.type_v() != GGML_TYPE_TURBO4_0) {
        return llama_kv_attention_backend_status::unsupported_kv_type;
    }
    if (device_type != GGML_BACKEND_DEVICE_TYPE_CPU) {
        return llama_kv_attention_backend_status::unsupported_backend;
    }
    return llama_kv_attention_backend_status::supported_reference;
}

llama_kv_attention_backend_status llama_kv_attention_operator_check_backend(
        ggml_backend_t backend,
        const llama_kv_attention_operator_metadata & metadata) noexcept {
    if (backend == nullptr) {
        return llama_kv_attention_backend_status::unsupported_backend;
    }
    return llama_kv_attention_operator_check_backend(
            ggml_backend_dev_type(ggml_backend_get_device(backend)), metadata);
}
