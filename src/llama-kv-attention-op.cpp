#include "llama-kv-attention-op.h"

#include <limits>
#include <new>
#include <utility>

struct llama_kv_attention_operator_metadata::state {
    llama_kv_attention_operator_params params;
    llama_kv_attention_view view;
};

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

uint32_t llama_kv_attention_operator_metadata::get_n_kv() const noexcept {
    return state_ ? state_->view.get_n_kv() : 0;
}

ggml_type llama_kv_attention_operator_metadata::type_k() const noexcept {
    return state_ ? state_->params.type_k : GGML_TYPE_COUNT;
}

ggml_type llama_kv_attention_operator_metadata::type_v() const noexcept {
    return state_ ? state_->params.type_v : GGML_TYPE_COUNT;
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
