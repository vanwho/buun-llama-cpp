#include "llama-kv-attention-execution.h"

#include "llama-impl.h"

#include <algorithm>
#include <limits>

namespace {

template<typename T>
T saturating_add(T a, T b) noexcept {
    if (b > std::numeric_limits<T>::max() - a) {
        return std::numeric_limits<T>::max();
    }
    return a + b;
}

void saturating_add_u64(uint64_t & target, uint64_t value) noexcept {
    target = saturating_add(target, value);
}

bool direct_shape(const llama_kv_attention_operator_metadata & metadata) noexcept {
    return metadata.causal() && metadata.type_k() == GGML_TYPE_TURBO4_0 &&
           metadata.type_v() == GGML_TYPE_TURBO4_0 && metadata.head_dim_k() == 256 &&
           metadata.head_dim_v() == 256 && metadata.n_query_tokens() == 1 &&
           metadata.n_batch() == 1 && metadata.n_head_kv() != 0 &&
           metadata.n_head_q() / metadata.n_head_kv() == 4 &&
           metadata.n_head_q() % metadata.n_head_kv() == 0;
}

} // namespace

const char * llama_kv_attention_execution_mode_name(
        llama_kv_attention_execution_mode mode) noexcept {
    switch (mode) {
        case llama_kv_attention_execution_mode::off:       return "off";
        case llama_kv_attention_execution_mode::observe:   return "observe";
        case llama_kv_attention_execution_mode::selective:return "selective";
        case llama_kv_attention_execution_mode::exact:    return "exact";
    }
    return "invalid";
}

const char * llama_kv_attention_execution_phase_name(
        llama_kv_attention_execution_phase phase) noexcept {
    switch (phase) {
        case llama_kv_attention_execution_phase::prefill: return "prefill";
        case llama_kv_attention_execution_phase::decode:  return "decode";
    }
    return "invalid";
}

const char * llama_kv_attention_execution_route_name(
        llama_kv_attention_execution_route route) noexcept {
    switch (route) {
        case llama_kv_attention_execution_route::dense:             return "dense";
        case llama_kv_attention_execution_route::observe:           return "observe";
        case llama_kv_attention_execution_route::selected_reference:return "selected reference";
        case llama_kv_attention_execution_route::selected_direct:  return "selected direct";
        case llama_kv_attention_execution_route::exact_reference:   return "exact reference";
        case llama_kv_attention_execution_route::exact_direct:      return "exact direct";
        case llama_kv_attention_execution_route::refusal:           return "refusal";
    }
    return "invalid";
}

const char * llama_kv_attention_execution_status_name(
        llama_kv_attention_execution_status status) noexcept {
    switch (status) {
        case llama_kv_attention_execution_status::ok:                    return "ok";
        case llama_kv_attention_execution_status::disabled:              return "disabled";
        case llama_kv_attention_execution_status::invalid_metadata:      return "invalid_metadata";
        case llama_kv_attention_execution_status::invalid_prefill_transition:return "invalid_prefill_transition";
        case llama_kv_attention_execution_status::overflow:               return "overflow";
        case llama_kv_attention_execution_status::not_configured:         return "not_configured";
    }
    return "invalid";
}

void llama_kv_attention_execution_metrics::record_exact_ledger(
        const llama_kv_attention_exact_ledger & ledger) noexcept {
    exact_plan_waves = ledger.waves;
    exact_plan_pages = ledger.logical_page_count;
    exact_resident_pages = ledger.resident_pages;
    exact_cold_pages = ledger.cold_pages;
    exact_pages_visited = ledger.pages_visited;
    exact_h2d_useful_bytes = ledger.h2d_useful_bytes;
    exact_h2d_aligned_bytes = ledger.h2d_aligned_bytes;
    exact_waits = ledger.waits;
    exact_peak_staging_pages = ledger.peak_staging_pages;
    exact_duplicate_pages = ledger.duplicate_pages;
    exact_missing_pages = ledger.missing_pages;
    exact_stale_pages = ledger.stale_pages;
    exact_faults = ledger.cold_pages;
}

uint64_t llama_kv_attention_scratch_request::required_rows() const noexcept {
    return saturating_add(saturating_add(resident_rows, transfer_rows), router_rows);
}

size_t llama_kv_attention_scratch_request::required_bytes() const noexcept {
    const uint64_t rows = required_rows();
    if (bytes_per_row != 0 && rows > uint64_t(std::numeric_limits<size_t>::max()) / bytes_per_row) {
        return std::numeric_limits<size_t>::max();
    }
    return size_t(rows) * bytes_per_row;
}

uint32_t llama_kv_attention_prefill_chunk_size(
        uint32_t configured_ubatch,
        uint32_t physical_page_count,
        uint32_t page_tokens) noexcept {
    if (configured_ubatch == 0 || physical_page_count == 0 || page_tokens == 0) {
        return configured_ubatch;
    }
    const uint64_t rows = uint64_t(physical_page_count) * page_tokens;
    return uint32_t(std::min<uint64_t>(configured_ubatch,
            std::min<uint64_t>(rows, std::numeric_limits<uint32_t>::max())));
}

llama_kv_attention_execution_status llama_kv_attention_prefill_admission::append(
        uint32_t logical_page, uint32_t row_count) noexcept {
    if (phase_ != llama_kv_attention_execution_phase::prefill || row_count == 0 ||
        row_count > VBR_GENERATION_PAGE_CELLS || decode_ready_) {
        return llama_kv_attention_execution_status::invalid_prefill_transition;
    }

    if (pages_.empty()) {
        if (logical_page != 0) {
            return llama_kv_attention_execution_status::invalid_prefill_transition;
        }
        pages_.push_back(0);
    } else if (logical_page >= pages_.size()) {
        if (logical_page != pages_.size() || pages_.back() != VBR_GENERATION_PAGE_CELLS) {
            return llama_kv_attention_execution_status::invalid_prefill_transition;
        }
        pages_.push_back(0);
    } else if (logical_page + 1 != pages_.size()) {
        return llama_kv_attention_execution_status::invalid_prefill_transition;
    }

    if (pages_[logical_page] > VBR_GENERATION_PAGE_CELLS - row_count) {
        return llama_kv_attention_execution_status::overflow;
    }
    pages_[logical_page] += row_count;
    if (resident_rows_ > std::numeric_limits<uint32_t>::max() - row_count) {
        return llama_kv_attention_execution_status::overflow;
    }
    resident_rows_ += row_count;
    tail_finished_ = false;
    return llama_kv_attention_execution_status::ok;
}

llama_kv_attention_execution_status llama_kv_attention_prefill_admission::finish_tail() noexcept {
    if (phase_ != llama_kv_attention_execution_phase::prefill || pages_.empty() ||
        pages_.back() == 0 || pages_.back() > VBR_GENERATION_PAGE_CELLS) {
        return llama_kv_attention_execution_status::invalid_prefill_transition;
    }
    tail_finished_ = true;
    return llama_kv_attention_execution_status::ok;
}

llama_kv_attention_execution_status llama_kv_attention_prefill_admission::begin_decode() noexcept {
    if (phase_ != llama_kv_attention_execution_phase::prefill || pages_.empty() ||
        pages_.back() == 0 || (pages_.back() != VBR_GENERATION_PAGE_CELLS && !tail_finished_)) {
        return llama_kv_attention_execution_status::invalid_prefill_transition;
    }
    phase_ = llama_kv_attention_execution_phase::decode;
    decode_ready_ = true;
    return llama_kv_attention_execution_status::ok;
}

llama_kv_attention_execution::llama_kv_attention_execution(
        llama_kv_attention_execution_mode mode) noexcept : mode_(mode) {}

void llama_kv_attention_execution::set_mode(llama_kv_attention_execution_mode mode) noexcept {
    if (mode_ != mode) {
        clear();
    }
    mode_ = mode;
}

bool llama_kv_attention_execution::same_graph(
        const llama_kv_attention_operator_metadata & metadata,
        llama_kv_attention_execution_phase phase,
        uint64_t representation_epoch,
        uint64_t shape_epoch,
        llama_kv_attention_execution_route route) const noexcept {
    return have_graph_ && metadata.graph_content_key() == metadata_.graph_content_key() &&
           metadata.table_epoch() == table_epoch_ &&
           phase == phase_ && representation_epoch == representation_epoch_ &&
           shape_epoch == shape_epoch_ && route == route_;
}

llama_kv_attention_execution_decision llama_kv_attention_execution::prepare(
        const llama_kv_attention_operator_metadata & metadata,
        llama_kv_attention_execution_phase phase,
        uint64_t representation_epoch,
        uint64_t shape_epoch,
        bool direct_capable,
        const llama_kv_attention_scratch_request & scratch) {
    llama_kv_attention_execution_decision result;
    result.phase = phase;
    result.representation_epoch = representation_epoch;
    result.shape_epoch = shape_epoch;
    result.scratch_rows = scratch.required_rows();
    result.scratch_bytes = scratch.required_bytes();

    if (mode_ == llama_kv_attention_execution_mode::off) {
        result.status = llama_kv_attention_execution_status::disabled;
        result.route = llama_kv_attention_execution_route::dense;
        result.reason = "feature disabled";
    } else if (mode_ == llama_kv_attention_execution_mode::observe) {
        result.status = llama_kv_attention_execution_status::ok;
        result.route = llama_kv_attention_execution_route::observe;
        result.reason = "observation preserves dense attention";
    } else if (mode_ == llama_kv_attention_execution_mode::exact) {
        result.status = llama_kv_attention_execution_status::ok;
        result.route = phase == llama_kv_attention_execution_phase::decode &&
                       direct_capable && direct_shape(metadata)
            ? llama_kv_attention_execution_route::exact_direct
            : llama_kv_attention_execution_route::exact_reference;
        result.reason = result.route == llama_kv_attention_execution_route::exact_direct
            ? "all-page Turbo4 direct exact route"
            : "all-page online-softmax reference";
    } else if ((scratch.required_rows() == std::numeric_limits<uint64_t>::max() &&
                (scratch.resident_rows != 0 || scratch.transfer_rows != 0 || scratch.router_rows != 0)) ||
               (scratch.required_bytes() == std::numeric_limits<size_t>::max() &&
                scratch.bytes_per_row != 0)) {
        result.status = llama_kv_attention_execution_status::overflow;
        result.route = llama_kv_attention_execution_route::refusal;
        result.reason = "selected scratch reservation overflows";
    } else if (!metadata.valid() || !metadata.enabled()) {
        result.status = llama_kv_attention_execution_status::invalid_metadata;
        result.route = llama_kv_attention_execution_route::refusal;
        result.reason = "selected metadata is invalid";
    } else {
        result.status = llama_kv_attention_execution_status::ok;
        result.route = phase == llama_kv_attention_execution_phase::decode &&
                       direct_capable && direct_shape(metadata)
            ? llama_kv_attention_execution_route::selected_direct
            : llama_kv_attention_execution_route::selected_reference;
        result.reason = result.route == llama_kv_attention_execution_route::selected_direct
            ? "qualified Turbo4 decode"
            : "compact selected reference";
        result.table_epoch = metadata.table_epoch();
    }

    result.graph_rebuild = result.status == llama_kv_attention_execution_status::ok &&
        !same_graph(metadata, phase, representation_epoch, shape_epoch, result.route);

    if (result.status == llama_kv_attention_execution_status::ok) {
        const bool rebuild = result.graph_rebuild;
        ++metrics_.graph_submission_count;
        if (rebuild) {
            ++metrics_.graph_capture_count;
            ++metrics_.graph_rebuild_count;
        } else {
            ++metrics_.graph_replay_count;
        }
        if (result.route == llama_kv_attention_execution_route::selected_direct && rebuild) {
            // This is the exact host-to-device page-table payload written by
            // llm_graph_input_attn_kv::set_input when a graph is captured. A
            // replay retains the immutable descriptor and does not upload it
            // again. The descriptor is four uint32 fields followed by one
            // native int64 position.
            constexpr uint64_t direct_page_bytes =
                4 * sizeof(uint32_t) + sizeof(int64_t);
            const uint64_t page_count = uint64_t(metadata.page_table().size());
            saturating_add_u64(metrics_.table_upload_bytes,
                    page_count > std::numeric_limits<uint64_t>::max() / direct_page_bytes
                        ? std::numeric_limits<uint64_t>::max()
                        : page_count * direct_page_bytes);
        }
        metrics_.scratch_high_water_rows = std::max(
                metrics_.scratch_high_water_rows, result.scratch_rows);
        metrics_.scratch_high_water_bytes = std::max(
                metrics_.scratch_high_water_bytes, uint64_t(result.scratch_bytes));
        metrics_.selected_pages = std::max<uint64_t>(metrics_.selected_pages,
                metadata.page_table().size());
        metadata_ = metadata;
        route_ = result.route;
        phase_ = phase;
        table_epoch_ = metadata.table_epoch();
        representation_epoch_ = representation_epoch;
        shape_epoch_ = shape_epoch;
        have_graph_ = true;
        graph_fences_.push_back(metadata.acquire_graph_fence());
    }

    LLAMA_LOG_DEBUG("kv-attention: %s path (%s, table=%llu, representation=%llu, shape=%llu, scratch_rows=%llu)\n",
            llama_kv_attention_execution_route_name(result.route),
            llama_kv_attention_execution_phase_name(phase),
            (unsigned long long) result.table_epoch,
            (unsigned long long) result.representation_epoch,
            (unsigned long long) result.shape_epoch,
            (unsigned long long) result.scratch_rows);
    return result;
}

void llama_kv_attention_execution::complete_one_graph() noexcept {
    if (graph_fences_.empty()) {
        return;
    }
    graph_fences_.front().release();
    graph_fences_.erase(graph_fences_.begin());
    ++metrics_.graph_completion_count;
}

void llama_kv_attention_execution::record_descriptor_prepare_us(uint64_t elapsed_us) noexcept {
    saturating_add_u64(metrics_.descriptor_prepare_us, elapsed_us);
}

void llama_kv_attention_execution::record_kernel_us(uint64_t elapsed_us) noexcept {
    saturating_add_u64(metrics_.kernel_us, elapsed_us);
}

void llama_kv_attention_execution::record_total_token_us(uint64_t elapsed_us) noexcept {
    saturating_add_u64(metrics_.total_token_us, elapsed_us);
}

void llama_kv_attention_execution::record_wait() noexcept {
    saturating_add_u64(metrics_.waits, 1);
}

void llama_kv_attention_execution::clear() noexcept {
    // A route/mode change invalidates the current graph key, but it does not
    // cancel work already submitted to the scheduler.  Keep every lease until
    // the normal completion boundary so an old page table cannot be reclaimed
    // while its graph is still consuming it.
    metadata_ = {};
    route_ = llama_kv_attention_execution_route::dense;
    phase_ = llama_kv_attention_execution_phase::prefill;
    table_epoch_ = representation_epoch_ = shape_epoch_ = 0;
    have_graph_ = false;
}
