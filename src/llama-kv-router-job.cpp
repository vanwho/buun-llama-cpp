#include "llama-kv-router-job.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <unordered_map>

bool llama_kv_router_completed_owner_result::make_exact_rerank_candidates(
        const llama_kv_router_query_identity & identity, bool terminal_success,
        const std::vector<llama_kv_router_exact_page_record> & records,
        std::vector<llama_kv_prefetch_candidate> & output) noexcept {
    if (!terminal_success || identity.sequence_id < 0 || identity.session_generation == 0 ||
            identity.turn_id == 0 || identity.query_generation == 0 ||
            identity.table_epoch == 0 ||
            identity.content_generation == 0 || records.empty()) return false;
    try {
        std::unordered_map<uint32_t, double> mean_mass_by_layer;
        std::set<std::pair<uint32_t, uint32_t>> seen_pages;
        std::vector<llama_kv_prefetch_candidate> validated;
        validated.reserve(records.size());
        for (const auto & record : records) {
            auto candidate = record.candidate;
            if (candidate.provenance != llama_kv_prefetch_candidate::score_kind::probe_softmax ||
                    candidate.identity.sequence_id != identity.sequence_id ||
                    candidate.identity.session_generation != identity.session_generation ||
                    candidate.generation != identity.query_generation ||
                    candidate.table_epoch != identity.table_epoch ||
                    candidate.rollback_generation != identity.rollback_generation ||
                    candidate.speculation_generation != candidate.identity.sequence_generation ||
                    candidate.identity.sequence_generation == 0 ||
                    candidate.identity.page_generation == 0 ||
                    candidate.content_version == 0 ||
                    record.descriptor_content_version != candidate.content_version ||
                    record.descriptor_summary_version != candidate.summary_version ||
                    candidate.summary_version != candidate.content_version ||
                    !seen_pages.emplace(candidate.attention_layer,
                        candidate.identity.logical_page).second ||
                    !std::isfinite(candidate.peak_probability) ||
                    !std::isfinite(candidate.mean_probability) ||
                    candidate.peak_probability < 0.0f || candidate.peak_probability > 1.0f ||
                    candidate.mean_probability < 0.0f || candidate.mean_probability > 1.0f) return false;
            mean_mass_by_layer[candidate.attention_layer] += candidate.mean_probability;
            candidate.provenance = llama_kv_prefetch_candidate::score_kind::exact_mass;
            candidate.score = candidate.peak_probability;
            validated.push_back(candidate);
        }
        for (const auto & layer : mean_mass_by_layer) {
            if (!std::isfinite(layer.second) || std::fabs(layer.second - 1.0) > 0.02) return false;
        }
        output = std::move(validated);
        return true;
    } catch (...) {
        return false;
    }
}

std::unique_ptr<llama_kv_router_job> llama_kv_router_job::create(
        ggml_backend_t backend, ggml_backend_buffer_type_t pinned_host_buft,
        size_t slot_bytes, llama_kv_rerank_stage_backend stage_backend,
        ggml_type encoded_type, uint32_t encoded_row_elements) noexcept {
    try {
        auto result = std::unique_ptr<llama_kv_router_job>(new llama_kv_router_job());
        if (!result->initialize(backend, pinned_host_buft, slot_bytes, stage_backend,
                encoded_type, encoded_row_elements)) {
            return nullptr;
        }
        return result;
    } catch (...) {
        return nullptr;
    }
}

bool llama_kv_router_job::initialize(ggml_backend_t backend,
        ggml_backend_buffer_type_t pinned_host_buft, size_t slot_bytes,
        llama_kv_rerank_stage_backend stage_backend, ggml_type encoded_type,
        uint32_t encoded_row_elements) noexcept {
    if (backend == nullptr || pinned_host_buft == nullptr || slot_bytes == 0 ||
            slot_bytes > LLAMA_KV_RERANK_STAGE_MAX_BYTES ||
            slot_bytes > size_t(std::numeric_limits<int64_t>::max()) ||
            !ggml_backend_buft_is_host(pinned_host_buft)) {
        return false;
    }
    backend_ = backend;
    if (encoded_type != GGML_TYPE_I8 && encoded_row_elements == 0) return false;
    if (encoded_type != GGML_TYPE_I8) {
        const size_t row_bytes = ggml_row_size(encoded_type, encoded_row_elements);
        if (row_bytes == 0 || row_bytes > slot_bytes) return false;
        slot_bytes = (slot_bytes / row_bytes) * row_bytes;
    }
    slot_bytes_ = slot_bytes;
    ledger_.alignment = ggml_backend_buft_get_alignment(pinned_host_buft);
    ledger_.pinned_requested = uint64_t(slot_bytes) * LLAMA_KV_RERANK_STAGE_SLOTS;
    pinned_buffer_ = ggml_backend_buft_alloc_buffer(pinned_host_buft,
            size_t(ledger_.pinned_requested));
    if (pinned_buffer_ == nullptr) return false;
    ledger_.pinned_realized = ggml_backend_buffer_get_size(pinned_buffer_);

    ggml_init_params params = {};
    params.mem_size = 4096;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    device_ctx_ = ggml_init(params);
    if (device_ctx_ == nullptr) return false;
    for (uint32_t i = 0; i < LLAMA_KV_RERANK_STAGE_SLOTS; ++i) {
        if (encoded_type == GGML_TYPE_I8) {
            device_slots_[i] = ggml_new_tensor_1d(device_ctx_, encoded_type,
                    int64_t(slot_bytes));
        } else {
            const size_t row_bytes = ggml_row_size(encoded_type, encoded_row_elements);
            device_slots_[i] = ggml_new_tensor_2d(device_ctx_, encoded_type,
                    encoded_row_elements, int64_t(slot_bytes / row_bytes));
        }
        if (device_slots_[i] == nullptr) return false;
    }
    const auto device_buft = ggml_backend_get_default_buffer_type(backend);
    ledger_.device_requested = ggml_backend_alloc_ctx_tensors_from_buft_size(
            device_ctx_, device_buft);
    device_buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(device_ctx_, device_buft);
    if (device_buffer_ == nullptr) return false;
    ledger_.device_realized = ggml_backend_buffer_get_size(device_buffer_);

    std::array<void *, LLAMA_KV_RERANK_STAGE_SLOTS> hosts{};
    std::array<void *, LLAMA_KV_RERANK_STAGE_SLOTS> devices{};
    auto * host_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(pinned_buffer_));
    for (uint32_t i = 0; i < LLAMA_KV_RERANK_STAGE_SLOTS; ++i) {
        hosts[i] = host_base + size_t(i) * slot_bytes;
        devices[i] = device_slots_[i]->data;
    }
    if (stage_backend.enqueue == nullptr && stage_backend.enqueue_task == nullptr) {
        stage_backend.context = this;
        stage_backend.enqueue_task = &llama_kv_router_job::enqueue_copy;
        stage_backend.poll = &llama_kv_router_job::poll_event;
        stage_backend.cancel = &llama_kv_router_job::cancel_event;
        stage_backend.release = &llama_kv_router_job::release_event;
    }
    if (!ring_.configure(hosts, devices, slot_bytes, stage_backend)) return false;
    return true;
}

bool llama_kv_router_job::enqueue_copy(void * context, uint32_t slot,
        const void * host, void *, size_t bytes,
        const llama_kv_rerank_chunk_task &, uint64_t * event_out) noexcept {
    auto * self = static_cast<llama_kv_router_job *>(context);
    if (self == nullptr || slot >= self->device_slots_.size() || host == nullptr ||
            bytes == 0 || bytes > self->slot_bytes_ || event_out == nullptr) return false;
    auto event = ggml_backend_event_new(ggml_backend_get_device(self->backend_));
    if (event == nullptr) return false;
    ggml_backend_tensor_set_async(self->backend_, self->device_slots_[slot], host, 0, bytes);
    ggml_backend_event_record(event, self->backend_);
    *event_out = uint64_t(reinterpret_cast<uintptr_t>(event));
    return true;
}

llama_kv_prefetch_poll llama_kv_router_job::poll_event(
        void *, uint64_t event_value) noexcept {
    if (event_value == 0) return llama_kv_prefetch_poll::failed;
    return ggml_backend_event_query(reinterpret_cast<ggml_backend_event_t>(
            uintptr_t(event_value)))
        ? llama_kv_prefetch_poll::completed : llama_kv_prefetch_poll::pending;
}

void llama_kv_router_job::cancel_event(void *, uint64_t event_value) noexcept {
    if (event_value != 0) ggml_backend_event_synchronize(
            reinterpret_cast<ggml_backend_event_t>(uintptr_t(event_value)));
}

void llama_kv_router_job::release_event(void *, uint64_t event_value) noexcept {
    if (event_value != 0) ggml_backend_event_free(
            reinterpret_cast<ggml_backend_event_t>(uintptr_t(event_value)));
}

llama_kv_router_job::~llama_kv_router_job() {
    cancel();
    if (device_buffer_ != nullptr) ggml_backend_buffer_free(device_buffer_);
    if (device_ctx_ != nullptr) ggml_free(device_ctx_);
    if (pinned_buffer_ != nullptr) ggml_backend_buffer_free(pinned_buffer_);
}

llama_kv_rerank_stage_status llama_kv_router_job::submit(
        const llama_kv_rerank_stage_source & source, uint64_t byte_offset,
        uint32_t rows, const llama_kv_rerank_chunk_task & task,
        uint32_t & slot, uint64_t * ticket) noexcept {
    if (failed_) return llama_kv_rerank_stage_status::enqueue_failed;
    if (rows == 0 || task.rows != rows || task.row_bytes == 0 ||
            rows > slot_bytes_ / task.row_bytes) {
        return llama_kv_rerank_stage_status::invalid_argument;
    }
    return ring_.submit(source, byte_offset, size_t(rows) * task.row_bytes,
            task, slot, ticket);
}

uint32_t llama_kv_router_job::poll(llama_kv_rerank_stage_ticket * output,
        uint32_t capacity) noexcept {
    llama_kv_rerank_stage_ticket local[LLAMA_KV_RERANK_STAGE_SLOTS];
    const uint32_t count = ring_.poll_completed(local, LLAMA_KV_RERANK_STAGE_SLOTS);
    for (uint32_t i = 0; i < count; ++i) {
        if (local[i].terminal != llama_kv_rerank_stage_terminal::succeeded) failed_ = true;
        if (output != nullptr && i < capacity) output[i] = local[i];
    }
    return count;
}

bool llama_kv_router_job::retain_coarse_shortlist(
        const llama_kv_router_query_identity & identity, uint32_t layer,
        const std::vector<llama_kv_prefetch_candidate> & candidates) noexcept {
    if (failed_ || identity.sequence_id < 0 || identity.session_generation == 0 ||
            identity.turn_id == 0 || identity.query_generation == 0 ||
            identity.table_epoch == 0 ||
            identity.content_generation == 0 || candidates.empty()) return false;
    if (query_active_ && (query_identity_.sequence_id != identity.sequence_id ||
            query_identity_.session_generation != identity.session_generation ||
            query_identity_.turn_id != identity.turn_id ||
            query_identity_.query_generation != identity.query_generation ||
            query_identity_.rollback_generation != identity.rollback_generation ||
            query_identity_.table_epoch != identity.table_epoch ||
            query_identity_.content_generation != identity.content_generation)) {
        abort_query();
    }
    try {
        query_identity_ = identity;
        query_active_ = true;
        if (coarse_shortlists_.size() <= layer) coarse_shortlists_.resize(size_t(layer) + 1);
        auto & shortlist = coarse_shortlists_[layer];
        if (!shortlist.empty()) return true;
        shortlist.reserve(candidates.size());
        for (const auto & candidate : candidates) {
            if (candidate.provenance != llama_kv_prefetch_candidate::score_kind::probe_softmax ||
                    candidate.attention_layer != layer ||
                    candidate.generation != identity.query_generation ||
                    candidate.table_epoch != identity.table_epoch ||
                    candidate.rollback_generation != identity.rollback_generation ||
                    candidate.identity.sequence_id != identity.sequence_id ||
                    candidate.identity.session_generation != identity.session_generation ||
                    candidate.content_version == 0 ||
                    candidate.summary_version != candidate.content_version) return false;
            shortlist.push_back(candidate);
        }
        owner_state_.identity = { identity.turn_id, identity.query_generation,
                identity.content_generation };
        owner_state_.shortlist.turn_generation = identity.turn_id;
        owner_state_.shortlist.query_generation = identity.query_generation;
        owner_state_.shortlist.content_generation = identity.content_generation;
        owner_state_.shortlist.pages.clear();
        for (const auto & pages : coarse_shortlists_) for (const auto & candidate : pages) {
            owner_state_.shortlist.pages.push_back({ int32_t(candidate.identity.logical_page),
                true, candidate.identity.page_generation, candidate.content_version,
                candidate.summary_version, 1u });
        }
        owner_state_.stage = llama_kv_router_owner_stage::coarse_pending;
        return true;
    } catch (...) {
        abort_query();
        return false;
    }
}

bool llama_kv_router_job::complete_query(
        const llama_kv_router_query_identity & identity, bool terminal_success,
        const std::vector<llama_kv_router_exact_page_record> & records) noexcept {
    if (query_active_ && owner_state_.stage == llama_kv_router_owner_stage::ready &&
            query_identity_.sequence_id == identity.sequence_id &&
            query_identity_.session_generation == identity.session_generation &&
            query_identity_.turn_id == identity.turn_id &&
            query_identity_.query_generation == identity.query_generation &&
            query_identity_.rollback_generation == identity.rollback_generation &&
            query_identity_.table_epoch == identity.table_epoch &&
            query_identity_.content_generation == identity.content_generation) return true;
    if (!query_active_ || failed_ ||
            query_identity_.sequence_id != identity.sequence_id ||
            query_identity_.session_generation != identity.session_generation ||
            query_identity_.turn_id != identity.turn_id ||
            query_identity_.query_generation != identity.query_generation ||
            query_identity_.rollback_generation != identity.rollback_generation ||
            query_identity_.table_epoch != identity.table_epoch ||
            query_identity_.content_generation != identity.content_generation ||
            owner_state_.shortlist.pages.empty()) {
        abort_query();
        return false;
    }
    std::vector<llama_kv_prefetch_candidate> exact;
    if (!llama_kv_router_completed_owner_result::make_exact_rerank_candidates(
            identity, terminal_success, records, exact)) {
        abort_query();
        return false;
    }
    for (const auto & candidate : exact) {
        if (!candidate.cold) continue;
        const uint32_t layer = candidate.attention_layer;
        if (layer >= coarse_shortlists_.size() ||
                std::none_of(coarse_shortlists_[layer].begin(), coarse_shortlists_[layer].end(),
                    [&](const auto & coarse) {
                return coarse.identity == candidate.identity &&
                    coarse.content_version == candidate.content_version;
            })) {
            abort_query();
            return false;
        }
    }
    exact_candidates_ = std::move(exact);
    owner_state_.stage = llama_kv_router_owner_stage::ready;
    return true;
}

void llama_kv_router_job::abort_query() noexcept {
    exact_candidates_.clear();
    coarse_shortlists_.clear();
    query_active_ = false;
    owner_state_ = {};
    cancel();
}

void llama_kv_router_job::cancel() noexcept {
    ring_.cancel();
}

ggml_tensor * llama_kv_router_job::encoded_key_slot(uint32_t slot) const noexcept {
    return slot < device_slots_.size() ? device_slots_[slot] : nullptr;
}
