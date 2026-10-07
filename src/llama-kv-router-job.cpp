#include "llama-kv-router-job.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <set>
#include <unordered_map>

static const char * llama_kv_router_finalize_reason_name(
        llama_kv_router_finalize_reason reason) noexcept {
    switch (reason) {
        case llama_kv_router_finalize_reason::none: return "none";
        case llama_kv_router_finalize_reason::inactive_query: return "inactive_query";
        case llama_kv_router_finalize_reason::terminal_failure: return "terminal_failure";
        case llama_kv_router_finalize_reason::identity_sequence: return "identity_sequence";
        case llama_kv_router_finalize_reason::identity_session: return "identity_session";
        case llama_kv_router_finalize_reason::identity_turn: return "identity_turn";
        case llama_kv_router_finalize_reason::identity_query: return "identity_query";
        case llama_kv_router_finalize_reason::identity_rollback: return "identity_rollback";
        case llama_kv_router_finalize_reason::identity_table: return "identity_table";
        case llama_kv_router_finalize_reason::identity_content: return "identity_content";
        case llama_kv_router_finalize_reason::layer_completion: return "layer_completion";
        case llama_kv_router_finalize_reason::provenance: return "provenance";
        case llama_kv_router_finalize_reason::duplicate_logical_page: return "duplicate_logical_page";
        case llama_kv_router_finalize_reason::invalid_probability: return "invalid_probability";
        case llama_kv_router_finalize_reason::descriptor_generation: return "descriptor_generation";
        case llama_kv_router_finalize_reason::descriptor_version: return "descriptor_version";
        case llama_kv_router_finalize_reason::layer_mass_sum: return "layer_mass_sum";
        case llama_kv_router_finalize_reason::cold_not_in_shortlist: return "cold_not_in_shortlist";
    }
    return "unknown";
}

llama_kv_router_execution_status llama_kv_router_query_executor::execute_once(
        const llama_kv_router_query_identity & identity, bool provisional_phase,
        const std::vector<llama_kv_router_layer_plan> & plans,
        std::vector<llama_kv_router_exact_page_record> & records,
        void * context, execute_fn execute) noexcept {
    records.clear();
    if (!provisional_phase) return llama_kv_router_execution_status::stale;
    if (identity.sequence_id < 0 || identity.session_generation == 0 ||
            identity.turn_id == 0 || identity.query_generation == 0 ||
            identity.table_epoch == 0 || identity.content_generation == 0 ||
            plans.empty() || execute == nullptr) return llama_kv_router_execution_status::stale;
    const bool same = identity_.sequence_id == identity.sequence_id &&
        identity_.session_generation == identity.session_generation &&
        identity_.turn_id == identity.turn_id &&
        identity_.query_generation == identity.query_generation &&
        identity_.rollback_generation == identity.rollback_generation &&
        identity_.table_epoch == identity.table_epoch &&
        identity_.content_generation == identity.content_generation;
    if (started_ && same) {
        if (status_ == llama_kv_router_execution_status::ready) records = records_;
        return status_;
    }
    identity_ = identity;
    started_ = true;
    records_.clear();
    status_ = execute(context, identity, plans, records_);
    if (status_ == llama_kv_router_execution_status::ready) records = records_;
    else records_.clear();
    return status_;
}

void llama_kv_router_query_executor::cancel() noexcept {
    started_ = false;
    identity_ = {};
    status_ = llama_kv_router_execution_status::cancelled;
    records_.clear();
}

llama_kv_router_cache_adapter::result llama_kv_router_cache_adapter::execute(
        llama_kv_router_query_executor & gate, int32_t sequence_id, uint64_t turn_id,
        void * prepare_context, prepare_fn prepare, void * execute_context,
        llama_kv_router_query_executor::execute_fn execute,
        void * terminal_context, terminal_fn terminal) noexcept {
    result output;
    std::vector<llama_kv_router_layer_plan> plans;
    if (prepare == nullptr || execute == nullptr || sequence_id < 0 || turn_id == 0 ||
            !prepare(prepare_context, sequence_id, turn_id, plans, output.plan_error)) {
        output.status = output.plan_error == llama_kv_router_plan_error::stale_query ||
                output.plan_error == llama_kv_router_plan_error::stale_page
            ? llama_kv_router_execution_status::stale : llama_kv_router_execution_status::failed;
        if (terminal != nullptr) terminal(terminal_context, output.status, output.records);
        return output;
    }
    if (plans.empty()) {
        output.plan_error = llama_kv_router_plan_error::invalid_geometry;
        output.status = llama_kv_router_execution_status::failed;
        if (terminal != nullptr) terminal(terminal_context, output.status, output.records);
        return output;
    }
    const auto & identity = plans.front().identity;
    output.layer = plans.front().compact_layer;
    for (const auto & plan : plans) {
        if (!llama_kv_router_layer_plan_metadata_valid(plan) ||
                plan.identity.sequence_id != sequence_id || plan.identity.turn_id != turn_id ||
                plan.identity.session_generation != identity.session_generation ||
                plan.identity.query_generation != identity.query_generation ||
                plan.identity.rollback_generation != identity.rollback_generation ||
                plan.identity.table_epoch != identity.table_epoch ||
                plan.identity.content_generation != identity.content_generation) {
            output.plan_error = llama_kv_router_plan_error::invalid_geometry;
            output.status = llama_kv_router_execution_status::failed;
            if (terminal != nullptr) terminal(terminal_context, output.status, output.records);
            return output;
        }
    }
    output.status = gate.execute_once(identity, true, plans, output.records,
            execute_context, execute);
    if (output.status != llama_kv_router_execution_status::ready) output.records.clear();
    if (terminal != nullptr) terminal(terminal_context, output.status, output.records);
    return output;
}

bool llama_kv_router_layer_plan_metadata_valid(
        const llama_kv_router_layer_plan & plan) noexcept {
    if (plan.compact_layer == UINT32_MAX || plan.model_layer == UINT32_MAX ||
            plan.probe_capture_generation == 0 || plan.page_tokens == 0 ||
            plan.query_heads == 0 || plan.kv_heads == 0 || plan.key_dimension == 0 ||
            !std::isfinite(plan.attention_scale) || plan.attention_scale == 0.0f ||
            !std::isfinite(plan.logit_softcap) || plan.logit_softcap < 0.0f) return false;
    for (size_t i = 0; i < plan.pages.size(); ++i) {
        const auto & page = plan.pages[i];
        if (page.valid_rows == 0 || page.valid_rows > plan.page_tokens ||
                page.stream == UINT32_MAX || page.candidate.content_version == 0 ||
                page.candidate.summary_version == 0 || page.page_generation == 0 ||
                page.content_generation == 0 || page.summary_generation == 0 ||
                (page.candidate.cold && (page.key_source.page_holder == nullptr ||
                    page.key_source.recheck == nullptr || page.key_source.read == nullptr)) ||
                (!page.candidate.cold && page.physical_slot == UINT32_MAX)) return false;
        for (size_t previous = 0; previous < i; ++previous) {
            if (plan.pages[previous].candidate.identity == page.candidate.identity) return false;
        }
    }
    return true;
}

std::shared_ptr<llama_kv_router_key_reader> llama_kv_router_key_reader::create(
        const llama_kv_page_id & identity, uint64_t content_version,
        uint64_t summary_version, const vbr_selected_page_unit_descriptor & unit,
        uint32_t valid_rows, void * current_context, current_fn is_current) noexcept {
    if (!llama_kv_page_id_valid(identity, llama_kv_page_id_is_tail(identity)) ||
            content_version == 0 || summary_version == 0 || valid_rows == 0 ||
            unit.side != vbr_artifact_side::key || unit.bytes == nullptr ||
            unit.row_bytes == 0 || unit.valid_rows < valid_rows ||
            uint64_t(unit.valid_rows) > std::numeric_limits<uint64_t>::max() / unit.row_bytes ||
            unit.bytes->size() < uint64_t(unit.valid_rows) * unit.row_bytes ||
            current_context == nullptr || is_current == nullptr) return nullptr;
    try {
        auto result = std::shared_ptr<llama_kv_router_key_reader>(new llama_kv_router_key_reader());
        result->identity_ = identity;
        result->content_version_ = content_version;
        result->summary_version_ = summary_version;
        result->valid_rows_ = valid_rows;
        result->unit_ = unit;
        result->current_context_ = current_context;
        result->is_current_ = is_current;
        return result;
    } catch (...) {
        return nullptr;
    }
}

llama_kv_rerank_stage_source llama_kv_router_key_reader::source() noexcept {
    llama_kv_rerank_stage_source result;
    result.page_holder = shared_from_this();
    result.context = this;
    result.recheck = recheck;
    result.read = read;
    return result;
}

bool llama_kv_router_key_reader::recheck(void * context, uint64_t content_version) noexcept {
    const auto * self = static_cast<const llama_kv_router_key_reader *>(context);
    return self != nullptr && content_version == self->content_version_ &&
        self->summary_version_ != 0 && self->is_current_ != nullptr &&
        self->is_current_(self->current_context_, self->identity_,
                self->content_version_, self->summary_version_);
}

bool llama_kv_router_key_reader::read(
        void * context, uint64_t offset, void * dst, size_t bytes) noexcept {
    const auto * self = static_cast<const llama_kv_router_key_reader *>(context);
    if (self == nullptr || dst == nullptr || self->unit_.bytes == nullptr ||
            self->unit_.row_bytes == 0 || offset % self->unit_.row_bytes != 0 ||
            uint64_t(bytes) % self->unit_.row_bytes != 0) return false;
    const uint64_t rows = offset / self->unit_.row_bytes;
    const uint64_t count = uint64_t(bytes) / self->unit_.row_bytes;
    if (rows > self->valid_rows_ || count > self->valid_rows_ - rows ||
            rows > std::numeric_limits<uint64_t>::max() / self->unit_.row_bytes) return false;
    const uint64_t byte_offset = rows * self->unit_.row_bytes;
    if (byte_offset > self->unit_.bytes->size() || bytes > self->unit_.bytes->size() - byte_offset)
        return false;
    return self->unit_.bytes->read(byte_offset, static_cast<uint8_t *>(dst), bytes);
}

bool llama_kv_router_completed_owner_result::make_exact_rerank_candidates(
        const llama_kv_router_query_identity & identity, bool terminal_success,
        const std::vector<llama_kv_router_exact_page_record> & records,
        std::vector<llama_kv_prefetch_candidate> & output,
        llama_kv_router_finalize_diagnostic & diagnostic) noexcept {
    auto reject = [&](llama_kv_router_finalize_reason reason,
            const llama_kv_router_exact_page_record * record, size_t index,
            double layer_sum = 0.0) {
        diagnostic.reason = reason;
        diagnostic.expected = identity;
        diagnostic.record_index = index <= UINT32_MAX ? uint32_t(index) : UINT32_MAX;
        diagnostic.measured_layer_sum = layer_sum;
        if (record != nullptr) {
            const auto & candidate = record->candidate;
            diagnostic.compact_layer = record->compact_layer != UINT32_MAX
                ? record->compact_layer : candidate.attention_layer;
            diagnostic.model_layer = record->model_layer;
            diagnostic.logical_page = candidate.identity.logical_page;
            diagnostic.actual = { candidate.identity.sequence_id,
                candidate.identity.session_generation, candidate.generation,
                candidate.generation, candidate.rollback_generation,
                candidate.table_epoch, candidate.content_version };
            diagnostic.expected_page_generation = candidate.speculation_generation;
            diagnostic.actual_page_generation = candidate.identity.sequence_generation;
            diagnostic.expected_content_version = candidate.content_version;
            diagnostic.actual_content_version = record->descriptor_content_version;
            diagnostic.expected_summary_version = candidate.summary_version;
            diagnostic.actual_summary_version = record->descriptor_summary_version;
            diagnostic.expected_provenance = uint8_t(
                llama_kv_prefetch_candidate::score_kind::probe_softmax);
            diagnostic.actual_provenance = uint8_t(candidate.provenance);
            diagnostic.peak_probability = candidate.peak_probability;
            diagnostic.mean_probability = candidate.mean_probability;
        }
        return false;
    };
    if (identity.sequence_id < 0 || identity.session_generation == 0 || identity.turn_id == 0 ||
            identity.query_generation == 0 || identity.table_epoch == 0 ||
            identity.content_generation == 0) return reject(
                llama_kv_router_finalize_reason::inactive_query, nullptr, 0);
    if (!terminal_success) return reject(
            llama_kv_router_finalize_reason::terminal_failure, nullptr, 0);
    try {
        std::unordered_map<uint32_t, double> mean_mass_by_layer;
        std::set<std::pair<uint32_t, uint32_t>> seen_pages;
        std::vector<llama_kv_prefetch_candidate> validated;
        validated.reserve(records.size());
        for (size_t index = 0; index < records.size(); ++index) {
            const auto & record = records[index];
            auto candidate = record.candidate;
            if (candidate.identity.sequence_id != identity.sequence_id) return reject(
                    llama_kv_router_finalize_reason::identity_sequence, &record, index);
            if (candidate.identity.session_generation != identity.session_generation) return reject(
                    llama_kv_router_finalize_reason::identity_session, &record, index);
            if (candidate.generation != identity.query_generation) return reject(
                    llama_kv_router_finalize_reason::identity_query, &record, index);
            if (candidate.rollback_generation != identity.rollback_generation) return reject(
                    llama_kv_router_finalize_reason::identity_rollback, &record, index);
            if (candidate.table_epoch != identity.table_epoch) return reject(
                    llama_kv_router_finalize_reason::identity_table, &record, index);
            if (candidate.content_version == 0 || candidate.speculation_generation !=
                    candidate.identity.sequence_generation ||
                    candidate.identity.sequence_generation == 0 ||
                    candidate.identity.page_generation == 0) return reject(
                    llama_kv_router_finalize_reason::descriptor_generation, &record, index);
            if (candidate.provenance != llama_kv_prefetch_candidate::score_kind::probe_softmax)
                return reject(llama_kv_router_finalize_reason::provenance, &record, index);
            if (record.descriptor_content_version != candidate.content_version ||
                    record.descriptor_summary_version != candidate.summary_version ||
                    candidate.summary_version != candidate.content_version) return reject(
                    llama_kv_router_finalize_reason::descriptor_version, &record, index);
            if (!seen_pages.emplace(candidate.attention_layer,
                    candidate.identity.logical_page).second) return reject(
                    llama_kv_router_finalize_reason::duplicate_logical_page, &record, index);
            if (!std::isfinite(candidate.peak_probability) ||
                    !std::isfinite(candidate.mean_probability) ||
                    candidate.peak_probability < 0.0f || candidate.peak_probability > 1.0f ||
                    candidate.mean_probability < 0.0f || candidate.mean_probability > 1.0f)
                return reject(llama_kv_router_finalize_reason::invalid_probability,
                        &record, index);
            mean_mass_by_layer[candidate.attention_layer] += candidate.mean_probability;
            candidate.provenance = llama_kv_prefetch_candidate::score_kind::exact_mass;
            candidate.score = candidate.peak_probability;
            validated.push_back(candidate);
        }
        for (const auto & layer : mean_mass_by_layer) {
            if (!std::isfinite(layer.second) || std::fabs(layer.second - 1.0) > 0.02) {
                const auto record = std::find_if(records.begin(), records.end(), [&](const auto & item) {
                    return item.candidate.attention_layer == layer.first;
                });
                const size_t index = size_t(record - records.begin());
                return reject(llama_kv_router_finalize_reason::layer_mass_sum,
                        record == records.end() ? nullptr : &*record, index, layer.second);
            }
        }
        output = std::move(validated);
        return true;
    } catch (...) {
        return reject(llama_kv_router_finalize_reason::layer_completion, nullptr, 0);
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

llama_kv_router_execution_status llama_kv_router_job::execute_query(
        const llama_kv_router_query_identity & identity,
        const std::vector<llama_kv_router_layer_plan> & layer_plans,
        std::vector<llama_kv_router_exact_page_record> & records) noexcept {
    records.clear();
    if (failed_) return llama_kv_router_execution_status::failed;
    const char * timing_setting = std::getenv("LLAMA_KV_ROUTER_TIMINGS");
    const bool timings_enabled = timing_setting != nullptr && timing_setting[0] == '1';
    struct timing_scope {
        execution_ledger * ledger;
        std::chrono::steady_clock::time_point start;
        bool enabled;
        ~timing_scope() {
            if (!enabled) return;
            ledger->total_us = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start).count());
            std::fprintf(stderr, "router_query_timing {\"total_us\":%" PRIu64
                    ",\"resident_graphs\":%u,\"cold_reader_graphs\":%u,\"reader_events\":%u"
                    ",\"cold_source_read_us\":%" PRIu64 ",\"reader_event_wait_us\":%" PRIu64
                    ",\"mass_graphs\":%u,\"mass_readbacks\":%u,\"mass_readback_bytes\":%" PRIu64
                    ",\"scratch_allocated_bytes\":%" PRIu64 ",\"key_h2d_bytes\":%" PRIu64
                    ",\"full_page_h2d_bytes\":%" PRIu64 ",\"staging_slots\":1}\n",
                    ledger->total_us, ledger->resident_rerank_graphs,
                    ledger->cold_reader_graphs, ledger->reader_events,
                    ledger->cold_source_read_us, ledger->reader_event_wait_us,
                    ledger->mass_graphs, ledger->mass_graphs,
                    ledger->mass_readback_bytes, ledger->scratch_allocated_bytes,
                    ledger->key_h2d_bytes, ledger->full_page_h2d_bytes);
        }
    } timing{&execution_, std::chrono::steady_clock::now(), timings_enabled};
    const char * failure_phase = "query_setup";
    uint32_t failure_layer = UINT32_MAX;
    auto fail = [this, &records, &failure_phase, &failure_layer]() {
        records.clear();
        failed_ = true;
        owner_state_.stage = llama_kv_router_owner_stage::failed;
        const auto & diagnostic = finalization_diagnostic_;
        std::fprintf(stderr, "router exact rerank failed phase=%s layer=%u resident=%u cold=%u events=%u mass=%u records=%u finalize=%s compact=%u model=%u page=%u index=%u mass_sum=%.9g peak=%.7g mean=%.7g\n",
                failure_phase, failure_layer, execution_.resident_rerank_graphs,
                execution_.cold_reader_graphs, execution_.reader_events,
                execution_.mass_graphs, execution_.output_records,
                llama_kv_router_finalize_reason_name(diagnostic.reason),
                diagnostic.compact_layer, diagnostic.model_layer, diagnostic.logical_page,
                diagnostic.record_index, diagnostic.measured_layer_sum,
                diagnostic.peak_probability, diagnostic.mean_probability);
        if (diagnostic.reason != llama_kv_router_finalize_reason::none &&
                diagnostic.record_count != 0) {
            std::fprintf(stderr, "router finalization failure snapshot {\"reason\":\"%s\",\"measured_layer_sum\":%.9g,\"records\":[",
                    llama_kv_router_finalize_reason_name(diagnostic.reason),
                    diagnostic.measured_layer_sum);
            for (uint32_t i = 0; i < diagnostic.record_count; ++i) {
                const auto & r = diagnostic.records[i];
                std::fprintf(stderr, "%s{\"compact\":%u,\"model\":%u,\"page\":%u,\"sequence\":%d,\"session\":%" PRIu64
                        ",\"sequence_generation\":%" PRIu64 ",\"page_generation\":%u,\"representation_epoch\":%" PRIu64
                        ",\"position_begin\":%" PRId64 ",\"position_end\":%" PRId64 ",\"attention_layer\":%u,\"stream\":%u,\"slot\":%u,\"valid_rows\":%u,\"first_row\":%" PRIu64
                        ",\"descriptor_page_generation\":%" PRIu64 ",\"content_version\":%" PRIu64 ",\"summary_version\":%" PRIu64
                        ",\"descriptor_content_generation\":%" PRIu64 ",\"descriptor_summary_generation\":%" PRIu64
                        ",\"key_unit\":%u,\"cold\":%u,\"shortlist_match\":%u,\"peak\":%.8g,\"mean\":%.8g}",
                        i == 0 ? "" : ",", r.compact_layer, r.model_layer,
                        r.identity.logical_page, r.identity.sequence_id,
                        r.identity.session_generation, r.identity.sequence_generation,
                        r.identity.page_generation, r.identity.representation_epoch,
                        int64_t(r.identity.position_begin), int64_t(r.identity.position_end),
                        r.identity.attention_layer, r.stream, r.physical_slot, r.valid_rows,
                        r.first_absolute_row, r.descriptor_page_generation, r.content_version,
                        r.summary_version, r.descriptor_content_generation,
                        r.descriptor_summary_generation, r.key_unit_id, r.cold,
                        r.shortlist_match, r.peak_probability, r.mean_probability);
            }
            std::fprintf(stderr, "],\"layers\":[");
            for (uint32_t i = 0; i < diagnostic.layer_count; ++i) {
                const auto & l = diagnostic.layers[i];
                std::fprintf(stderr, "%s{\"compact\":%u,\"model\":%u,\"capture_generation\":%" PRIu64
                        ",\"page_tokens\":%u,\"query_heads\":%u,\"kv_heads\":%u,\"key_dimension\":%u,\"resident_streams\":%u"
                        ",\"resident_key_ne\":[%" PRId64 ",%" PRId64 ",%" PRId64 "],\"validity_read\":%u,\"validity\":[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]"
                        ",\"mean_mass_sum\":%.9g,\"state_valid_mask\":[\"%016" PRIx64 "\",\"%016" PRIx64 "\",\"%016" PRIx64 "\",\"%016" PRIx64 "\"]"
                        ",\"state_mass_mask\":[\"%016" PRIx64 "\",\"%016" PRIx64 "\",\"%016" PRIx64 "\",\"%016" PRIx64 "\"]"
                        ",\"state_nonzero_count\":%u,\"state_finite_count\":%u,\"state_snapshot_read\":%u,\"rerank_eligible_pages\":%u,\"mass_eligible_pages\":%u,\"valid_probe_count\":%u"
                        ",\"probe_finite_count\":%u,\"probe_nonzero_count\":%u,\"probe_nonfinite_count\":%u,\"probe_snapshot_read\":%u}",
                        i == 0 ? "" : ",", l.compact_layer, l.model_layer,
                        l.probe_capture_generation, l.page_tokens, l.query_heads,
                        l.kv_heads, l.key_dimension, l.resident_streams,
                        l.resident_key_ne[0], l.resident_key_ne[1], l.resident_key_ne[2],
                        l.validity_read, l.validity[0], l.validity[1], l.validity[2],
                        l.validity[3], l.validity[4], l.validity[5], l.validity[6],
                        l.validity[7], l.validity[8], l.mean_mass_sum,
                        l.state_valid_channel_mask[0], l.state_valid_channel_mask[1],
                        l.state_valid_channel_mask[2], l.state_valid_channel_mask[3],
                        l.state_has_mass_channel_mask[0], l.state_has_mass_channel_mask[1],
                        l.state_has_mass_channel_mask[2], l.state_has_mass_channel_mask[3],
                        l.state_nonzero_count, l.state_finite_count, l.state_snapshot_read,
                        l.rerank_eligible_pages, l.mass_eligible_pages, l.valid_probe_count,
                        l.probe_finite_count, l.probe_nonzero_count,
                        l.probe_nonfinite_count, l.probe_snapshot_read);
            }
            std::fprintf(stderr, "]}\n");
        }
        return llama_kv_router_execution_status::failed;
    };
    if (!query_active_ || identity.turn_id == 0 || identity.sequence_id < 0 ||
            layer_plans.empty()) return llama_kv_router_execution_status::stale;
    const bool same_completed_identity = identity.sequence_id == query_identity_.sequence_id &&
        identity.session_generation == query_identity_.session_generation &&
        identity.turn_id == query_identity_.turn_id &&
        identity.query_generation == query_identity_.query_generation &&
        identity.rollback_generation == query_identity_.rollback_generation &&
        identity.table_epoch == query_identity_.table_epoch &&
        identity.content_generation == query_identity_.content_generation;
    if (owner_state_.stage == llama_kv_router_owner_stage::ready &&
            completed_turn_id_ == identity.turn_id) {
        if (!same_completed_identity) return llama_kv_router_execution_status::stale;
        records = completed_records_;
        return llama_kv_router_execution_status::ready;
    }
    execution_ = {};
    if (identity.sequence_id != query_identity_.sequence_id ||
            identity.session_generation != query_identity_.session_generation ||
            identity.turn_id != query_identity_.turn_id ||
            identity.query_generation != query_identity_.query_generation ||
            identity.rollback_generation != query_identity_.rollback_generation ||
            identity.table_epoch != query_identity_.table_epoch ||
            identity.content_generation != query_identity_.content_generation) {
        return llama_kv_router_execution_status::stale;
    }

    std::array<llama_kv_router_finalize_diagnostic::layer_snapshot, 64> mass_snapshots{};
    std::array<bool, 64> has_mass_snapshot{};
    try {
        owner_state_.stage = llama_kv_router_owner_stage::rerank_pending;
        static std::atomic<uint64_t> next_job_serial { 1 };
        const uint64_t serial = next_job_serial.fetch_add(1, std::memory_order_relaxed);
        if (serial == 0 || serial > uint64_t(INT64_MAX)) {
            return fail();
        }
        std::vector<llama_kv_router_exact_page_record> completed;
        for (const auto & plan : layer_plans) {
            const size_t plan_index = size_t(&plan - layer_plans.data());
            failure_layer = plan.model_layer;
            failure_phase = "plan_validation";
            if (!llama_kv_router_layer_plan_metadata_valid(plan) ||
                    plan.identity.sequence_id != identity.sequence_id ||
                    plan.identity.session_generation != identity.session_generation ||
                    plan.identity.turn_id != identity.turn_id ||
                    plan.identity.query_generation != identity.query_generation ||
                    plan.identity.rollback_generation != identity.rollback_generation ||
                    plan.identity.table_epoch != identity.table_epoch ||
                    plan.identity.content_generation != identity.content_generation ||
                    plan.probes == nullptr || plan.validity == nullptr ||
                    plan.resident_keys == nullptr || plan.probes->ne[0] != plan.key_dimension ||
                    plan.probes->ne[1] != plan.query_heads || plan.probes->ne[2] != 4 ||
                    plan.resident_keys->ne[0] != int64_t(plan.key_dimension) * plan.kv_heads ||
                    plan.resident_keys->type != GGML_TYPE_TURBO4_0 ||
                    slot_bytes_ / ggml_row_size(GGML_TYPE_TURBO4_0,
                        plan.resident_keys->ne[0]) == 0 ||
                    device_slots_[0]->type != GGML_TYPE_TURBO4_0) {
                return fail();
            }
            const size_t row_bytes = ggml_row_size(GGML_TYPE_TURBO4_0,
                    plan.resident_keys->ne[0]);
            const uint32_t rows_per_chunk = uint32_t(slot_bytes_ / row_bytes);
            const size_t count = plan.pages.size();
            if (count > size_t(INT32_MAX) ||
                    plan.probe_capture_generation > uint64_t(INT64_MAX)) {
                return fail();
            }
            if (count == 0) continue;

            ggml_init_params params = {};
            params.mem_size = std::max<size_t>(1 << 20,
                    count * 512 + plan.query_heads * 4 * 16 + 4096);
            execution_.scratch_allocated_bytes += params.mem_size;
            params.no_alloc = true;
            ggml_context * ctx = ggml_init(params);
            if (ctx == nullptr) { failure_phase = "context_allocation"; return fail(); }
            const auto buft = ggml_backend_get_default_buffer_type(backend_);
            ggml_backend_buffer_t buffer = nullptr;
            ggml_cgraph * graph = nullptr;
            bool layer_ok = false;
            do {
                failure_phase = "tensor_creation";
                auto * descriptors = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 10, int64_t(count));
                auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 2, int64_t(count + 1));
                auto * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2,
                        int64_t(count), plan.query_heads, 4);
                if (descriptors == nullptr || ids == nullptr || state == nullptr) break;
                auto * rerank = ggml_kv_page_rerank(ctx, plan.probes, plan.resident_keys,
                        device_slots_[0], descriptors, ids, plan.validity, state,
                        plan.attention_scale, plan.logit_softcap);
                failure_phase = "rerank_support";
                if (rerank == nullptr || !ggml_backend_supports_op(backend_, rerank)) break;
                auto * mass = ggml_kv_page_mass(ctx, state, descriptors, ids, plan.validity);
                failure_phase = "mass_support";
                if (mass == nullptr || !ggml_backend_supports_op(backend_, mass)) break;
                buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
                failure_phase = "buffer_allocation";
                if (buffer == nullptr) break;
                execution_.scratch_allocated_bytes += ggml_backend_buffer_get_size(buffer);

                std::vector<int64_t> host_ids(2 * (count + 1), 0);
                host_ids[0] = int64_t(plan.probe_capture_generation);
                host_ids[1] = int64_t(serial);
                std::vector<int64_t> host_desc(10 * count, 0);
                bool descriptors_ok = true;
                for (size_t i = 0; i < count; ++i) {
                    const auto & page = plan.pages[i];
                    const auto & candidate = page.candidate;
                    if (candidate.identity.logical_page > uint32_t(INT32_MAX) ||
                            page.page_generation > uint64_t(INT64_MAX) ||
                            candidate.content_version > uint64_t(INT64_MAX) ||
                            page.first_absolute_row > uint64_t(INT64_MAX)) {
                        descriptors_ok = false;
                        break;
                    }
                    host_desc[10 * i + 0] = candidate.identity.logical_page;
                    host_desc[10 * i + 1] = page.physical_slot;
                    host_desc[10 * i + 2] = page.valid_rows;
                    host_desc[10 * i + 3] = page.stream;
                    host_desc[10 * i + 4] = plan.page_tokens;
                    host_desc[10 * i + 5] = int64_t(page.page_generation);
                    host_desc[10 * i + 6] = int64_t(candidate.content_version);
                    host_desc[10 * i + 7] = 1;
                    host_desc[10 * i + 8] = int64_t(page.first_absolute_row);
                    host_desc[10 * i + 9] = candidate.cold ? 1 : 0;
                    host_ids[2 * (i + 1)] = int64_t(page.page_generation);
                    host_ids[2 * (i + 1) + 1] = int64_t(candidate.content_version);
                }
                failure_phase = "descriptor_validation";
                if (!descriptors_ok) break;
                std::vector<float> initial(size_t(2) * count * plan.query_heads * 4);
                for (size_t i = 0; i < initial.size(); i += 2) {
                    initial[i] = -std::numeric_limits<float>::infinity();
                    initial[i + 1] = 0.0f;
                }
                ggml_backend_tensor_set(descriptors, host_desc.data(), 0,
                        host_desc.size() * sizeof(int64_t));
                ggml_backend_tensor_set(ids, host_ids.data(), 0,
                        host_ids.size() * sizeof(int64_t));
                ggml_backend_tensor_set(state, initial.data(), 0,
                        initial.size() * sizeof(float));
                std::vector<int64_t> resident_desc = host_desc;
                for (size_t i = 0; i < count; ++i) {
                    if (plan.pages[i].candidate.cold) resident_desc[10 * i + 7] = 0;
                }
                ggml_backend_tensor_set(descriptors, resident_desc.data(), 0,
                        resident_desc.size() * sizeof(int64_t));
                graph = ggml_new_graph(ctx);
                failure_phase = "resident_graph_creation";
                if (graph == nullptr) break;
                ggml_build_forward_expand(graph, rerank);
                failure_phase = "resident_graph_compute";
                if (ggml_backend_graph_compute(backend_, graph) != GGML_STATUS_SUCCESS) break;
                ++execution_.resident_rerank_graphs;

                size_t layer_record_start = completed.size();
                bool cold_ok = true;
                failure_phase = "cold_source_validation";
                for (size_t i = 0; i < count && cold_ok; ++i) {
                    const auto & page = plan.pages[i];
                    if (!page.candidate.cold) continue;
                    if (page.key_source.recheck == nullptr || page.key_source.read == nullptr ||
                            page.key_source.page_holder == nullptr ||
                            !page.key_source.recheck(page.key_source.context,
                                page.candidate.content_version)) {
                        failure_phase = "cold_source_recheck";
                        cold_ok = false;
                        break;
                    }
                    for (uint32_t row = 0; row < page.valid_rows;) {
                        const uint32_t chunk_rows = std::min(rows_per_chunk, page.valid_rows - row);
                        if (row > UINT64_MAX / row_bytes ||
                                uint64_t(chunk_rows) > SIZE_MAX / row_bytes) {
                            failure_phase = "cold_row_overflow";
                            cold_ok = false;
                            break;
                        }
                        std::vector<uint8_t> bytes(size_t(chunk_rows) * row_bytes);
                        auto source_read_started = std::chrono::steady_clock::now();
                        if (!page.key_source.read(page.key_source.context,
                                uint64_t(row) * row_bytes, bytes.data(), bytes.size())) {
                            execution_.cold_source_read_us += uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - source_read_started).count());
                            failure_phase = "cold_source_read";
                            cold_ok = false;
                            break;
                        }
                        execution_.cold_source_read_us += uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - source_read_started).count());
                        std::vector<int64_t> chunk_desc(10 * count, 0);
                        chunk_desc[10 * i + 0] = host_desc[10 * i + 0];
                        chunk_desc[10 * i + 1] = 0;
                        chunk_desc[10 * i + 2] = chunk_rows;
                        chunk_desc[10 * i + 3] = 0;
                        chunk_desc[10 * i + 4] = host_desc[10 * i + 4];
                        chunk_desc[10 * i + 5] = host_desc[10 * i + 5];
                        chunk_desc[10 * i + 6] = host_desc[10 * i + 6];
                        chunk_desc[10 * i + 7] = 1;
                        chunk_desc[10 * i + 8] = host_desc[10 * i + 8] + row;
                        chunk_desc[10 * i + 9] = 1;
                        ggml_backend_tensor_set(descriptors, chunk_desc.data(), 0,
                                chunk_desc.size() * sizeof(int64_t));
                        ggml_backend_event_t reader_event = ggml_backend_event_new(
                                ggml_backend_get_device(backend_));
                        if (reader_event == nullptr) {
                            failure_phase = "reader_event_allocation";
                            cold_ok = false;
                            break;
                        }
                        ggml_backend_tensor_set_async(backend_, device_slots_[0],
                                bytes.data(), 0, bytes.size());
                        execution_.key_h2d_bytes += bytes.size();
                        const ggml_status submit_status =
                                ggml_backend_graph_compute_async(backend_, graph);
                        if (submit_status != GGML_STATUS_SUCCESS) {
                            // H2D may already be queued. Drain the owned ranking
                            // stream before the host bytes or slot can be reused.
                            ggml_backend_synchronize(backend_);
                            ggml_backend_event_free(reader_event);
                            failure_phase = "cold_graph_submit";
                            cold_ok = false;
                            break;
                        }
                        ggml_backend_event_record(reader_event, backend_);
                        auto wait_started = std::chrono::steady_clock::now();
                        ggml_backend_event_synchronize(reader_event);
                        execution_.reader_event_wait_us += uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - wait_started).count());
                        ggml_backend_event_free(reader_event);
                        ++execution_.cold_reader_graphs;
                        ++execution_.reader_events;
                        row += chunk_rows;
                    }
                }
                if (!cold_ok) break;
                std::vector<int64_t> final_desc = host_desc;
                for (size_t i = 0; i < count; ++i) final_desc[10 * i + 7] = 1;
                ggml_backend_tensor_set(descriptors, final_desc.data(), 0,
                        final_desc.size() * sizeof(int64_t));
                graph = ggml_new_graph(ctx);
                failure_phase = "mass_graph_creation";
                if (graph == nullptr) break;
                ggml_build_forward_expand(graph, mass);
                failure_phase = "mass_graph_compute";
                if (ggml_backend_graph_compute(backend_, graph) != GGML_STATUS_SUCCESS) break;
                ++execution_.mass_graphs;
                std::vector<ggml_kv_page_rank_record> host_records(count);
                execution_.mass_readback_bytes += host_records.size() * sizeof(ggml_kv_page_rank_record);
                ggml_backend_tensor_get(mass, host_records.data(), 0,
                        host_records.size() * sizeof(ggml_kv_page_rank_record));
                for (size_t i = 0; i < count; ++i) {
                    failure_phase = "mass_output_validation";
                    if (host_records[i].logical_page != int32_t(plan.pages[i].candidate.identity.logical_page) ||
                            host_records[i].validity_flags == 0 ||
                            !std::isfinite(host_records[i].peak_probability) ||
                            !std::isfinite(host_records[i].mean_probability)) {
                        std::fprintf(stderr, "router exact rerank invalid record model_layer=%u index=%zu"
                                " expected_page=%u actual_page=%d flags=%u peak=%g mean=%g count=%zu\n",
                                plan.model_layer, i, plan.pages[i].candidate.identity.logical_page,
                                host_records[i].logical_page, host_records[i].validity_flags,
                                host_records[i].peak_probability, host_records[i].mean_probability,
                                count);
                        break;
                    }
                    llama_kv_router_exact_page_record record;
                    record.candidate = plan.pages[i].candidate;
                    record.candidate.peak_probability = host_records[i].peak_probability;
                    record.candidate.mean_probability = host_records[i].mean_probability;
                    record.descriptor_content_version = plan.pages[i].candidate.content_version;
                    record.descriptor_summary_version = plan.pages[i].candidate.summary_version;
                    record.compact_layer = plan.compact_layer;
                    record.model_layer = plan.model_layer;
                    record.stream = plan.pages[i].stream;
                    record.physical_slot = plan.pages[i].physical_slot;
                    record.valid_rows = plan.pages[i].valid_rows;
                    record.key_unit_id = plan.pages[i].key_unit_id;
                    record.first_absolute_row = plan.pages[i].first_absolute_row;
                    record.descriptor_page_generation = plan.pages[i].page_generation;
                    record.descriptor_content_generation = plan.pages[i].content_generation;
                    record.descriptor_summary_generation = plan.pages[i].summary_generation;
                    completed.push_back(std::move(record));
                    ++execution_.output_records;
                }
                double layer_mass_sum = 0.0;
                for (const auto & item : host_records) layer_mass_sum += item.mean_probability;
                if (layer_mass_sum == 0.0 && plan_index < mass_snapshots.size()) {
                    auto & snapshot = mass_snapshots[plan_index];
                    snapshot.compact_layer = plan.compact_layer;
                    snapshot.model_layer = plan.model_layer;
                    snapshot.probe_capture_generation = plan.probe_capture_generation;
                    snapshot.page_tokens = plan.page_tokens;
                    snapshot.query_heads = plan.query_heads;
                    snapshot.kv_heads = plan.kv_heads;
                    snapshot.key_dimension = plan.key_dimension;
                    snapshot.resident_streams = uint32_t(plan.resident_keys->ne[2]);
                    for (int axis = 0; axis < 3; ++axis)
                        snapshot.resident_key_ne[axis] = plan.resident_keys->ne[axis];
                    snapshot.mean_mass_sum = layer_mass_sum;
                    ggml_backend_tensor_get(plan.validity, snapshot.validity,
                            0, sizeof(snapshot.validity));
                    snapshot.validity_read = 1;
                    for (uint32_t q = 0; q < 4; ++q) {
                        if (snapshot.validity[5 + q] != 0) ++snapshot.valid_probe_count;
                    }
                    uint32_t rerank_eligible = 0;
                    uint32_t mass_eligible = 0;
                    for (size_t i = 0; i < count; ++i) {
                        const auto * d = host_desc.data() + 10 * i;
                        const auto * expected = host_ids.data() + 2 * (i + 1);
                        const bool mass_ok = d[7] && d[0] >= 0 && d[5] == expected[0] &&
                            d[6] == expected[1] && snapshot.validity[0] == host_ids[0];
                        if (mass_ok) ++mass_eligible;
                        const bool staged = d[9] != 0;
                        const int64_t rows = staged ? int64_t(rows_per_chunk)
                            : plan.resident_keys->ne[1];
                        const int64_t streams = staged ? device_slots_[0]->ne[2]
                            : plan.resident_keys->ne[2];
                        const int64_t slot = staged ? 0 : d[1];
                        const int64_t stream = staged ? 0 : d[3];
                        if (d[7] && d[0] >= 0 && d[2] > 0 && d[4] > 0 && d[2] <= d[4] &&
                                stream >= 0 && stream < streams && slot >= 0 && rows >= d[2] &&
                                slot <= (rows - d[2]) / d[4] && d[5] == expected[0] &&
                                d[6] == expected[1] && snapshot.validity[0] == host_ids[0]) {
                            ++rerank_eligible;
                        }
                    }
                    snapshot.rerank_eligible_pages = rerank_eligible;
                    snapshot.mass_eligible_pages = mass_eligible;
                    std::vector<float> state_values(size_t(2) * count * plan.query_heads * 4);
                    ggml_backend_tensor_get(state, state_values.data(), 0,
                            state_values.size() * sizeof(float));
                    snapshot.state_snapshot_read = 1;
                    for (uint32_t q = 0; q < 4; ++q) for (uint32_t h = 0;
                            h < plan.query_heads; ++h) {
                        const uint32_t channel = q * plan.query_heads + h;
                        const uint32_t mask_word = channel / 64;
                        const uint32_t mask_bit = channel % 64;
                        if (channel < 256 && snapshot.validity[5 + q] != 0)
                            snapshot.state_valid_channel_mask[mask_word] |= uint64_t(1) << mask_bit;
                        bool channel_mass = false;
                        for (size_t p = 0; p < count; ++p) {
                            const size_t index = 2 * (p + count * (h + plan.query_heads * q));
                            const float m = state_values[index];
                            const float z = state_values[index + 1];
                            if (std::isfinite(m) && std::isfinite(z)) ++snapshot.state_finite_count;
                            if (z > 0.0f && std::isfinite(m) && std::isfinite(z)) {
                                ++snapshot.state_nonzero_count;
                                channel_mass = true;
                            }
                        }
                        if (channel < 256 && channel_mass)
                            snapshot.state_has_mass_channel_mask[mask_word] |= uint64_t(1) << mask_bit;
                    }
                    std::vector<float> probe_values(size_t(plan.key_dimension) *
                            plan.query_heads * 4);
                    ggml_backend_tensor_get(plan.probes, probe_values.data(), 0,
                            probe_values.size() * sizeof(float));
                    snapshot.probe_snapshot_read = 1;
                    for (float value : probe_values) {
                        if (!std::isfinite(value)) ++snapshot.probe_nonfinite_count;
                        else {
                            ++snapshot.probe_finite_count;
                            if (value != 0.0f) ++snapshot.probe_nonzero_count;
                        }
                    }
                    has_mass_snapshot[plan_index] = true;
                }
                layer_ok = completed.size() - layer_record_start == count;
            } while (false);
            if (buffer != nullptr) ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            if (!layer_ok) {
                return fail();
            }
        }
        failure_phase = "query_finalize";
        const auto invalid_probe_layer = std::find_if(mass_snapshots.begin(),
                mass_snapshots.end(), [](const auto & snapshot) {
            return snapshot.probe_snapshot_read != 0 &&
                snapshot.probe_finite_count == 0 && snapshot.probe_nonfinite_count != 0;
        });
        if (invalid_probe_layer != mass_snapshots.end()) {
            // Non-finite model Q is execution failure, never an authenticated
            // empty retrieval result. Preserve the diagnostic and abort without
            // publishing a new selection or advancing a successful-turn cache.
            failure_phase = "nonfinite_query_probe";
            return fail();
        }
        if (!complete_query(identity, true, completed)) {
            auto & diagnostic = finalization_diagnostic_;
            diagnostic.layer_count = std::min<uint32_t>(
                    uint32_t(layer_plans.size()), uint32_t(std::size(diagnostic.layers)));
            for (uint32_t i = 0; i < diagnostic.layer_count; ++i) {
                const auto & plan = layer_plans[i];
                auto & layer = diagnostic.layers[i];
                layer.compact_layer = plan.compact_layer;
                layer.model_layer = plan.model_layer;
                layer.probe_capture_generation = plan.probe_capture_generation;
                layer.page_tokens = plan.page_tokens;
                layer.query_heads = plan.query_heads;
                layer.kv_heads = plan.kv_heads;
                layer.key_dimension = plan.key_dimension;
                layer.resident_streams = plan.resident_keys != nullptr
                    ? uint32_t(plan.resident_keys->ne[2]) : 0;
                if (plan.resident_keys != nullptr) {
                    for (int axis = 0; axis < 3; ++axis)
                        layer.resident_key_ne[axis] = plan.resident_keys->ne[axis];
                }
                if (plan.validity != nullptr && plan.validity->type == GGML_TYPE_I64 &&
                        plan.validity->ne[0] >= 9) {
                    ggml_backend_tensor_get(plan.validity, layer.validity,
                            0, sizeof(layer.validity));
                    layer.validity_read = 1;
                }
                if (i < mass_snapshots.size() && has_mass_snapshot[i])
                    layer = mass_snapshots[i];
            }
            return fail();
        }
        completed_records_ = completed;
        completed_turn_id_ = identity.turn_id;
        records = std::move(completed);
        return llama_kv_router_execution_status::ready;
    } catch (...) {
        failure_phase = "exception";
        return fail();
    }
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
    if (failed_) {
        const bool same_query = query_identity_.sequence_id == identity.sequence_id &&
            query_identity_.session_generation == identity.session_generation &&
            query_identity_.turn_id == identity.turn_id &&
            query_identity_.query_generation == identity.query_generation &&
            query_identity_.rollback_generation == identity.rollback_generation &&
            query_identity_.table_epoch == identity.table_epoch &&
            query_identity_.content_generation == identity.content_generation;
        if (same_query) return false;
        abort_query();
    }
    if (identity.sequence_id < 0 || identity.session_generation == 0 ||
            identity.turn_id == 0 || identity.query_generation == 0 ||
            identity.table_epoch == 0 || identity.content_generation == 0) return false;
    if (query_active_ && (query_identity_.sequence_id != identity.sequence_id ||
            query_identity_.session_generation != identity.session_generation ||
            query_identity_.turn_id != identity.turn_id ||
            query_identity_.query_generation != identity.query_generation ||
            query_identity_.rollback_generation != identity.rollback_generation ||
            query_identity_.table_epoch != identity.table_epoch ||
            query_identity_.content_generation != identity.content_generation)) {
        abort_query();
    }
    if (!query_active_) finalization_diagnostic_ = {};
    try {
        query_identity_ = identity;
        query_active_ = true;
        if (coarse_shortlists_.size() <= layer) coarse_shortlists_.resize(size_t(layer) + 1);
        if (coarse_shortlist_completed_.size() <= layer)
            coarse_shortlist_completed_.resize(size_t(layer) + 1, false);
        auto & shortlist = coarse_shortlists_[layer];
        if (coarse_shortlist_completed_[layer]) return true;
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
        coarse_shortlist_completed_[layer] = true;
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
    if (!query_active_ || failed_) {
        finalization_diagnostic_.reason = llama_kv_router_finalize_reason::inactive_query;
        finalization_diagnostic_.expected = query_identity_;
        finalization_diagnostic_.actual = identity;
        abort_query();
        return false;
    }
    finalization_diagnostic_ = {};
    finalization_diagnostic_.expected = query_identity_;
    finalization_diagnostic_.actual = identity;
    if (query_identity_.sequence_id != identity.sequence_id) finalization_diagnostic_.reason =
            llama_kv_router_finalize_reason::identity_sequence;
    else if (query_identity_.session_generation != identity.session_generation)
        finalization_diagnostic_.reason = llama_kv_router_finalize_reason::identity_session;
    else if (query_identity_.turn_id != identity.turn_id)
        finalization_diagnostic_.reason = llama_kv_router_finalize_reason::identity_turn;
    else if (query_identity_.query_generation != identity.query_generation)
        finalization_diagnostic_.reason = llama_kv_router_finalize_reason::identity_query;
    else if (query_identity_.rollback_generation != identity.rollback_generation)
        finalization_diagnostic_.reason = llama_kv_router_finalize_reason::identity_rollback;
    else if (query_identity_.table_epoch != identity.table_epoch)
        finalization_diagnostic_.reason = llama_kv_router_finalize_reason::identity_table;
    else if (query_identity_.content_generation != identity.content_generation)
        finalization_diagnostic_.reason = llama_kv_router_finalize_reason::identity_content;
    else if (std::none_of(coarse_shortlist_completed_.begin(), coarse_shortlist_completed_.end(),
                [](bool completed) { return completed; }))
        finalization_diagnostic_.reason = llama_kv_router_finalize_reason::layer_completion;
    if (finalization_diagnostic_.reason != llama_kv_router_finalize_reason::none) {
        abort_query();
        return false;
    }
    std::vector<llama_kv_prefetch_candidate> exact;
    if (!llama_kv_router_completed_owner_result::make_exact_rerank_candidates(
            identity, terminal_success, records, exact, finalization_diagnostic_)) {
        auto & diagnostic = finalization_diagnostic_;
        diagnostic.record_count = std::min<uint32_t>(uint32_t(records.size()),
                uint32_t(std::size(diagnostic.records)));
        for (uint32_t i = 0; i < diagnostic.record_count; ++i) {
            const auto & source = records[i];
            const auto & candidate = source.candidate;
            auto & target = diagnostic.records[i];
            target.identity = candidate.identity;
            target.compact_layer = source.compact_layer != UINT32_MAX
                ? source.compact_layer : candidate.attention_layer;
            target.model_layer = source.model_layer;
            target.stream = source.stream;
            target.physical_slot = source.physical_slot;
            target.valid_rows = source.valid_rows;
            target.key_unit_id = source.key_unit_id;
            target.first_absolute_row = source.first_absolute_row;
            target.descriptor_page_generation = source.descriptor_page_generation;
            target.descriptor_content_generation = source.descriptor_content_generation;
            target.descriptor_summary_generation = source.descriptor_summary_generation;
            target.content_version = candidate.content_version;
            target.summary_version = candidate.summary_version;
            target.cold = candidate.cold ? 1u : 0u;
            target.peak_probability = candidate.peak_probability;
            target.mean_probability = candidate.mean_probability;
            if (candidate.cold && candidate.attention_layer < coarse_shortlists_.size()) {
                target.shortlist_match = std::any_of(
                        coarse_shortlists_[candidate.attention_layer].begin(),
                        coarse_shortlists_[candidate.attention_layer].end(), [&](const auto & coarse) {
                    return coarse.identity == candidate.identity &&
                        coarse.content_version == candidate.content_version &&
                        coarse.summary_version == candidate.summary_version;
                }) ? 1u : 0u;
            }
        }
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
            finalization_diagnostic_.reason = llama_kv_router_finalize_reason::cold_not_in_shortlist;
            finalization_diagnostic_.compact_layer = layer;
            finalization_diagnostic_.logical_page = candidate.identity.logical_page;
            finalization_diagnostic_.expected = identity;
            finalization_diagnostic_.actual = { candidate.identity.sequence_id,
                candidate.identity.session_generation, candidate.generation,
                candidate.generation, candidate.rollback_generation,
                candidate.table_epoch, candidate.content_version };
            abort_query();
            return false;
        }
    }
    exact_candidates_ = std::move(exact);
    owner_state_.stage = llama_kv_router_owner_stage::ready;
    return true;
}

bool llama_kv_router_job::complete_empty_query(
        const llama_kv_router_query_identity & identity,
        uint32_t layer_count) noexcept {
    if (identity.sequence_id < 0 || identity.session_generation == 0 ||
            identity.turn_id == 0 || identity.query_generation == 0 ||
            identity.table_epoch == 0 || identity.content_generation == 0 ||
            layer_count == 0) return false;

    const bool same_query = query_active_ &&
        query_identity_.sequence_id == identity.sequence_id &&
        query_identity_.session_generation == identity.session_generation &&
        query_identity_.turn_id == identity.turn_id &&
        query_identity_.query_generation == identity.query_generation &&
        query_identity_.rollback_generation == identity.rollback_generation &&
        query_identity_.table_epoch == identity.table_epoch &&
        query_identity_.content_generation == identity.content_generation;
    if (query_active_ && !same_query) abort_query();
    if (failed_) return false;

    // The cache calls this only after proving that no historical page is
    // eligible. Never let an observed candidate disappear into an empty
    // terminal result.
    for (const auto & shortlist : coarse_shortlists_) {
        if (!shortlist.empty()) return false;
    }
    for (size_t i = layer_count; i < coarse_shortlists_.size(); ++i) {
        if (!coarse_shortlists_[i].empty()) return false;
    }

    query_identity_ = identity;
    query_active_ = true;
    coarse_shortlists_.resize(layer_count);
    coarse_shortlist_completed_.resize(layer_count, false);
    std::fill(coarse_shortlist_completed_.begin(), coarse_shortlist_completed_.end(), true);
    exact_candidates_.clear();
    completed_records_.clear();
    completed_turn_id_ = 0;
    owner_state_ = {};
    owner_state_.identity = { identity.turn_id, identity.query_generation,
            identity.content_generation };
    owner_state_.shortlist.turn_generation = identity.turn_id;
    owner_state_.shortlist.query_generation = identity.query_generation;
    owner_state_.shortlist.content_generation = identity.content_generation;
    owner_state_.stage = llama_kv_router_owner_stage::coarse_pending;
    return complete_query(identity, true, {});
}

void llama_kv_router_job::abort_query() noexcept {
    exact_candidates_.clear();
    completed_records_.clear();
    completed_turn_id_ = 0;
    coarse_shortlists_.clear();
    coarse_shortlist_completed_.clear();
    query_active_ = false;
    owner_state_ = {};
    cancel();
    failed_ = false;
}

void llama_kv_router_job::cancel() noexcept {
    ring_.cancel();
}

ggml_tensor * llama_kv_router_job::encoded_key_slot(uint32_t slot) const noexcept {
    return slot < device_slots_.size() ? device_slots_[slot] : nullptr;
}
