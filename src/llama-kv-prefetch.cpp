#include "llama-kv-prefetch.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace {

bool add_u64(uint64_t a, uint64_t b, uint64_t & out) noexcept {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    out = a + b;
    return true;
}

bool has_identity(const llama_kv_prefetch_intent & intent) noexcept {
    return intent.identity.session_generation != 0 ||
           intent.identity.sequence_generation != 0 ||
           intent.identity.sequence_id >= 0 ||
           intent.identity.logical_page != 0 ||
           intent.identity.page_generation != 0;
}

bool same_identity(const llama_kv_prefetch_intent & lhs,
                   const llama_kv_prefetch_intent & rhs) noexcept {
    if (has_identity(lhs) || has_identity(rhs)) {
        return has_identity(lhs) && has_identity(rhs) &&
               lhs.identity == rhs.identity &&
               lhs.attention_layer == rhs.attention_layer;
    }
    return lhs.page_id == rhs.page_id &&
           lhs.attention_layer == rhs.attention_layer;
}

bool same_candidate(const llama_kv_prefetch_candidate & lhs,
                    const llama_kv_prefetch_candidate & rhs) noexcept {
    return lhs.identity == rhs.identity &&
           lhs.attention_layer == rhs.attention_layer;
}

} // namespace

bool llama_kv_prefetch_expand_selector_ids(
        const int32_t * raw_ids, uint32_t raw_count,
        const std::vector<llama_kv_prefetch_selector_segment> & segments,
        uint32_t mailbox_capacity,
        std::vector<int32_t> & copied_ids,
        llama_kv_prefetch_candidate * records,
        uint32_t & written) noexcept {
    if (raw_ids == nullptr || records == nullptr || raw_count == 0 ||
            raw_count > mailbox_capacity || written > mailbox_capacity) return false;
    try {
        copied_ids.assign(raw_ids, raw_ids + raw_count);
    } catch (...) {
        return false;
    }
    for (const auto & segment : segments) {
        if (segment.record_format != llama_kv_prefetch_selector_segment::format::legacy_i32_ids ||
                segment.byte_stride != sizeof(int32_t) || segment.pages == nullptr || segment.count == 0 ||
                segment.raw_offset > raw_count || segment.count > raw_count - segment.raw_offset ||
                segment.byte_offset != uint64_t(segment.raw_offset) * sizeof(int32_t) ||
                segment.byte_count != uint64_t(segment.count) * sizeof(int32_t) ||
                segment.count != segment.resident_count + segment.cold_count ||
                segment.resident_offset > segment.pages->size() ||
                segment.cold_offset > segment.pages->size() ||
                segment.resident_count > segment.pages->size() - segment.resident_offset ||
                segment.cold_count > segment.pages->size() - segment.cold_offset) return false;
        for (uint32_t rank = 0; rank < segment.count; ++rank) {
            const int32_t raw_index = copied_ids[segment.raw_offset + rank];
            const bool cold = rank >= segment.resident_count;
            const size_t region_rank = cold ? rank - segment.resident_count : rank;
            const size_t region_count = cold ? segment.cold_count : segment.resident_count;
            if (raw_index < 0 || size_t(raw_index) >= segment.pages->size() ||
                    region_rank >= region_count) continue;
            const auto & page = (*segment.pages)[size_t(raw_index)];
            if (page.identity.sequence_id != segment.sequence_id ||
                    page.identity.session_generation != segment.session_generation ||
                    page.identity.sequence_generation == 0 ||
                    page.identity.page_generation == 0 || page.content_version == 0 ||
                    page.summary_version != page.content_version) continue;
            llama_kv_prefetch_candidate candidate;
            candidate.identity = page.identity;
            candidate.attention_layer = segment.attention_layer;
            candidate.selector_rank = uint32_t(region_rank);
            candidate.generation = segment.query_generation;
            candidate.table_epoch = segment.table_epoch;
            candidate.query_position = segment.query_position;
            candidate.speculation_generation = segment.sequence_generation;
            candidate.rollback_generation = segment.rollback_generation;
            candidate.cold = cold;
            candidate.score = 1.0f / (1.0f + float(region_rank));
            candidate.requested_bytes = segment.requested_bytes;
            candidate.content_version = page.content_version;
            candidate.summary_version = page.summary_version;
            if (std::find_if(records, records + written, [&](const auto & old) {
                    return old.identity == candidate.identity &&
                        old.attention_layer == candidate.attention_layer;
                }) != records + written) continue;
            if (written >= mailbox_capacity) return false;
            records[written++] = candidate;
        }
    }
    return true;
}

bool llama_kv_prefetch_expand_rank_records(
        const ggml_kv_page_rank_record * raw_records, uint32_t raw_count,
        uint64_t raw_bytes,
        const std::vector<llama_kv_prefetch_selector_segment> & segments,
        uint32_t mailbox_capacity,
        std::vector<ggml_kv_page_rank_record> & copied_records,
        llama_kv_prefetch_candidate * records,
        uint32_t & written) noexcept {
    constexpr uint64_t record_bytes = sizeof(ggml_kv_page_rank_record);
    if (raw_records == nullptr || records == nullptr || raw_count == 0 ||
            raw_count > mailbox_capacity || written > mailbox_capacity ||
            raw_count > UINT64_MAX / record_bytes || raw_bytes != uint64_t(raw_count) * record_bytes) return false;
    try {
        copied_records.resize(raw_count);
        std::memcpy(copied_records.data(), raw_records, size_t(raw_bytes));
    } catch (...) { return false; }
    for (const auto & segment : segments) {
        if (segment.record_format != llama_kv_prefetch_selector_segment::format::packed_rank_records ||
                segment.pages == nullptr || segment.count == 0 || segment.byte_stride != record_bytes ||
                segment.raw_offset > raw_count || segment.count > raw_count - segment.raw_offset ||
                segment.count != segment.resident_count + segment.cold_count ||
                segment.byte_offset > raw_bytes || segment.byte_count != uint64_t(segment.count) * record_bytes ||
                segment.byte_count > raw_bytes - segment.byte_offset ||
                segment.resident_offset > segment.pages->size() || segment.cold_offset > segment.pages->size() ||
                segment.resident_count > segment.pages->size() - segment.resident_offset ||
                segment.cold_count > segment.pages->size() - segment.cold_offset) return false;
        for (uint32_t rank = 0; rank < segment.count; ++rank) {
            const auto & raw = copied_records[segment.raw_offset + rank];
            if (raw.logical_page < 0 || raw.validity_flags != 1u ||
                    !std::isfinite(raw.peak_probability) || !std::isfinite(raw.mean_probability) ||
                    raw.peak_probability < 0.0f || raw.mean_probability < 0.0f ||
                    raw.peak_probability > 1.0f || raw.mean_probability > 1.0f) continue;
            const bool cold = rank >= segment.resident_count;
            if (size_t(raw.logical_page) >= segment.pages->size()) continue;
            const auto & page = (*segment.pages)[size_t(raw.logical_page)];
            if (page.identity.logical_page != uint32_t(raw.logical_page) ||
                    page.identity.sequence_id != segment.sequence_id ||
                    page.identity.session_generation != segment.session_generation ||
                    page.identity.sequence_generation == 0 || page.identity.page_generation == 0 ||
                    page.content_version == 0 || page.summary_version != page.content_version) continue;
            llama_kv_prefetch_candidate candidate;
            candidate.identity = page.identity;
            candidate.attention_layer = segment.attention_layer;
            candidate.selector_rank = rank - (cold ? segment.resident_count : 0);
            candidate.generation = segment.query_generation;
            candidate.table_epoch = segment.table_epoch;
            candidate.query_position = segment.query_position;
            candidate.speculation_generation = segment.sequence_generation;
            candidate.rollback_generation = segment.rollback_generation;
            candidate.cold = cold;
            candidate.provenance = llama_kv_prefetch_candidate::score_kind::probe_softmax;
            candidate.peak_probability = raw.peak_probability;
            candidate.mean_probability = raw.mean_probability;
            candidate.score = raw.peak_probability;
            candidate.requested_bytes = segment.requested_bytes;
            candidate.content_version = page.content_version;
            candidate.summary_version = page.summary_version;
            if (std::find_if(records, records + written, [&](const auto & old) {
                    return old.identity == candidate.identity && old.attention_layer == candidate.attention_layer;
                }) != records + written) continue;
            if (written >= mailbox_capacity) return false;
            records[written++] = candidate;
        }
    }
    return true;
}

bool llama_kv_prefetch_expand_selector_segments(
        const void * raw_bytes, uint32_t raw_count, uint64_t byte_count,
        const std::vector<llama_kv_prefetch_selector_segment> & segments,
        uint32_t mailbox_capacity,
        std::vector<int32_t> & copied_ids,
        llama_kv_prefetch_candidate * records,
        uint32_t & written) noexcept {
    constexpr uint64_t packed_stride = sizeof(ggml_kv_page_rank_record);
    if (raw_bytes == nullptr || records == nullptr || raw_count == 0 ||
            raw_count > mailbox_capacity || written > mailbox_capacity ||
            byte_count > uint64_t(mailbox_capacity) * packed_stride || segments.empty()) return false;
    std::vector<uint8_t> snapshot;
    try {
        snapshot.resize(size_t(byte_count));
        std::memcpy(snapshot.data(), raw_bytes, size_t(byte_count));
        copied_ids.assign(raw_count, -1);
    } catch (...) { return false; }
    uint64_t end = 0;
    uint32_t record_end = 0;
    for (const auto & segment : segments) {
        const uint32_t expected_stride = segment.record_format ==
                llama_kv_prefetch_selector_segment::format::packed_rank_records
            ? uint32_t(packed_stride) : uint32_t(sizeof(int32_t));
        if (segment.pages == nullptr || segment.count == 0 ||
                segment.byte_stride != expected_stride ||
                segment.byte_offset != end || segment.byte_count != uint64_t(segment.count) * expected_stride ||
                segment.byte_offset > byte_count ||
                segment.byte_count > byte_count - segment.byte_offset ||
                segment.raw_offset != record_end || segment.raw_offset > raw_count ||
                segment.count > raw_count - segment.raw_offset ||
                segment.count != segment.resident_count + segment.cold_count ||
                segment.resident_offset > segment.pages->size() || segment.cold_offset > segment.pages->size() ||
                segment.resident_count > segment.pages->size() - segment.resident_offset ||
                segment.cold_count > segment.pages->size() - segment.cold_offset) return false;
        end += segment.byte_count;
        record_end += segment.count;
        for (uint32_t rank = 0; rank < segment.count; ++rank) {
            const uint8_t * source = snapshot.data() + segment.byte_offset + uint64_t(rank) * expected_stride;
            int32_t logical_page = -1;
            ggml_kv_page_rank_record packed{};
            if (segment.record_format == llama_kv_prefetch_selector_segment::format::packed_rank_records) {
                std::memcpy(&packed, source, sizeof(packed));
                logical_page = packed.logical_page;
            } else {
                std::memcpy(&logical_page, source, sizeof(logical_page));
            }
            copied_ids[segment.raw_offset + rank] = logical_page;
            if (logical_page < 0 || size_t(logical_page) >= segment.pages->size()) continue;
            if (segment.record_format == llama_kv_prefetch_selector_segment::format::packed_rank_records &&
                    (packed.validity_flags != 1u || !std::isfinite(packed.peak_probability) ||
                     !std::isfinite(packed.mean_probability) || packed.peak_probability < 0.0f ||
                     packed.mean_probability < 0.0f || packed.peak_probability > 1.0f ||
                     packed.mean_probability > 1.0f)) continue;
            const auto & page = (*segment.pages)[size_t(logical_page)];
            if (page.identity.logical_page != uint32_t(logical_page) ||
                    page.identity.sequence_id != segment.sequence_id ||
                    page.identity.session_generation != segment.session_generation ||
                    page.identity.sequence_generation == 0 || page.identity.page_generation == 0 ||
                    page.content_version == 0 || page.summary_version != page.content_version) continue;
            const bool cold = rank >= segment.resident_count;
            llama_kv_prefetch_candidate candidate;
            candidate.identity = page.identity;
            candidate.attention_layer = segment.attention_layer;
            candidate.selector_rank = rank - (cold ? segment.resident_count : 0);
            candidate.generation = segment.query_generation;
            candidate.table_epoch = segment.table_epoch;
            candidate.query_position = segment.query_position;
            candidate.speculation_generation = segment.sequence_generation;
            candidate.rollback_generation = segment.rollback_generation;
            candidate.cold = cold;
            candidate.requested_bytes = segment.requested_bytes;
            candidate.content_version = page.content_version;
            candidate.summary_version = page.summary_version;
            if (segment.record_format == llama_kv_prefetch_selector_segment::format::packed_rank_records) {
                candidate.provenance = llama_kv_prefetch_candidate::score_kind::probe_softmax;
                candidate.peak_probability = packed.peak_probability;
                candidate.mean_probability = packed.mean_probability;
                candidate.score = packed.peak_probability;
            } else {
                candidate.score = 1.0f / (1.0f + float(candidate.selector_rank));
            }
            if (std::find_if(records, records + written, [&](const auto & old) {
                    return old.identity == candidate.identity &&
                        old.attention_layer == candidate.attention_layer;
                }) != records + written) continue;
            if (written >= mailbox_capacity) return false;
            records[written++] = candidate;
        }
    }
    return end == byte_count && record_end == raw_count;
}

const char * llama_kv_prefetch_mailbox_status_name(
        llama_kv_prefetch_mailbox_status status) noexcept {
    switch (status) {
        case llama_kv_prefetch_mailbox_status::ok: return "ok";
        case llama_kv_prefetch_mailbox_status::not_configured: return "not_configured";
        case llama_kv_prefetch_mailbox_status::invalid_argument: return "invalid_argument";
        case llama_kv_prefetch_mailbox_status::full: return "full";
        case llama_kv_prefetch_mailbox_status::stale_generation: return "stale_generation";
        case llama_kv_prefetch_mailbox_status::cancelled: return "cancelled";
        case llama_kv_prefetch_mailbox_status::_count: break;
    }
    return "invalid";
}

llama_kv_prefetch_mailbox::llama_kv_prefetch_mailbox(
        const llama_kv_prefetch_mailbox_config & config) noexcept
    : capacity_(0) {
    (void) configure(config);
}

bool llama_kv_prefetch_mailbox::configure(
        const llama_kv_prefetch_mailbox_config & config) noexcept {
    if (config.slot_count == 0 || config.candidates_per_slot == 0) return false;
    for (const auto & value : slots_) {
        if (value.state != slot_state::free || value.external_records != nullptr) return false;
    }
    try {
        std::vector<slot> next(config.slot_count);
        for (auto & value : next) {
            value.records.resize(config.candidates_per_slot);
        }
        slots_.swap(next);
        capacity_ = config.candidates_per_slot;
        return true;
    } catch (...) {
        // Preserve the previous configuration if allocation fails.
        return false;
    }
}

uint32_t llama_kv_prefetch_mailbox::pending_slots() const noexcept {
    return uint32_t(std::count_if(slots_.begin(), slots_.end(),
            [](const auto & slot) {
        return slot.state == slot_state::pending;
    }));
}

uint32_t llama_kv_prefetch_mailbox::ready_slots() const noexcept {
    return uint32_t(std::count_if(slots_.begin(), slots_.end(),
            [](const auto & slot) {
        return slot.state == slot_state::ready;
    }));
}

bool llama_kv_prefetch_mailbox::validate(
        const llama_kv_prefetch_candidate & candidate,
        uint64_t generation, uint64_t table_epoch) const noexcept {
    return candidate.attention_layer != UINT32_MAX &&
           candidate.generation != 0 &&
           (generation == 0 || candidate.generation == generation) &&
           (table_epoch == 0 || candidate.table_epoch == 0 ||
            candidate.table_epoch == table_epoch) &&
           candidate.requested_bytes != 0 && std::isfinite(candidate.score) &&
           !candidate.speculation_rejected &&
           llama_kv_page_id_valid(candidate.identity,
                                  llama_kv_page_id_is_tail(candidate.identity));
}

void llama_kv_prefetch_mailbox::release(slot & value) noexcept {
    value.count = 0;
    value.generation = 0;
    value.event = 0;
    value.state = slot_state::free;
}

llama_kv_prefetch_mailbox_status llama_kv_prefetch_mailbox::acquire(
        uint32_t & slot, llama_kv_prefetch_candidate *& records) noexcept {
    slot = UINT32_MAX;
    records = nullptr;
    if (!configured()) return llama_kv_prefetch_mailbox_status::not_configured;
    for (uint32_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].state != slot_state::free) continue;
        slots_[i].state = slot_state::writing;
        slots_[i].count = 0;
        slot = i;
        records = this->records(slots_[i]);
        return llama_kv_prefetch_mailbox_status::ok;
    }
    return llama_kv_prefetch_mailbox_status::full;
}

llama_kv_prefetch_candidate * llama_kv_prefetch_mailbox::data(
        uint32_t slot) noexcept {
    return slot < slots_.size() && slots_[slot].state == slot_state::writing
        ? records(slots_[slot]) : nullptr;
}

const llama_kv_prefetch_candidate * llama_kv_prefetch_mailbox::data(
        uint32_t slot) const noexcept {
    return slot < slots_.size() && slots_[slot].state != slot_state::free
        ? records(slots_[slot]) : nullptr;
}

bool llama_kv_prefetch_mailbox::attach_storage(
        uint32_t slot, llama_kv_prefetch_candidate * storage) noexcept {
    if (slot >= slots_.size() || storage == nullptr ||
            slots_[slot].state != slot_state::free) {
        return false;
    }
    slots_[slot].external_records = storage;
    return true;
}

llama_kv_prefetch_mailbox_status llama_kv_prefetch_mailbox::publish_pending(
        uint32_t slot, uint32_t count, uint64_t generation,
        uint64_t event) noexcept {
    if (slot >= slots_.size() || count == 0 || count > capacity_ ||
        generation == 0 || event == 0 || !backend_.poll ||
        slots_[slot].state != slot_state::writing) {
        return !configured() || slot >= slots_.size()
            ? llama_kv_prefetch_mailbox_status::not_configured
            : llama_kv_prefetch_mailbox_status::invalid_argument;
    }
    auto & value = slots_[slot];
    // A completion hook owns decoding for raw compact producer output.  The
    // legacy path still validates records at publication for CPU callers.
    if (backend_.complete == nullptr) {
        for (uint32_t i = 0; i < count; ++i) {
            if (!validate(records(value)[i], generation, 0)) {
                release(value);
                return llama_kv_prefetch_mailbox_status::invalid_argument;
            }
            for (uint32_t j = 0; j < i; ++j) {
                if (same_candidate(records(value)[i], records(value)[j])) {
                    release(value);
                    return llama_kv_prefetch_mailbox_status::invalid_argument;
                }
            }
        }
    }
    value.count = count;
    value.generation = generation;
    value.event = event;
    value.state = slot_state::pending;
    return llama_kv_prefetch_mailbox_status::ok;
}

llama_kv_prefetch_mailbox_status llama_kv_prefetch_mailbox::publish_ready(
        uint32_t slot, uint32_t count, uint64_t generation) noexcept {
    if (slot >= slots_.size() || count == 0 || count > capacity_ ||
        generation == 0 || slots_[slot].state != slot_state::writing) {
        return !configured() || slot >= slots_.size()
            ? llama_kv_prefetch_mailbox_status::not_configured
            : llama_kv_prefetch_mailbox_status::invalid_argument;
    }
    auto & value = slots_[slot];
    for (uint32_t i = 0; i < count; ++i) {
        if (!validate(records(value)[i], generation, 0)) {
            release(value);
            return llama_kv_prefetch_mailbox_status::invalid_argument;
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (same_candidate(records(value)[i], records(value)[j])) {
                release(value);
                return llama_kv_prefetch_mailbox_status::invalid_argument;
            }
        }
    }
    value.count = count;
    value.generation = generation;
    value.event = 0;
    value.state = slot_state::ready;
    return llama_kv_prefetch_mailbox_status::ok;
}

void llama_kv_prefetch_mailbox::abandon(uint32_t slot) noexcept {
    if (slot < slots_.size() && slots_[slot].state == slot_state::writing) {
        release(slots_[slot]);
    }
}

llama_kv_prefetch_mailbox_status llama_kv_prefetch_mailbox::poll(
        uint64_t generation, uint64_t table_epoch) noexcept {
    if (!configured()) return llama_kv_prefetch_mailbox_status::not_configured;
    llama_kv_prefetch_mailbox_status result = llama_kv_prefetch_mailbox_status::ok;
    for (auto & value : slots_) {
        if (value.state != slot_state::pending) continue;
        if (generation != 0 && value.generation != generation) {
            if (backend_.cancel && value.event) backend_.cancel(backend_.context, value.event);
            if (backend_.release && value.event) backend_.release(backend_.context, value.event);
            release(value);
            result = llama_kv_prefetch_mailbox_status::stale_generation;
            continue;
        }
        const auto state = backend_.poll(backend_.context, value.event);
        if (state == llama_kv_prefetch_mailbox_poll::pending) continue;
        if (state != llama_kv_prefetch_mailbox_poll::completed) {
            if (backend_.cancel && value.event) backend_.cancel(backend_.context, value.event);
            if (backend_.release && value.event) backend_.release(backend_.context, value.event);
            release(value);
            result = state == llama_kv_prefetch_mailbox_poll::stale_generation
                ? llama_kv_prefetch_mailbox_status::stale_generation
                : llama_kv_prefetch_mailbox_status::cancelled;
            continue;
        }
        uint32_t decoded_count = value.count;
        if (backend_.complete != nullptr && !backend_.complete(
                backend_.context, uint32_t(&value - slots_.data()),
                records(value), records(value), &decoded_count,
                value.generation)) {
            if (backend_.cancel && value.event) backend_.cancel(backend_.context, value.event);
            if (backend_.release && value.event) backend_.release(backend_.context, value.event);
            release(value);
            result = llama_kv_prefetch_mailbox_status::cancelled;
            continue;
        }
        if (decoded_count == 0 || decoded_count > value.count) {
            if (backend_.cancel && value.event) backend_.cancel(backend_.context, value.event);
            if (backend_.release && value.event) backend_.release(backend_.context, value.event);
            release(value);
            result = llama_kv_prefetch_mailbox_status::cancelled;
            continue;
        }
        value.count = decoded_count;
        bool valid = true;
        for (uint32_t i = 0; i < value.count; ++i) {
            if (!validate(records(value)[i], generation, table_epoch)) {
                valid = false;
                break;
            }
            for (uint32_t j = 0; j < i; ++j) {
                if (same_candidate(records(value)[i], records(value)[j])) {
                    valid = false;
                    break;
                }
            }
            if (!valid) break;
        }
        if (!valid) {
            if (backend_.cancel && value.event) backend_.cancel(backend_.context, value.event);
            if (backend_.release && value.event) backend_.release(backend_.context, value.event);
            release(value);
            result = llama_kv_prefetch_mailbox_status::stale_generation;
            continue;
        }
        if (backend_.release && value.event) backend_.release(backend_.context, value.event);
        value.event = 0;
        value.state = slot_state::ready;
    }
    return result;
}

size_t llama_kv_prefetch_mailbox::take_ready(
        std::vector<llama_kv_prefetch_candidate> & output,
        uint32_t max_candidates) noexcept {
    if (!configured() || max_candidates == 0) return 0;
    const size_t before = output.size();
    try {
        for (auto & value : slots_) {
            if (value.state != slot_state::ready) continue;
            for (uint32_t i = 0; i < value.count &&
                    output.size() - before < max_candidates; ++i) {
                output.push_back(records(value)[i]);
            }
            release(value);
            if (output.size() - before >= max_candidates) break;
        }
    } catch (...) {
        return output.size() - before;
    }
    return output.size() - before;
}

void llama_kv_prefetch_mailbox::cancel() noexcept {
    for (auto & value : slots_) {
        if (value.state == slot_state::pending && backend_.cancel && value.event) {
            backend_.cancel(backend_.context, value.event);
        }
        if (value.state == slot_state::pending && backend_.release && value.event) {
            backend_.release(backend_.context, value.event);
        }
        release(value);
    }
}

llama_kv_prefetch_predictor::llama_kv_prefetch_predictor(
        uint32_t capacity) noexcept : capacity_(capacity) {
    try {
        previous_.reserve(capacity_);
    } catch (...) {
        capacity_ = 0;
    }
}

bool llama_kv_prefetch_predictor::observe(
        uint64_t query_generation, uint32_t layer, uint64_t token,
        const std::vector<llama_kv_prefetch_intent> & ranked) noexcept {
    if (query_generation == 0 || capacity_ == 0) return false;
    try {
        std::vector<llama_kv_prefetch_intent> candidates;
        candidates.reserve(std::min<size_t>(ranked.size(), capacity_));
        for (const auto & input : ranked) {
            if (input.page_id == 0 || input.useful_bytes == 0 ||
                input.aligned_bytes < input.useful_bytes ||
                std::find_if(candidates.begin(), candidates.end(),
                    [&](const auto & value) { return same_identity(value, input); }) !=
                    candidates.end()) {
                continue;
            }
            auto intent = input;
            intent.required = false;
            intent.prediction = false;
            intent.prediction_useful_counted = false;
            intent.prediction_hit_counted = false;
            const auto position = std::find_if(candidates.begin(), candidates.end(),
                    [&](const auto & value) { return value.priority < intent.priority; });
            candidates.insert(position, intent);
            if (candidates.size() > capacity_) candidates.pop_back();
        }
        previous_.clear();
        previous_.reserve(candidates.size());
        for (const auto & intent : candidates) {
            previous_.push_back({ intent, query_generation, layer, token });
        }
        return true;
    } catch (...) {
        previous_.clear();
        return false;
    }
}

std::vector<llama_kv_prefetch_intent> llama_kv_prefetch_predictor::predict(
        uint64_t generation, uint32_t layer, uint64_t token,
        uint32_t limit) const noexcept {
    std::vector<llama_kv_prefetch_intent> result;
    if (generation == 0 || limit == 0 || previous_.empty()) return result;
    try {
        const size_t count = std::min<size_t>(
                std::min<uint32_t>(limit, capacity_), previous_.size());
        result.reserve(count);
        for (const auto & item : previous_) {
            if (item.layer != layer) continue;
            auto intent = item.intent;
            intent.generation = generation;
            intent.required = false;
            intent.source_query_generation = item.query_generation;
            intent.source_query_layer = item.layer;
            intent.source_query_token = item.token;
            intent.needed_by_layer = layer;
            intent.needed_by_token = token;
            intent.prediction = true;
            intent.prediction_useful_counted = false;
            intent.prediction_hit_counted = false;
            result.push_back(intent);
            if (result.size() == count) break;
        }
    } catch (...) {
        result.clear();
    }
    return result;
}

void llama_kv_prefetch_predictor::clear() noexcept {
    previous_.clear();
}

const char * llama_kv_prefetch_status_name(llama_kv_prefetch_status status) noexcept {
    switch (status) {
        case llama_kv_prefetch_status::ok: return "ok";
        case llama_kv_prefetch_status::not_configured: return "not_configured";
        case llama_kv_prefetch_status::invalid_argument: return "invalid_argument";
        case llama_kv_prefetch_status::backpressure: return "backpressure";
        case llama_kv_prefetch_status::queue_full: return "queue_full";
        case llama_kv_prefetch_status::event_full: return "event_full";
        case llama_kv_prefetch_status::staging_full: return "staging_full";
        case llama_kv_prefetch_status::host_miss: return "host_miss";
        case llama_kv_prefetch_status::transfer_failed: return "transfer_failed";
        case llama_kv_prefetch_status::cancelled: return "cancelled";
        case llama_kv_prefetch_status::stale_generation: return "stale_generation";
        case llama_kv_prefetch_status::dirty_page: return "dirty_page";
        case llama_kv_prefetch_status::shutdown: return "shutdown";
        case llama_kv_prefetch_status::not_ready: return "not_ready";
        case llama_kv_prefetch_status::_count: break;
    }
    return "invalid";
}

const char * llama_kv_prefetch_timeline_kind_name(
        llama_kv_prefetch_timeline_kind kind) noexcept {
    switch (kind) {
        case llama_kv_prefetch_timeline_kind::enqueue: return "enqueue";
        case llama_kv_prefetch_timeline_kind::needed: return "needed";
        case llama_kv_prefetch_timeline_kind::copy_begin: return "copy_begin";
        case llama_kv_prefetch_timeline_kind::copy_end: return "copy_end";
        case llama_kv_prefetch_timeline_kind::wait: return "wait";
        case llama_kv_prefetch_timeline_kind::consumed: return "consumed";
        case llama_kv_prefetch_timeline_kind::cancelled: return "cancelled";
        case llama_kv_prefetch_timeline_kind::_count: break;
    }
    return "invalid";
}

std::unique_ptr<llama_kv_prefetch_scheduler> llama_kv_prefetch_scheduler::create(
        const llama_kv_prefetch_config & config,
        const llama_kv_prefetch_backend & backend,
        llama_kv_prefetch_status & status) noexcept {
    status = llama_kv_prefetch_status::invalid_argument;
    try {
        if (config.max_queued_pages == 0 || config.max_queued_bytes == 0 ||
            config.max_events == 0 || config.max_pinned_slots == 0 ||
            config.staging_slots < 2 ||
            config.max_pinned_slots < config.max_events ||
            config.min_resident_pages > config.max_pinned_slots ||
            !backend.submit || !backend.poll || !backend.publish_complete) {
            status = (!backend.submit || !backend.poll || !backend.publish_complete)
                ? llama_kv_prefetch_status::not_configured
                : llama_kv_prefetch_status::invalid_argument;
            return nullptr;
        }
        auto result = std::unique_ptr<llama_kv_prefetch_scheduler>(
            new llama_kv_prefetch_scheduler(config, backend));
        status = llama_kv_prefetch_status::ok;
        return result;
    } catch (...) {
        status = llama_kv_prefetch_status::not_configured;
        return nullptr;
    }
}

llama_kv_prefetch_scheduler::llama_kv_prefetch_scheduler(
        const llama_kv_prefetch_config & config,
        const llama_kv_prefetch_backend & backend)
    : config_(config), backend_(backend), predictor_(config.prefetch_depth) {
    queue_.reserve(config_.max_queued_pages);
    active_.reserve(config_.max_events);
    ready_.reserve(config_.max_pinned_slots);
    timeline_.reserve(config_.max_timeline_events);
}

llama_kv_prefetch_scheduler::~llama_kv_prefetch_scheduler() {
    shutdown();
}

llama_kv_prefetch_status llama_kv_prefetch_scheduler::validate_intent(
        const llama_kv_prefetch_intent & intent) const noexcept {
    if (intent.page_id == 0 || intent.generation == 0 || intent.useful_bytes == 0 ||
        intent.aligned_bytes < intent.useful_bytes || intent.speculation_rejected) {
        return llama_kv_prefetch_status::invalid_argument;
    }
    if (intent.attention_layer != UINT32_MAX && intent.identity.attention_layer != UINT32_MAX &&
        intent.identity.attention_layer != intent.attention_layer) {
        return llama_kv_prefetch_status::invalid_argument;
    }
    return llama_kv_prefetch_status::ok;
}

bool llama_kv_prefetch_scheduler::is_ready(
        const llama_kv_prefetch_intent & wanted) const noexcept {
    for (const auto & ready : ready_) {
        if (ready.generation == wanted.generation &&
            same_identity(ready, wanted)) return true;
    }
    return false;
}

uint64_t llama_kv_prefetch_scheduler::now_us() const noexcept {
    return backend_.timestamp_us ? backend_.timestamp_us(backend_.context) : 0;
}

void llama_kv_prefetch_scheduler::mark_failure() noexcept {
    ++counters_.failed;
}

void llama_kv_prefetch_scheduler::record_timeline(
        llama_kv_prefetch_timeline_kind kind,
        const llama_kv_prefetch_intent & intent, uint64_t ticket) noexcept {
    if (config_.max_timeline_events == 0) return;
    try {
        if (timeline_.size() >= config_.max_timeline_events) {
            timeline_.erase(timeline_.begin());
        }
        timeline_.push_back({ kind, intent.page_id, intent.generation, ticket,
                              now_us(), intent.needed_by_layer,
                              intent.needed_by_token, intent.table_epoch,
                              intent.destination_slot, intent.host_offset,
                              intent.host_bytes });
    } catch (...) {
        // Telemetry is bounded and non-authoritative.
    }
}

bool llama_kv_prefetch_scheduler::mark_prediction_useful(
        llama_kv_prefetch_intent & intent) noexcept {
    if (!intent.prediction || intent.prediction_hit_counted) return false;
    intent.prediction_hit_counted = true;
    ++counters_.prediction_hits;
    return true;
}

void llama_kv_prefetch_scheduler::complete_prediction_useful(
        llama_kv_prefetch_intent & intent) noexcept {
    if (!intent.prediction || !intent.prediction_hit_counted ||
        intent.prediction_useful_counted) return;
    intent.prediction_useful_counted = true;
    if (counters_.prediction_useful_bytes > UINT64_MAX - intent.useful_bytes) {
        counters_.prediction_useful_bytes = UINT64_MAX;
    } else {
        counters_.prediction_useful_bytes += intent.useful_bytes;
    }
}

void llama_kv_prefetch_scheduler::mark_prediction_wasted(
        const llama_kv_prefetch_intent & intent) noexcept {
    if (!intent.prediction || intent.prediction_useful_counted) return;
    if (counters_.prediction_wasted_bytes > UINT64_MAX - intent.useful_bytes) {
        counters_.prediction_wasted_bytes = UINT64_MAX;
    } else {
        counters_.prediction_wasted_bytes += intent.useful_bytes;
    }
}

bool llama_kv_prefetch_scheduler::erase_queued(
        const llama_kv_prefetch_intent & wanted) noexcept {
    for (auto it = queue_.begin(); it != queue_.end(); ++it) {
        if (it->generation != wanted.generation || !same_identity(*it, wanted)) continue;
        mark_prediction_wasted(*it);
        queued_bytes_ -= it->aligned_bytes;
        queue_.erase(it);
        return true;
    }
    return false;
}

bool llama_kv_prefetch_scheduler::cancel_active(size_t index) noexcept {
    if (index >= active_.size()) return false;
    if (backend_.cancel) backend_.cancel(backend_.context, active_[index].ticket);
    mark_prediction_wasted(active_[index].intent);
    record_timeline(llama_kv_prefetch_timeline_kind::cancelled,
                    active_[index].intent, active_[index].ticket);
    ++counters_.cancellations;
    active_.erase(active_.begin() + index);
    return true;
}

llama_kv_prefetch_status llama_kv_prefetch_scheduler::enqueue(
        const llama_kv_prefetch_intent & intent) noexcept {
    try {
        if (stopped_) return llama_kv_prefetch_status::shutdown;
        const auto valid = validate_intent(intent);
        if (valid != llama_kv_prefetch_status::ok) return valid;
        ++counters_.requested;

        for (auto it = ready_.begin(); it != ready_.end();) {
            if (same_identity(*it, intent) && it->generation != intent.generation) {
                mark_prediction_wasted(*it);
                it = ready_.erase(it);
                ++counters_.stale_generation_rejects;
            } else {
                ++it;
            }
        }

        if (is_ready(intent)) {
            if (intent.required) {
                ++counters_.prefetch_hits;
                for (auto & ready : ready_) {
                    if (same_identity(ready, intent) &&
                        ready.generation == intent.generation) {
                        mark_prediction_useful(ready);
                        complete_prediction_useful(ready);
                        break;
                    }
                }
            }
            return llama_kv_prefetch_status::ok;
        }

        for (auto & queued : queue_) {
            if (!same_identity(queued, intent)) continue;
            if (queued.generation != intent.generation) {
                erase_queued(queued);
                ++counters_.cancellations;
                break;
            }
            if (intent.aligned_bytes > queued.aligned_bytes) {
                uint64_t new_bytes = 0;
                if (!add_u64(queued_bytes_, intent.aligned_bytes - queued.aligned_bytes, new_bytes) ||
                    new_bytes > config_.max_queued_bytes) {
                    return llama_kv_prefetch_status::backpressure;
                }
                queued_bytes_ = new_bytes;
                queued.aligned_bytes = intent.aligned_bytes;
                queued.useful_bytes = std::max(queued.useful_bytes, intent.useful_bytes);
            }
            queued.priority = std::max(queued.priority, intent.priority);
            queued.required = queued.required || intent.required;
            if (intent.required) mark_prediction_useful(queued);
            if (intent.required) ++counters_.faults;
            return llama_kv_prefetch_status::ok;
        }
        for (size_t i = 0; i < active_.size(); ++i) {
            if (!same_identity(active_[i].intent, intent)) continue;
            if (active_[i].intent.generation == intent.generation) {
                active_[i].intent.required = active_[i].intent.required || intent.required;
                active_[i].intent.priority = std::max(active_[i].intent.priority, intent.priority);
                if (intent.required) mark_prediction_useful(active_[i].intent);
                if (intent.required) ++counters_.faults;
                return llama_kv_prefetch_status::ok;
            }
            cancel_active(i);
            ++counters_.stale_generation_rejects;
            break;
        }
        if (intent.required) ++counters_.faults;
        if (!intent.required && refresh_id_ != 0) {
            if (config_.max_cold_pages_per_refresh != 0 &&
                refresh_pages_ >= config_.max_cold_pages_per_refresh) {
                return llama_kv_prefetch_status::backpressure;
            }
            if (config_.bytes_per_refresh != 0 &&
                (refresh_bytes_ > config_.bytes_per_refresh ||
                 intent.aligned_bytes > config_.bytes_per_refresh - refresh_bytes_)) {
                return llama_kv_prefetch_status::backpressure;
            }
        }
        if (queue_.size() >= config_.max_queued_pages) return llama_kv_prefetch_status::queue_full;
        if (pinned_slots() >= config_.max_pinned_slots) return llama_kv_prefetch_status::backpressure;
        uint64_t new_bytes = 0;
        if (!add_u64(queued_bytes_, intent.aligned_bytes, new_bytes) ||
            new_bytes > config_.max_queued_bytes) {
            return llama_kv_prefetch_status::backpressure;
        }
        const auto position = std::find_if(queue_.begin(), queue_.end(),
                [&](const auto & value) { return value.priority < intent.priority; });
        queue_.insert(position, intent);
        queued_bytes_ = new_bytes;
        if (!intent.required && refresh_id_ != 0) {
            ++refresh_pages_;
            refresh_bytes_ += intent.aligned_bytes;
        }
        ++counters_.queued;
        if (intent.prediction) ++counters_.prediction_requested;
        record_timeline(llama_kv_prefetch_timeline_kind::enqueue, intent);
        return pump();
    } catch (...) {
        return llama_kv_prefetch_status::not_configured;
    }
}

llama_kv_prefetch_status llama_kv_prefetch_scheduler::pump() noexcept {
    if (stopped_) return llama_kv_prefetch_status::shutdown;
    llama_kv_prefetch_status result = llama_kv_prefetch_status::ok;
    try {
        while (!queue_.empty() && active_.size() < config_.max_events &&
               pinned_slots() < config_.max_pinned_slots) {
            uint32_t staging_slot = UINT32_MAX;
            for (uint32_t slot = 0; slot < config_.staging_slots; ++slot) {
                bool used = false;
                for (const auto & active : active_) if (active.staging_slot == slot) used = true;
                if (!used) { staging_slot = slot; break; }
            }
            if (staging_slot == UINT32_MAX) {
                result = llama_kv_prefetch_status::staging_full;
                break;
            }
            const auto intent = queue_.front();
            queue_.erase(queue_.begin());
            queued_bytes_ -= intent.aligned_bytes;
            if (backend_.host_available &&
                !backend_.host_available(backend_.context, intent)) {
                mark_failure();
                mark_prediction_wasted(intent);
                record_timeline(llama_kv_prefetch_timeline_kind::cancelled,
                                intent);
                result = llama_kv_prefetch_status::host_miss;
                continue;
            }
            const uint64_t ticket = next_ticket_++;
            record_timeline(llama_kv_prefetch_timeline_kind::copy_begin,
                            intent, ticket);
            if (ticket == 0 || !backend_.submit(
                    backend_.context, intent, staging_slot, ticket, true)) {
                mark_failure();
                mark_prediction_wasted(intent);
                record_timeline(llama_kv_prefetch_timeline_kind::cancelled,
                                intent, ticket);
                result = llama_kv_prefetch_status::transfer_failed;
                continue;
            }
            active_.push_back({ intent, ticket, now_us(), staging_slot });
            ++counters_.submitted;
            if (!add_u64(counters_.useful_bytes, intent.useful_bytes, counters_.useful_bytes) ||
                !add_u64(counters_.aligned_bytes, intent.aligned_bytes, counters_.aligned_bytes)) {
                mark_failure();
                result = llama_kv_prefetch_status::transfer_failed;
                cancel_active(active_.size() - 1);
                break;
            }
        }
        if (!queue_.empty() && active_.size() >= config_.max_events &&
            result == llama_kv_prefetch_status::ok) result = llama_kv_prefetch_status::event_full;
        if (!queue_.empty() && pinned_slots() >= config_.max_pinned_slots &&
            result == llama_kv_prefetch_status::ok) result = llama_kv_prefetch_status::backpressure;
        return result;
    } catch (...) {
        return llama_kv_prefetch_status::transfer_failed;
    }
}

llama_kv_prefetch_status llama_kv_prefetch_scheduler::prefetch(
        const std::vector<llama_kv_prefetch_intent> & intents) noexcept {
    if (stopped_) return llama_kv_prefetch_status::shutdown;
    llama_kv_prefetch_status result = llama_kv_prefetch_status::ok;
    try {
        const size_t count = std::min<size_t>(intents.size(), config_.prefetch_depth);
        for (size_t i = 0; i < count; ++i) {
            auto intent = intents[i];
            intent.required = false;
            const auto status = enqueue(intent);
            if (status != llama_kv_prefetch_status::ok &&
                result == llama_kv_prefetch_status::ok) result = status;
        }
        return result;
    } catch (...) {
        return llama_kv_prefetch_status::backpressure;
    }
}

llama_kv_prefetch_status llama_kv_prefetch_scheduler::begin_refresh(
        uint64_t refresh_id, bool force) noexcept {
    if (stopped_) return llama_kv_prefetch_status::shutdown;
    if (refresh_id == 0) return llama_kv_prefetch_status::invalid_argument;
    if (!force && refresh_id == refresh_id_) return llama_kv_prefetch_status::ok;
    if (!force && refresh_id_ != 0 && refresh_id < refresh_id_) {
        return llama_kv_prefetch_status::stale_generation;
    }
    refresh_id_ = refresh_id;
    refresh_pages_ = 0;
    refresh_bytes_ = 0;
    return llama_kv_prefetch_status::ok;
}

bool llama_kv_prefetch_scheduler::observe_query(
        uint64_t query_generation, uint32_t layer, uint64_t token,
        const std::vector<llama_kv_prefetch_intent> & ranked) noexcept {
    return predictor_.observe(query_generation, layer, token, ranked);
}

std::vector<llama_kv_prefetch_intent> llama_kv_prefetch_scheduler::predict_next(
        uint64_t generation, uint32_t layer, uint64_t token,
        uint32_t limit) const noexcept {
    return predictor_.predict(generation, layer, token, limit);
}

llama_kv_prefetch_status llama_kv_prefetch_scheduler::advance() noexcept {
    if (stopped_) return llama_kv_prefetch_status::shutdown;
    llama_kv_prefetch_status result = pump();
    try {
        for (size_t i = 0; i < active_.size();) {
            const auto active = active_[i];
            const auto state = backend_.poll(backend_.context, active.ticket);
            if (state == llama_kv_prefetch_poll::pending) { ++i; continue; }
            auto intent = active.intent;
            const uint64_t submitted_us = active.submitted_us;
            if (state == llama_kv_prefetch_poll::completed) {
                const bool current = !backend_.generation_current ||
                    backend_.generation_current(backend_.context, intent);
                const bool ready_capacity = ready_.size() < config_.max_pinned_slots ||
                    ready_.size() > config_.min_resident_pages;
                if (!current || intent.speculation_rejected) {
                    ++counters_.stale_generation_rejects;
                    mark_failure();
                    result = llama_kv_prefetch_status::stale_generation;
                    if (backend_.discard_complete) {
                        backend_.discard_complete(backend_.context, intent);
                    }
                    cancel_active(i);
                } else if (!ready_capacity) {
                    mark_failure();
                    result = llama_kv_prefetch_status::backpressure;
                    if (backend_.discard_complete) {
                        backend_.discard_complete(backend_.context, intent);
                    }
                    cancel_active(i);
                } else if (!backend_.publish_complete(backend_.context, intent)) {
                    mark_failure();
                    result = llama_kv_prefetch_status::transfer_failed;
                    cancel_active(i);
                } else {
                    record_timeline(llama_kv_prefetch_timeline_kind::copy_end,
                                    intent, active.ticket);
                    if (intent.prediction) ++counters_.prediction_completed;
                    complete_prediction_useful(intent);
                    if (ready_.size() >= config_.max_pinned_slots) ready_.erase(ready_.begin());
                    ready_.push_back(intent);
                    ++counters_.completed;
                    const uint64_t completed_us = now_us();
                    if (completed_us >= submitted_us) counters_.stage_latency_us += completed_us - submitted_us;
                    active_.erase(active_.begin() + i);
                }
            } else if (state == llama_kv_prefetch_poll::stale_generation) {
                ++counters_.stale_generation_rejects;
                mark_failure();
                result = llama_kv_prefetch_status::stale_generation;
                cancel_active(i);
            } else {
                mark_failure();
                result = llama_kv_prefetch_status::transfer_failed;
                cancel_active(i);
            }
        }
        const auto pump_result = pump();
        if (result == llama_kv_prefetch_status::ok) result = pump_result;
        return result;
    } catch (...) {
        return llama_kv_prefetch_status::transfer_failed;
    }
}

llama_kv_prefetch_status llama_kv_prefetch_scheduler::cancel(
        uint64_t page_id, uint64_t generation) noexcept {
    if (stopped_) return llama_kv_prefetch_status::shutdown;
    bool found = false;
    for (auto it = queue_.begin(); it != queue_.end();) {
        if (it->page_id != page_id || it->generation != generation) {
            ++it;
            continue;
        }
        mark_prediction_wasted(*it);
        queued_bytes_ -= it->aligned_bytes;
        it = queue_.erase(it);
        found = true;
    }
    for (size_t i = active_.size(); i > 0; --i) {
        const auto & active = active_[i - 1];
        if (active.intent.page_id == page_id && active.intent.generation == generation) {
            cancel_active(i - 1);
            found = true;
        }
    }
    for (auto it = ready_.begin(); it != ready_.end();) {
        if (it->page_id == page_id && it->generation == generation) {
            mark_prediction_wasted(*it);
            record_timeline(llama_kv_prefetch_timeline_kind::cancelled, *it);
            it = ready_.erase(it);
            found = true;
        } else ++it;
    }
    return found ? llama_kv_prefetch_status::cancelled : llama_kv_prefetch_status::not_ready;
}

void llama_kv_prefetch_scheduler::shutdown() noexcept {
    if (stopped_) return;
    for (size_t i = active_.size(); i > 0; --i) cancel_active(i - 1);
    queue_.clear();
    ready_.clear();
    queued_bytes_ = 0;
    stopped_ = true;
}

llama_kv_prefetch_resolution llama_kv_prefetch_scheduler::ensure_ready(
        const std::vector<llama_kv_prefetch_intent> & required,
        const std::vector<uint64_t> & previous_hot_set,
        uint32_t wait_budget_steps) noexcept {
    llama_kv_prefetch_resolution result;
    try {
        if (stopped_) {
            result.readiness = llama_kv_prefetch_readiness::cancelled;
            return result;
        }
        if (wait_budget_steps == UINT32_MAX) wait_budget_steps = config_.wait_budget_steps;
        std::vector<uint64_t> required_ids;
        required_ids.reserve(required.size());
        for (const auto & intent : required) {
            if (validate_intent(intent) != llama_kv_prefetch_status::ok) {
                result.readiness = llama_kv_prefetch_readiness::cancelled;
                return result;
            }
            for (const auto & prior : required) {
                if (&prior == &intent) break;
                if (prior.page_id == intent.page_id && prior.generation != intent.generation) {
                    result.readiness = llama_kv_prefetch_readiness::cancelled;
                    return result;
                }
            }
            if (std::find(required_ids.begin(), required_ids.end(), intent.page_id) == required_ids.end()) {
                required_ids.push_back(intent.page_id);
            }
            auto required_intent = intent;
            required_intent.required = true;
            const auto status = enqueue(required_intent);
            record_timeline(llama_kv_prefetch_timeline_kind::needed, intent);
            if (status == llama_kv_prefetch_status::shutdown) {
                result.readiness = llama_kv_prefetch_readiness::cancelled;
                return result;
            }
        }
        auto collect = [&]() {
            result.ready.clear();
            for (const auto & intent : required) {
                if (is_ready(intent) &&
                    std::find(result.ready.begin(), result.ready.end(), intent.page_id) == result.ready.end()) {
                    result.ready.push_back(intent.page_id);
                }
            }
            return result.ready.size() == required_ids.size();
        };
        if (collect()) {
            for (const auto & intent : required) {
                record_timeline(llama_kv_prefetch_timeline_kind::consumed, intent);
            }
            result.readiness = llama_kv_prefetch_readiness::ready;
            return result;
        }
        for (uint32_t step = 0; step < wait_budget_steps; ++step) {
            ++counters_.late_waits;
            for (const auto & intent : required) {
                record_timeline(llama_kv_prefetch_timeline_kind::wait, intent);
            }
            advance();
            if (collect()) {
                for (const auto & intent : required) {
                    if (std::find(result.ready.begin(), result.ready.end(), intent.page_id) !=
                        result.ready.end()) {
                        record_timeline(llama_kv_prefetch_timeline_kind::consumed, intent);
                    }
                }
                result.readiness = llama_kv_prefetch_readiness::waited_ready;
                return result;
            }
        }
        if (!previous_hot_set.empty()) {
            result.readiness = llama_kv_prefetch_readiness::reuse_old_hot_set;
            result.fallback = previous_hot_set;
        } else {
            result.readiness = llama_kv_prefetch_readiness::fallback_larger_union;
            result.fallback = required_ids;
        }
        return result;
    } catch (...) {
        result.readiness = llama_kv_prefetch_readiness::cancelled;
        result.ready.clear();
        result.fallback.clear();
        return result;
    }
}

llama_kv_prefetch_status llama_kv_prefetch_scheduler::evict(
        const llama_kv_prefetch_eviction & request) noexcept {
    if (stopped_) return llama_kv_prefetch_status::shutdown;
    if (validate_intent(request.page) != llama_kv_prefetch_status::ok) {
        return llama_kv_prefetch_status::invalid_argument;
    }
    if (request.dirty) {
        if (!backend_.reseal_dirty || !backend_.reseal_dirty(backend_.context, request.page)) {
            return llama_kv_prefetch_status::dirty_page;
        }
        ++counters_.reseals;
    }
    if (!backend_.evict_clean || !backend_.evict_clean(backend_.context, request.page)) {
        return llama_kv_prefetch_status::transfer_failed;
    }
    ++counters_.evictions;
    return llama_kv_prefetch_status::ok;
}

llama_kv_rerank_stage_ring::~llama_kv_rerank_stage_ring() {
    cancel();
}

bool llama_kv_rerank_stage_ring::configure(
        const std::array<void *, LLAMA_KV_RERANK_STAGE_SLOTS> & host,
        const std::array<void *, LLAMA_KV_RERANK_STAGE_SLOTS> & device,
        size_t slot_bytes, llama_kv_rerank_stage_backend backend) noexcept {
    if (busy_slots() != 0 || slot_bytes == 0 || slot_bytes > LLAMA_KV_RERANK_STAGE_MAX_BYTES ||
            (!backend.enqueue && !backend.enqueue_task) || !backend.poll ||
            !backend.cancel || !backend.release) return false;
    for (uint32_t i = 0; i < LLAMA_KV_RERANK_STAGE_SLOTS; ++i) {
        if (host[i] == nullptr || device[i] == nullptr) return false;
    }
    host_ = host;
    device_ = device;
    slot_bytes_ = slot_bytes;
    backend_ = backend;
    next_slot_ = 0;
    return true;
}

llama_kv_rerank_stage_status llama_kv_rerank_stage_ring::submit(
        const llama_kv_rerank_stage_source & source, uint64_t offset, size_t bytes,
        uint64_t query_generation, uint64_t content_version, uint32_t & slot_index) noexcept {
    llama_kv_rerank_chunk_task task;
    task.rows = 1;
    task.query_generation = query_generation;
    task.content_version = content_version;
    return submit(source, offset, bytes, task, slot_index, nullptr);
}

llama_kv_rerank_stage_status llama_kv_rerank_stage_ring::submit(
        const llama_kv_rerank_stage_source & source, uint64_t offset, size_t bytes,
        const llama_kv_rerank_chunk_task & task, uint32_t & slot_index,
        uint64_t * ticket_out) noexcept {
    if (slot_bytes_ == 0 || (!backend_.enqueue && !backend_.enqueue_task) ||
            source.page_holder == nullptr ||
            !source.recheck || !source.read || bytes == 0 || bytes > slot_bytes_ ||
            task.query_generation == 0 || task.content_version == 0 || task.rows == 0) {
        return slot_bytes_ == 0 ? llama_kv_rerank_stage_status::not_configured
                                : llama_kv_rerank_stage_status::invalid_argument;
    }
    uint32_t chosen = LLAMA_KV_RERANK_STAGE_SLOTS;
    for (uint32_t step = 0; step < LLAMA_KV_RERANK_STAGE_SLOTS; ++step) {
        const uint32_t i = (next_slot_ + step) % LLAMA_KV_RERANK_STAGE_SLOTS;
        if (!slots_[i].busy) { chosen = i; break; }
    }
    if (chosen == LLAMA_KV_RERANK_STAGE_SLOTS) return llama_kv_rerank_stage_status::backpressure;
    if (!source.recheck(source.context, task.content_version)) return llama_kv_rerank_stage_status::stale_source;
    if (!source.read(source.context, offset, host_[chosen], bytes)) return llama_kv_rerank_stage_status::read_failed;
    if (!source.recheck(source.context, task.content_version)) return llama_kv_rerank_stage_status::stale_source;
    // Publish ownership before enqueue: enqueue may launch a reader immediately.
    const uint64_t ticket = next_ticket_++;
    if (next_ticket_ == 0) next_ticket_ = 1;
    slots_[chosen].page_holder = source.page_holder;
    slots_[chosen].ticket = ticket;
    slots_[chosen].task = task;
    slots_[chosen].busy = true;
    uint64_t event = 0;
    const bool enqueued = backend_.enqueue_task
        ? backend_.enqueue_task(backend_.context, chosen, host_[chosen], device_[chosen], bytes,
                task, &event)
        : backend_.enqueue(backend_.context, chosen, host_[chosen], device_[chosen], bytes,
                task.query_generation, task.content_version, &event);
    if (!enqueued || event == 0) {
        if (event != 0) {
            if (backend_.cancel) backend_.cancel(backend_.context, event);
            if (backend_.release) backend_.release(backend_.context, event);
        }
        slots_[chosen] = {};
        return llama_kv_rerank_stage_status::enqueue_failed;
    }
    slots_[chosen].event = event;
    slot_index = chosen;
    if (ticket_out) *ticket_out = ticket;
    next_slot_ = (chosen + 1) % LLAMA_KV_RERANK_STAGE_SLOTS;
    return llama_kv_rerank_stage_status::ok;
}

uint32_t llama_kv_rerank_stage_ring::poll() noexcept {
    return poll_completed(nullptr, 0);
}

uint32_t llama_kv_rerank_stage_ring::poll_completed(
        llama_kv_rerank_stage_ticket * output, uint32_t capacity) noexcept {
    uint32_t retired = 0;
    while (cancelled_count_ != 0) {
        if (output != nullptr && retired < capacity) output[retired] = cancelled_[0];
        for (uint32_t i = 1; i < cancelled_count_; ++i) cancelled_[i - 1] = cancelled_[i];
        --cancelled_count_;
        ++retired;
    }
    for (auto & slot : slots_) {
        if (!slot.busy) continue;
        const auto status = backend_.poll(backend_.context, slot.event);
        if (status == llama_kv_prefetch_poll::pending) continue;
        if (output != nullptr && retired < capacity) {
            output[retired].ticket = slot.ticket;
            output[retired].slot = uint32_t(&slot - slots_.data());
            output[retired].task = slot.task;
            output[retired].terminal = status == llama_kv_prefetch_poll::completed
                ? llama_kv_rerank_stage_terminal::succeeded
                : status == llama_kv_prefetch_poll::failed
                    ? llama_kv_rerank_stage_terminal::failed
                    : llama_kv_rerank_stage_terminal::cancelled;
        }
        backend_.release(backend_.context, slot.event);
        slot = {};
        ++retired;
    }
    return retired;
}

uint32_t llama_kv_rerank_stage_ring::busy_slots() const noexcept {
    uint32_t busy = 0;
    for (const auto & slot : slots_) busy += slot.busy;
    return busy;
}

void llama_kv_rerank_stage_ring::cancel() noexcept {
    for (auto & slot : slots_) {
        if (!slot.busy) continue;
        if (backend_.cancel) backend_.cancel(backend_.context, slot.event);
        if (backend_.release) backend_.release(backend_.context, slot.event);
        if (cancelled_count_ < cancelled_.size()) {
            auto & result = cancelled_[cancelled_count_++];
            result.ticket = slot.ticket;
            result.slot = uint32_t(&slot - slots_.data());
            result.task = slot.task;
            result.terminal = llama_kv_rerank_stage_terminal::cancelled;
        }
        slot = {};
    }
}
