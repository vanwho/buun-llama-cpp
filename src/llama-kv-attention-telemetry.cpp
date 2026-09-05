#include "llama-kv-attention-telemetry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

namespace {
bool valid_ratio(float value) noexcept {
    return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
}

bool stride_product_fits(size_t stride, uint32_t count) noexcept {
    return count == 0 || stride <= std::numeric_limits<size_t>::max() / count;
}

uint64_t elapsed_us(const std::chrono::steady_clock::time_point begin) noexcept {
    return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin).count());
}

void saturating_add(uint64_t & target, uint64_t value) noexcept {
    if (value > std::numeric_limits<uint64_t>::max() - target) {
        target = std::numeric_limits<uint64_t>::max();
    } else {
        target += value;
    }
}

uint64_t saturating_mul(uint64_t value, uint64_t factor) noexcept {
    if (factor != 0 && value > std::numeric_limits<uint64_t>::max() / factor) {
        return std::numeric_limits<uint64_t>::max();
    }
    return value * factor;
}

bool valid_state(llama_kv_page_state state) noexcept {
    return state != llama_kv_page_state::absent && state != llama_kv_page_state::invalid;
}
}

const char * llama_kv_attention_telemetry_status_name(
        llama_kv_attention_telemetry_status status) noexcept {
    switch (status) {
        case llama_kv_attention_telemetry_status::ok: return "ok";
        case llama_kv_attention_telemetry_status::disabled: return "disabled";
        case llama_kv_attention_telemetry_status::invalid_argument: return "invalid_argument";
        case llama_kv_attention_telemetry_status::stale_epoch: return "stale_epoch";
        case llama_kv_attention_telemetry_status::invalid_page: return "invalid_page";
        case llama_kv_attention_telemetry_status::sampling_skipped: return "sampling_skipped";
        case llama_kv_attention_telemetry_status::overflow: return "overflow";
    }
    return "invalid";
}

llama_kv_attention_telemetry::llama_kv_attention_telemetry(
        const llama_kv_attention_telemetry_config & config) noexcept :
        mode_(config.mode),
        logical_page_count_(config.logical_page_count),
        sample_interval_tokens_(config.sample_interval_tokens),
        layer_index_(config.layer_index),
        head_begin_(config.head_begin),
        head_count_(config.head_count),
        ema_alpha_(config.ema_alpha),
        peak_decay_(config.peak_decay) {
    if (logical_page_count_ == 0 || pages_.max_size() < logical_page_count_ ||
        sample_interval_tokens_ == 0 || !valid_ratio(ema_alpha_) || !valid_ratio(peak_decay_)) {
        mode_ = llama_kv_attention_telemetry_mode::off;
        logical_page_count_ = 0;
        return;
    }
    try {
        pages_.resize(logical_page_count_);
    } catch (...) {
        mode_ = llama_kv_attention_telemetry_mode::off;
        logical_page_count_ = 0;
        pages_.clear();
    }
}

bool llama_kv_attention_telemetry::valid_page(uint32_t logical_page) const noexcept {
    return logical_page < logical_page_count_;
}

bool llama_kv_attention_telemetry::snapshot_matches(
        const llama_kv_residency_snapshot & snapshot) const noexcept {
    if (snapshot.epoch() != table_epoch_) return false;
    for (uint32_t logical = 0; logical < logical_page_count_; ++logical) {
        const auto & stored = pages_[logical].value;
        if (!stored.known || !stored.resident) continue;
        const auto it = std::find_if(snapshot.pages().begin(), snapshot.pages().end(),
                [&](const auto & page) { return page.id.logical_page == logical; });
        if (it == snapshot.pages().end() || it->id != stored.id) return false;
    }
    return true;
}

llama_kv_attention_telemetry_status llama_kv_attention_telemetry::reject_stale() noexcept {
    ++counters_.stale_dropped;
    return llama_kv_attention_telemetry_status::stale_epoch;
}

llama_kv_attention_telemetry_status llama_kv_attention_telemetry::reject_invalid() noexcept {
    ++counters_.invalid_dropped;
    return llama_kv_attention_telemetry_status::invalid_argument;
}

llama_kv_attention_telemetry_status llama_kv_attention_telemetry::initialize(
        const llama_kv_residency_snapshot & snapshot) noexcept {
    return initialize(snapshot, snapshot.pages());
}

llama_kv_attention_telemetry_status llama_kv_attention_telemetry::initialize(
        const llama_kv_residency_snapshot & snapshot,
        const std::vector<llama_kv_page_record> & inventory) noexcept {
    if (mode_ == llama_kv_attention_telemetry_mode::off) {
        return llama_kv_attention_telemetry_status::disabled;
    }
    if (snapshot.epoch() == 0 || logical_page_count_ == 0 ||
        inventory.empty() || inventory.size() > pages_.max_size()) {
        return reject_invalid();
    }
    for (const auto & resident : snapshot.pages()) {
        const auto it = std::find_if(inventory.begin(), inventory.end(),
                [&](const auto & page) { return page.id == resident.id; });
        if (it == inventory.end()) return reject_invalid();
    }
    clear();
    table_epoch_ = snapshot.epoch();
    for (const auto & page : inventory) {
        const bool tail = page.state == llama_kv_page_state::filling_gpu;
        if (!valid_state(page.state) || !llama_kv_page_id_valid(page.id, tail) ||
            (page.physical_slot == UINT32_MAX &&
             (page.state != llama_kv_page_state::host_clean || !page.host_valid))) {
            clear();
            return reject_invalid();
        }
        if (!valid_page(page.id.logical_page) || pages_[page.id.logical_page].value.known) {
            clear();
            return reject_invalid();
        }
        auto & state = pages_[page.id.logical_page].value;
        state = {};
        state.known = true;
        state.id = page.id;
        state.resident = std::find_if(snapshot.pages().begin(), snapshot.pages().end(),
                [&](const auto & resident) { return resident.id == page.id; }) != snapshot.pages().end();
    }
    return llama_kv_attention_telemetry_status::ok;
}

llama_kv_attention_telemetry_status llama_kv_attention_telemetry::reconcile(
        const llama_kv_residency_snapshot & snapshot) noexcept {
    if (mode_ == llama_kv_attention_telemetry_mode::off) {
        return llama_kv_attention_telemetry_status::disabled;
    }
    if (snapshot.epoch() == 0 || logical_page_count_ == 0 ||
        snapshot.pages().size() > pages_.max_size()) {
        return reject_invalid();
    }
    // The graph construction seam only has the physical snapshot. Preserve
    // previously authenticated host-only identities here; the cache boundary
    // later calls the inventory overload to replace this conservative view
    // with the complete catalog.
    std::vector<llama_kv_page_record> inventory = snapshot.pages();
    try {
        for (const auto & old_slot : pages_) {
            const auto & old = old_slot.value;
            if (!old.known) continue;
            const bool logical_reused = std::find_if(snapshot.pages().begin(), snapshot.pages().end(),
                    [&](const auto & page) { return page.id.logical_page == old.id.logical_page; }) !=
                snapshot.pages().end();
            if (logical_reused) continue;
            llama_kv_page_record cold;
            cold.id = old.id;
            cold.physical_slot = UINT32_MAX;
            cold.state = llama_kv_page_state::host_clean;
            cold.host_valid = true;
            inventory.push_back(cold);
        }
    } catch (...) {
        return reject_invalid();
    }
    return reconcile(snapshot, inventory);
}

llama_kv_attention_telemetry_status llama_kv_attention_telemetry::reconcile(
        const llama_kv_residency_snapshot & snapshot,
        const std::vector<llama_kv_page_record> & inventory) noexcept {
    if (mode_ == llama_kv_attention_telemetry_mode::off) {
        return llama_kv_attention_telemetry_status::disabled;
    }
    if (snapshot.epoch() == 0 || logical_page_count_ == 0 ||
        inventory.empty() || inventory.size() > pages_.max_size()) {
        return reject_invalid();
    }
    for (const auto & resident : snapshot.pages()) {
        const auto it = std::find_if(inventory.begin(), inventory.end(),
                [&](const auto & page) { return page.id == resident.id; });
        if (it == inventory.end()) return reject_invalid();
    }

    std::vector<slot> next;
    try {
        next.resize(logical_page_count_);
    } catch (...) {
        return reject_invalid();
    }
    for (const auto & page : inventory) {
        const bool tail = page.state == llama_kv_page_state::filling_gpu;
        if (!valid_state(page.state) || !llama_kv_page_id_valid(page.id, tail) ||
            (page.physical_slot == UINT32_MAX &&
             (page.state != llama_kv_page_state::host_clean || !page.host_valid))) {
            return reject_invalid();
        }
        if (!valid_page(page.id.logical_page) || next[page.id.logical_page].value.known) {
            return reject_invalid();
        }
        auto & state = next[page.id.logical_page].value;
        state.known = true;
        state.id = page.id;
        state.resident = std::find_if(snapshot.pages().begin(), snapshot.pages().end(),
                [&](const auto & resident) { return resident.id == page.id; }) != snapshot.pages().end();
        const auto & old = pages_[page.id.logical_page].value;
        if (old.known && old.id == page.id) {
            state.observed = old.observed;
            state.normalized_ema = old.normalized_ema;
            state.recent_peak = old.recent_peak;
            state.sample_count = old.sample_count;
            state.frequency = old.frequency;
            state.last_observed_token = old.last_observed_token;
        }
    }
    pages_.swap(next);
    table_epoch_ = snapshot.epoch();
    return llama_kv_attention_telemetry_status::ok;
}

llama_kv_attention_telemetry_status llama_kv_attention_telemetry::publish_completed(
        const llama_kv_residency_snapshot & snapshot,
        const llama_kv_attention_telemetry_sample & sample) noexcept {
    if (mode_ == llama_kv_attention_telemetry_mode::off) {
        return llama_kv_attention_telemetry_status::disabled;
    }
    if (sample.token_count == 0 || sample.layer_count == 0 || sample.head_count == 0 ||
        sample.page_mass == nullptr || sample.pages == nullptr || sample.page_count == 0 ||
        sample.table_epoch != table_epoch_ || snapshot.epoch() != sample.table_epoch ||
        sample.token_index % sample_interval_tokens_ != 0 || !snapshot_matches(snapshot)) {
        if (sample.table_epoch != table_epoch_ || snapshot.epoch() != table_epoch_ ||
            !snapshot_matches(snapshot)) return reject_stale();
        if (sample.token_count != 0 && sample.token_index % sample_interval_tokens_ != 0) {
            saturating_add(counters_.skipped, 1);
            return llama_kv_attention_telemetry_status::sampling_skipped;
        }
        return reject_invalid();
    }
    if (sample.page_count > logical_page_count_ ||
        sample.head_stride_bytes < sizeof(float) * logical_page_count_ ||
        !stride_product_fits(sample.head_stride_bytes, sample.head_count) ||
        !stride_product_fits(sample.layer_stride_bytes, sample.layer_count) ||
        !stride_product_fits(sample.token_stride_bytes, sample.token_count) ||
        sample.layer_stride_bytes < sample.head_stride_bytes * sample.head_count ||
        sample.token_stride_bytes < sample.layer_stride_bytes * sample.layer_count) {
        return reject_invalid();
    }
    const auto publish_begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < sample.page_count; ++i) {
        const auto & page = sample.pages[i];
        if (!valid_page(page.id.logical_page) || !pages_[page.id.logical_page].value.known ||
            pages_[page.id.logical_page].value.id != page.id) return reject_stale();
        for (size_t j = 0; j < i; ++j) {
            if (sample.pages[j].id.logical_page == page.id.logical_page) return reject_invalid();
        }
    }

    // Validate the complete event result before touching any EMA. A failed or
    // stale graph therefore cannot publish a partial attention observation.
    for (uint32_t token = 0; token < sample.token_count; ++token) {
        const char * token_base = reinterpret_cast<const char *>(sample.page_mass) +
            size_t(token) * sample.token_stride_bytes;
        for (uint32_t layer = 0; layer < sample.layer_count; ++layer) {
            const char * layer_base = token_base + size_t(layer) * sample.layer_stride_bytes;
            for (uint32_t head = 0; head < sample.head_count; ++head) {
                const float * head_base = reinterpret_cast<const float *>(
                    layer_base + size_t(head) * sample.head_stride_bytes);
                for (size_t i = 0; i < sample.page_count; ++i) {
                    const float mass = head_base[sample.pages[i].id.logical_page];
                    if (!std::isfinite(mass) || mass < 0.0f || mass > 1.0f) return reject_invalid();
                }
            }
        }
    }

    for (size_t i = 0; i < sample.page_count; ++i) {
        const uint32_t logical = sample.pages[i].id.logical_page;
        double sum = 0.0;
        float peak = 0.0f;
        for (uint32_t token = 0; token < sample.token_count; ++token) {
            const char * token_base = reinterpret_cast<const char *>(sample.page_mass) +
                size_t(token) * sample.token_stride_bytes;
            for (uint32_t layer = 0; layer < sample.layer_count; ++layer) {
                const char * layer_base = token_base + size_t(layer) * sample.layer_stride_bytes;
                float layer_sum = 0.0f;
                for (uint32_t head = 0; head < sample.head_count; ++head) {
                    const float mass = reinterpret_cast<const float *>(
                        layer_base + size_t(head) * sample.head_stride_bytes)[logical];
                    layer_sum += mass;
                }
                const float layer_average = layer_sum / float(sample.head_count);
                sum += double(layer_average);
                peak = std::max(peak, layer_average);
            }
        }
        auto & state = pages_[logical].value;
        state.resident = true;
        const float normalized = float(sum / double(sample.token_count) / double(sample.layer_count));
        state.normalized_ema = state.observed
            ? ema_alpha_ * normalized + (1.0f - ema_alpha_) * state.normalized_ema
            : normalized;
        state.recent_peak = std::max(peak, peak_decay_ * state.recent_peak);
        state.observed = true;
        state.last_observed_token = sample.token_index;
        ++state.sample_count;
        if (state.frequency != std::numeric_limits<uint64_t>::max()) ++state.frequency;
    }
    saturating_add(counters_.samples, 1);
    saturating_add(counters_.sampled_tokens, sample.token_count);
    saturating_add(counters_.sampled_pages, sample.page_count);
    saturating_add(counters_.sampled_layers,
            saturating_mul(sample.token_count, sample.layer_count));
    saturating_add(counters_.sampled_heads,
            saturating_mul(saturating_mul(sample.token_count, sample.layer_count), sample.head_count));
    saturating_add(counters_.gpu_reduction_us, sample.gpu_reduction_us);
    saturating_add(counters_.d2h_bytes, sample.d2h_bytes);
    saturating_add(counters_.d2h_time_us, sample.d2h_time_us);
    saturating_add(counters_.publish_time_us, elapsed_us(publish_begin));
    return llama_kv_attention_telemetry_status::ok;
}

void llama_kv_attention_telemetry::clear() noexcept {
    for (auto & page : pages_) page.value = {};
    table_epoch_ = 0;
    counters_ = {};
}

void llama_kv_attention_telemetry::set_mode(
        llama_kv_attention_telemetry_mode mode) noexcept {
    if (mode_ != mode) clear();
    mode_ = mode;
}

void llama_kv_attention_telemetry::record_observe_overhead(uint64_t elapsed) noexcept {
    saturating_add(counters_.observe_overhead_us, elapsed);
}

void llama_kv_attention_telemetry::record_skipped_sample() noexcept {
    saturating_add(counters_.skipped, 1);
}

bool llama_kv_attention_telemetry::page_state(
        uint32_t logical_page, llama_kv_attention_telemetry_page & output) const noexcept {
    if (!valid_page(logical_page)) return false;
    output = pages_[logical_page].value;
    return output.known;
}

bool llama_kv_attention_telemetry::copy_scores(
        float * ema, float * peak, uint32_t count) const noexcept {
    if (ema == nullptr || peak == nullptr || count != logical_page_count_) return false;
    for (uint32_t logical = 0; logical < count; ++logical) {
        ema[logical] = pages_[logical].value.observed ? pages_[logical].value.normalized_ema : 0.0f;
        peak[logical] = pages_[logical].value.observed ? pages_[logical].value.recent_peak : 0.0f;
    }
    return true;
}

llama_kv_attention_telemetry_accounting llama_kv_attention_telemetry::accounting() const noexcept {
    llama_kv_attention_telemetry_accounting result;
    result.logical_page_count = logical_page_count_;
    result.ema_bytes = saturating_mul(logical_page_count_, sizeof(float));
    result.peak_bytes = saturating_mul(logical_page_count_, sizeof(float));
    result.controller_score_bytes = result.ema_bytes >
            std::numeric_limits<uint64_t>::max() - result.peak_bytes
        ? std::numeric_limits<uint64_t>::max()
        : result.ema_bytes + result.peak_bytes;
    result.metadata_bytes = saturating_mul(logical_page_count_, sizeof(llama_kv_page_id));
    result.device_capture_bytes = saturating_mul(logical_page_count_, sizeof(float));
    return result;
}
