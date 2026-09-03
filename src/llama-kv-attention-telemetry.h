#pragma once

#include "llama-kv-residency.h"

#include <cstddef>
#include <cstdint>
#include <vector>

enum class llama_kv_attention_telemetry_mode : uint8_t {
    off = 0,
    observe,
    selective,
};

enum class llama_kv_attention_telemetry_status : uint8_t {
    ok = 0,
    disabled,
    invalid_argument,
    stale_epoch,
    invalid_page,
    sampling_skipped,
    overflow,
};

const char * llama_kv_attention_telemetry_status_name(
        llama_kv_attention_telemetry_status status) noexcept;

struct llama_kv_attention_telemetry_config {
    llama_kv_attention_telemetry_mode mode = llama_kv_attention_telemetry_mode::off;
    uint32_t logical_page_count = 0;
    uint32_t sample_interval_tokens = 1;
    float ema_alpha = 0.25f;
    float peak_decay = 0.90f;
};

// `page_mass` is a completed GPU reduction, laid out as
// [token][layer][head][logical_page].  The caller invokes publish_completed
// only after the graph/event that produced the buffer has completed.
struct llama_kv_attention_telemetry_sample {
    uint64_t table_epoch = 0;
    uint64_t token_index = 0;
    uint32_t token_count = 1;
    uint32_t layer_count = 1;
    uint32_t head_count = 1;
    size_t head_stride_bytes = 0;
    size_t layer_stride_bytes = 0;
    size_t token_stride_bytes = 0;
    const float * page_mass = nullptr;

    // In selective mode this is the selected page subset. In observe mode it
    // is normally the complete dense page list. The full page id is required
    // so a reused logical slot cannot update an older generation's EMA.
    const llama_kv_page_record * pages = nullptr;
    size_t page_count = 0;
};

struct llama_kv_attention_telemetry_page {
    bool known = false;
    bool observed = false;
    llama_kv_page_id id;
    float normalized_ema = 0.0f;
    float recent_peak = 0.0f;
    uint64_t sample_count = 0;
};

struct llama_kv_attention_telemetry_counters {
    uint64_t sampled_tokens = 0;
    uint64_t sampled_pages = 0;
    uint64_t stale_dropped = 0;
    uint64_t invalid_dropped = 0;
};

struct llama_kv_attention_telemetry_accounting {
    uint32_t logical_page_count = 0;
    uint64_t ema_bytes = 0;
    uint64_t peak_bytes = 0;
    uint64_t controller_score_bytes = 0;
    uint64_t metadata_bytes = 0;
};

class llama_kv_attention_telemetry {
public:
    explicit llama_kv_attention_telemetry(
            const llama_kv_attention_telemetry_config & config = {}) noexcept;

    llama_kv_attention_telemetry_status initialize(
            const llama_kv_residency_snapshot & snapshot) noexcept;
    llama_kv_attention_telemetry_status publish_completed(
            const llama_kv_residency_snapshot & snapshot,
            const llama_kv_attention_telemetry_sample & sample) noexcept;

    void clear() noexcept;
    void set_mode(llama_kv_attention_telemetry_mode mode) noexcept;
    llama_kv_attention_telemetry_mode mode() const noexcept { return mode_; }
    uint64_t table_epoch() const noexcept { return table_epoch_; }

    bool page_state(uint32_t logical_page,
                    llama_kv_attention_telemetry_page & output) const noexcept;
    // Copy only the bounded controller vectors. `count` is normally the
    // configured logical page count and never follows the logical token tail.
    bool copy_scores(float * ema, float * peak, uint32_t count) const noexcept;

    const llama_kv_attention_telemetry_counters & counters() const noexcept { return counters_; }
    llama_kv_attention_telemetry_accounting accounting() const noexcept;

private:
    struct slot {
        llama_kv_attention_telemetry_page value;
    };

    bool valid_page(uint32_t logical_page) const noexcept;
    bool snapshot_matches(const llama_kv_residency_snapshot & snapshot) const noexcept;
    llama_kv_attention_telemetry_status reject_stale() noexcept;
    llama_kv_attention_telemetry_status reject_invalid() noexcept;

    llama_kv_attention_telemetry_mode mode_;
    uint32_t logical_page_count_ = 0;
    uint32_t sample_interval_tokens_ = 1;
    float ema_alpha_ = 0.25f;
    float peak_decay_ = 0.90f;
    uint64_t table_epoch_ = 0;
    // Sized from the resolved logical page count. The caller owns the bound;
    // no model-family context or page-count constant is encoded here.
    std::vector<slot> pages_;
    llama_kv_attention_telemetry_counters counters_;
};
