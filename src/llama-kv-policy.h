#pragma once

#include <cstdint>
#include <vector>

constexpr uint32_t LLAMA_KV_POLICY_TRACE_VERSION = 1;
constexpr uint32_t LLAMA_KV_POLICY_RATIO_SCALE = 1000000;
constexpr uint32_t LLAMA_KV_POLICY_WEIGHT_SCALE = 1000000;

enum class llama_kv_policy_status : uint8_t {
    ok = 0,
    pin_overflow,
    invalid_trace,
    unavailable,
    _count,
};

struct llama_kv_policy_page {
    uint64_t id = 0;
    uint32_t attention_layer = 0;
    uint64_t age = 0;
    uint64_t recency = 0;
    uint64_t attention_ema_q = 0;
    uint64_t retrieval_hits = 0;
    uint64_t reuse_count = 0;
    uint64_t dirty_cost = 0;
    uint64_t fault_cost = 0;
    uint64_t recent_peak_q = 0;
    uint64_t hysteresis_q = 0;
    uint64_t attention_sample_count = 0;
    uint64_t attention_last_observed = 0;
    bool attention_observed = false;
    bool recent = false;
    bool anchor = false;
    bool structural = false;
    bool current = false;
    bool application_pin = false;
    bool inflight_pin = false;
    bool speculative_pin = false;
    bool resident = false;
};

struct llama_kv_policy_trace {
    uint32_t version = LLAMA_KV_POLICY_TRACE_VERSION;
    uint32_t capacity_pages = 0;
    uint64_t epoch = 0;
    uint64_t write_page = 0;
    std::vector<llama_kv_policy_page> pages;
    std::vector<uint64_t> summary_top_k;
    std::vector<uint64_t> exploration;
};

struct llama_kv_policy_result {
    llama_kv_policy_status status = llama_kv_policy_status::unavailable;
    uint32_t version = LLAMA_KV_POLICY_TRACE_VERSION;
    std::vector<uint64_t> retrieve;
    std::vector<uint64_t> victims;
};

enum class llama_kv_policy_reason : uint8_t {
    mandatory = 0,
    recent,
    structural,
    summary,
    exploration,
    retention,
    keep,
    victim,
    unavailable,
};

struct llama_kv_policy_decision_entry {
    uint64_t id = 0;
    llama_kv_policy_reason reason = llama_kv_policy_reason::unavailable;
    uint64_t normalized_keep_q = 0;
};

struct llama_kv_policy_controller_config {
    // Capacity is supplied by runtime admission; zero is not configured.
    uint32_t capacity_pages = 0;

    // Nonzero page counts are explicit overrides. Zero selects the corresponding
    // capacity-relative ratio, so no preferred context size is encoded here.
    uint32_t recent_pages = 0;
    uint32_t structural_pages = 0;
    uint32_t historical_pages = 0;
    uint32_t transient_pages = 0;

    uint32_t recent_ratio = 400000;
    uint32_t structural_ratio = 100000;
    uint32_t historical_ratio = 350000;
    uint32_t transient_ratio = 150000;

    uint32_t recent_min_pages = 1;
    uint32_t structural_min_pages = 0;
    uint32_t historical_min_pages = 0;
    uint32_t transient_min_pages = 1;

    // Normalized retention evidence weights.  The release candidate gives
    // completed attention EMA the most influence, while retaining peak,
    // frequency, and recency as independent recovery signals.
    uint32_t attention_ema_weight = 500000;
    uint32_t recent_peak_weight = 200000;
    uint32_t frequency_weight = 150000;
    uint32_t recency_weight = 150000;
    uint64_t hysteresis_q = 100000;
};

// Return the phase-13 release candidate. `capacity_pages` is supplied by
// runtime admission and is intentionally the only runtime-sized field here.
llama_kv_policy_controller_config llama_kv_policy_release_defaults(
        uint32_t capacity_pages = 0) noexcept;

struct llama_kv_policy_decision {
    llama_kv_policy_status status = llama_kv_policy_status::unavailable;
    uint32_t version = LLAMA_KV_POLICY_TRACE_VERSION;
    uint64_t epoch = 0;
    std::vector<uint64_t> target;
    std::vector<uint64_t> adds;
    std::vector<uint64_t> keeps;
    std::vector<uint64_t> victims;
    std::vector<llama_kv_policy_decision_entry> records;
    uint32_t unavailable_evidence = 0;
};

// Decode-boundary policy. It only computes a bounded target and records; it
// does not transfer pages or call a graph/backend. `previous_target` is the
// target published at the preceding boundary and is never mutated.
llama_kv_policy_decision llama_kv_policy_decide(
        const llama_kv_policy_trace & trace,
        const llama_kv_policy_controller_config & config,
        const std::vector<uint64_t> & previous_target = {}) noexcept;

// Pure, deterministic replay. The input is never modified and the output contains a
// deduplicated retrieval union followed by the coldest evictable resident pages.
llama_kv_policy_result llama_kv_policy_replay(
        const llama_kv_policy_trace & trace) noexcept;

const char * llama_kv_policy_status_name(llama_kv_policy_status status) noexcept;
