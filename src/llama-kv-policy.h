#pragma once

#include <cstdint>
#include <vector>

constexpr uint32_t LLAMA_KV_POLICY_TRACE_VERSION = 1;
constexpr uint32_t LLAMA_KV_POLICY_MAX_PAGES = 4096;
constexpr uint32_t LLAMA_KV_POLICY_MAX_SUMMARY = 256;
constexpr uint32_t LLAMA_KV_POLICY_DEFAULT_CAPACITY = 304;

enum class llama_kv_policy_status : uint8_t {
    ok = 0,
    pin_overflow,
    invalid_trace,
    unavailable,
    _count,
};

struct llama_kv_policy_page {
    uint64_t id = 0;
    uint64_t age = 0;
    uint64_t recency = 0;
    uint64_t attention_ema_q = 0;
    uint64_t retrieval_hits = 0;
    uint64_t dirty_cost = 0;
    uint64_t fault_cost = 0;
    uint64_t recent_peak_q = 0;
    uint64_t hysteresis_q = 0;
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
    uint32_t capacity_pages = LLAMA_KV_POLICY_DEFAULT_CAPACITY;
    uint32_t recent_pages = 96;
    uint32_t structural_pages = 32;
    uint32_t historical_pages = 140;
    uint32_t transient_pages = 36;
    uint64_t hysteresis_q = 0;
};

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
