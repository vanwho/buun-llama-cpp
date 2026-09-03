#pragma once

#include <cstdint>
#include <vector>

constexpr uint32_t LLAMA_KV_POLICY_TRACE_VERSION = 1;
constexpr uint32_t LLAMA_KV_POLICY_MAX_PAGES = 4096;
constexpr uint32_t LLAMA_KV_POLICY_MAX_SUMMARY = 256;

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
    bool attention_observed = false;
    bool recent = false;
    bool anchor = false;
    bool application_pin = false;
    bool inflight_pin = false;
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

// Pure, deterministic replay. The input is never modified and the output contains a
// deduplicated retrieval union followed by the coldest evictable resident pages.
llama_kv_policy_result llama_kv_policy_replay(
        const llama_kv_policy_trace & trace) noexcept;

const char * llama_kv_policy_status_name(llama_kv_policy_status status) noexcept;
