#pragma once

#include "llama-kv-routing-summary.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// Owned query probes survive graph allocator reuse and stream fences.
struct llama_kv_router_probes {
    std::vector<int32_t> positions;
    std::vector<uint8_t> valid;
    std::vector<float> values;
    uint32_t query_heads = 0;
    uint32_t head_dim = 0;
};

// Contains authenticated scalar identity only; it never retains pointers into
// a summary catalogue, graph arena, or recyclable transfer descriptor.
struct llama_kv_router_shortlist_page {
    int32_t logical_page = -1;
    bool authenticated = false;
    uint32_t page_generation = 0;
    uint64_t content_version = 0;
    uint64_t summary_version = 0;
    uint32_t validity_flags = 0;
};

struct llama_kv_router_shortlist {
    uint64_t turn_generation = 0;
    uint64_t query_generation = 0;
    uint64_t content_generation = 0;
    std::vector<llama_kv_router_shortlist_page> pages;
};

// Stable streaming log-sum-exp accumulator for one page/head/probe tile.
struct llama_kv_router_rerank_state {
    float maximum = -std::numeric_limits<float>::infinity();
    float scaled_sum = 0.0f;
    uint32_t valid_rows = 0;
};

using llama_kv_router_page_probability = ggml_kv_page_probability_record;

inline bool llama_kv_router_probability_record_valid(
        const llama_kv_router_page_probability & record) noexcept {
    return record.logical_page < 0 ||
            (std::isfinite(record.peak_probability) &&
             std::isfinite(record.mean_probability));
}

enum class llama_kv_router_owner_stage : uint8_t {
    idle = 0,
    coarse_pending,
    keys_pending,
    rerank_pending,
    promotion_pending,
    ready,
    failed,
    cancelled,
};

struct llama_kv_router_identity {
    uint64_t turn_generation = 0;
    uint64_t query_generation = 0;
    uint64_t content_generation = 0;
};

struct llama_kv_router_owner_state {
    llama_kv_router_owner_stage stage = llama_kv_router_owner_stage::idle;
    llama_kv_router_identity identity;
    llama_kv_router_probes probes;
    llama_kv_router_shortlist shortlist;
    std::vector<llama_kv_router_rerank_state> rerank;
    std::vector<llama_kv_router_page_probability> probabilities;
    llama_kv_router_allocation_ledger allocations;
};
