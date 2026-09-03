#pragma once

#include "llama-kv-routing-summary.h"

#include <cstdint>
#include <vector>

constexpr uint32_t LLAMA_KV_ROUTING_RETRIEVAL_VERSION = 1;

enum class llama_kv_routing_retrieval_status : uint8_t {
    ok = 0,
    invalid_argument,
    stale_query,
    stale_summary,
    unavailable_summary,
    mandatory_overflow,
    overflow,
};

const char * llama_kv_routing_retrieval_status_name(
        llama_kv_routing_retrieval_status status) noexcept;

// The query is the pre-attention representation for one token and one
// summary layer/head.  It is deliberately tagged independently of the
// summary payload so a query from an older table or representation cannot
// select pages from the current route.
struct llama_kv_routing_query {
    std::vector<float> values;
    uint64_t query_generation = 0;
    uint64_t token_index = 0;
    uint64_t table_epoch = 0;
    uint64_t model_identity = 0;
    uint64_t topology_identity = 0;
    uint64_t representation_epoch = 0;
    uint64_t session_generation = 0;
    uint64_t sequence_generation = 0;
    int32_t sequence_id = -1;
    llama_pos position = -1;
    uint32_t layer_index = 0;
    uint32_t head_index = 0;
};

// Page attributes are supplied by the controller/application at the decode
// boundary. The page id is copied from the same residency snapshot passed to
// retrieve; no logical page number is resolved without its full identity.
struct llama_kv_routing_page_attributes {
    llama_kv_page_id id;
    bool current = false;
    bool mandatory = false;
    bool structural = false;
    bool recent = false;
    bool application_pin = false;
    bool inflight_pin = false;
    bool speculative_pin = false;
};

struct llama_kv_routing_retrieval_config {
    uint32_t capacity_pages = 0;       // resolved hot capacity H
    uint32_t summary_top_k = 0;
    uint32_t exploration_pages = 0;
    uint64_t exploration_seed = 0;
    uint64_t exploration_turn = 0;
};

enum class llama_kv_routing_retrieval_reason : uint8_t {
    mandatory = 0,
    structural,
    recent,
    summary,
    exploration,
    fallback,
};

struct llama_kv_routing_retrieval_entry {
    llama_kv_page_id id;
    llama_kv_routing_retrieval_reason reason =
        llama_kv_routing_retrieval_reason::fallback;
    float score = 0.0f;
    bool score_available = false;
    bool upper_bound = false;
    uint64_t page_distance = 0;
};

struct llama_kv_routing_retrieval_metrics {
    uint64_t valid_pages = 0;
    uint64_t summary_pages_scored = 0;
    uint64_t summary_comparisons = 0;
    uint64_t summary_bytes = 0;
    uint64_t selected_pages = 0;
    uint64_t mandatory_pages = 0;
    uint64_t exploration_pages = 0;
    uint64_t score_time_us = 0;
    uint64_t union_time_us = 0;
    uint64_t total_time_us = 0;
    bool summary_complete = false;
    bool fallback_used = false;
};

struct llama_kv_routing_retrieval_result {
    llama_kv_routing_retrieval_status status =
        llama_kv_routing_retrieval_status::invalid_argument;
    uint32_t version = LLAMA_KV_ROUTING_RETRIEVAL_VERSION;
    uint64_t table_epoch = 0;
    uint64_t query_generation = 0;
    uint64_t token_index = 0;
    uint64_t model_identity = 0;
    uint64_t topology_identity = 0;
    uint64_t representation_epoch = 0;
    uint64_t session_generation = 0;
    uint64_t sequence_generation = 0;
    int32_t sequence_id = -1;
    llama_pos position = -1;
    uint32_t layer_index = 0;
    uint32_t head_index = 0;
    std::vector<llama_kv_routing_retrieval_entry> selected;
    llama_kv_routing_retrieval_metrics metrics;
};

// All-page query scoring and bounded candidate union. The input page list is
// validated against the snapshot before any score is accepted. Structural
// and current pages are inserted first, followed by recent pages, summary
// top-K, and a deterministic rotating exploration slice.
llama_kv_routing_retrieval_result llama_kv_routing_retrieve(
        const llama_kv_residency_snapshot & snapshot,
        const llama_kv_routing_summary_store & summaries,
        const llama_kv_routing_query & query,
        const llama_kv_routing_retrieval_config & config,
        const std::vector<llama_kv_routing_page_attributes> & attributes,
        const std::vector<llama_kv_page_id> & previous_target = {}) noexcept;

const char * llama_kv_routing_retrieval_reason_name(
        llama_kv_routing_retrieval_reason reason) noexcept;
