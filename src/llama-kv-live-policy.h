#pragma once

#include "llama-kv-policy.h"
#include "llama-kv-routing-retrieval.h"
#include "llama-kv-residency-transaction.h"

#include <cstdint>
#include <vector>

constexpr uint32_t LLAMA_KV_LIVE_POLICY_VERSION = 1;

enum class llama_kv_live_policy_status : uint8_t {
    committed = 0,
    no_change,
    safe_fallback,
    not_configured,
    invalid_argument,
    stale_snapshot,
    unavailable_inventory,
    mandatory_overflow,
    missing_host_source,
    all_pinned,
    dirty_victim,
    transaction_failed,
    _count,
};

const char * llama_kv_live_policy_status_name(
        llama_kv_live_policy_status status) noexcept;

// One immutable logical-page description at a decode boundary. `record` is
// copied from either the current residency table or the canonical host catalog;
// a cold record has UINT32_MAX as its physical slot. The numeric fields are
// deliberately boundary inputs rather than policy-owned history.
struct llama_kv_live_policy_page {
    llama_kv_page_record record;
    uint32_t attention_layer = 0;
    uint64_t age = 0;
    uint64_t recency = 0;
    uint64_t fault_cost = 0;
    uint64_t dirty_cost = 0;
    uint64_t recent_peak_q = 0;
    uint64_t retrieval_hits = 0;
    uint64_t reuse_count = 0;
    uint64_t hysteresis_q = 0;
    uint64_t attention_sample_count = 0;
    uint64_t attention_last_observed = 0;
    bool attention_observed = false;
    uint64_t attention_ema_q = 0;
    bool recent = false;
    bool anchor = false;
    bool structural = false;
    bool current = false;
    bool application_pin = false;
    bool inflight_pin = false;
    bool speculative_pin = false;
};

// The caller supplies the complete logical inventory for the sequence. This
// is the trace boundary: residency is immutable at `snapshot.epoch()`, while
// `pages` may also contain authenticated host-only pages for cold promotion.
// `retrieval` is query evidence and is never folded into retention scores.
struct llama_kv_live_policy_boundary {
    uint32_t version = LLAMA_KV_LIVE_POLICY_VERSION;
    llama_kv_residency_snapshot snapshot;
    uint32_t hot_capacity = 0;       // runtime-admitted H
    uint32_t logical_page_count = 0; // runtime-resolved L
    llama_kv_page_id write_page;
    bool has_write_page = false;
    std::vector<llama_kv_live_policy_page> pages;
    llama_kv_routing_retrieval_result retrieval;
    std::vector<llama_kv_page_id> previous_target;

    llama_kv_policy_controller_config policy;

    // The transfer owner prepares bounded plans for the target identities.
    // The live controller supplies desired_pages after deterministic slot
    // assignment and leaves all transfer bytes opaque to policy.
    llama_kv_residency_transaction_request transaction;
};

struct llama_kv_live_policy_trace_page {
    llama_kv_page_id id;
    uint32_t attention_layer = 0;
    bool resident = false;
    bool host_valid = false;
    uint32_t policy_id = 0;
    float retrieval_score = 0.0f;
    bool retrieval_score_available = false;
};

struct llama_kv_live_policy_trace {
    uint32_t version = LLAMA_KV_LIVE_POLICY_VERSION;
    uint64_t epoch = 0;
    uint32_t hot_capacity = 0;
    uint32_t logical_page_count = 0;
    uint32_t mandatory_pages = 0;
    uint32_t target_pages = 0;
    uint32_t unavailable_attention_pages = 0;
    bool retrieval_fallback = false;
    std::vector<llama_kv_live_policy_trace_page> pages;
};

struct llama_kv_live_policy_decision_entry {
    llama_kv_page_id id;
    llama_kv_policy_reason selection_reason = llama_kv_policy_reason::unavailable;
    uint64_t retention_score_q = 0;
    float retrieval_score = 0.0f;
    bool retrieval_score_available = false;
    bool added = false;
    bool kept = false;
    bool victim = false;
};

struct llama_kv_live_policy_result {
    llama_kv_live_policy_status status = llama_kv_live_policy_status::invalid_argument;
    uint32_t version = LLAMA_KV_LIVE_POLICY_VERSION;
    uint64_t base_epoch = 0;
    uint64_t published_epoch = 0;
    bool publication_attempted = false;
    bool published = false;
    bool retrieval_fallback = false;
    llama_kv_live_policy_trace trace;
    llama_kv_policy_decision policy;
    std::vector<llama_kv_page_record> target_pages;
    std::vector<llama_kv_live_policy_decision_entry> decisions;
    llama_kv_residency_transaction_result transaction;
};

// Build and validate the one-boundary trace without touching a table, pool, or
// backend. `output` contains one stable policy ordinal per full page identity.
bool llama_kv_live_policy_build_trace(
        const llama_kv_live_policy_boundary & boundary,
        llama_kv_live_policy_trace & output,
        llama_kv_policy_trace & policy_trace,
        std::vector<llama_kv_policy_page> & policy_pages) noexcept;

// Apply one complete target. A successful transfer/event sequence is the only
// path to table publication; every failure leaves the previous graph-usable
// table authoritative through the transaction rollback contract.
llama_kv_live_policy_result llama_kv_live_policy_apply(
        llama_kv_residency_table & table,
        llama_kv_residency_pool & pool,
        const llama_kv_live_policy_boundary & boundary,
        const llama_kv_residency_pool_backend & backend,
        const llama_kv_residency_transfer_transport & transport,
        const llama_kv_residency_transaction_hooks & hooks = {}) noexcept;
