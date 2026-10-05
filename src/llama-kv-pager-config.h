#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class llama_kv_pager_mode : uint8_t {
    off = 0,
    observe,
    selective,
    exact,
};

enum class llama_kv_retrieval_policy : uint8_t {
    turn = 0,
    cadence,
};

enum class llama_kv_router_mode : uint8_t {
    legacy = 0,
    probe_rerank,
};

struct llama_kv_pager_auto_size {
    bool automatic = true;
    uint64_t bytes = 0;
};

struct llama_kv_pager_auto_count {
    bool automatic = true;
    uint32_t value = 0;
};

// This is the single normalized pager configuration shared by the common
// parser, server, and target context.  It deliberately contains no runtime
// allocations or backend handles.
struct llama_kv_pager_config {
    llama_kv_pager_mode mode = llama_kv_pager_mode::off;
    llama_kv_router_mode router = llama_kv_router_mode::legacy;
    uint32_t page_size = 256;
    llama_kv_pager_auto_count generation_tail_tokens;
    llama_kv_pager_auto_count retrieval_pages;
    llama_kv_retrieval_policy retrieval_policy = llama_kv_retrieval_policy::turn;
    llama_kv_pager_auto_size vram_budget;
    llama_kv_pager_auto_size host_budget;
    llama_kv_pager_auto_size safety_headroom;
    llama_kv_pager_auto_count pin_recent;
    std::string hotset_policy = "attention";
    llama_kv_pager_auto_count hot_pages;
    // Release-candidate routing defaults. Capacity and page count remain
    // runtime-derived; these values only bound evidence and lookahead.
    uint32_t router_top_k = 8;
    // Refresh the GPU routing catalogue after this many accepted target
    // tokens.  This is a policy cadence, not a graph/submission ID.
    uint32_t router_refresh_tokens = 8;
    // Per-layer rows attended by selective attention, inclusive of current,
    // recent and sink rows. Zero lets the runtime derive a bounded value from
    // the admitted hot-page capacity.
    uint32_t attention_tokens = 0;
    uint32_t router_explore = 2;
    uint32_t prefetch_depth = 2;
    bool debug = false;
    bool telemetry = true;
    uint32_t telemetry_interval_tokens = 4;
    uint32_t telemetry_layer = 0;
    uint32_t telemetry_head_begin = 0;
    uint32_t telemetry_head_count = 0;
    // Opt-in internal test seam. The model pressure driver may request one
    // known logical page so promotion mechanics can be proved independently
    // of attention-score quality; no command-line production path sets it.
    uint32_t test_force_logical_page = UINT32_MAX;

    bool enabled() const noexcept { return mode != llama_kv_pager_mode::off; }
    bool validate(std::string & error) const;
    std::string mode_name() const;
    std::string summary() const;
};

// Request cadence is evaluated against the committed target-token watermark,
// while the submission ID remains an independent graph/mailbox identity.
bool llama_kv_pager_refresh_due(
        uint64_t submission_id, uint64_t accepted_tokens,
        uint64_t refresh_watermark, bool policy_dirty,
        uint32_t cadence) noexcept;

enum class llama_kv_pager_capability_reason : uint8_t {
    ok = 0,
    backend,
    model_architecture,
    non_causal,
    cache_type,
    attention_geometry,
    page_geometry,
    device_topology,
    sequence_layout,
    host_budget,
    mtp,
    conflicting_vbr,
};

struct llama_kv_pager_capability_result {
    bool supported = false;
    std::vector<llama_kv_pager_capability_reason> reasons;
    std::string diagnostic;
};

const char * llama_kv_pager_capability_reason_name(llama_kv_pager_capability_reason reason) noexcept;
llama_kv_pager_capability_result llama_kv_pager_evaluate_capability(
        const llama_kv_pager_config & config,
        bool backend,
        bool model_architecture,
        bool causal,
        bool turbo4_kv,
        bool attention_geometry,
        bool page_geometry,
        bool one_device,
        bool sequence_layout,
        bool host_budget,
        bool mtp,
        bool conflicting_vbr);
bool llama_kv_pager_parse_size(const std::string & raw, llama_kv_pager_auto_size & out);
bool llama_kv_pager_parse_count(const std::string & raw, llama_kv_pager_auto_count & out);
bool llama_kv_pager_parse_mode(const std::string & raw, llama_kv_pager_mode & out);
bool llama_kv_pager_parse_retrieval_policy(const std::string & raw, llama_kv_retrieval_policy & out);
bool llama_kv_router_parse_mode(const std::string & raw, llama_kv_router_mode & out);

struct llama_kv_pager_turn_geometry {
    uint32_t hot_pages = 0;
    uint32_t retrieval_pages = 0;
    uint32_t generation_pages = 0;
    uint32_t generation_tokens = 0;
};

bool llama_kv_pager_derive_turn_geometry(
        uint32_t hot_pages, uint32_t page_tokens,
        uint32_t generation_tail_tokens, uint32_t mandatory_anchor_pages,
        uint32_t mandatory_slack_pages, llama_kv_pager_turn_geometry & output,
        uint32_t requested_retrieval_pages = UINT32_MAX) noexcept;
uint32_t llama_kv_pager_generation_available_pages(
        const llama_kv_pager_turn_geometry & geometry,
        uint32_t selected_history_pages) noexcept;
