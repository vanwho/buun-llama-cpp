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
    uint32_t page_size = 256;
    llama_kv_pager_auto_size vram_budget;
    llama_kv_pager_auto_size host_budget;
    llama_kv_pager_auto_size safety_headroom;
    llama_kv_pager_auto_count pin_recent;
    std::string hotset_policy = "attention";
    llama_kv_pager_auto_count hot_pages;
    uint32_t router_top_k = 0;
    uint32_t router_explore = 0;
    uint32_t prefetch_depth = 0;
    bool debug = false;
    bool telemetry = true;

    bool enabled() const noexcept { return mode != llama_kv_pager_mode::off; }
    bool validate(std::string & error) const;
    std::string mode_name() const;
    std::string summary() const;
};

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
