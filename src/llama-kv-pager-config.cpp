#include "llama-kv-pager-config.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>

static std::string pager_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return value;
}

bool llama_kv_pager_refresh_due(
        uint64_t submission_id, uint64_t accepted_tokens,
        uint64_t refresh_watermark, bool policy_dirty,
        uint32_t cadence) noexcept {
    return submission_id == 1 || policy_dirty ||
        (accepted_tokens >= refresh_watermark &&
         accepted_tokens - refresh_watermark >= std::max<uint64_t>(1, cadence));
}

bool llama_kv_pager_parse_size(const std::string & raw, llama_kv_pager_auto_size & out) {
    const std::string s = pager_lower(raw);
    if (s == "auto") { out = {}; return true; }
    if (s.empty()) return false;
    char * end = nullptr;
    errno = 0;
    const double value = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || errno == ERANGE || !std::isfinite(value) || value <= 0) return false;
    const std::string suffix = end;
    double multiplier = 1;
    if (suffix == "b") multiplier = 1;
    else if (suffix == "k" || suffix == "kb" || suffix == "kib") multiplier = 1024;
    else if (suffix == "m" || suffix == "mb" || suffix == "mib") multiplier = 1024 * 1024;
    else if (suffix == "g" || suffix == "gb" || suffix == "gib") multiplier = 1024 * 1024 * 1024;
    else return false;
    const double bytes = value * multiplier;
    if (!std::isfinite(bytes) || bytes > double(std::numeric_limits<uint64_t>::max())) return false;
    out.automatic = false;
    out.bytes = uint64_t(bytes);
    return out.bytes != 0;
}

bool llama_kv_pager_parse_count(const std::string & raw, llama_kv_pager_auto_count & out) {
    if (pager_lower(raw) == "auto") { out = {}; return true; }
    if (raw.empty() || raw[0] == '-') return false;
    char * end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(raw.c_str(), &end, 10);
    if (end == raw.c_str() || *end || errno == ERANGE || value > std::numeric_limits<uint32_t>::max()) return false;
    out.automatic = false;
    out.value = uint32_t(value);
    return true;
}

bool llama_kv_pager_parse_mode(const std::string & raw, llama_kv_pager_mode & out) {
    const std::string s = pager_lower(raw);
    if (s == "off") out = llama_kv_pager_mode::off;
    else if (s == "observe") out = llama_kv_pager_mode::observe;
    else if (s == "selective") out = llama_kv_pager_mode::selective;
    else if (s == "exact") out = llama_kv_pager_mode::exact;
    else return false;
    return true;
}

bool llama_kv_pager_parse_retrieval_policy(
        const std::string & raw, llama_kv_retrieval_policy & out) {
    const std::string s = pager_lower(raw);
    if (s == "turn") out = llama_kv_retrieval_policy::turn;
    else if (s == "cadence") out = llama_kv_retrieval_policy::cadence;
    else return false;
    return true;
}

bool llama_kv_router_parse_mode(const std::string & raw, llama_kv_router_mode & out) {
    if (raw == "legacy") {
        out = llama_kv_router_mode::legacy;
        return true;
    }
    if (raw == "probe-rerank") {
        out = llama_kv_router_mode::probe_rerank;
        return true;
    }
    return false;
}

bool llama_kv_pager_derive_turn_geometry(
        uint32_t hot_pages, uint32_t page_tokens,
        uint32_t generation_tail_tokens, uint32_t mandatory_anchor_pages,
        uint32_t mandatory_slack_pages, llama_kv_pager_turn_geometry & output,
        uint32_t requested_retrieval_pages) noexcept {
    output = {};
    if (hot_pages == 0 || page_tokens == 0) return false;
    const uint64_t minimum_generation_pages = uint64_t(1) + mandatory_slack_pages;
    const uint64_t requested_pages =
        (uint64_t(generation_tail_tokens) + page_tokens - 1) / page_tokens;
    const uint64_t generation_pages = std::max(minimum_generation_pages, requested_pages);
    if (generation_pages > hot_pages || generation_pages > UINT32_MAX) return false;
    const uint64_t available_retrieval_pages = uint64_t(hot_pages) - generation_pages;
    const uint64_t retrieval_pages = requested_retrieval_pages == UINT32_MAX
        ? available_retrieval_pages
        : requested_retrieval_pages;
    if (retrieval_pages > available_retrieval_pages) return false;
    if (mandatory_anchor_pages > retrieval_pages) return false;
    const uint64_t generation_tokens = generation_pages * page_tokens;
    if (generation_tokens > UINT32_MAX) return false;
    output.hot_pages = hot_pages;
    output.generation_pages = uint32_t(generation_pages);
    output.retrieval_pages = uint32_t(retrieval_pages);
    output.generation_tokens = uint32_t(generation_tokens);
    return uint64_t(output.retrieval_pages) + output.generation_pages <= output.hot_pages;
}

uint32_t llama_kv_pager_generation_available_pages(
        const llama_kv_pager_turn_geometry & geometry,
        uint32_t selected_history_pages) noexcept {
    if (selected_history_pages > geometry.retrieval_pages ||
        selected_history_pages > geometry.hot_pages) return 0;
    const uint32_t available = geometry.hot_pages - selected_history_pages;
    return available < geometry.generation_pages ? 0 : available;
}

bool llama_kv_pager_config::validate(std::string & error) const {
    if (mode == llama_kv_pager_mode::off) return true;
    if (page_size == 0 || page_size % 256 != 0) { error = "page geometry requires a nonzero 256-token multiple"; return false; }
    if (!generation_tail_tokens.automatic && generation_tail_tokens.value == 0) {
        error = "generation tail must be auto or positive"; return false;
    }
    if (!retrieval_pages.automatic && retrieval_pages.value == 0) {
        error = "retrieval page budget must be auto or positive"; return false;
    }
    if (hot_pages.automatic == false && hot_pages.value == 0) { error = "hot-page cap must be auto or positive"; return false; }
    if (hot_pages.automatic == false && hot_pages.value > 0 && vram_budget.automatic == false && vram_budget.bytes == 0) {
        error = "hot-page cap contradicts an empty VRAM budget"; return false;
    }
    if (hotset_policy.empty()) { error = "hot-page policy must not be empty"; return false; }
    if (router_refresh_tokens == 0) { error = "router refresh cadence must be positive"; return false; }
    if (telemetry_interval_tokens == 0) { error = "telemetry interval must be positive"; return false; }
    return true;
}

std::string llama_kv_pager_config::mode_name() const {
    switch (mode) { case llama_kv_pager_mode::off: return "off"; case llama_kv_pager_mode::observe: return "observe";
        case llama_kv_pager_mode::selective: return "selective"; case llama_kv_pager_mode::exact: return "exact"; }
    return "off";
}

std::string llama_kv_pager_config::summary() const {
    const auto size_name = [](const llama_kv_pager_auto_size & value) {
        return value.automatic ? std::string("auto") : std::to_string(value.bytes);
    };
    const auto count_name = [](const llama_kv_pager_auto_count & value) {
        return value.automatic ? std::string("auto") : std::to_string(value.value);
    };

    // Keep this ordering stable: the string is emitted in startup diagnostics
    // and is also useful as a machine-readable key/value line in logs.
    return "mode=" + mode_name() +
           " page_size_tokens=" + std::to_string(page_size) +
           " generation_tail_tokens=" + count_name(generation_tail_tokens) +
           " retrieval_pages=" + count_name(retrieval_pages) +
           " retrieval_policy=" + (retrieval_policy == llama_kv_retrieval_policy::turn ? "turn" : "cadence") +
           " vram_budget_bytes=" + size_name(vram_budget) +
           " host_budget_bytes=" + size_name(host_budget) +
           " safety_headroom_bytes=" + size_name(safety_headroom) +
           " pin_recent_tokens=" + count_name(pin_recent) +
           " hotset_policy=" + hotset_policy +
           " hot_pages_cap=" + count_name(hot_pages) +
           " router_top_k=" + std::to_string(router_top_k) +
           " router_refresh_tokens=" + std::to_string(router_refresh_tokens) +
           " attention_tokens=" + std::to_string(attention_tokens) +
           " router_explore=" + std::to_string(router_explore) +
           " prefetch_depth=" + std::to_string(prefetch_depth) +
           " telemetry=" + (telemetry ? "on" : "off") +
           " telemetry_interval_tokens=" + std::to_string(telemetry_interval_tokens) +
           " telemetry_layer=" + std::to_string(telemetry_layer) +
           " telemetry_heads=" + std::to_string(telemetry_head_begin) + ":" +
           std::to_string(telemetry_head_count) +
           " debug=" + (debug ? "on" : "off");
}

const char * llama_kv_pager_capability_reason_name(llama_kv_pager_capability_reason reason) noexcept {
    static const char * names[] = { "ok", "backend", "model_architecture", "non_causal", "cache_type",
        "attention_geometry", "page_geometry", "device_topology", "sequence_layout", "host_budget", "mtp", "conflicting_vbr" };
    const auto index = size_t(reason);
    return index < sizeof(names) / sizeof(names[0]) ? names[index] : "unknown";
}

llama_kv_pager_capability_result llama_kv_pager_evaluate_capability(
        const llama_kv_pager_config & config, bool backend, bool model_architecture, bool causal,
        bool turbo4_kv, bool attention_geometry, bool page_geometry, bool one_device,
        bool sequence_layout, bool host_budget, bool mtp, bool conflicting_vbr) {
    llama_kv_pager_capability_result result;
    if (!config.enabled()) { result.supported = true; result.reasons.push_back(llama_kv_pager_capability_reason::ok); return result; }
    const bool checks[] = { backend, model_architecture, causal, turbo4_kv, attention_geometry,
        page_geometry, one_device, sequence_layout, host_budget, mtp, !conflicting_vbr };
    const llama_kv_pager_capability_reason reasons[] = {
        llama_kv_pager_capability_reason::backend, llama_kv_pager_capability_reason::model_architecture,
        llama_kv_pager_capability_reason::non_causal, llama_kv_pager_capability_reason::cache_type,
        llama_kv_pager_capability_reason::attention_geometry, llama_kv_pager_capability_reason::page_geometry,
        llama_kv_pager_capability_reason::device_topology, llama_kv_pager_capability_reason::sequence_layout,
        llama_kv_pager_capability_reason::host_budget, llama_kv_pager_capability_reason::mtp,
        llama_kv_pager_capability_reason::conflicting_vbr };
    for (size_t i = 0; i < sizeof(checks) / sizeof(checks[0]); ++i) {
        if (!checks[i]) result.reasons.push_back(reasons[i]);
    }
    result.supported = result.reasons.empty();
    for (const auto reason : result.reasons) {
        if (!result.diagnostic.empty()) result.diagnostic += ",";
        result.diagnostic += llama_kv_pager_capability_reason_name(reason);
    }
    return result;
}
