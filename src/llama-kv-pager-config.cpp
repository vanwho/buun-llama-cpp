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

bool llama_kv_pager_config::validate(std::string & error) const {
    if (mode == llama_kv_pager_mode::off) return true;
    if (page_size == 0 || page_size % 256 != 0) { error = "page geometry requires a nonzero 256-token multiple"; return false; }
    if (hot_pages.automatic == false && hot_pages.value == 0) { error = "hot-page cap must be auto or positive"; return false; }
    if (hot_pages.automatic == false && hot_pages.value > 0 && vram_budget.automatic == false && vram_budget.bytes == 0) {
        error = "hot-page cap contradicts an empty VRAM budget"; return false;
    }
    if (hotset_policy.empty()) { error = "hot-page policy must not be empty"; return false; }
    return true;
}

std::string llama_kv_pager_config::mode_name() const {
    switch (mode) { case llama_kv_pager_mode::off: return "off"; case llama_kv_pager_mode::observe: return "observe";
        case llama_kv_pager_mode::selective: return "selective"; case llama_kv_pager_mode::exact: return "exact"; }
    return "off";
}

std::string llama_kv_pager_config::summary() const {
    return "mode=" + mode_name() + " page_size=" + std::to_string(page_size) +
           " vram=" + (vram_budget.automatic ? "auto" : std::to_string(vram_budget.bytes)) +
           " host=" + (host_budget.automatic ? "auto" : std::to_string(host_budget.bytes));
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
