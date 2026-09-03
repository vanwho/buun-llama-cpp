#include "llama-kv-policy.h"

#include <algorithm>
#include <limits>
#include <set>

namespace {

const llama_kv_policy_page * find_page(
        const llama_kv_policy_trace & trace, uint64_t id) {
    for (const auto & page : trace.pages) {
        if (page.id == id) return &page;
    }
    return nullptr;
}

bool add_unique(std::vector<uint64_t> & ids, uint64_t id) {
    if (id == 0) return false;
    if (std::find(ids.begin(), ids.end(), id) != ids.end()) return true;
    ids.push_back(id);
    return true;
}

bool pinned(const llama_kv_policy_page & page) {
    return page.application_pin || page.inflight_pin;
}

bool colder(const llama_kv_policy_page & a, const llama_kv_policy_page & b) {
    // Unobserved attention is unavailable, not a zero score. Both pages therefore
    // remain ordered by the explicit safe fallbacks until evidence arrives.
    if (a.attention_observed != b.attention_observed) return a.attention_observed;
    if (a.attention_observed && a.attention_ema_q != b.attention_ema_q) {
        return a.attention_ema_q < b.attention_ema_q;
    }
    if (a.retrieval_hits != b.retrieval_hits) return a.retrieval_hits < b.retrieval_hits;
    if (a.recent != b.recent) return a.recent;
    if (a.dirty_cost != b.dirty_cost) return a.dirty_cost > b.dirty_cost;
    if (a.age != b.age) return a.age > b.age;
    if (a.recency != b.recency) return a.recency < b.recency;
    return a.id < b.id;
}

} // namespace

namespace {

uint64_t normalized(uint64_t value, uint64_t maximum) {
    if (maximum == 0) return 0;
    return static_cast<uint64_t>((static_cast<long double>(value) * 1000000.0L) / maximum);
}

bool contains(const std::vector<uint64_t> & values, uint64_t id) {
    return std::find(values.begin(), values.end(), id) != values.end();
}

bool controller_pinned(const llama_kv_policy_page & page) {
    return page.current || page.recent || page.anchor || page.application_pin || page.inflight_pin ||
           page.speculative_pin;
}

llama_kv_policy_decision_entry record(uint64_t id, llama_kv_policy_reason reason,
                                      uint64_t score = 0) {
    llama_kv_policy_decision_entry entry;
    entry.id = id;
    entry.reason = reason;
    entry.normalized_keep_q = score;
    return entry;
}

} // namespace

llama_kv_policy_decision llama_kv_policy_decide(
        const llama_kv_policy_trace & trace,
        const llama_kv_policy_controller_config & config,
        const std::vector<uint64_t> & previous_target) noexcept {
    llama_kv_policy_decision out;
    out.epoch = trace.epoch;
    try {
        const uint64_t partition = static_cast<uint64_t>(config.recent_pages) + config.structural_pages +
                                   config.historical_pages + config.transient_pages;
        if (trace.version != LLAMA_KV_POLICY_TRACE_VERSION || trace.epoch == 0 ||
            trace.pages.empty() || trace.pages.size() > LLAMA_KV_POLICY_MAX_PAGES ||
            config.capacity_pages == 0 || config.capacity_pages > LLAMA_KV_POLICY_MAX_PAGES ||
            partition == 0 ||
            trace.summary_top_k.size() > LLAMA_KV_POLICY_MAX_SUMMARY ||
            trace.exploration.size() > LLAMA_KV_POLICY_MAX_SUMMARY) {
            out.status = llama_kv_policy_status::invalid_trace;
            return out;
        }

        std::set<uint64_t> ids;
        for (const auto & page : trace.pages) {
            if (page.id == 0 || !ids.insert(page.id).second) {
                out.status = llama_kv_policy_status::invalid_trace;
                return out;
            }
        }
        auto find = [&](uint64_t id) -> const llama_kv_policy_page * {
            for (const auto & page : trace.pages) if (page.id == id) return &page;
            return nullptr;
        };
        auto add = [&](const llama_kv_policy_page & page, llama_kv_policy_reason reason,
                       uint64_t score = 0) {
            if (contains(out.target, page.id) || out.target.size() >= config.capacity_pages) return false;
            out.target.push_back(page.id);
            out.records.push_back(record(page.id, reason, score));
            return true;
        };

        uint32_t mandatory = 0;
        for (const auto & page : trace.pages) {
            mandatory += controller_pinned(page) || page.id == trace.write_page;
        }
        if (mandatory > config.capacity_pages) {
            out.status = llama_kv_policy_status::pin_overflow;
            return out;
        }
        auto mandatory_add = [&](uint64_t id) {
            const auto * page = find(id);
            if (page) add(*page, llama_kv_policy_reason::mandatory);
        };
        mandatory_add(trace.write_page);
        for (const auto & page : trace.pages) if (controller_pinned(page)) add(page, llama_kv_policy_reason::mandatory);

        uint64_t max_ema = 0, max_peak = 0, max_frequency = 0, max_recency = 0;
        uint64_t max_fault = 0, max_dirty = 0, max_age = 0;
        for (const auto & page : trace.pages) {
            max_ema = std::max(max_ema, page.attention_ema_q);
            max_peak = std::max(max_peak, page.recent_peak_q);
            max_frequency = std::max(max_frequency, page.retrieval_hits);
            max_recency = std::max(max_recency, page.recency);
            max_fault = std::max(max_fault, page.fault_cost);
            max_dirty = std::max(max_dirty, page.dirty_cost);
            max_age = std::max(max_age, page.age);
        }
        auto score = [&](const llama_kv_policy_page & page) {
            // Inputs are normalized independently. Equal contribution is an
            // intentionally provisional rank, not a calibrated coefficient.
            uint64_t value = normalized(page.attention_ema_q, max_ema) +
                normalized(page.recent_peak_q, max_peak) + normalized(page.retrieval_hits, max_frequency) +
                normalized(page.recency, max_recency) + page.hysteresis_q;
            if (contains(previous_target, page.id)) value += config.hysteresis_q;
            const uint64_t cost = normalized(page.fault_cost, max_fault) + normalized(page.dirty_cost, max_dirty);
            const uint64_t decay = normalized(page.age, max_age);
            return value > cost + decay ? value - cost - decay : 0;
        };
        auto rank = [&](bool (*predicate)(const llama_kv_policy_page &)) {
            std::vector<const llama_kv_policy_page *> values;
            for (const auto & page : trace.pages) if (predicate(page) && !contains(out.target, page.id)) values.push_back(&page);
            std::sort(values.begin(), values.end(), [&](const auto * lhs, const auto * rhs) {
                const uint64_t ls = score(*lhs), rs = score(*rhs);
                if (ls != rs) return ls > rs;
                return lhs->id < rhs->id;
            });
            return values;
        };
        auto take = [&](const std::vector<const llama_kv_policy_page *> & values,
                        uint32_t limit, llama_kv_policy_reason reason) {
            for (const auto * page : values) {
                if (limit == 0 || out.target.size() >= config.capacity_pages) break;
                if (add(*page, reason, score(*page))) --limit;
            }
        };
        auto is_recent = [](const llama_kv_policy_page & p) { return p.recent; };
        auto is_structural = [](const llama_kv_policy_page & p) { return p.structural || p.anchor; };
        auto is_historical = [](const llama_kv_policy_page & p) { return p.resident || p.attention_observed; };
        auto is_transient = [](const llama_kv_policy_page & p) { return !p.resident && !p.attention_observed; };
        take(rank(is_recent), config.recent_pages, llama_kv_policy_reason::recent);
        take(rank(is_structural), config.structural_pages, llama_kv_policy_reason::structural);

        auto include_ids = [&](const std::vector<uint64_t> & values, llama_kv_policy_reason reason) {
            for (uint64_t id : values) {
                const auto * page = find(id);
                if (page && out.target.size() < config.capacity_pages) add(*page, reason, score(*page));
            }
        };
        include_ids(trace.summary_top_k, llama_kv_policy_reason::summary);
        take(rank(is_historical), config.historical_pages, llama_kv_policy_reason::retention);
        include_ids(trace.exploration, llama_kv_policy_reason::exploration);
        take(rank(is_transient), config.transient_pages, llama_kv_policy_reason::exploration);

        auto has_record = [&](uint64_t id) {
            for (const auto & entry : out.records) if (entry.id == id) return true;
            return false;
        };
        for (const auto & page : trace.pages) {
            if (!page.attention_observed) {
                ++out.unavailable_evidence;
                if (!has_record(page.id) && out.records.size() < LLAMA_KV_POLICY_MAX_PAGES) {
                    out.records.push_back(record(page.id, llama_kv_policy_reason::unavailable));
                }
            }
            if (out.target.size() >= config.capacity_pages) break;
            if (!contains(out.target, page.id)) add(page, llama_kv_policy_reason::retention, score(page));
        }

        for (uint64_t id : out.target) {
            if (contains(previous_target, id)) out.keeps.push_back(id);
            else out.adds.push_back(id);
        }
        std::vector<const llama_kv_policy_page *> victim_pages;
        for (const auto & page : trace.pages) {
            if (page.resident && !contains(out.target, page.id) && !controller_pinned(page)) {
                victim_pages.push_back(&page);
            }
        }
        std::sort(victim_pages.begin(), victim_pages.end(), [&](const auto * lhs, const auto * rhs) {
            const uint64_t ls = score(*lhs), rs = score(*rhs);
            if (ls != rs) return ls < rs;
            return lhs->id < rhs->id;
        });
        for (const auto * page : victim_pages) {
            out.victims.push_back(page->id);
            out.records.push_back(record(page->id, llama_kv_policy_reason::victim, score(*page)));
        }
        out.status = llama_kv_policy_status::ok;
        return out;
    } catch (...) {
        out.status = llama_kv_policy_status::unavailable;
        out.target.clear(); out.adds.clear(); out.keeps.clear(); out.victims.clear(); out.records.clear();
        return out;
    }
}

llama_kv_policy_result llama_kv_policy_replay(
        const llama_kv_policy_trace & trace) noexcept {
    llama_kv_policy_result out;
    try {
        if (trace.version != LLAMA_KV_POLICY_TRACE_VERSION || trace.epoch == 0 ||
            trace.capacity_pages == 0 || trace.capacity_pages > LLAMA_KV_POLICY_MAX_PAGES ||
            trace.pages.empty() || trace.pages.size() > LLAMA_KV_POLICY_MAX_PAGES ||
            trace.summary_top_k.size() > LLAMA_KV_POLICY_MAX_SUMMARY ||
            trace.exploration.size() > LLAMA_KV_POLICY_MAX_SUMMARY) {
            out.status = llama_kv_policy_status::invalid_trace;
            return out;
        }
        for (size_t i = 0; i < trace.pages.size(); ++i) {
            if (trace.pages[i].id == 0 ||
                find_page(trace, trace.pages[i].id) != &trace.pages[i]) {
                out.status = llama_kv_policy_status::invalid_trace;
                return out;
            }
        }
        out.retrieve.reserve(trace.pages.size());
        auto include = [&](uint64_t id) {
            const auto * page = find_page(trace, id);
            return page && add_unique(out.retrieve, id);
        };
        include(trace.write_page);
        for (const auto & page : trace.pages) {
            if (page.recent || page.anchor || page.application_pin || page.inflight_pin) {
                include(page.id);
            }
        }
        for (const uint64_t id : trace.summary_top_k) include(id);
        for (const uint64_t id : trace.exploration) include(id);

        size_t mandatory = 0;
        for (const auto & page : trace.pages) {
            if (pinned(page)) mandatory++;
        }
        if (mandatory > trace.capacity_pages) {
            out.status = llama_kv_policy_status::pin_overflow;
            out.retrieve.clear();
            return out;
        }

        std::vector<const llama_kv_policy_page *> residents;
        for (const auto & page : trace.pages) {
            if (page.resident && !pinned(page)) residents.push_back(&page);
        }
        std::sort(residents.begin(), residents.end(),
            [](const auto * a, const auto * b) { return colder(*a, *b); });
        size_t resident_count = 0;
        for (const auto & page : trace.pages) resident_count += page.resident;
        if (resident_count > trace.capacity_pages) {
            const size_t need = resident_count - trace.capacity_pages;
            for (size_t i = 0; i < need && i < residents.size(); ++i) {
                out.victims.push_back(residents[i]->id);
            }
            if (out.victims.size() != need) {
                out.status = llama_kv_policy_status::pin_overflow;
                out.retrieve.clear(); out.victims.clear(); return out;
            }
        }
        out.status = llama_kv_policy_status::ok;
        return out;
    } catch (...) {
        out.status = llama_kv_policy_status::unavailable;
        out.retrieve.clear(); out.victims.clear();
        return out;
    }
}

const char * llama_kv_policy_status_name(llama_kv_policy_status status) noexcept {
    switch (status) {
        case llama_kv_policy_status::ok: return "ok";
        case llama_kv_policy_status::pin_overflow: return "pin_overflow";
        case llama_kv_policy_status::invalid_trace: return "invalid_trace";
        case llama_kv_policy_status::unavailable: return "unavailable";
        case llama_kv_policy_status::_count: return "invalid";
    }
    return "invalid";
}
