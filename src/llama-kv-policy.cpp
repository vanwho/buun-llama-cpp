#include "llama-kv-policy.h"

#include <algorithm>
#include <limits>

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
