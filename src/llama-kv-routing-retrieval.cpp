#include "llama-kv-routing-retrieval.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>

namespace {

struct candidate {
    const llama_kv_page_record * record = nullptr;
    llama_kv_routing_page_attributes attributes;
    bool has_score = false;
    float score = 0.0f;
    bool upper_bound = false;
};

bool valid_state(llama_kv_page_state state) noexcept {
    return state != llama_kv_page_state::absent && state != llama_kv_page_state::invalid;
}

bool valid_page(const llama_kv_page_record & record) noexcept {
    const bool tail = llama_kv_page_id_is_tail(record.id);
    if (!valid_state(record.state) || !llama_kv_page_id_valid(record.id, tail)) return false;
    return record.physical_slot != UINT32_MAX ||
        (record.state == llama_kv_page_state::host_clean && record.host_valid);
}

bool identity_matches(const llama_kv_page_id & id,
                      const llama_kv_routing_query & query) noexcept {
    if (query.sequence_id < 0 || id.sequence_id != query.sequence_id) return false;
    if (query.session_generation != 0 && id.session_generation != query.session_generation) return false;
    if (query.sequence_generation != 0 && id.sequence_generation != query.sequence_generation) return false;
    if (query.model_identity != 0 && id.model_identity != query.model_identity) return false;
    if (query.topology_identity != 0 && id.topology_identity != query.topology_identity) return false;
    if (query.representation_epoch != 0 && id.representation_epoch != query.representation_epoch) return false;
    return true;
}

bool same_logical(const llama_kv_page_id & lhs, const llama_kv_page_id & rhs) noexcept {
    return lhs.session_generation == rhs.session_generation &&
           lhs.sequence_id == rhs.sequence_id &&
           lhs.sequence_generation == rhs.sequence_generation &&
           lhs.logical_page == rhs.logical_page;
}

uint64_t page_distance(const llama_kv_page_id & id, llama_pos query_position) noexcept {
    if (query_position < 0) return 0;
    if (query_position < id.position_begin) {
        return uint64_t(id.position_begin) - uint64_t(query_position);
    }
    if (query_position > id.position_end) {
        return uint64_t(query_position) - uint64_t(id.position_end);
    }
    return 0;
}

bool recent_order(const candidate & lhs, const candidate & rhs) noexcept {
    if (lhs.record->id.position_end != rhs.record->id.position_end) {
        return lhs.record->id.position_end > rhs.record->id.position_end;
    }
    return lhs.record->id.logical_page < rhs.record->id.logical_page;
}

bool logical_order(const candidate & lhs, const candidate & rhs) noexcept {
    if (lhs.record->id.logical_page != rhs.record->id.logical_page) {
        return lhs.record->id.logical_page < rhs.record->id.logical_page;
    }
    return lhs.record->id.page_generation < rhs.record->id.page_generation;
}

bool has_id(const std::vector<llama_kv_routing_retrieval_entry> & selected,
            const llama_kv_page_id & id) noexcept {
    for (const auto & entry : selected) {
        if (entry.id == id) return true;
    }
    return false;
}

} // namespace

const char * llama_kv_routing_retrieval_status_name(
        llama_kv_routing_retrieval_status status) noexcept {
    switch (status) {
        case llama_kv_routing_retrieval_status::ok:                  return "ok";
        case llama_kv_routing_retrieval_status::invalid_argument:   return "invalid_argument";
        case llama_kv_routing_retrieval_status::stale_query:        return "stale_query";
        case llama_kv_routing_retrieval_status::stale_summary:      return "stale_summary";
        case llama_kv_routing_retrieval_status::unavailable_summary:return "unavailable_summary";
        case llama_kv_routing_retrieval_status::mandatory_overflow: return "mandatory_overflow";
        case llama_kv_routing_retrieval_status::overflow:            return "overflow";
    }
    return "invalid";
}

const char * llama_kv_routing_retrieval_reason_name(
        llama_kv_routing_retrieval_reason reason) noexcept {
    switch (reason) {
        case llama_kv_routing_retrieval_reason::mandatory:   return "mandatory";
        case llama_kv_routing_retrieval_reason::structural:  return "structural";
        case llama_kv_routing_retrieval_reason::recent:      return "recent";
        case llama_kv_routing_retrieval_reason::summary:     return "summary";
        case llama_kv_routing_retrieval_reason::exploration: return "exploration";
        case llama_kv_routing_retrieval_reason::fallback:    return "fallback";
    }
    return "invalid";
}

llama_kv_routing_retrieval_result llama_kv_routing_retrieve(
        const llama_kv_residency_snapshot & snapshot,
        const llama_kv_routing_summary_store & summaries,
        const llama_kv_routing_query & query,
        const llama_kv_routing_retrieval_config & config,
        const std::vector<llama_kv_routing_page_attributes> & attributes,
        const std::vector<llama_kv_page_id> & previous_target) noexcept {
    return llama_kv_routing_retrieve(snapshot, snapshot.pages(), summaries, query,
            config, attributes, previous_target);
}

llama_kv_routing_retrieval_result llama_kv_routing_retrieve(
        const llama_kv_residency_snapshot & snapshot,
        const llama_kv_routing_page_inventory & inventory,
        const llama_kv_routing_summary_store & summaries,
        const llama_kv_routing_query & query,
        const llama_kv_routing_retrieval_config & config,
        const std::vector<llama_kv_routing_page_attributes> & attributes,
        const std::vector<llama_kv_page_id> & previous_target) noexcept {
    llama_kv_routing_retrieval_result result;
    result.table_epoch = query.table_epoch;
    result.query_generation = query.query_generation;
    result.token_index = query.token_index;
    result.model_identity = query.model_identity;
    result.topology_identity = query.topology_identity;
    result.representation_epoch = query.representation_epoch;
    result.session_generation = query.session_generation;
    result.sequence_generation = query.sequence_generation;
    result.sequence_id = query.sequence_id;
    result.position = query.position;
    result.layer_index = query.layer_index;
    result.head_index = query.head_index;
    result.coordinate_identity = query.coordinate_identity;
    const auto started = std::chrono::steady_clock::now();
    try {
        if (snapshot.epoch() == 0 || query.table_epoch == 0 || query.query_generation == 0 ||
            query.sequence_id < 0 || query.position < -1 || query.values.empty() ||
            config.capacity_pages == 0) {
            result.status = llama_kv_routing_retrieval_status::invalid_argument;
            return result;
        }
        if (query.table_epoch != snapshot.epoch()) {
            result.status = llama_kv_routing_retrieval_status::stale_query;
            return result;
        }
        for (const float value : query.values) {
            if (!std::isfinite(value)) {
                result.status = llama_kv_routing_retrieval_status::invalid_argument;
                return result;
            }
        }
        if (summaries.valid() && (query.layer_index != summaries.layer_index() ||
                                  query.head_index != summaries.head_index())) {
            result.status = llama_kv_routing_retrieval_status::stale_query;
            return result;
        }
        if (summaries.valid() && query.coordinate_identity != 0 &&
            summaries.coordinate_identity() != 0 &&
            query.coordinate_identity != summaries.coordinate_identity()) {
            result.status = llama_kv_routing_retrieval_status::stale_query;
            return result;
        }
        if (summaries.valid() && query.values.size() != summaries.accounting().vector_dim) {
            result.status = llama_kv_routing_retrieval_status::invalid_argument;
            return result;
        }

        for (size_t i = 0; i < inventory.size(); ++i) {
            if (!valid_page(inventory[i])) {
                result.status = llama_kv_routing_retrieval_status::invalid_argument;
                return result;
            }
            for (size_t j = i + 1; j < inventory.size(); ++j) {
                if (same_logical(inventory[i].id, inventory[j].id)) {
                    result.status = llama_kv_routing_retrieval_status::invalid_argument;
                    return result;
                }
            }
        }
        for (const auto & resident : snapshot.pages()) {
            if (!valid_page(resident)) continue;
            const auto it = std::find_if(inventory.begin(), inventory.end(),
                    [&](const auto & record) { return same_logical(record.id, resident.id); });
            if (it == inventory.end() || it->id != resident.id ||
                it->physical_slot != resident.physical_slot) {
                result.status = llama_kv_routing_retrieval_status::stale_query;
                return result;
            }
        }

        std::vector<candidate> pages;
        pages.reserve(inventory.size());
        for (const auto & record : inventory) {
            if (!valid_page(record)) continue;
            if (!identity_matches(record.id, query)) {
                result.status = llama_kv_routing_retrieval_status::stale_query;
                return result;
            }
            candidate page;
            page.record = &record;
            pages.push_back(page);
        }
        if (pages.empty()) {
            result.status = llama_kv_routing_retrieval_status::invalid_argument;
            return result;
        }
        result.metrics.valid_pages = pages.size();

        for (size_t i = 0; i < attributes.size(); ++i) {
            const auto & attribute = attributes[i];
            if (attribute.id.sequence_id != query.sequence_id) {
                result.status = llama_kv_routing_retrieval_status::invalid_argument;
                return result;
            }
            for (size_t j = i + 1; j < attributes.size(); ++j) {
                if (same_logical(attribute.id, attributes[j].id)) {
                    result.status = llama_kv_routing_retrieval_status::invalid_argument;
                    return result;
                }
            }
            const auto it = std::find_if(pages.begin(), pages.end(), [&](const candidate & page) {
                return page.record->id == attribute.id;
            });
            if (it == pages.end()) {
                result.status = llama_kv_routing_retrieval_status::stale_query;
                return result;
            }
            it->attributes = attribute;
        }

        bool query_stale = false;
        if (summaries.valid() && summaries.layer_index() == query.layer_index &&
            summaries.head_index() == query.head_index) {
            const auto score_started = std::chrono::steady_clock::now();
            const uint32_t score_limit = uint32_t(std::min<size_t>(pages.size(), UINT32_MAX));
            const auto scored = summaries.score(snapshot, inventory, query.values, score_limit);
            result.metrics.summary_pages_scored = scored.pages_scored;
            result.metrics.summary_comparisons = scored.comparisons;
            result.metrics.score_time_us = scored.latency_us;
            result.metrics.summary_bytes = summaries.accounting().charged_bytes;
            if (scored.status == llama_kv_routing_summary_status::ok) {
                result.metrics.summary_complete = true;
                for (const auto & score : scored.top_pages) {
                    const auto it = std::find_if(pages.begin(), pages.end(), [&](const candidate & page) {
                        return page.record->id.logical_page == score.logical_page &&
                               page.record->id.page_generation == score.page_generation;
                    });
                    if (it != pages.end()) {
                        it->has_score = true;
                        it->score = score.score;
                        it->upper_bound = score.upper_bound;
                    }
                }
            } else {
                query_stale = scored.status == llama_kv_routing_summary_status::stale_summary;
            }
            (void) score_started;
        } else {
            result.metrics.summary_bytes = summaries.accounting().charged_bytes;
            query_stale = summaries.valid();
        }

        const auto union_started = std::chrono::steady_clock::now();
        auto add = [&](const candidate & page, llama_kv_routing_retrieval_reason reason) {
            if (has_id(result.selected, page.record->id) || result.selected.size() >= config.capacity_pages) {
                return false;
            }
            result.selected.push_back({ page.record->id, reason, page.score, page.has_score,
                    page.upper_bound, page_distance(page.record->id, query.position) });
            return true;
        };
        auto find_candidate = [&](const llama_kv_page_id & id) -> const candidate * {
            for (const auto & page : pages) if (page.record->id == id) return &page;
            return nullptr;
        };

        std::vector<const candidate *> mandatory;
        std::vector<const candidate *> structural;
        std::vector<const candidate *> recent;
        for (const auto & page : pages) {
            const auto & a = page.attributes;
            const bool pinned = page.record->pin_count != 0 || a.application_pin ||
                a.inflight_pin || a.speculative_pin;
            if (a.current || a.mandatory || pinned) mandatory.push_back(&page);
            else if (a.structural) structural.push_back(&page);
            if (a.recent) recent.push_back(&page);
        }
        std::sort(mandatory.begin(), mandatory.end(), [](const candidate * lhs, const candidate * rhs) {
            if (lhs->attributes.current != rhs->attributes.current) return lhs->attributes.current;
            if (lhs->attributes.mandatory != rhs->attributes.mandatory) return lhs->attributes.mandatory;
            return logical_order(*lhs, *rhs);
        });
        std::sort(structural.begin(), structural.end(), [](const candidate * lhs, const candidate * rhs) {
            return logical_order(*lhs, *rhs);
        });
        result.metrics.mandatory_pages = mandatory.size() + structural.size();
        if (result.metrics.mandatory_pages > config.capacity_pages) {
            result.selected.clear();
            result.status = llama_kv_routing_retrieval_status::mandatory_overflow;
            return result;
        }
        for (const auto * page : mandatory) add(*page, llama_kv_routing_retrieval_reason::mandatory);
        for (const auto * page : structural) add(*page, llama_kv_routing_retrieval_reason::structural);

        std::sort(recent.begin(), recent.end(), [](const candidate * lhs, const candidate * rhs) {
            return recent_order(*lhs, *rhs);
        });
        for (const auto * page : recent) add(*page, llama_kv_routing_retrieval_reason::recent);

        if (result.metrics.summary_complete) {
            std::vector<const candidate *> ranked;
            for (const auto & page : pages) if (page.has_score) ranked.push_back(&page);
            std::sort(ranked.begin(), ranked.end(), [](const candidate * lhs, const candidate * rhs) {
                if (lhs->score != rhs->score) return lhs->score > rhs->score;
                return logical_order(*lhs, *rhs);
            });
            uint32_t remaining = config.summary_top_k;
            for (const auto * page : ranked) {
                if (remaining == 0 || result.selected.size() >= config.capacity_pages) break;
                if (add(*page, llama_kv_routing_retrieval_reason::summary)) --remaining;
            }
        }

        if (!result.metrics.summary_complete) {
            result.metrics.fallback_used = true;
            for (const auto & id : previous_target) {
                const auto * page = find_candidate(id);
                if (page) add(*page, llama_kv_routing_retrieval_reason::fallback);
            }
        }

        std::vector<const candidate *> exploration;
        for (const auto & page : pages) {
            if (!has_id(result.selected, page.record->id)) exploration.push_back(&page);
        }
        std::sort(exploration.begin(), exploration.end(), [](const candidate * lhs, const candidate * rhs) {
            return logical_order(*lhs, *rhs);
        });
        if (!exploration.empty() && config.exploration_pages != 0) {
            const size_t offset = size_t((config.exploration_seed % exploration.size() +
                    config.exploration_turn % exploration.size()) % exploration.size());
            uint32_t remaining = config.exploration_pages;
            for (size_t i = 0; i < exploration.size() && remaining != 0; ++i) {
                if (add(*exploration[(offset + i) % exploration.size()],
                        llama_kv_routing_retrieval_reason::exploration)) --remaining;
            }
            result.metrics.exploration_pages = config.exploration_pages - remaining;
        }

        // When the physical capacity can hold the complete valid inventory,
        // return it even if the configured candidate budgets are smaller. This
        // is the dense-equivalent small-context safety rule.
        if (config.capacity_pages >= pages.size()) {
            std::vector<const candidate *> all;
            for (const auto & page : pages) all.push_back(&page);
            std::sort(all.begin(), all.end(), [](const candidate * lhs, const candidate * rhs) {
                return logical_order(*lhs, *rhs);
            });
            for (const auto * page : all) {
                if (!has_id(result.selected, page->record->id)) {
                    add(*page, page->has_score ? llama_kv_routing_retrieval_reason::summary :
                        llama_kv_routing_retrieval_reason::fallback);
                }
            }
        }

        result.metrics.selected_pages = result.selected.size();
        result.metrics.union_time_us = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - union_started).count());
        result.metrics.total_time_us = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count());
        result.status = result.metrics.summary_complete
            ? llama_kv_routing_retrieval_status::ok
            : (query_stale ? llama_kv_routing_retrieval_status::stale_summary
                           : llama_kv_routing_retrieval_status::unavailable_summary);
        return result;
    } catch (const std::bad_alloc &) {
        result.status = llama_kv_routing_retrieval_status::overflow;
        result.selected.clear();
        return result;
    } catch (...) {
        result.status = llama_kv_routing_retrieval_status::overflow;
        result.selected.clear();
        return result;
    }
}
