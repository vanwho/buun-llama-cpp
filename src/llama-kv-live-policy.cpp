#include "llama-kv-live-policy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace {

bool same_logical_page(const llama_kv_page_id & a, const llama_kv_page_id & b) noexcept {
    return a.session_generation == b.session_generation &&
           a.sequence_id == b.sequence_id &&
           a.sequence_generation == b.sequence_generation &&
           a.logical_page == b.logical_page;
}

const llama_kv_live_policy_page * find_page(
        const std::vector<llama_kv_live_policy_page> & pages,
        const llama_kv_page_id & id) noexcept {
    for (const auto & page : pages) if (page.record.id == id) return &page;
    return nullptr;
}

const llama_kv_routing_retrieval_entry * find_retrieval(
        const llama_kv_routing_retrieval_result & retrieval,
        const llama_kv_page_id & id) noexcept {
    for (const auto & entry : retrieval.selected) if (entry.id == id) return &entry;
    return nullptr;
}

uint64_t identity_hash(const llama_kv_page_id & id) noexcept {
    uint64_t hash = 1469598103934665603ull;
    const auto mix = [&](uint64_t value) {
        for (size_t i = 0; i < sizeof(value); ++i) {
            hash ^= uint8_t(value >> (i * 8));
            hash *= 1099511628211ull;
        }
    };
    mix(id.session_generation);
    mix(uint32_t(id.sequence_id));
    mix(id.sequence_generation);
    mix(id.logical_page);
    mix(id.page_generation);
    mix(id.representation_epoch);
    mix(id.model_identity);
    mix(id.topology_identity);
    mix(id.codec_digest);
    mix(id.codebook_digest);
    mix(id.rotation_digest);
    mix(id.meansub_digest);
    mix(uint64_t(id.position_begin));
    mix(uint64_t(id.position_end));
    return hash;
}

bool contains_id(const std::vector<llama_kv_page_id> & values,
                 const llama_kv_page_id & id) noexcept {
    return std::find(values.begin(), values.end(), id) != values.end();
}

const llama_kv_policy_decision_entry * find_policy_record(
        const llama_kv_policy_decision & decision, uint64_t id) noexcept {
    for (const auto & record : decision.records) if (record.id == id) return &record;
    return nullptr;
}

bool same_record(const llama_kv_page_record & a,
                 const llama_kv_page_record & b) noexcept {
    return a.id == b.id && a.physical_slot == b.physical_slot &&
           a.state == b.state && a.host_valid == b.host_valid &&
           a.dirty == b.dirty && a.pin_count == b.pin_count;
}

bool same_snapshot(const llama_kv_residency_snapshot & a,
                   const llama_kv_residency_snapshot & b) noexcept {
    if (a.epoch() != b.epoch() || a.slot_capacity() != b.slot_capacity() ||
        a.pages().size() != b.pages().size()) return false;
    for (const auto & page : a.pages()) {
        const auto it = std::find_if(b.pages().begin(), b.pages().end(),
                [&](const auto & other) { return other.id == page.id; });
        if (it == b.pages().end() || !same_record(page, *it)) return false;
    }
    return true;
}

llama_kv_live_policy_status transaction_status(
        llama_kv_residency_transaction_status status) noexcept {
    switch (status) {
        case llama_kv_residency_transaction_status::committed: return llama_kv_live_policy_status::committed;
        case llama_kv_residency_transaction_status::stale_epoch:
        case llama_kv_residency_transaction_status::stale_generation: return llama_kv_live_policy_status::stale_snapshot;
        case llama_kv_residency_transaction_status::missing_host_source: return llama_kv_live_policy_status::missing_host_source;
        case llama_kv_residency_transaction_status::all_pinned: return llama_kv_live_policy_status::all_pinned;
        case llama_kv_residency_transaction_status::dirty_victim: return llama_kv_live_policy_status::dirty_victim;
        case llama_kv_residency_transaction_status::_count: break;
        default: break;
    }
    return llama_kv_live_policy_status::transaction_failed;
}

} // namespace

const char * llama_kv_live_policy_status_name(
        llama_kv_live_policy_status status) noexcept {
    switch (status) {
        case llama_kv_live_policy_status::committed: return "committed";
        case llama_kv_live_policy_status::no_change: return "no_change";
        case llama_kv_live_policy_status::safe_fallback: return "safe_fallback";
        case llama_kv_live_policy_status::not_configured: return "not_configured";
        case llama_kv_live_policy_status::invalid_argument: return "invalid_argument";
        case llama_kv_live_policy_status::stale_snapshot: return "stale_snapshot";
        case llama_kv_live_policy_status::unavailable_inventory: return "unavailable_inventory";
        case llama_kv_live_policy_status::mandatory_overflow: return "mandatory_overflow";
        case llama_kv_live_policy_status::query_capacity_refused: return "query_capacity_refused";
        case llama_kv_live_policy_status::missing_host_source: return "missing_host_source";
        case llama_kv_live_policy_status::all_pinned: return "all_pinned";
        case llama_kv_live_policy_status::dirty_victim: return "dirty_victim";
        case llama_kv_live_policy_status::transaction_failed: return "transaction_failed";
        case llama_kv_live_policy_status::_count: break;
    }
    return "invalid";
}

void llama_kv_live_policy_reconcile_residency_records(
        const llama_kv_residency_snapshot & snapshot,
        std::vector<llama_kv_page_record> & records) noexcept {
    for (auto & record : records) {
        const auto resident = std::find_if(snapshot.pages().begin(), snapshot.pages().end(),
                [&](const auto & current) { return current.id == record.id; });
        if (resident != snapshot.pages().end()) {
            record.physical_slot = resident->physical_slot;
            record.state = resident->state;
            record.host_valid = resident->host_valid;
            record.dirty = resident->dirty;
            record.pin_count = resident->pin_count;
        } else if (record.physical_slot != UINT32_MAX) {
            // A merged cold-catalogue entry can retain a formerly owned slot.
            // It may be selected only as host-cold; never let that stale slot
            // alias the page that currently owns it in the published table.
            record.physical_slot = UINT32_MAX;
            record.state = llama_kv_page_state::host_clean;
            record.dirty = false;
            record.pin_count = 0;
        }
    }
}

bool llama_kv_query_commit_add_candidate(
        llama_kv_query_commit & commit, const llama_kv_page_id & canonical_identity,
        uint64_t content_version, uint32_t attention_layer) noexcept {
    if (!commit.enabled || content_version == 0 || attention_layer == UINT32_MAX) return false;
    const auto existing = std::find_if(commit.selected.begin(), commit.selected.end(),
            [&](const auto & page) { return same_logical_page(page.identity, canonical_identity); });
    if (existing != commit.selected.end()) {
        if (existing->identity != canonical_identity || existing->content_version != content_version) return false;
        if (std::find(existing->attention_layers.begin(), existing->attention_layers.end(),
                attention_layer) == existing->attention_layers.end()) {
            existing->attention_layers.push_back(attention_layer);
        }
        return true;
    }
    if (commit.retrieval_budget == 0 || commit.selected.size() >= commit.retrieval_budget) return false;
    llama_kv_query_commit_page page;
    page.identity = canonical_identity;
    page.content_version = content_version;
    page.attention_layers.push_back(attention_layer);
    commit.selected.push_back(std::move(page));
    return true;
}

bool llama_kv_live_policy_prepare_query_target(
        const llama_kv_live_policy_boundary & boundary,
        std::vector<llama_kv_page_record> & target) noexcept {
    target.clear();
    try {
        const auto & commit = boundary.query_commit;
        if (!commit.enabled || commit.query_generation == 0 || commit.turn_id == 0 ||
            commit.retrieval_epoch == 0 || commit.table_epoch != boundary.snapshot.epoch() ||
            commit.query_position == 0 || commit.rollback_generation == 0 ||
            commit.model_identity == 0 || commit.session_generation == 0 ||
            commit.sequence_generation == 0 || commit.sequence_id < 0 ||
            commit.representation_epoch == 0 || commit.retrieval_budget == 0 ||
            uint64_t(commit.retrieval_budget) + commit.generation_budget > boundary.hot_capacity ||
            boundary.retrieval.status != llama_kv_routing_retrieval_status::ok ||
            boundary.retrieval.table_epoch != commit.table_epoch ||
            boundary.retrieval.position < 0 ||
            uint64_t(boundary.retrieval.position) != commit.query_position ||
            boundary.retrieval.query_generation != commit.query_generation ||
            boundary.retrieval.model_identity != commit.model_identity ||
            boundary.retrieval.session_generation != commit.session_generation ||
            boundary.retrieval.sequence_generation != commit.sequence_generation ||
            boundary.retrieval.sequence_id != commit.sequence_id ||
            boundary.retrieval.representation_epoch != commit.representation_epoch) return false;

        const auto get = [&](const llama_kv_page_id & id) -> const llama_kv_live_policy_page * {
            const auto found = std::find_if(boundary.pages.begin(), boundary.pages.end(),
                    [&](const auto & page) { return page.record.id == id; });
            return found == boundary.pages.end() ? nullptr : &*found;
        };
        uint64_t inventory_rollback_generation = 0;
        for (const auto & page : boundary.pages) {
            inventory_rollback_generation = std::max<uint64_t>(
                    inventory_rollback_generation, page.record.id.page_generation);
        }
        if (commit.rollback_generation != inventory_rollback_generation) return false;
        std::vector<const llama_kv_live_policy_page *> retrieval;
        retrieval.reserve(commit.selected.size());
        for (size_t i = 0; i < commit.selected.size(); ++i) {
            const auto & item = commit.selected[i];
            const auto * page = get(item.identity);
            if (page == nullptr || item.content_version == 0 ||
                page->record.content_version != item.content_version ||
                item.identity.model_identity != commit.model_identity ||
                item.identity.session_generation != commit.session_generation ||
                item.identity.sequence_generation != commit.sequence_generation ||
                item.identity.sequence_id != commit.sequence_id ||
                item.identity.representation_epoch != commit.representation_epoch ||
                (!page->record.host_valid && page->record.physical_slot == UINT32_MAX)) return false;
            for (size_t j = 0; j < i; ++j) {
                if (same_logical_page(commit.selected[j].identity, item.identity)) return false;
            }
            retrieval.push_back(page);
        }

        uint32_t generation_mandatory = 0;
        uint32_t retrieval_mandatory = 0;
        std::vector<const llama_kv_live_policy_page *> mandatory;
        std::vector<llama_kv_page_id> mandatory_ids;
        std::vector<bool> mandatory_generation;
        for (const auto & page : boundary.pages) {
            const bool write = boundary.has_write_page && page.record.id == boundary.write_page;
            const bool generation = page.current || write || page.inflight_pin ||
                page.speculative_pin || page.record.pin_count != 0;
            const bool pinned = generation || page.anchor || page.application_pin;
            if (!pinned) continue;
            const auto existing = std::find(mandatory_ids.begin(), mandatory_ids.end(),
                    page.record.id);
            if (existing == mandatory_ids.end()) {
                mandatory_ids.push_back(page.record.id);
                mandatory_generation.push_back(generation);
                mandatory.push_back(&page);
                if (generation) ++generation_mandatory;
                else ++retrieval_mandatory;
            } else if (generation) {
                const size_t index = size_t(existing - mandatory_ids.begin());
                if (!mandatory_generation[index]) {
                    mandatory_generation[index] = true;
                    --retrieval_mandatory;
                    ++generation_mandatory;
                }
            }
        }
        if (generation_mandatory > commit.generation_budget ||
            retrieval_mandatory > commit.retrieval_budget) return false;
        const uint32_t remaining_retrieval = commit.retrieval_budget - retrieval_mandatory;
        const size_t selected_history = std::count_if(retrieval.begin(), retrieval.end(),
                [&](const auto * page) {
            return std::find(mandatory_ids.begin(), mandatory_ids.end(),
                    page->record.id) == mandatory_ids.end();
        });
        if (selected_history > remaining_retrieval) return false;
        // A selected resident may already be mandatory (for example the
        // current generated page). Count it once when checking capacity; the
        // ordered target below also deduplicates these identities.
        if (mandatory.size() + selected_history > boundary.hot_capacity) return false;

        std::vector<const llama_kv_live_policy_page *> ordered;
        std::vector<llama_kv_page_id> ordered_ids;
        ordered.reserve(mandatory.size() + retrieval.size());
        ordered_ids.reserve(mandatory.size() + retrieval.size());
        for (const auto * page : mandatory) {
            if (std::find(ordered_ids.begin(), ordered_ids.end(), page->record.id) ==
                    ordered_ids.end()) {
                ordered.push_back(page);
                ordered_ids.push_back(page->record.id);
            }
        }
        for (const auto * page : retrieval) {
            if (std::find(ordered_ids.begin(), ordered_ids.end(), page->record.id) ==
                    ordered_ids.end()) {
                ordered.push_back(page);
                ordered_ids.push_back(page->record.id);
            }
        }
        std::vector<bool> used(boundary.snapshot.slot_capacity(), false);
        for (const auto * page : ordered) {
            if (page->record.physical_slot == UINT32_MAX) continue;
            const uint32_t slot = page->record.physical_slot;
            if (slot >= used.size() || used[slot]) return false;
            const auto resident = std::find_if(boundary.snapshot.pages().begin(),
                    boundary.snapshot.pages().end(), [&](const auto & current) {
                return current.id == page->record.id;
            });
            if (resident == boundary.snapshot.pages().end() ||
                resident->physical_slot != slot ||
                resident->content_version != page->record.content_version) return false;
            used[slot] = true;
        }
        target.reserve(ordered.size());
        for (const auto * page : ordered) {
            auto record = page->record;
            if (record.physical_slot == UINT32_MAX) {
                if (!record.host_valid) return false;
                uint32_t slot = UINT32_MAX;
                for (uint32_t i = 0; i < used.size(); ++i) if (!used[i]) { slot = i; break; }
                if (slot == UINT32_MAX) {
                    for (uint32_t i = 0; i < used.size(); ++i) {
                        const bool old_slot = std::any_of(boundary.snapshot.pages().begin(),
                                boundary.snapshot.pages().end(), [&](const auto & old) {
                            return old.physical_slot == i && std::none_of(ordered.begin(), ordered.end(),
                                    [&](const auto * keep) { return keep->record.id == old.id; });
                        });
                        if (old_slot && !used[i]) { slot = i; break; }
                    }
                }
                // Recycle a slot owned by an omitted clean resident.
                if (slot == UINT32_MAX) {
                    for (const auto & old : boundary.snapshot.pages()) {
                        const bool retained = std::any_of(ordered.begin(), ordered.end(),
                                [&](const auto * keep) { return keep->record.id == old.id; });
                        if (!retained && old.pin_count == 0 && !old.dirty) {
                            slot = old.physical_slot;
                            break;
                        }
                    }
                }
                if (slot == UINT32_MAX || slot >= used.size()) return false;
                record.physical_slot = slot;
                record.state = llama_kv_page_state::gpu_host_clean;
                record.dirty = false;
                record.pin_count = 0;
                used[slot] = true;
            }
            target.push_back(record);
        }
        return true;
    } catch (...) {
        target.clear();
        return false;
    }
}

bool llama_kv_live_policy_build_trace(
        const llama_kv_live_policy_boundary & boundary,
        llama_kv_live_policy_trace & output,
        llama_kv_policy_trace & policy_trace,
        std::vector<llama_kv_policy_page> & policy_pages) noexcept {
    output = {};
    policy_trace = {};
    policy_pages.clear();
    try {
        if (boundary.version != LLAMA_KV_LIVE_POLICY_VERSION ||
            boundary.snapshot.epoch() == 0 || boundary.hot_capacity == 0 ||
            boundary.logical_page_count == 0 || boundary.hot_capacity > boundary.snapshot.slot_capacity() ||
            boundary.pages.empty() || boundary.pages.size() > UINT32_MAX ||
            boundary.logical_page_count < boundary.pages.size()) return false;

        std::unordered_map<uint64_t, std::vector<size_t>> identity_buckets;
        std::vector<bool> physical_slots(boundary.snapshot.slot_capacity(), false);
        for (size_t page_index = 0; page_index < boundary.pages.size(); ++page_index) {
            const auto & page = boundary.pages[page_index];
            if (!llama_kv_page_id_valid(page.record.id,
                    llama_kv_page_id_is_tail(page.record.id)) ||
                (page.record.physical_slot == UINT32_MAX &&
                 page.record.state != llama_kv_page_state::host_clean)) return false;
            auto & bucket = identity_buckets[identity_hash(page.record.id)];
            for (const size_t prior_index : bucket) {
                const auto & prior = boundary.pages[prior_index].record.id;
                if (prior == page.record.id || same_logical_page(prior, page.record.id)) return false;
            }
            bucket.push_back(page_index);
            if (page.record.physical_slot != UINT32_MAX &&
                (page.record.physical_slot >= physical_slots.size() ||
                 physical_slots[page.record.physical_slot])) return false;
            if (page.record.physical_slot != UINT32_MAX) physical_slots[page.record.physical_slot] = true;
        }
        for (const auto & resident : boundary.snapshot.pages()) {
            const auto * page = find_page(boundary.pages, resident.id);
            if (!page || page->record.physical_slot != resident.physical_slot ||
                page->record.id != resident.id) return false;
        }
        for (size_t i = 0; i < boundary.previous_target.size(); ++i) {
            if (find_page(boundary.pages, boundary.previous_target[i]) == nullptr) return false;
            for (size_t j = 0; j < i; ++j) {
                if (boundary.previous_target[i] == boundary.previous_target[j]) return false;
            }
        }

        output.version = boundary.version;
        output.epoch = boundary.snapshot.epoch();
        output.hot_capacity = boundary.hot_capacity;
        output.logical_page_count = boundary.logical_page_count;
        output.pages.reserve(boundary.pages.size());
        policy_pages.reserve(boundary.pages.size());
        policy_trace.version = LLAMA_KV_POLICY_TRACE_VERSION;
        policy_trace.epoch = boundary.snapshot.epoch();
        policy_trace.capacity_pages = boundary.hot_capacity;
        policy_trace.pages.reserve(boundary.pages.size());

        uint32_t write_policy_id = 0;
        for (size_t i = 0; i < boundary.pages.size(); ++i) {
            const auto & source = boundary.pages[i];
            const uint32_t id = uint32_t(i + 1);
            llama_kv_policy_page page;
            page.id = id;
            page.attention_layer = source.attention_layer;
            page.age = source.age;
            page.recency = source.recency;
            page.attention_ema_q = source.attention_ema_q;
            page.retrieval_hits = source.retrieval_hits;
            page.reuse_count = source.reuse_count;
            page.dirty_cost = source.dirty_cost;
            page.fault_cost = source.fault_cost;
            page.recent_peak_q = source.recent_peak_q;
            page.hysteresis_q = source.hysteresis_q;
            page.attention_sample_count = source.attention_sample_count;
            page.attention_last_observed = source.attention_last_observed;
            page.attention_observed = source.attention_observed;
            page.recent = source.recent;
            page.anchor = source.anchor;
            page.structural = source.structural;
            page.current = source.current;
            page.application_pin = source.application_pin;
            page.inflight_pin = source.inflight_pin || source.record.pin_count != 0;
            page.speculative_pin = source.speculative_pin;
            page.resident = source.record.physical_slot != UINT32_MAX;
            policy_pages.push_back(page);
            policy_trace.pages.push_back(page);
            const auto * retrieval = find_retrieval(boundary.retrieval, source.record.id);
            llama_kv_live_policy_trace_page trace_page;
            trace_page.id = source.record.id;
            trace_page.attention_layer = source.attention_layer;
            trace_page.resident = page.resident;
            trace_page.host_valid = source.record.host_valid;
            trace_page.policy_id = id;
            if (retrieval != nullptr && retrieval->score_available) {
                if (!std::isfinite(retrieval->score)) return false;
                trace_page.retrieval_score = retrieval->score;
                trace_page.retrieval_score_available = true;
            }
            output.pages.push_back(trace_page);
            if (boundary.has_write_page && source.record.id == boundary.write_page) {
                write_policy_id = id;
            }
            const bool mandatory = page.current || page.anchor || page.application_pin ||
                page.inflight_pin || page.speculative_pin ||
                (boundary.has_write_page && source.record.id == boundary.write_page);
            output.mandatory_pages += mandatory;
        }
        if (boundary.retrieval.status == llama_kv_routing_retrieval_status::ok &&
            (boundary.retrieval.table_epoch == 0 ||
             boundary.retrieval.table_epoch != boundary.snapshot.epoch())) return false;
        if (boundary.has_write_page && write_policy_id == 0) return false;
        policy_trace.write_page = write_policy_id;

        for (const auto & id : boundary.previous_target) {
            const auto * page = find_page(boundary.pages, id);
            if (page == nullptr) return false;
            const auto it = std::find_if(output.pages.begin(), output.pages.end(),
                    [&](const auto & value) { return value.id == id; });
            if (it == output.pages.end()) return false;
            policy_trace.pages[it->policy_id - 1].hysteresis_q =
                std::max(policy_trace.pages[it->policy_id - 1].hysteresis_q,
                         boundary.policy.hysteresis_q);
        }
        if (boundary.retrieval.status != llama_kv_routing_retrieval_status::ok) {
            output.retrieval_fallback = true;
        }
        if (boundary.retrieval.status == llama_kv_routing_retrieval_status::ok) {
            for (const auto & entry : boundary.retrieval.selected) {
                const auto it = std::find_if(output.pages.begin(), output.pages.end(),
                        [&](const auto & value) { return value.id == entry.id; });
                if (it != output.pages.end()) policy_trace.summary_top_k.push_back(it->policy_id);
            }
            policy_trace.exploration.reserve(boundary.retrieval.selected.size());
        }
        for (const auto & entry : boundary.retrieval.selected) {
            if (boundary.retrieval.status != llama_kv_routing_retrieval_status::ok ||
                entry.reason != llama_kv_routing_retrieval_reason::exploration) continue;
            const auto it = std::find_if(output.pages.begin(), output.pages.end(),
                    [&](const auto & value) { return value.id == entry.id; });
            if (it != output.pages.end()) policy_trace.exploration.push_back(it->policy_id);
        }
        for (const auto & page : policy_trace.pages) {
            if (!page.attention_observed) ++output.unavailable_attention_pages;
        }
        return true;
    } catch (...) {
        output = {};
        policy_trace = {};
        policy_pages.clear();
        return false;
    }
}

llama_kv_live_policy_result llama_kv_live_policy_apply(
        llama_kv_residency_table & table,
        llama_kv_residency_pool & pool,
        const llama_kv_live_policy_boundary & boundary,
        const llama_kv_residency_pool_backend & backend,
        const llama_kv_residency_transfer_transport & transport,
        const llama_kv_residency_transaction_hooks & hooks) noexcept {
    llama_kv_live_policy_result output;
    output.base_epoch = table.snapshot().epoch();
    llama_kv_policy_trace policy_trace;
    std::vector<llama_kv_policy_page> policy_pages;
    if (!llama_kv_live_policy_build_trace(boundary, output.trace, policy_trace, policy_pages)) {
        output.status = llama_kv_live_policy_status::invalid_argument;
        return output;
    }
    const auto current = table.snapshot();
    if (!same_snapshot(current, boundary.snapshot)) {
        output.status = llama_kv_live_policy_status::stale_snapshot;
        return output;
    }
    output.base_epoch = current.epoch();

    auto policy_config = boundary.policy;
    policy_config.capacity_pages = boundary.hot_capacity;
    policy_trace.pages = policy_pages;
    std::vector<uint64_t> previous_policy_target;
    previous_policy_target.reserve(boundary.previous_target.size());
    for (const auto & id : boundary.previous_target) {
        const auto it = std::find_if(output.trace.pages.begin(), output.trace.pages.end(),
                [&](const auto & page) { return page.id == id; });
        if (it == output.trace.pages.end()) {
            output.status = llama_kv_live_policy_status::stale_snapshot;
            return output;
        }
        previous_policy_target.push_back(it->policy_id);
    }
    std::vector<llama_kv_page_record> committed_target;
    if (boundary.query_commit.enabled) {
        if (!llama_kv_live_policy_prepare_query_target(boundary, committed_target)) {
            output.status = llama_kv_live_policy_status::query_capacity_refused;
            return output;
        }
        output.policy.status = llama_kv_policy_status::ok;
        for (const auto & record : committed_target) {
            const auto found = std::find_if(output.trace.pages.begin(), output.trace.pages.end(),
                    [&](const auto & page) { return page.id == record.id; });
            if (found == output.trace.pages.end()) {
                output.status = llama_kv_live_policy_status::stale_snapshot;
                return output;
            }
            output.policy.target.push_back(found->policy_id);
            const auto prior = std::find(boundary.previous_target.begin(),
                    boundary.previous_target.end(), record.id);
            if (prior != boundary.previous_target.end()) output.policy.keeps.push_back(found->policy_id);
            else output.policy.adds.push_back(found->policy_id);
            output.policy.records.push_back({ found->policy_id,
                std::find_if(boundary.query_commit.selected.begin(), boundary.query_commit.selected.end(),
                    [&](const auto & page) { return page.identity == record.id; }) !=
                    boundary.query_commit.selected.end()
                    ? llama_kv_policy_reason::summary : llama_kv_policy_reason::mandatory, 0 });
        }
        for (const auto & old : current.pages()) {
            if (std::none_of(committed_target.begin(), committed_target.end(),
                    [&](const auto & page) { return page.id == old.id; })) {
                const auto found = std::find_if(output.trace.pages.begin(), output.trace.pages.end(),
                        [&](const auto & page) { return page.id == old.id; });
                if (found != output.trace.pages.end()) {
                    output.policy.victims.push_back(found->policy_id);
                    output.policy.records.push_back({ found->policy_id, llama_kv_policy_reason::victim, 0 });
                }
            }
        }
    } else {
        output.policy = llama_kv_policy_decide(policy_trace, policy_config, previous_policy_target);
    }
    output.trace.target_pages = uint32_t(output.policy.target.size());
    output.retrieval_fallback = output.trace.retrieval_fallback;
    if (output.policy.status == llama_kv_policy_status::pin_overflow) {
        output.status = llama_kv_live_policy_status::mandatory_overflow;
        return output;
    }
    if (output.policy.status != llama_kv_policy_status::ok ||
        (!boundary.query_commit.enabled &&
         output.policy.target.size() != std::min<uint32_t>(boundary.hot_capacity,
                                                            boundary.logical_page_count))) {
        output.status = llama_kv_live_policy_status::unavailable_inventory;
        return output;
    }

    // A missing or stale routing summary may not evict a previously complete
    // safe table. Reuse it when it still covers all mandatory pages.
    const bool telemetry_unavailable = output.trace.unavailable_attention_pages == output.trace.pages.size();
    // A first useful query commonly arrives before any page has published an
    // attention sample.  Do not let the conservative anchor fallback erase a
    // valid retrieval decision for a host-backed cold page: that decision is
    // precisely the evidence which can make the page resident for the next
    // causal operation.  The fallback remains active when there is no such
    // cold retrieval target.
    const bool selected_cold_page = boundary.retrieval.status ==
            llama_kv_routing_retrieval_status::ok &&
        std::any_of(boundary.retrieval.selected.begin(),
            boundary.retrieval.selected.end(), [&](const auto & entry) {
                const auto * page = find_page(boundary.pages, entry.id);
                return page != nullptr && page->record.physical_slot == UINT32_MAX &&
                    page->record.host_valid;
            });
    if (!boundary.query_commit.enabled && (output.retrieval_fallback ||
            (telemetry_unavailable && !selected_cold_page)) &&
        boundary.previous_target.size() == output.policy.target.size()) {
        bool safe = true;
        for (const auto & page : boundary.pages) {
            const bool mandatory = page.current || page.anchor || page.application_pin ||
                page.inflight_pin || page.record.pin_count != 0 || page.speculative_pin ||
                (boundary.has_write_page && page.record.id == boundary.write_page);
            if (mandatory && !contains_id(boundary.previous_target, page.record.id)) safe = false;
        }
        if (safe) {
            output.policy.target.clear();
            output.policy.adds.clear();
            output.policy.keeps.clear();
            output.policy.victims.clear();
            output.policy.records.clear();
            for (const auto & id : boundary.previous_target) {
                const auto it = std::find_if(output.trace.pages.begin(), output.trace.pages.end(),
                        [&](const auto & page) { return page.id == id; });
                if (it == output.trace.pages.end()) { safe = false; break; }
                output.policy.target.push_back(it->policy_id);
            }
            if (safe) {
                for (const auto policy_id : output.policy.target) {
                    output.policy.keeps.push_back(policy_id);
                    output.policy.records.push_back({ policy_id, llama_kv_policy_reason::keep, 0 });
                }
                output.status = llama_kv_live_policy_status::safe_fallback;
            }
        }
    }

    std::vector<bool> used(pool.slot_capacity(), false);
    std::vector<llama_kv_page_record> desired;
    if (boundary.query_commit.enabled) {
        desired = committed_target;
    }
    desired.reserve(output.policy.target.size());
    // Preserve the physical slots of pages that remain resident before
    // assigning a slot to a cold promotion. This permits an occupied pool to
    // replace a victim without accidentally assigning the new page to a slot
    // still owned by a retained page.
    if (!boundary.query_commit.enabled) for (const uint64_t policy_id : output.policy.target) {
        if (policy_id == 0 || policy_id > output.trace.pages.size()) {
            output.status = llama_kv_live_policy_status::unavailable_inventory;
            return output;
        }
        const auto & trace_page = output.trace.pages[size_t(policy_id - 1)];
        const auto * source = find_page(boundary.pages, trace_page.id);
        if (source == nullptr) {
            output.status = llama_kv_live_policy_status::unavailable_inventory;
            return output;
        }
        if (source->record.physical_slot != UINT32_MAX) {
            if (source->record.physical_slot >= used.size() ||
                used[source->record.physical_slot]) {
                output.status = llama_kv_live_policy_status::invalid_argument;
                return output;
            }
            used[source->record.physical_slot] = true;
        }
    }
    if (!boundary.query_commit.enabled) for (const uint64_t policy_id : output.policy.target) {
        if (policy_id == 0 || policy_id > output.trace.pages.size()) {
            output.status = llama_kv_live_policy_status::unavailable_inventory;
            return output;
        }
        const auto & trace_page = output.trace.pages[size_t(policy_id - 1)];
        const auto * source = find_page(boundary.pages, trace_page.id);
        if (source == nullptr) {
            output.status = llama_kv_live_policy_status::unavailable_inventory;
            return output;
        }
        llama_kv_page_record record = source->record;
        if (record.physical_slot != UINT32_MAX) {
            if (record.physical_slot >= used.size()) {
                output.status = llama_kv_live_policy_status::invalid_argument;
                return output;
            }
        } else {
            uint32_t slot = UINT32_MAX;
            for (uint32_t i = 0; i < used.size(); ++i) {
                if (!used[i]) { slot = i; break; }
            }
            if (slot == UINT32_MAX) {
                output.status = llama_kv_live_policy_status::unavailable_inventory;
                return output;
            }
            if (!record.host_valid) {
                output.status = llama_kv_live_policy_status::missing_host_source;
                return output;
            }
            record.physical_slot = slot;
            record.state = llama_kv_page_state::gpu_host_clean;
            record.dirty = false;
            record.pin_count = 0;
            used[slot] = true;
        }
        desired.push_back(record);
    }

    for (size_t i = 0; i < desired.size(); ++i) {
        const auto * policy_record = find_policy_record(output.policy, output.policy.target[i]);
        const auto previous = std::find(boundary.previous_target.begin(), boundary.previous_target.end(),
                                        desired[i].id);
        llama_kv_live_policy_decision_entry record;
        record.id = desired[i].id;
        record.selection_reason = policy_record != nullptr
            ? policy_record->reason : llama_kv_policy_reason::keep;
        record.retention_score_q = policy_record != nullptr ? policy_record->normalized_keep_q : 0;
        const auto * retrieval = find_retrieval(boundary.retrieval, desired[i].id);
        if (retrieval != nullptr) {
            record.retrieval_score = retrieval->score;
            record.retrieval_score_available = retrieval->score_available;
        }
        record.kept = previous != boundary.previous_target.end();
        record.added = !record.kept;
        output.decisions.push_back(record);
    }
    for (const uint64_t policy_id : output.policy.victims) {
        if (policy_id == 0 || policy_id > output.trace.pages.size()) continue;
        llama_kv_live_policy_decision_entry record;
        record.id = output.trace.pages[size_t(policy_id - 1)].id;
        record.selection_reason = llama_kv_policy_reason::victim;
        const auto * policy_record = find_policy_record(output.policy, policy_id);
        record.retention_score_q = policy_record != nullptr ? policy_record->normalized_keep_q : 0;
        record.victim = true;
        output.decisions.push_back(record);
    }
    output.target_pages = desired;

    // A transfer plan is an opaque byte schedule, but its page identity and
    // physical destination are still controller invariants. Check these before
    // entering the transaction so a malformed plan cannot copy a valid page
    // into another target's slot.
    const auto desired_page = [&](const llama_kv_page_id & id) {
        return std::find_if(desired.begin(), desired.end(),
                [&](const auto & page) { return page.id == id; });
    };
    const auto current_page = [&](const llama_kv_page_id & id) {
        return std::find_if(current.pages().begin(), current.pages().end(),
                [&](const auto & page) { return page.id == id; });
    };
    for (const auto & plan : boundary.transaction.transfers) {
        for (const auto & transfer_page : plan.pages) {
            const auto target = desired_page(transfer_page.page);
            const auto old = current_page(transfer_page.page);
            const auto match = plan.direction == llama_kv_residency_transfer_direction::h2d_promotion
                ? target : old;
            if (match == (plan.direction == llama_kv_residency_transfer_direction::h2d_promotion
                    ? desired.end() : current.pages().end()) ||
                transfer_page.physical_slot != match->physical_slot ||
                transfer_page.table_epoch != current.epoch()) {
                output.status = llama_kv_live_policy_status::invalid_argument;
                return output;
            }
        }
    }
    for (const auto & target : desired) {
        if (current_page(target.id) != current.pages().end()) continue;
        const bool has_promotion = std::any_of(
                boundary.transaction.transfers.begin(), boundary.transaction.transfers.end(),
                [&](const auto & plan) {
                    return plan.direction == llama_kv_residency_transfer_direction::h2d_promotion &&
                        std::any_of(plan.pages.begin(), plan.pages.end(),
                            [&](const auto & page) { return page.page == target.id; });
                });
        if (!has_promotion) {
            output.status = llama_kv_live_policy_status::missing_host_source;
            return output;
        }
    }

    bool changed = desired.size() != current.pages().size();
    if (!changed) {
        for (const auto & page : desired) {
            const auto it = std::find_if(current.pages().begin(), current.pages().end(),
                    [&](const auto & value) { return value.id == page.id && value.physical_slot == page.physical_slot; });
            if (it == current.pages().end()) { changed = true; break; }
        }
    }
    if (!changed) {
        if (output.status != llama_kv_live_policy_status::safe_fallback) {
            output.status = llama_kv_live_policy_status::no_change;
        }
        output.published_epoch = current.epoch();
        return output;
    }

    auto request = boundary.transaction;
    request.desired_pages = desired;
    output.publication_attempted = true;
    output.transaction = llama_kv_residency_execute_transaction(
            table, pool, request, backend, transport, hooks);
    output.published = output.transaction.published;
    output.published_epoch = output.transaction.published_epoch;
    if (output.transaction.status == llama_kv_residency_transaction_status::committed) {
        output.status = llama_kv_live_policy_status::committed;
    } else {
        output.status = transaction_status(output.transaction.status);
    }
    return output;
}
