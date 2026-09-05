#include "llama-kv-live-lifecycle.h"

#include <algorithm>
#include <limits>
#include <new>

namespace {

bool same_generation(const llama_kv_live_lifecycle_generation & a,
                     const llama_kv_live_lifecycle_generation & b) noexcept {
    return a.session_generation == b.session_generation &&
           a.sequence_id == b.sequence_id &&
           a.sequence_generation == b.sequence_generation &&
           a.operation_generation == b.operation_generation;
}

bool page_matches(const llama_kv_page_id & page,
                  const llama_kv_live_lifecycle_generation & generation) noexcept {
    // A logical-page match alone would let a reused slot or an old
    // model/codec image enter a new generation.
    return llama_kv_page_id_valid(page, llama_kv_page_id_is_tail(page)) &&
           page.page_generation != 0 && page.representation_epoch != 0 &&
           page.model_identity != 0 && page.topology_identity != 0 &&
           page.codec_digest != 0 && page.codebook_digest != 0 &&
           page.rotation_digest != 0 && page.meansub_digest != 0 &&
           page.session_generation == generation.session_generation &&
           page.sequence_id == generation.sequence_id &&
           page.sequence_generation == generation.sequence_generation;
}

} // namespace

bool llama_kv_live_lifecycle_frontier::valid() const noexcept {
    if (target_tokens < 0 || recurrent_tokens < 0 || mtp_tokens < 0 ||
        target_tokens != recurrent_tokens || target_tokens != mtp_tokens ||
        speculative_accepted > speculative_proposed ||
        speculative_rejected != speculative_proposed - speculative_accepted) {
        return false;
    }
    return true;
}

const char * llama_kv_live_lifecycle_status_name(
        llama_kv_live_lifecycle_status status) noexcept {
    switch (status) {
        case llama_kv_live_lifecycle_status::committed: return "committed";
        case llama_kv_live_lifecycle_status::ready: return "ready";
        case llama_kv_live_lifecycle_status::waited_ready: return "waited_ready";
        case llama_kv_live_lifecycle_status::reuse_old_hot_set: return "reuse_old_hot_set";
        case llama_kv_live_lifecycle_status::fallback_larger_union: return "fallback_larger_union";
        case llama_kv_live_lifecycle_status::invalid_argument: return "invalid_argument";
        case llama_kv_live_lifecycle_status::not_configured: return "not_configured";
        case llama_kv_live_lifecycle_status::stale_generation: return "stale_generation";
        case llama_kv_live_lifecycle_status::backpressure: return "backpressure";
        case llama_kv_live_lifecycle_status::prefetch_failed: return "prefetch_failed";
        case llama_kv_live_lifecycle_status::policy_failed: return "policy_failed";
        case llama_kv_live_lifecycle_status::companion_rejected: return "companion_rejected";
        case llama_kv_live_lifecycle_status::cancelled: return "cancelled";
        case llama_kv_live_lifecycle_status::shutdown: return "shutdown";
        case llama_kv_live_lifecycle_status::unsupported_slots: return "unsupported_slots";
        case llama_kv_live_lifecycle_status::_count: break;
    }
    return "invalid";
}

std::unique_ptr<llama_kv_live_lifecycle> llama_kv_live_lifecycle::create(
        const llama_kv_live_lifecycle_config & config,
        const llama_kv_prefetch_backend & prefetch_backend,
        const llama_kv_live_lifecycle_hooks & hooks,
        llama_kv_live_lifecycle_status & status) noexcept {
    status = llama_kv_live_lifecycle_status::invalid_argument;
    if (config.slot_count != 1) {
        status = llama_kv_live_lifecycle_status::unsupported_slots;
        return nullptr;
    }
    if (!hooks.publish_companions) {
        status = llama_kv_live_lifecycle_status::not_configured;
        return nullptr;
    }
    llama_kv_prefetch_status prefetch_status;
    auto prefetch = llama_kv_prefetch_scheduler::create(
            config.prefetch, prefetch_backend, prefetch_status);
    if (!prefetch) {
        status = prefetch_status == llama_kv_prefetch_status::not_configured
            ? llama_kv_live_lifecycle_status::not_configured
            : llama_kv_live_lifecycle_status::invalid_argument;
        return nullptr;
    }
    try {
        auto result = std::unique_ptr<llama_kv_live_lifecycle>(
                new llama_kv_live_lifecycle(config, std::move(prefetch), hooks));
        status = llama_kv_live_lifecycle_status::ready;
        return result;
    } catch (...) {
        status = llama_kv_live_lifecycle_status::not_configured;
        return nullptr;
    }
}

llama_kv_live_lifecycle::llama_kv_live_lifecycle(
        const llama_kv_live_lifecycle_config & config,
        std::unique_ptr<llama_kv_prefetch_scheduler> prefetch,
        llama_kv_live_lifecycle_hooks hooks)
    : config_(config), prefetch_(std::move(prefetch)), hooks_(hooks) {}

llama_kv_live_lifecycle::~llama_kv_live_lifecycle() {
    shutdown();
}

bool llama_kv_live_lifecycle::current() const noexcept {
    return active_ && !stopped_ && generation_.valid() &&
           (!hooks_.generation_current || hooks_.generation_current(
                   hooks_.context, generation_));
}

bool llama_kv_live_lifecycle::matches_current(uint64_t generation) const noexcept {
    return generation == 0 || (current() && generation == generation_.operation_generation);
}

void llama_kv_live_lifecycle::cancel_tracked() noexcept {
    for (const auto & intent : tracked_) {
        prefetch_->cancel(intent.page_id, intent.generation);
    }
    tracked_.clear();
}

llama_kv_live_lifecycle_status llama_kv_live_lifecycle::start(
        const llama_kv_live_lifecycle_generation & generation) noexcept {
    if (stopped_) return llama_kv_live_lifecycle_status::shutdown;
    if (!generation.valid()) return llama_kv_live_lifecycle_status::invalid_argument;
    if (generation_.valid() &&
            generation.operation_generation <= generation_.operation_generation) {
        return llama_kv_live_lifecycle_status::stale_generation;
    }
    if (hooks_.generation_current && !hooks_.generation_current(hooks_.context, generation)) {
        return llama_kv_live_lifecycle_status::stale_generation;
    }
    cancel_tracked();
    generation_ = generation;
    frontier_ = {};
    active_ = true;
    return llama_kv_live_lifecycle_status::ready;
}

llama_kv_live_lifecycle_status llama_kv_live_lifecycle::set_frontier(
        const llama_kv_live_lifecycle_frontier & frontier) noexcept {
    if (!current()) return stopped_ ? llama_kv_live_lifecycle_status::shutdown
                                    : llama_kv_live_lifecycle_status::stale_generation;
    if (!frontier.valid()) return llama_kv_live_lifecycle_status::invalid_argument;
    // The committed target/recurrent/MTP frontier is monotone within one
    // operation generation. Speculative proposed tokens may shrink after a
    // rejection, but an accepted prefix may never move backwards.
    if (frontier.target_tokens < frontier_.target_tokens ||
        frontier.recurrent_tokens < frontier_.recurrent_tokens ||
        frontier.mtp_tokens < frontier_.mtp_tokens ||
        frontier.speculative_accepted < frontier_.speculative_accepted) {
        return llama_kv_live_lifecycle_status::stale_generation;
    }
    frontier_ = frontier;
    return llama_kv_live_lifecycle_status::ready;
}

llama_kv_live_lifecycle_status llama_kv_live_lifecycle::map_prefetch_status(
        llama_kv_prefetch_status status) noexcept {
    switch (status) {
        case llama_kv_prefetch_status::ok: return llama_kv_live_lifecycle_status::ready;
        case llama_kv_prefetch_status::shutdown: return llama_kv_live_lifecycle_status::shutdown;
        case llama_kv_prefetch_status::stale_generation: return llama_kv_live_lifecycle_status::stale_generation;
        case llama_kv_prefetch_status::queue_full:
        case llama_kv_prefetch_status::event_full:
        case llama_kv_prefetch_status::staging_full:
        case llama_kv_prefetch_status::backpressure: return llama_kv_live_lifecycle_status::backpressure;
        case llama_kv_prefetch_status::cancelled: return llama_kv_live_lifecycle_status::cancelled;
        default: return llama_kv_live_lifecycle_status::prefetch_failed;
    }
}

llama_kv_live_lifecycle_status llama_kv_live_lifecycle::prefetch(
        const std::vector<llama_kv_prefetch_intent> & ranked) noexcept {
    if (!current()) return stopped_ ? llama_kv_live_lifecycle_status::shutdown
                                    : llama_kv_live_lifecycle_status::stale_generation;
    try {
        std::vector<llama_kv_prefetch_intent> normalized;
        normalized.reserve(std::min<size_t>(ranked.size(), config_.prefetch.prefetch_depth));
        for (const auto & input : ranked) {
            if (input.page_id == 0 || input.useful_bytes == 0 ||
                input.aligned_bytes < input.useful_bytes ||
                !matches_current(input.generation)) {
                return llama_kv_live_lifecycle_status::stale_generation;
            }
            auto it = std::find_if(normalized.begin(), normalized.end(),
                    [&](const auto & value) { return value.page_id == input.page_id; });
            if (it == normalized.end()) {
                auto value = input;
                value.generation = generation_.operation_generation;
                value.required = false;
                normalized.push_back(value);
            } else {
                it->priority = std::max(it->priority, input.priority);
                it->useful_bytes = std::max(it->useful_bytes, input.useful_bytes);
                it->aligned_bytes = std::max(it->aligned_bytes, input.aligned_bytes);
            }
        }
        std::stable_sort(normalized.begin(), normalized.end(),
                [](const auto & a, const auto & b) { return a.priority > b.priority; });
        const auto status = prefetch_->prefetch(normalized);
        if (status == llama_kv_prefetch_status::ok ||
            status == llama_kv_prefetch_status::event_full ||
            status == llama_kv_prefetch_status::backpressure) {
            tracked_.insert(tracked_.end(), normalized.begin(), normalized.end());
        }
        return map_prefetch_status(status);
    } catch (...) {
        return llama_kv_live_lifecycle_status::prefetch_failed;
    }
}

llama_kv_live_lifecycle_status llama_kv_live_lifecycle::observe_query(
        uint32_t layer, uint64_t token,
        const std::vector<llama_kv_prefetch_intent> & ranked) noexcept {
    if (!current()) return stopped_ ? llama_kv_live_lifecycle_status::shutdown
                                    : llama_kv_live_lifecycle_status::stale_generation;
    try {
        std::vector<llama_kv_prefetch_intent> normalized;
        normalized.reserve(ranked.size());
        for (const auto & input : ranked) {
            if (input.page_id == 0 || input.useful_bytes == 0 ||
                input.aligned_bytes < input.useful_bytes ||
                !matches_current(input.generation)) {
                return llama_kv_live_lifecycle_status::stale_generation;
            }
            auto value = input;
            value.generation = generation_.operation_generation;
            value.required = false;
            value.prediction = false;
            value.prediction_useful_counted = false;
            value.prediction_hit_counted = false;
            normalized.push_back(value);
        }
        return prefetch_->observe_query(generation_.operation_generation,
                                         layer, token, normalized)
            ? llama_kv_live_lifecycle_status::ready
            : llama_kv_live_lifecycle_status::prefetch_failed;
    } catch (...) {
        return llama_kv_live_lifecycle_status::prefetch_failed;
    }
}

std::vector<llama_kv_prefetch_intent> llama_kv_live_lifecycle::predict_next(
        uint32_t layer, uint64_t token, uint32_t limit) const noexcept {
    if (!current()) return {};
    return prefetch_->predict_next(generation_.operation_generation,
                                   layer, token, limit);
}

llama_kv_live_lifecycle_resolution llama_kv_live_lifecycle::ensure_ready(
        const std::vector<llama_kv_prefetch_intent> & required,
        const std::vector<uint64_t> & previous_hot_set,
        uint32_t wait_budget_steps) noexcept {
    llama_kv_live_lifecycle_resolution output;
    if (!current()) {
        output.status = stopped_ ? llama_kv_live_lifecycle_status::shutdown
                                  : llama_kv_live_lifecycle_status::stale_generation;
        return output;
    }
    try {
        for (size_t i = 0; i < previous_hot_set.size(); ++i) {
            if (previous_hot_set[i] == 0 || std::find(
                    previous_hot_set.begin(), previous_hot_set.begin() + i,
                    previous_hot_set[i]) != previous_hot_set.begin() + i) {
                output.status = llama_kv_live_lifecycle_status::invalid_argument;
                return output;
            }
        }
        std::vector<llama_kv_prefetch_intent> normalized;
        normalized.reserve(required.size());
        for (const auto & input : required) {
            if (input.page_id == 0 || input.useful_bytes == 0 ||
                input.aligned_bytes < input.useful_bytes ||
                !matches_current(input.generation)) {
                output.status = llama_kv_live_lifecycle_status::stale_generation;
                return output;
            }
            auto value = input;
            value.generation = generation_.operation_generation;
            value.required = true;
            normalized.push_back(value);
        }
        output.prefetch = prefetch_->ensure_ready(
                normalized, previous_hot_set, wait_budget_steps);
        for (const auto & intent : normalized) {
            tracked_.push_back(intent);
        }
        switch (output.prefetch.readiness) {
            case llama_kv_prefetch_readiness::ready:
                output.status = llama_kv_live_lifecycle_status::ready; break;
            case llama_kv_prefetch_readiness::waited_ready:
                output.status = llama_kv_live_lifecycle_status::waited_ready; break;
            case llama_kv_prefetch_readiness::reuse_old_hot_set:
                output.status = llama_kv_live_lifecycle_status::reuse_old_hot_set; break;
            case llama_kv_prefetch_readiness::fallback_larger_union:
                output.status = llama_kv_live_lifecycle_status::fallback_larger_union; break;
            case llama_kv_prefetch_readiness::not_configured:
                output.status = llama_kv_live_lifecycle_status::not_configured; break;
            case llama_kv_prefetch_readiness::cancelled:
                output.status = llama_kv_live_lifecycle_status::cancelled; break;
        }
        return output;
    } catch (...) {
        output.status = llama_kv_live_lifecycle_status::prefetch_failed;
        return output;
    }
}

llama_kv_live_lifecycle_status llama_kv_live_lifecycle::advance() noexcept {
    if (!current()) return stopped_ ? llama_kv_live_lifecycle_status::shutdown
                                    : llama_kv_live_lifecycle_status::stale_generation;
    return map_prefetch_status(prefetch_->advance());
}

struct llama_kv_live_lifecycle::transaction_context {
    llama_kv_live_lifecycle * owner = nullptr;
    llama_kv_residency_transaction_hooks upstream;
    llama_kv_live_lifecycle_generation generation;
    llama_kv_live_lifecycle_frontier frontier;
    bool companion_published = false;
    bool companion_attempted = false;
};

bool llama_kv_live_lifecycle::transaction_phase(void * opaque,
        llama_kv_residency_transaction_phase phase) noexcept {
    auto & context = *static_cast<transaction_context *>(opaque);
    if (context.upstream.phase && !context.upstream.phase(context.upstream.context, phase)) {
        return false;
    }
    if (phase != llama_kv_residency_transaction_phase::publish) return true;
    if (!context.owner->current() ||
        !same_generation(context.owner->generation(), context.generation)) {
        return false;
    }
    context.companion_attempted = true;
    if (!context.owner->hooks_.publish_companions(context.owner->hooks_.context,
                context.generation, context.frontier)) return false;
    context.companion_published = true;
    return true;
}

bool llama_kv_live_lifecycle::transaction_pin(void * opaque, const llama_kv_page_id & page) noexcept {
    auto & c = *static_cast<transaction_context *>(opaque);
    return !c.upstream.pin || c.upstream.pin(c.upstream.context, page);
}
void llama_kv_live_lifecycle::transaction_unpin(void * opaque, const llama_kv_page_id & page) noexcept {
    auto & c = *static_cast<transaction_context *>(opaque);
    if (c.upstream.unpin) c.upstream.unpin(c.upstream.context, page);
}
bool llama_kv_live_lifecycle::transaction_drop(void * opaque, const llama_kv_page_record & page) noexcept {
    auto & c = *static_cast<transaction_context *>(opaque);
    return !c.upstream.drop_clean || c.upstream.drop_clean(c.upstream.context, page);
}
bool llama_kv_live_lifecycle::transaction_restore(void * opaque, const llama_kv_page_record & page) noexcept {
    auto & c = *static_cast<transaction_context *>(opaque);
    return !c.upstream.restore_clean || c.upstream.restore_clean(c.upstream.context, page);
}
void llama_kv_live_lifecycle::transaction_retire(void * opaque, const llama_kv_page_record & page) noexcept {
    auto & c = *static_cast<transaction_context *>(opaque);
    if (c.upstream.retire) c.upstream.retire(c.upstream.context, page);
}
bool llama_kv_live_lifecycle::transaction_host(void * opaque, const llama_kv_page_id & page,
        uint64_t bytes) noexcept {
    auto & c = *static_cast<transaction_context *>(opaque);
    return !c.upstream.has_clean_host || c.upstream.has_clean_host(c.upstream.context, page, bytes);
}
bool llama_kv_live_lifecycle::transaction_recheck(void * opaque, uint64_t epoch,
        const std::vector<llama_kv_page_record> & desired) noexcept {
    auto & c = *static_cast<transaction_context *>(opaque);
    if (!c.owner->current() || !same_generation(c.owner->generation(), c.generation)) return false;
    return !c.upstream.recheck || c.upstream.recheck(c.upstream.context, epoch, desired);
}

llama_kv_live_lifecycle_result llama_kv_live_lifecycle::apply_policy(
        llama_kv_residency_table & table,
        llama_kv_residency_pool & pool,
        const llama_kv_live_policy_boundary & boundary,
        const llama_kv_residency_pool_backend & backend,
        const llama_kv_residency_transfer_transport & transport,
        const llama_kv_residency_transaction_hooks & hooks) noexcept {
    llama_kv_live_lifecycle_result output;
    output.generation = generation_;
    output.frontier = frontier_;
    if (!current()) {
        output.status = stopped_ ? llama_kv_live_lifecycle_status::shutdown
                                 : llama_kv_live_lifecycle_status::stale_generation;
        return output;
    }
    if (boundary.version != LLAMA_KV_LIVE_POLICY_VERSION ||
        !frontier_.valid() || boundary.snapshot.epoch() == 0 ||
        boundary.pages.empty()) {
        output.status = llama_kv_live_lifecycle_status::invalid_argument;
        return output;
    }
    for (const auto & page : boundary.pages) {
        if (!page_matches(page.record.id, generation_)) {
            output.status = llama_kv_live_lifecycle_status::stale_generation;
            return output;
        }
    }

    transaction_context context;
    context.owner = this;
    context.upstream = hooks;
    context.generation = generation_;
    context.frontier = frontier_;
    llama_kv_residency_transaction_hooks fenced;
    fenced.context = &context;
    fenced.phase = transaction_phase;
    fenced.pin = transaction_pin;
    fenced.unpin = transaction_unpin;
    fenced.drop_clean = transaction_drop;
    fenced.restore_clean = transaction_restore;
    fenced.retire = transaction_retire;
    fenced.has_clean_host = transaction_host;
    fenced.recheck = transaction_recheck;

    output.policy = llama_kv_live_policy_apply(
            table, pool, boundary, backend, transport, fenced);
    output.companion_published = context.companion_published;
    output.generation = generation_;
    output.frontier = frontier_;
    if (!output.policy.published) {
        if (output.policy.status == llama_kv_live_policy_status::no_change) {
            output.status = llama_kv_live_lifecycle_status::committed;
            return output;
        }
        if (output.policy.status == llama_kv_live_policy_status::safe_fallback) {
            output.status = llama_kv_live_lifecycle_status::reuse_old_hot_set;
            return output;
        }
        if (context.companion_attempted && hooks_.rollback_companions) {
            hooks_.rollback_companions(hooks_.context, generation_, frontier_);
            output.companion_rolled_back = true;
        }
        output.status = output.policy.status == llama_kv_live_policy_status::stale_snapshot
            ? llama_kv_live_lifecycle_status::stale_generation
            : (context.companion_attempted
                ? llama_kv_live_lifecycle_status::companion_rejected
                : llama_kv_live_lifecycle_status::policy_failed);
        return output;
    }
    output.status = llama_kv_live_lifecycle_status::committed;
    return output;
}

llama_kv_live_lifecycle_status llama_kv_live_lifecycle::rotate(
        llama_kv_live_lifecycle_event event,
        const llama_kv_live_lifecycle_generation * next) noexcept {
    if (stopped_) return llama_kv_live_lifecycle_status::shutdown;
    if (next && (!next->valid() || (generation_.valid() &&
            next->operation_generation <= generation_.operation_generation))) {
        return llama_kv_live_lifecycle_status::stale_generation;
    }
    cancel_tracked();
    if (hooks_.event && !hooks_.event(hooks_.context, event, generation_)) {
        // A rejected destructive transition leaves the old image unusable
        // until the owner starts a fresh operation. This prevents a late
        // completion from reviving the pre-transition generation.
        active_ = false;
        frontier_ = {};
        return llama_kv_live_lifecycle_status::companion_rejected;
    }
    active_ = false;
    frontier_ = {};
    if (next) {
        generation_ = *next;
        active_ = true;
    }
    return event == llama_kv_live_lifecycle_event::cancel
        ? llama_kv_live_lifecycle_status::cancelled
        : llama_kv_live_lifecycle_status::ready;
}

llama_kv_live_lifecycle_status llama_kv_live_lifecycle::checkpoint_save() noexcept {
    if (!current()) return stopped_ ? llama_kv_live_lifecycle_status::shutdown
                                    : llama_kv_live_lifecycle_status::stale_generation;
    // A checkpoint is a stable artifact, not a snapshot of a half-uploaded
    // page. Unpublished predictive work is disposable; cancelling it here
    // gives the artifact owner a quiet table and prevents a late completion
    // from appearing in the saved lineage.
    cancel_tracked();
    if (hooks_.event && !hooks_.event(hooks_.context,
            llama_kv_live_lifecycle_event::checkpoint_save, generation_)) {
        return llama_kv_live_lifecycle_status::companion_rejected;
    }
    return llama_kv_live_lifecycle_status::ready;
}

llama_kv_live_lifecycle_status llama_kv_live_lifecycle::prompt_adopt(
        const llama_kv_live_lifecycle_generation & generation) noexcept {
    return rotate(llama_kv_live_lifecycle_event::prompt_adopt, &generation);
}

llama_kv_live_lifecycle_status llama_kv_live_lifecycle::slot_reuse(
        const llama_kv_live_lifecycle_generation & generation) noexcept {
    return rotate(llama_kv_live_lifecycle_event::slot_reuse, &generation);
}

llama_kv_live_lifecycle_status llama_kv_live_lifecycle::checkpoint_restore(
        const llama_kv_live_lifecycle_generation & generation) noexcept {
    return rotate(llama_kv_live_lifecycle_event::checkpoint_restore, &generation);
}

llama_kv_live_lifecycle_status llama_kv_live_lifecycle::clear() noexcept {
    return rotate(llama_kv_live_lifecycle_event::clear, nullptr);
}

llama_kv_live_lifecycle_status llama_kv_live_lifecycle::cancel() noexcept {
    return rotate(llama_kv_live_lifecycle_event::cancel, nullptr);
}

void llama_kv_live_lifecycle::shutdown() noexcept {
    if (stopped_) return;
    cancel_tracked();
    if (hooks_.event) hooks_.event(hooks_.context,
            llama_kv_live_lifecycle_event::shutdown, generation_);
    prefetch_->shutdown();
    active_ = false;
    stopped_ = true;
}
