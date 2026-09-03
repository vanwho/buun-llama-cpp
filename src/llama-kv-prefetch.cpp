#include "llama-kv-prefetch.h"

#include <algorithm>
#include <limits>
#include <new>
#include <utility>

namespace {

bool add_u64(uint64_t a, uint64_t b, uint64_t & out) noexcept {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    out = a + b;
    return true;
}

} // namespace

const char * llama_kv_prefetch_status_name(llama_kv_prefetch_status status) noexcept {
    switch (status) {
        case llama_kv_prefetch_status::ok: return "ok";
        case llama_kv_prefetch_status::not_configured: return "not_configured";
        case llama_kv_prefetch_status::invalid_argument: return "invalid_argument";
        case llama_kv_prefetch_status::backpressure: return "backpressure";
        case llama_kv_prefetch_status::queue_full: return "queue_full";
        case llama_kv_prefetch_status::event_full: return "event_full";
        case llama_kv_prefetch_status::staging_full: return "staging_full";
        case llama_kv_prefetch_status::host_miss: return "host_miss";
        case llama_kv_prefetch_status::transfer_failed: return "transfer_failed";
        case llama_kv_prefetch_status::cancelled: return "cancelled";
        case llama_kv_prefetch_status::stale_generation: return "stale_generation";
        case llama_kv_prefetch_status::dirty_page: return "dirty_page";
        case llama_kv_prefetch_status::shutdown: return "shutdown";
        case llama_kv_prefetch_status::not_ready: return "not_ready";
        case llama_kv_prefetch_status::_count: break;
    }
    return "invalid";
}

std::unique_ptr<llama_kv_prefetch_scheduler> llama_kv_prefetch_scheduler::create(
        const llama_kv_prefetch_config & config,
        const llama_kv_prefetch_backend & backend,
        llama_kv_prefetch_status & status) noexcept {
    status = llama_kv_prefetch_status::invalid_argument;
    try {
        if (config.max_queued_pages == 0 || config.max_queued_bytes == 0 ||
            config.max_events == 0 || config.max_pinned_slots == 0 ||
            config.staging_slots < 2 ||
            config.max_pinned_slots < config.max_events ||
            !backend.submit || !backend.poll || !backend.publish_complete) {
            status = (!backend.submit || !backend.poll || !backend.publish_complete)
                ? llama_kv_prefetch_status::not_configured
                : llama_kv_prefetch_status::invalid_argument;
            return nullptr;
        }
        auto result = std::unique_ptr<llama_kv_prefetch_scheduler>(
            new llama_kv_prefetch_scheduler(config, backend));
        status = llama_kv_prefetch_status::ok;
        return result;
    } catch (...) {
        status = llama_kv_prefetch_status::not_configured;
        return nullptr;
    }
}

llama_kv_prefetch_scheduler::llama_kv_prefetch_scheduler(
        const llama_kv_prefetch_config & config,
        const llama_kv_prefetch_backend & backend)
    : config_(config), backend_(backend) {
    queue_.reserve(config_.max_queued_pages);
    active_.reserve(config_.max_events);
    ready_.reserve(config_.max_pinned_slots);
}

llama_kv_prefetch_scheduler::~llama_kv_prefetch_scheduler() {
    shutdown();
}

llama_kv_prefetch_status llama_kv_prefetch_scheduler::validate_intent(
        const llama_kv_prefetch_intent & intent) const noexcept {
    if (intent.page_id == 0 || intent.generation == 0 || intent.useful_bytes == 0 ||
        intent.aligned_bytes < intent.useful_bytes) {
        return llama_kv_prefetch_status::invalid_argument;
    }
    return llama_kv_prefetch_status::ok;
}

bool llama_kv_prefetch_scheduler::is_ready(
        uint64_t page_id, uint64_t generation) const noexcept {
    for (const auto & intent : ready_) {
        if (intent.page_id == page_id && intent.generation == generation) return true;
    }
    return false;
}

uint64_t llama_kv_prefetch_scheduler::now_us() const noexcept {
    return backend_.timestamp_us ? backend_.timestamp_us(backend_.context) : 0;
}

void llama_kv_prefetch_scheduler::mark_failure() noexcept {
    ++counters_.failed;
}

bool llama_kv_prefetch_scheduler::erase_queued(
        uint64_t page_id, uint64_t generation) noexcept {
    for (auto it = queue_.begin(); it != queue_.end(); ++it) {
        if (it->page_id != page_id || it->generation != generation) continue;
        queued_bytes_ -= it->aligned_bytes;
        queue_.erase(it);
        return true;
    }
    return false;
}

bool llama_kv_prefetch_scheduler::cancel_active(size_t index) noexcept {
    if (index >= active_.size()) return false;
    if (backend_.cancel) backend_.cancel(backend_.context, active_[index].ticket);
    ++counters_.cancellations;
    active_.erase(active_.begin() + index);
    return true;
}

llama_kv_prefetch_status llama_kv_prefetch_scheduler::enqueue(
        const llama_kv_prefetch_intent & intent) noexcept {
    try {
        if (stopped_) return llama_kv_prefetch_status::shutdown;
        const auto valid = validate_intent(intent);
        if (valid != llama_kv_prefetch_status::ok) return valid;
        ++counters_.requested;

        for (auto it = ready_.begin(); it != ready_.end();) {
            if (it->page_id == intent.page_id && it->generation != intent.generation) {
                it = ready_.erase(it);
                ++counters_.stale_generation_rejects;
            } else {
                ++it;
            }
        }

        if (is_ready(intent.page_id, intent.generation)) {
            if (intent.required) ++counters_.prefetch_hits;
            return llama_kv_prefetch_status::ok;
        }

        for (auto & queued : queue_) {
            if (queued.page_id != intent.page_id) continue;
            if (queued.generation != intent.generation) {
                erase_queued(queued.page_id, queued.generation);
                ++counters_.cancellations;
                break;
            }
            if (intent.aligned_bytes > queued.aligned_bytes) {
                uint64_t new_bytes = 0;
                if (!add_u64(queued_bytes_, intent.aligned_bytes - queued.aligned_bytes, new_bytes) ||
                    new_bytes > config_.max_queued_bytes) {
                    return llama_kv_prefetch_status::backpressure;
                }
                queued_bytes_ = new_bytes;
                queued.aligned_bytes = intent.aligned_bytes;
                queued.useful_bytes = std::max(queued.useful_bytes, intent.useful_bytes);
            }
            queued.priority = std::max(queued.priority, intent.priority);
            queued.required = queued.required || intent.required;
            if (intent.required) ++counters_.faults;
            return llama_kv_prefetch_status::ok;
        }
        for (size_t i = 0; i < active_.size(); ++i) {
            if (active_[i].intent.page_id != intent.page_id) continue;
            if (active_[i].intent.generation == intent.generation) {
                active_[i].intent.required = active_[i].intent.required || intent.required;
                active_[i].intent.priority = std::max(active_[i].intent.priority, intent.priority);
                if (intent.required) ++counters_.faults;
                return llama_kv_prefetch_status::ok;
            }
            cancel_active(i);
            ++counters_.stale_generation_rejects;
            break;
        }
        if (intent.required) ++counters_.faults;
        if (queue_.size() >= config_.max_queued_pages) return llama_kv_prefetch_status::queue_full;
        if (pinned_slots() >= config_.max_pinned_slots) return llama_kv_prefetch_status::backpressure;
        uint64_t new_bytes = 0;
        if (!add_u64(queued_bytes_, intent.aligned_bytes, new_bytes) ||
            new_bytes > config_.max_queued_bytes) {
            return llama_kv_prefetch_status::backpressure;
        }
        queue_.push_back(intent);
        queued_bytes_ = new_bytes;
        ++counters_.queued;
        return pump();
    } catch (...) {
        return llama_kv_prefetch_status::not_configured;
    }
}

llama_kv_prefetch_status llama_kv_prefetch_scheduler::pump() noexcept {
    if (stopped_) return llama_kv_prefetch_status::shutdown;
    llama_kv_prefetch_status result = llama_kv_prefetch_status::ok;
    try {
        while (!queue_.empty() && active_.size() < config_.max_events &&
               pinned_slots() <= config_.max_pinned_slots) {
            uint32_t staging_slot = UINT32_MAX;
            for (uint32_t slot = 0; slot < config_.staging_slots; ++slot) {
                bool used = false;
                for (const auto & active : active_) if (active.staging_slot == slot) used = true;
                if (!used) { staging_slot = slot; break; }
            }
            if (staging_slot == UINT32_MAX) {
                result = llama_kv_prefetch_status::staging_full;
                break;
            }
            const auto intent = queue_.front();
            queue_.erase(queue_.begin());
            queued_bytes_ -= intent.aligned_bytes;
            if (backend_.host_available &&
                !backend_.host_available(backend_.context, intent)) {
                mark_failure();
                result = llama_kv_prefetch_status::host_miss;
                continue;
            }
            const uint64_t ticket = next_ticket_++;
            if (ticket == 0 || !backend_.submit(
                    backend_.context, intent, staging_slot, ticket, true)) {
                mark_failure();
                result = llama_kv_prefetch_status::transfer_failed;
                continue;
            }
            active_.push_back({ intent, ticket, now_us(), staging_slot });
            ++counters_.submitted;
            if (!add_u64(counters_.useful_bytes, intent.useful_bytes, counters_.useful_bytes) ||
                !add_u64(counters_.aligned_bytes, intent.aligned_bytes, counters_.aligned_bytes)) {
                mark_failure();
                result = llama_kv_prefetch_status::transfer_failed;
                cancel_active(active_.size() - 1);
                break;
            }
        }
        if (!queue_.empty() && active_.size() >= config_.max_events &&
            result == llama_kv_prefetch_status::ok) result = llama_kv_prefetch_status::event_full;
        if (!queue_.empty() && pinned_slots() >= config_.max_pinned_slots &&
            result == llama_kv_prefetch_status::ok) result = llama_kv_prefetch_status::backpressure;
        return result;
    } catch (...) {
        return llama_kv_prefetch_status::transfer_failed;
    }
}

llama_kv_prefetch_status llama_kv_prefetch_scheduler::prefetch(
        const std::vector<llama_kv_prefetch_intent> & intents) noexcept {
    if (stopped_) return llama_kv_prefetch_status::shutdown;
    llama_kv_prefetch_status result = llama_kv_prefetch_status::ok;
    try {
        const size_t count = std::min<size_t>(intents.size(), config_.prefetch_depth);
        for (size_t i = 0; i < count; ++i) {
            auto intent = intents[i];
            intent.required = false;
            const auto status = enqueue(intent);
            if (status != llama_kv_prefetch_status::ok &&
                result == llama_kv_prefetch_status::ok) result = status;
        }
        return result;
    } catch (...) {
        return llama_kv_prefetch_status::backpressure;
    }
}

llama_kv_prefetch_status llama_kv_prefetch_scheduler::advance() noexcept {
    if (stopped_) return llama_kv_prefetch_status::shutdown;
    llama_kv_prefetch_status result = pump();
    try {
        for (size_t i = 0; i < active_.size();) {
            auto & active = active_[i];
            const auto state = backend_.poll(backend_.context, active.ticket);
            if (state == llama_kv_prefetch_poll::pending) { ++i; continue; }
            const auto intent = active.intent;
            const uint64_t submitted_us = active.submitted_us;
            if (state == llama_kv_prefetch_poll::completed) {
                if (!backend_.publish_complete(backend_.context, intent)) {
                    mark_failure();
                    result = llama_kv_prefetch_status::transfer_failed;
                } else {
                    if (ready_.size() >= config_.max_queued_pages) ready_.erase(ready_.begin());
                    ready_.push_back(intent);
                    ++counters_.completed;
                    const uint64_t completed_us = now_us();
                    if (completed_us >= submitted_us) counters_.stage_latency_us += completed_us - submitted_us;
                }
            } else if (state == llama_kv_prefetch_poll::stale_generation) {
                ++counters_.stale_generation_rejects;
                mark_failure();
                result = llama_kv_prefetch_status::stale_generation;
            } else {
                mark_failure();
                result = llama_kv_prefetch_status::transfer_failed;
            }
            active_.erase(active_.begin() + i);
        }
        const auto pump_result = pump();
        if (result == llama_kv_prefetch_status::ok) result = pump_result;
        return result;
    } catch (...) {
        return llama_kv_prefetch_status::transfer_failed;
    }
}

llama_kv_prefetch_status llama_kv_prefetch_scheduler::cancel(
        uint64_t page_id, uint64_t generation) noexcept {
    if (stopped_) return llama_kv_prefetch_status::shutdown;
    bool found = erase_queued(page_id, generation);
    for (size_t i = active_.size(); i > 0; --i) {
        const auto & active = active_[i - 1];
        if (active.intent.page_id == page_id && active.intent.generation == generation) {
            cancel_active(i - 1);
            found = true;
        }
    }
    for (auto it = ready_.begin(); it != ready_.end();) {
        if (it->page_id == page_id && it->generation == generation) {
            it = ready_.erase(it);
            found = true;
        } else ++it;
    }
    return found ? llama_kv_prefetch_status::cancelled : llama_kv_prefetch_status::not_ready;
}

void llama_kv_prefetch_scheduler::shutdown() noexcept {
    if (stopped_) return;
    for (size_t i = active_.size(); i > 0; --i) cancel_active(i - 1);
    queue_.clear();
    ready_.clear();
    queued_bytes_ = 0;
    stopped_ = true;
}

llama_kv_prefetch_resolution llama_kv_prefetch_scheduler::ensure_ready(
        const std::vector<llama_kv_prefetch_intent> & required,
        const std::vector<uint64_t> & previous_hot_set,
        uint32_t wait_budget_steps) noexcept {
    llama_kv_prefetch_resolution result;
    try {
        if (stopped_) {
            result.readiness = llama_kv_prefetch_readiness::cancelled;
            return result;
        }
        if (wait_budget_steps == UINT32_MAX) wait_budget_steps = config_.wait_budget_steps;
        std::vector<uint64_t> required_ids;
        required_ids.reserve(required.size());
        for (const auto & intent : required) {
            if (validate_intent(intent) != llama_kv_prefetch_status::ok) {
                result.readiness = llama_kv_prefetch_readiness::cancelled;
                return result;
            }
            for (const auto & prior : required) {
                if (&prior == &intent) break;
                if (prior.page_id == intent.page_id && prior.generation != intent.generation) {
                    result.readiness = llama_kv_prefetch_readiness::cancelled;
                    return result;
                }
            }
            if (std::find(required_ids.begin(), required_ids.end(), intent.page_id) == required_ids.end()) {
                required_ids.push_back(intent.page_id);
            }
            const auto status = enqueue({ intent.page_id, intent.generation,
                                          intent.useful_bytes, intent.aligned_bytes,
                                          intent.priority, true });
            if (status == llama_kv_prefetch_status::shutdown) {
                result.readiness = llama_kv_prefetch_readiness::cancelled;
                return result;
            }
        }
        auto collect = [&]() {
            result.ready.clear();
            for (const auto & intent : required) {
                if (is_ready(intent.page_id, intent.generation) &&
                    std::find(result.ready.begin(), result.ready.end(), intent.page_id) == result.ready.end()) {
                    result.ready.push_back(intent.page_id);
                }
            }
            return result.ready.size() == required_ids.size();
        };
        if (collect()) {
            result.readiness = llama_kv_prefetch_readiness::ready;
            return result;
        }
        for (uint32_t step = 0; step < wait_budget_steps; ++step) {
            ++counters_.late_waits;
            advance();
            if (collect()) {
                result.readiness = llama_kv_prefetch_readiness::waited_ready;
                return result;
            }
        }
        if (!previous_hot_set.empty()) {
            result.readiness = llama_kv_prefetch_readiness::reuse_old_hot_set;
            result.fallback = previous_hot_set;
        } else {
            result.readiness = llama_kv_prefetch_readiness::fallback_larger_union;
            result.fallback = required_ids;
        }
        return result;
    } catch (...) {
        result.readiness = llama_kv_prefetch_readiness::cancelled;
        result.ready.clear();
        result.fallback.clear();
        return result;
    }
}

llama_kv_prefetch_status llama_kv_prefetch_scheduler::evict(
        const llama_kv_prefetch_eviction & request) noexcept {
    if (stopped_) return llama_kv_prefetch_status::shutdown;
    if (validate_intent(request.page) != llama_kv_prefetch_status::ok) {
        return llama_kv_prefetch_status::invalid_argument;
    }
    if (request.dirty) {
        if (!backend_.reseal_dirty || !backend_.reseal_dirty(backend_.context, request.page)) {
            return llama_kv_prefetch_status::dirty_page;
        }
        ++counters_.reseals;
    }
    if (!backend_.evict_clean || !backend_.evict_clean(backend_.context, request.page)) {
        return llama_kv_prefetch_status::transfer_failed;
    }
    ++counters_.evictions;
    return llama_kv_prefetch_status::ok;
}
