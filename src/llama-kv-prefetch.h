#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// The scheduler is deliberately a small owner-side seam.  It does not know
// about a backend stream or a residency table; those are supplied by these
// callbacks so CPU replay and a device implementation use identical queue,
// cancellation, and readiness rules.
struct llama_kv_prefetch_intent {
    uint64_t page_id = 0;
    uint64_t generation = 0;
    uint64_t useful_bytes = 0;
    uint64_t aligned_bytes = 0;
    uint32_t priority = 0;
    bool required = false;
};

enum class llama_kv_prefetch_poll : uint8_t {
    pending = 0,
    completed,
    failed,
    stale_generation,
};

enum class llama_kv_prefetch_status : uint8_t {
    ok = 0,
    not_configured,
    invalid_argument,
    backpressure,
    queue_full,
    event_full,
    staging_full,
    host_miss,
    transfer_failed,
    cancelled,
    stale_generation,
    dirty_page,
    shutdown,
    not_ready,
    _count,
};

const char * llama_kv_prefetch_status_name(llama_kv_prefetch_status status) noexcept;

struct llama_kv_prefetch_config {
    uint32_t max_queued_pages = 72;
    uint64_t max_queued_bytes = uint64_t(72) * 1024 * 1024;
    uint32_t max_events = 8;
    uint32_t max_pinned_slots = 16;
    uint32_t staging_slots = 2;
    uint32_t prefetch_depth = 2;
    uint32_t wait_budget_steps = 2;
};

struct llama_kv_prefetch_backend {
    void * context = nullptr;

    // `staging_slot` belongs to this intent until its completion callback
    // returns. `asynchronous` is always true for a configured scheduler.
    bool (*submit)(void * context, const llama_kv_prefetch_intent & intent,
                   uint32_t staging_slot, uint64_t ticket,
                   bool asynchronous) noexcept = nullptr;
    llama_kv_prefetch_poll (*poll)(void * context, uint64_t ticket) noexcept = nullptr;
    void (*cancel)(void * context, uint64_t ticket) noexcept = nullptr;

    // The scheduler calls publish only after the complete intent has
    // completed. A partial page has no callback path to publication.
    bool (*publish_complete)(void * context,
                             const llama_kv_prefetch_intent & intent) noexcept = nullptr;
    bool (*host_available)(void * context,
                           const llama_kv_prefetch_intent & intent) noexcept = nullptr;

    // Clean eviction is mapping-only. Dirty pages must pass through reseal
    // first and are counted separately from eviction.
    bool (*reseal_dirty)(void * context,
                         const llama_kv_prefetch_intent & intent) noexcept = nullptr;
    bool (*evict_clean)(void * context,
                        const llama_kv_prefetch_intent & intent) noexcept = nullptr;

    uint64_t (*timestamp_us)(void * context) noexcept = nullptr;
};

struct llama_kv_prefetch_counters {
    uint64_t requested = 0;
    uint64_t queued = 0;
    uint64_t submitted = 0;
    uint64_t completed = 0;
    uint64_t failed = 0;
    uint64_t faults = 0;
    uint64_t prefetch_hits = 0;
    uint64_t late_waits = 0;
    uint64_t evictions = 0;
    uint64_t reseals = 0;
    uint64_t cancellations = 0;
    uint64_t stale_generation_rejects = 0;
    uint64_t useful_bytes = 0;
    uint64_t aligned_bytes = 0;
    uint64_t stage_latency_us = 0;
};

enum class llama_kv_prefetch_readiness : uint8_t {
    ready = 0,
    waited_ready,
    reuse_old_hot_set,
    fallback_larger_union,
    not_configured,
    cancelled,
};

struct llama_kv_prefetch_resolution {
    llama_kv_prefetch_readiness readiness = llama_kv_prefetch_readiness::cancelled;
    std::vector<uint64_t> ready;
    std::vector<uint64_t> fallback;
};

struct llama_kv_prefetch_eviction {
    llama_kv_prefetch_intent page;
    bool dirty = false;
};

class llama_kv_prefetch_scheduler {
public:
    static std::unique_ptr<llama_kv_prefetch_scheduler> create(
            const llama_kv_prefetch_config & config,
            const llama_kv_prefetch_backend & backend,
            llama_kv_prefetch_status & status) noexcept;
    ~llama_kv_prefetch_scheduler();

    llama_kv_prefetch_scheduler(const llama_kv_prefetch_scheduler &) = delete;
    llama_kv_prefetch_scheduler & operator=(const llama_kv_prefetch_scheduler &) = delete;

    llama_kv_prefetch_status enqueue(
            const llama_kv_prefetch_intent & intent) noexcept;
    // Enqueue the next-token candidates in priority order, bounded by the
    // configured predictive depth. Required pages use ensure_ready().
    llama_kv_prefetch_status prefetch(
            const std::vector<llama_kv_prefetch_intent> & intents) noexcept;
    llama_kv_prefetch_status pump() noexcept;
    llama_kv_prefetch_status advance() noexcept;
    llama_kv_prefetch_status cancel(
            uint64_t page_id, uint64_t generation) noexcept;
    void shutdown() noexcept;

    llama_kv_prefetch_resolution ensure_ready(
            const std::vector<llama_kv_prefetch_intent> & required,
            const std::vector<uint64_t> & previous_hot_set,
            uint32_t wait_budget_steps = UINT32_MAX) noexcept;
    llama_kv_prefetch_status evict(
            const llama_kv_prefetch_eviction & request) noexcept;

    uint32_t queued_pages() const noexcept { return uint32_t(queue_.size()); }
    uint32_t active_events() const noexcept { return uint32_t(active_.size()); }
    uint32_t pinned_slots() const noexcept {
        return uint32_t(queue_.size() + active_.size());
    }
    uint64_t queued_bytes() const noexcept { return queued_bytes_; }
    bool stopped() const noexcept { return stopped_; }
    const llama_kv_prefetch_counters & counters() const noexcept { return counters_; }

private:
    struct active_intent {
        llama_kv_prefetch_intent intent;
        uint64_t ticket = 0;
        uint64_t submitted_us = 0;
        uint32_t staging_slot = UINT32_MAX;
    };

    llama_kv_prefetch_scheduler(const llama_kv_prefetch_config & config,
                                const llama_kv_prefetch_backend & backend);
    llama_kv_prefetch_status validate_intent(
            const llama_kv_prefetch_intent & intent) const noexcept;
    bool is_ready(uint64_t page_id, uint64_t generation) const noexcept;
    bool erase_queued(uint64_t page_id, uint64_t generation) noexcept;
    bool cancel_active(size_t index) noexcept;
    void mark_failure() noexcept;
    uint64_t now_us() const noexcept;

    llama_kv_prefetch_config config_;
    llama_kv_prefetch_backend backend_;
    std::vector<llama_kv_prefetch_intent> queue_;
    std::vector<active_intent> active_;
    std::vector<llama_kv_prefetch_intent> ready_;
    uint64_t queued_bytes_ = 0;
    uint64_t next_ticket_ = 1;
    llama_kv_prefetch_counters counters_;
    bool stopped_ = false;
};
