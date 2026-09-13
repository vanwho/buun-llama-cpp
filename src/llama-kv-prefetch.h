#pragma once

#include "llama-kv-residency.h"

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
    // These fields are metadata only.  They let the owner distinguish a
    // previous-query lookahead from an authoritative current-query fault and
    // let a backend attach the ticket to its exact layer/token deadline.
    uint64_t source_query_generation = 0;
    uint32_t source_query_layer = UINT32_MAX;
    uint64_t source_query_token = UINT64_MAX;
    uint32_t needed_by_layer = UINT32_MAX;
    uint64_t needed_by_token = UINT64_MAX;
    bool prediction = false;
    bool prediction_useful_counted = false;
    // Backend ownership metadata carried by the ticket.  Zero/UINT32_MAX
    // means that the owner uses its implicit page mapping.
    uint64_t table_epoch = 0;
    uint32_t destination_slot = UINT32_MAX;
    uint64_t host_offset = 0;
    uint64_t host_bytes = 0;
    bool prediction_hit_counted = false;
    // Layer identity is part of the residency key.  The legacy page_id field
    // remains valid for owner-side callers that use compact test identities.
    uint32_t attention_layer = UINT32_MAX;
    llama_kv_page_id identity;
    uint64_t speculation_generation = 0;
    bool speculation_rejected = false;
};

// GPU ranking publishes only compact metadata.  The mailbox owns two fixed
// host slots; producers fill a slot through their backend's asynchronous
// device-to-host operation and publish the completion event.  Consumers poll
// that event at a boundary and never synchronize the compute stream.
struct llama_kv_prefetch_candidate {
    llama_kv_page_id identity;
    uint32_t attention_layer = UINT32_MAX;
    uint64_t generation = 0;
    uint64_t table_epoch = 0;
    float score = 0.0f;
    uint64_t requested_bytes = 0;
    uint64_t content_version = 0;
    uint64_t speculation_generation = 0;
    bool speculation_rejected = false;
};

enum class llama_kv_prefetch_mailbox_poll : uint8_t {
    pending = 0,
    completed,
    failed,
    stale_generation,
};

enum class llama_kv_prefetch_mailbox_status : uint8_t {
    ok = 0,
    not_configured,
    invalid_argument,
    full,
    stale_generation,
    cancelled,
    _count,
};

const char * llama_kv_prefetch_mailbox_status_name(
        llama_kv_prefetch_mailbox_status status) noexcept;

struct llama_kv_prefetch_mailbox_backend {
    void * context = nullptr;
    llama_kv_prefetch_mailbox_poll (*poll)(
            void * context, uint64_t event) noexcept = nullptr;
    void (*cancel)(void * context, uint64_t event) noexcept = nullptr;
    void (*release)(void * context, uint64_t event) noexcept = nullptr;
};

struct llama_kv_prefetch_mailbox_config {
    uint32_t slot_count = 2;
    uint32_t candidates_per_slot = 16;
};

class llama_kv_prefetch_mailbox {
public:
    explicit llama_kv_prefetch_mailbox(
            const llama_kv_prefetch_mailbox_config & config = {}) noexcept;
    ~llama_kv_prefetch_mailbox() = default;

    llama_kv_prefetch_mailbox(const llama_kv_prefetch_mailbox &) = delete;
    llama_kv_prefetch_mailbox & operator=(
            const llama_kv_prefetch_mailbox &) = delete;

    bool configured() const noexcept { return !slots_.empty(); }
    uint32_t slot_count() const noexcept { return uint32_t(slots_.size()); }
    uint32_t candidates_per_slot() const noexcept { return capacity_; }
    uint32_t pending_slots() const noexcept;
    uint32_t ready_slots() const noexcept;

    // Reserve a writable fixed slot for a GPU producer. The returned pointer
    // is stable until publish_pending(), publish_ready(), or abandon().
    llama_kv_prefetch_mailbox_status acquire(
            uint32_t & slot, llama_kv_prefetch_candidate *& records) noexcept;
    llama_kv_prefetch_candidate * data(uint32_t slot) noexcept;
    const llama_kv_prefetch_candidate * data(uint32_t slot) const noexcept;
    llama_kv_prefetch_mailbox_status publish_pending(
            uint32_t slot, uint32_t count, uint64_t generation,
            uint64_t event) noexcept;
    llama_kv_prefetch_mailbox_status publish_ready(
            uint32_t slot, uint32_t count, uint64_t generation) noexcept;
    void abandon(uint32_t slot) noexcept;

    // Polls backend events without waiting. A stale generation or rejected
    // speculative result is discarded before it reaches the ready queue.
    llama_kv_prefetch_mailbox_status poll(
            uint64_t generation, uint64_t table_epoch = 0) noexcept;
    size_t take_ready(
            std::vector<llama_kv_prefetch_candidate> & output,
            uint32_t max_candidates = UINT32_MAX) noexcept;
    void cancel() noexcept;
    void set_backend(llama_kv_prefetch_mailbox_backend backend) noexcept {
        backend_ = backend;
    }

private:
    enum class slot_state : uint8_t { free = 0, writing, pending, ready };
    struct slot {
        std::vector<llama_kv_prefetch_candidate> records;
        uint32_t count = 0;
        uint64_t generation = 0;
        uint64_t event = 0;
        slot_state state = slot_state::free;
    };

    bool validate(const llama_kv_prefetch_candidate & candidate,
                  uint64_t generation, uint64_t table_epoch) const noexcept;
    void release(slot & value) noexcept;

    llama_kv_prefetch_mailbox_backend backend_;
    uint32_t capacity_ = 0;
    std::vector<slot> slots_;
};

// A bounded previous-query record.  The scheduler never treats this as proof
// for the current query; callers must still pass authoritative pages to
// ensure_ready() before attention consumes them.
class llama_kv_prefetch_predictor {
public:
    explicit llama_kv_prefetch_predictor(uint32_t capacity = 0) noexcept;

    bool observe(uint64_t query_generation, uint32_t layer, uint64_t token,
                 const std::vector<llama_kv_prefetch_intent> & ranked) noexcept;
    std::vector<llama_kv_prefetch_intent> predict(
            uint64_t generation, uint32_t layer, uint64_t token,
            uint32_t limit = UINT32_MAX) const noexcept;
    void clear() noexcept;
    uint32_t capacity() const noexcept { return capacity_; }

private:
    struct evidence {
        llama_kv_prefetch_intent intent;
        uint64_t query_generation = 0;
        uint32_t layer = UINT32_MAX;
        uint64_t token = UINT64_MAX;
    };

    uint32_t capacity_ = 0;
    std::vector<evidence> previous_;
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

enum class llama_kv_prefetch_timeline_kind : uint8_t {
    enqueue = 0,
    needed,
    copy_begin,
    copy_end,
    wait,
    consumed,
    cancelled,
    _count,
};

const char * llama_kv_prefetch_timeline_kind_name(
        llama_kv_prefetch_timeline_kind kind) noexcept;

struct llama_kv_prefetch_timeline_event {
    llama_kv_prefetch_timeline_kind kind = llama_kv_prefetch_timeline_kind::enqueue;
    uint64_t page_id = 0;
    uint64_t generation = 0;
    uint64_t ticket = 0;
    uint64_t timestamp_us = 0;
    uint32_t layer = UINT32_MAX;
    uint64_t token = UINT64_MAX;
    uint64_t table_epoch = 0;
    uint32_t destination_slot = UINT32_MAX;
    uint64_t host_offset = 0;
    uint64_t host_bytes = 0;
};

struct llama_kv_prefetch_config {
    uint32_t max_queued_pages = 72;
    uint64_t max_queued_bytes = uint64_t(72) * 1024 * 1024;
    uint32_t max_events = 8;
    uint32_t max_pinned_slots = 16;
    uint32_t staging_slots = 2;
    uint32_t prefetch_depth = 2;
    uint32_t wait_budget_steps = 2;
    uint32_t max_timeline_events = 256;
    // These refresh controls apply to predictive (non-required) requests.
    // A zero byte budget disables only that optional admission limit; demand
    // misses remain independently bounded by max_events/max_pinned_slots.
    uint32_t refresh_cadence = 8;
    uint32_t max_cold_pages_per_refresh = 1;
    uint64_t bytes_per_refresh = 0;
    uint32_t min_resident_pages = 0;
    uint32_t hysteresis_priority = 0;
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
    bool (*generation_current)(void * context,
                               const llama_kv_prefetch_intent & intent) noexcept = nullptr;
    void (*discard_complete)(void * context,
                             const llama_kv_prefetch_intent & intent) noexcept = nullptr;
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
    uint64_t prediction_requested = 0;
    uint64_t prediction_completed = 0;
    uint64_t prediction_hits = 0;
    uint64_t prediction_useful_bytes = 0;
    uint64_t prediction_wasted_bytes = 0;
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
    // Starts a bounded refresh window. Refresh identifiers are monotonic; a
    // repeated identifier coalesces candidates without resetting budgets.
    llama_kv_prefetch_status begin_refresh(
            uint64_t refresh_id, bool force = false) noexcept;
    bool observe_query(uint64_t query_generation, uint32_t layer, uint64_t token,
                       const std::vector<llama_kv_prefetch_intent> & ranked) noexcept;
    std::vector<llama_kv_prefetch_intent> predict_next(
            uint64_t generation, uint32_t layer, uint64_t token,
            uint32_t limit = UINT32_MAX) const noexcept;
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
        return uint32_t(queue_.size() + active_.size() + ready_.size());
    }
    uint64_t queued_bytes() const noexcept { return queued_bytes_; }
    bool stopped() const noexcept { return stopped_; }
    uint64_t refresh_id() const noexcept { return refresh_id_; }
    const llama_kv_prefetch_counters & counters() const noexcept { return counters_; }
    const std::vector<llama_kv_prefetch_timeline_event> & timeline() const noexcept {
        return timeline_;
    }

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
    bool is_ready(const llama_kv_prefetch_intent & intent) const noexcept;
    bool erase_queued(const llama_kv_prefetch_intent & intent) noexcept;
    bool cancel_active(size_t index) noexcept;
    void mark_failure() noexcept;
    void record_timeline(llama_kv_prefetch_timeline_kind kind,
                         const llama_kv_prefetch_intent & intent,
                         uint64_t ticket = 0) noexcept;
    bool mark_prediction_useful(llama_kv_prefetch_intent & intent) noexcept;
    void complete_prediction_useful(llama_kv_prefetch_intent & intent) noexcept;
    void mark_prediction_wasted(const llama_kv_prefetch_intent & intent) noexcept;
    uint64_t now_us() const noexcept;

    llama_kv_prefetch_config config_;
    llama_kv_prefetch_backend backend_;
    std::vector<llama_kv_prefetch_intent> queue_;
    std::vector<active_intent> active_;
    std::vector<llama_kv_prefetch_intent> ready_;
    uint64_t queued_bytes_ = 0;
    uint64_t next_ticket_ = 1;
    llama_kv_prefetch_counters counters_;
    llama_kv_prefetch_predictor predictor_;
    std::vector<llama_kv_prefetch_timeline_event> timeline_;
    bool stopped_ = false;
    uint64_t refresh_id_ = 0;
    uint32_t refresh_pages_ = 0;
    uint64_t refresh_bytes_ = 0;
};
