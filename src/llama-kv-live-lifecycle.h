#pragma once

#include "llama-kv-live-policy.h"
#include "llama-kv-prefetch.h"

#include <cstdint>
#include <memory>
#include <vector>

constexpr uint32_t LLAMA_KV_LIVE_LIFECYCLE_VERSION = 1;

// A lifecycle generation is separate from a page generation.  The operation
// component changes for every prompt/checkpoint/slot transition, so a late
// transfer cannot become valid merely because its logical page number was
// reused.
struct llama_kv_live_lifecycle_generation {
    uint64_t session_generation = 0;
    int32_t sequence_id = -1;
    uint64_t sequence_generation = 0;
    uint64_t operation_generation = 0;

    bool valid() const noexcept {
        return session_generation != 0 && sequence_id >= 0 &&
               sequence_generation != 0 && operation_generation != 0;
    }
};

enum class llama_kv_live_lifecycle_status : uint8_t {
    committed = 0,
    ready,
    waited_ready,
    reuse_old_hot_set,
    fallback_larger_union,
    invalid_argument,
    not_configured,
    stale_generation,
    backpressure,
    prefetch_failed,
    policy_failed,
    companion_rejected,
    cancelled,
    shutdown,
    unsupported_slots,
    _count,
};

const char * llama_kv_live_lifecycle_status_name(
        llama_kv_live_lifecycle_status status) noexcept;

// These frontiers are deliberately one value per companion.  A successful
// boundary can therefore not advance target KV while leaving recurrent state,
// MTP KV, or speculative state on an older branch.
struct llama_kv_live_lifecycle_frontier {
    int64_t target_tokens = 0;
    int64_t recurrent_tokens = 0;
    int64_t mtp_tokens = 0;
    uint64_t speculative_proposed = 0;
    uint64_t speculative_accepted = 0;
    uint64_t speculative_rejected = 0;

    bool valid() const noexcept;
};

enum class llama_kv_live_lifecycle_event : uint8_t {
    prompt_adopt = 0,
    checkpoint_save,
    checkpoint_restore,
    clear,
    cancel,
    slot_reuse,
    shutdown,
};

// The callback is an owner-side seam for target, recurrent, MTP, and
// speculative state. It runs from the residency transaction's publish phase,
// immediately before the immutable table publish. A false result aborts the
// transaction, so no companion can publish without its page table.
struct llama_kv_live_lifecycle_hooks {
    void * context = nullptr;
    bool (*publish_companions)(
            void * context,
            const llama_kv_live_lifecycle_generation & generation,
            const llama_kv_live_lifecycle_frontier & frontier) noexcept = nullptr;
    void (*rollback_companions)(
            void * context,
            const llama_kv_live_lifecycle_generation & generation,
            const llama_kv_live_lifecycle_frontier & frontier) noexcept = nullptr;
    bool (*event)(
            void * context,
            llama_kv_live_lifecycle_event event,
            const llama_kv_live_lifecycle_generation & generation) noexcept = nullptr;
    bool (*generation_current)(
            void * context,
            const llama_kv_live_lifecycle_generation & generation) noexcept = nullptr;
};

struct llama_kv_live_lifecycle_config {
    uint32_t slot_count = 1;
    llama_kv_prefetch_config prefetch;
};

struct llama_kv_live_lifecycle_resolution {
    llama_kv_live_lifecycle_status status =
        llama_kv_live_lifecycle_status::invalid_argument;
    llama_kv_prefetch_resolution prefetch;
};

struct llama_kv_live_lifecycle_result {
    llama_kv_live_lifecycle_status status =
        llama_kv_live_lifecycle_status::invalid_argument;
    llama_kv_live_lifecycle_generation generation;
    llama_kv_live_lifecycle_frontier frontier;
    llama_kv_live_lifecycle_resolution readiness;
    llama_kv_live_policy_result policy;
    bool companion_published = false;
    bool companion_rolled_back = false;
};

class llama_kv_live_lifecycle {
public:
    static std::unique_ptr<llama_kv_live_lifecycle> create(
            const llama_kv_live_lifecycle_config & config,
            const llama_kv_prefetch_backend & prefetch_backend,
            const llama_kv_live_lifecycle_hooks & hooks,
            llama_kv_live_lifecycle_status & status) noexcept;
    ~llama_kv_live_lifecycle();

    llama_kv_live_lifecycle(const llama_kv_live_lifecycle &) = delete;
    llama_kv_live_lifecycle & operator=(const llama_kv_live_lifecycle &) = delete;

    llama_kv_live_lifecycle_status start(
            const llama_kv_live_lifecycle_generation & generation) noexcept;
    llama_kv_live_lifecycle_status set_frontier(
            const llama_kv_live_lifecycle_frontier & frontier) noexcept;

    // `generation == 0` is accepted as shorthand for the current operation
    // generation. Other values must match it. The owner stamps every accepted
    // intent with the current operation generation before queueing.
    llama_kv_live_lifecycle_status prefetch(
            const std::vector<llama_kv_prefetch_intent> & ranked) noexcept;
    // Prediction is lookahead evidence only. The lifecycle stamps the
    // observed query with the active operation generation; callers must still
    // pass authoritative pages to ensure_ready() before attention consumes.
    llama_kv_live_lifecycle_status observe_query(
            uint32_t layer, uint64_t token,
            const std::vector<llama_kv_prefetch_intent> & ranked) noexcept;
    std::vector<llama_kv_prefetch_intent> predict_next(
            uint32_t layer, uint64_t token,
            uint32_t limit = UINT32_MAX) const noexcept;
    llama_kv_live_lifecycle_resolution ensure_ready(
            const std::vector<llama_kv_prefetch_intent> & required,
            const std::vector<uint64_t> & previous_hot_set,
            uint32_t wait_budget_steps = UINT32_MAX) noexcept;
    llama_kv_live_lifecycle_status advance() noexcept;

    // The caller first resolves decode readiness, then supplies the complete
    // live-policy boundary. The lifecycle wrapper forwards ordinary transfer
    // hooks while fencing companion publication and the generation recheck.
    llama_kv_live_lifecycle_result apply_policy(
            llama_kv_residency_table & table,
            llama_kv_residency_pool & pool,
            const llama_kv_live_policy_boundary & boundary,
            const llama_kv_residency_pool_backend & backend,
            const llama_kv_residency_transfer_transport & transport,
            const llama_kv_residency_transaction_hooks & hooks = {}) noexcept;

    llama_kv_live_lifecycle_status checkpoint_save() noexcept;
    llama_kv_live_lifecycle_status prompt_adopt(
            const llama_kv_live_lifecycle_generation & generation) noexcept;
    llama_kv_live_lifecycle_status slot_reuse(
            const llama_kv_live_lifecycle_generation & generation) noexcept;
    llama_kv_live_lifecycle_status checkpoint_restore(
            const llama_kv_live_lifecycle_generation & generation) noexcept;
    llama_kv_live_lifecycle_status clear() noexcept;
    llama_kv_live_lifecycle_status cancel() noexcept;
    void shutdown() noexcept;

    bool active() const noexcept { return active_; }
    bool stopped() const noexcept { return stopped_; }
    const llama_kv_live_lifecycle_generation & generation() const noexcept {
        return generation_;
    }
    const llama_kv_live_lifecycle_frontier & frontier() const noexcept {
        return frontier_;
    }
    const llama_kv_prefetch_counters & prefetch_counters() const noexcept {
        return prefetch_->counters();
    }

private:
    struct transaction_context;

    llama_kv_live_lifecycle(
            const llama_kv_live_lifecycle_config & config,
            std::unique_ptr<llama_kv_prefetch_scheduler> prefetch,
            llama_kv_live_lifecycle_hooks hooks);

    bool current() const noexcept;
    bool matches_current(uint64_t generation) const noexcept;
    llama_kv_live_lifecycle_status rotate(
            llama_kv_live_lifecycle_event event,
            const llama_kv_live_lifecycle_generation * next) noexcept;
    void cancel_tracked() noexcept;
    static llama_kv_live_lifecycle_status map_prefetch_status(
            llama_kv_prefetch_status status) noexcept;
    static bool transaction_phase(void *, llama_kv_residency_transaction_phase) noexcept;
    static bool transaction_pin(void *, const llama_kv_page_id &) noexcept;
    static void transaction_unpin(void *, const llama_kv_page_id &) noexcept;
    static bool transaction_drop(void *, const llama_kv_page_record &) noexcept;
    static bool transaction_restore(void *, const llama_kv_page_record &) noexcept;
    static void transaction_retire(void *, const llama_kv_page_record &) noexcept;
    static bool transaction_host(void *, const llama_kv_page_id &, uint64_t) noexcept;
    static bool transaction_recheck(void *, uint64_t,
            const std::vector<llama_kv_page_record> &) noexcept;

    llama_kv_live_lifecycle_config config_;
    std::unique_ptr<llama_kv_prefetch_scheduler> prefetch_;
    llama_kv_live_lifecycle_hooks hooks_;
    llama_kv_live_lifecycle_generation generation_;
    llama_kv_live_lifecycle_frontier frontier_;
    std::vector<llama_kv_prefetch_intent> tracked_;
    bool active_ = false;
    bool stopped_ = false;
};
