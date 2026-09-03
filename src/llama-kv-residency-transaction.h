#pragma once

#include "llama-kv-residency-transfer.h"

#include <cstdint>
#include <vector>

// A composite residency operation is intentionally explicit about its
// linearization point. Before publish, every callback must be reversible or
// owned by a move-only transfer claim. After publish, old resources are only
// retired and the new immutable table is authoritative.
enum class llama_kv_residency_transaction_phase : uint8_t {
    snapshot = 0,
    plan,
    reserve,
    pin,
    reseal,
    drop,
    load,
    fence,
    recheck,
    publish,
    retire,
    release,
    _count,
};

const char * llama_kv_residency_transaction_phase_name(
    llama_kv_residency_transaction_phase phase) noexcept;

enum class llama_kv_residency_transaction_status : uint8_t {
    committed = 0,
    invalid_argument,
    shutdown,
    stale_generation,
    stale_epoch,
    insufficient_slots,
    all_pinned,
    dirty_victim,
    missing_host_source,
    short_page,
    phase_failed,
    transfer_failed,
    publish_failed,
    rollback_failed,
    internal_error,
    _count,
};

const char * llama_kv_residency_transaction_status_name(
    llama_kv_residency_transaction_status status) noexcept;

// The desired page vector is the output of policy. It is a complete target
// resident set, not a mutation log; this makes occupied replacement and
// reverse-map validation deterministic from one snapshot.
struct llama_kv_residency_transaction_request {
    std::vector<llama_kv_page_record> desired_pages;
    std::vector<llama_kv_residency_transfer_plan> transfers;
    uint64_t staging_capacity = 0;
    llama_kv_residency_catalog_reservation catalog;
    bool shutting_down = false;
};

struct llama_kv_residency_transaction_hooks {
    void * context = nullptr;

    // A false return injects a failure at that phase. The callback runs
    // without the table or pool mutex held.
    bool (*phase)(
        void * context,
        llama_kv_residency_transaction_phase phase) noexcept = nullptr;

    // Pinning is external to the immutable table (for example, a graph
    // consumer fence). Every successful pin receives exactly one unpin.
    bool (*pin)(
        void * context, const llama_kv_page_id & page) noexcept = nullptr;
    void (*unpin)(
        void * context, const llama_kv_page_id & page) noexcept = nullptr;

    // Clean victim mapping drops happen before loading a replacement so an
    // occupied slot can be reused. restore_clean must restore every drop if
    // a later pre-publish phase fails.
    bool (*drop_clean)(
        void * context, const llama_kv_page_record & page) noexcept = nullptr;
    bool (*restore_clean)(
        void * context, const llama_kv_page_record & page) noexcept = nullptr;
    void (*retire)(
        void * context, const llama_kv_page_record & page) noexcept = nullptr;

    // Optional catalog identity check performed before the bounded H2D read.
    // A false return means the canonical clean host source is absent.
    bool (*has_clean_host)(
        void * context, const llama_kv_page_id & page,
        uint64_t useful_bytes) noexcept = nullptr;

    // Recheck the complete page identity and table epoch after all events
    // have completed, before the table publish linearization point.
    bool (*recheck)(
        void * context, uint64_t base_epoch,
        const std::vector<llama_kv_page_record> & desired) noexcept = nullptr;
};

struct llama_kv_residency_transaction_result {
    llama_kv_residency_transaction_status status =
        llama_kv_residency_transaction_status::internal_error;
    llama_kv_residency_transaction_phase failed_phase =
        llama_kv_residency_transaction_phase::snapshot;
    uint64_t base_epoch = 0;
    uint64_t published_epoch = 0;
    uint32_t pinned_pages = 0;
    uint32_t dropped_pages = 0;
    uint32_t loaded_pages = 0;
    bool published = false;
    bool rollback_complete = false;
    llama_kv_residency_transfer_counters transfer_counters;
};

// Executes one failure-atomic residency update. The caller supplies the
// policy target and transfer plans; this function owns ordering, rollback,
// generation checks, and exactly-once table publication.
llama_kv_residency_transaction_result
llama_kv_residency_execute_transaction(
    llama_kv_residency_table & table,
    llama_kv_residency_pool & pool,
    const llama_kv_residency_transaction_request & request,
    const llama_kv_residency_pool_backend & backend,
    const llama_kv_residency_transfer_transport & transport,
    const llama_kv_residency_transaction_hooks & hooks = {}) noexcept;

