#pragma once

#include "llama-kv-residency.h"
#include "llama-vbr-artifact-capture.h"
#include "llama-vbr-artifact-stage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// This is the backend residency axis. It deliberately does not share the VBR
// representation pool: VBR answers which encoding owns a row, while this
// object answers which logical page has a bounded device slot.
enum class llama_kv_residency_transfer_direction : uint8_t {
    d2h_seal = 0,
    d2h_reseal,
    h2d_promotion,
    _count,
};

struct llama_kv_residency_completion {
    llama_kv_page_id page;
    uint64_t table_epoch = 0;
    uint32_t physical_slot = UINT32_MAX;
    uint32_t run_index = UINT32_MAX;
};

struct llama_kv_residency_transfer_run {
    uint32_t page_index = UINT32_MAX;
    uint32_t lane = 0;
    uint32_t layer = UINT32_MAX;
    uint8_t side = 0;
    uint32_t first_physical_row = 0;
    uint32_t row_count = 0;
    uint64_t row_bytes = 0;
    uint64_t host_offset = 0;
    uint64_t device_offset = 0;

    uint64_t useful_bytes() const noexcept;
};

struct llama_kv_residency_transfer_page {
    llama_kv_page_id page;
    uint64_t table_epoch = 0;
    uint32_t physical_slot = UINT32_MAX;
    std::vector<llama_kv_residency_transfer_run> runs;
};

struct llama_kv_residency_transfer_limits {
    uint32_t max_pages = 1024;
    uint32_t max_runs = 1048576;
    uint64_t max_useful_bytes = uint64_t(16)*1024*1024*1024;
};

struct llama_kv_residency_transfer_plan {
    llama_kv_residency_transfer_direction direction =
        llama_kv_residency_transfer_direction::d2h_seal;
    uint64_t useful_bytes = 0;
    uint64_t aligned_bytes = 0;
    uint32_t event_count = 0;
    std::vector<llama_kv_residency_transfer_page> pages;
    std::vector<llama_kv_residency_transfer_run> runs;
};

// Inputs are copied into a bounded plan. Runs are coalesced only within one
// page when layer, side, physical rows, and both byte spaces are consecutive.
bool llama_kv_residency_build_transfer_plan(
    llama_kv_residency_transfer_direction direction,
    const std::vector<llama_kv_residency_transfer_page> & pages,
    uint64_t alignment,
    const llama_kv_residency_transfer_limits & limits,
    llama_kv_residency_transfer_plan & output) noexcept;

enum class llama_kv_residency_pool_status : uint8_t {
    ok = 0,
    invalid_argument,
    not_configured,
    backend_unavailable,
    slot_unavailable,
    event_unavailable,
    staging_unavailable,
    catalog_unavailable,
    transfer_failed,
    cancelled,
    stale_completion,
    dirty_page,
    not_found,
    transaction_closed,
    internal_error,
    _count,
};

const char * llama_kv_residency_pool_status_name(
    llama_kv_residency_pool_status status) noexcept;

struct llama_kv_residency_pool_config {
    uint32_t slot_capacity = 0;
    uint64_t bytes_per_slot = 0;
    uint64_t allocation_alignment = 1;
    uint32_t event_capacity = 0;
    uint64_t max_transfer_bytes = uint64_t(16)*1024*1024*1024;
};

struct llama_kv_residency_pool_backend {
    void * context = nullptr;

    // Optional whole-pool physical reservation. A backend without these
    // callbacks is still useful for plan-only CPU tests, but cannot execute a
    // device mapping operation.
    bool (*reserve_slots)(void * context, uint32_t slots,
                          uint64_t bytes_per_slot) noexcept = nullptr;
    void (*release_slots)(void * context, uint32_t slots,
                          uint64_t bytes_per_slot) noexcept = nullptr;
    bool (*map_slot)(void * context, uint32_t slot) noexcept = nullptr;
    bool (*drop_slot)(void * context, uint32_t slot) noexcept = nullptr;

    // `host` points into the bounded pinned ring. The callback may retain it
    // only until complete_copy/cancel_copy for the same ticket.
    bool (*issue_copy)(
        void * context,
        llama_kv_residency_transfer_direction direction,
        const llama_kv_residency_completion & completion,
        uint64_t device_offset,
        void * host,
        size_t size,
        uint64_t ticket,
        bool asynchronous) noexcept = nullptr;
    bool (*complete_copy)(void * context, uint64_t ticket) noexcept = nullptr;
    void (*cancel_copy)(void * context, uint64_t ticket) noexcept = nullptr;

    // Set for a real ggml-backed adapter.  Tensor transfers use the backend
    // and slot tensor directly; the callback copy seam above remains for
    // deterministic CPU/fault-injection backends.  Appending these fields
    // preserves the positional initializer contract of the fake backend.
    ggml_backend_t backend = nullptr;
    ggml_tensor * (*slot_tensor)(void * context, uint32_t slot) noexcept = nullptr;
};

enum class llama_kv_residency_ggml_status : uint8_t {
    ok = 0,
    invalid_argument,
    unavailable,
    allocation_failed,
    event_failed,
    _count,
};

const char * llama_kv_residency_ggml_status_name(
        llama_kv_residency_ggml_status status) noexcept;

struct llama_kv_residency_ggml_adapter_config {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr; // borrowed; owner outlives adapter
    uint32_t slot_capacity = 0;
    uint64_t bytes_per_slot = 0;
    bool force_synchronous = false;
};

// Owns the slot tensor views and backend events for one compact residency
// allocation.  The allocation itself remains owned by the pager; this object
// only creates bounded views over it and supplies the transfer-pool callbacks.
// Real transfers take the tensor route in llama_kv_residency_execute_transfer;
// issue_copy is retained as a synchronous-compatible compatibility door for
// callers that do not use the extended tensor route.
class llama_kv_residency_ggml_adapter {
public:
    static std::unique_ptr<llama_kv_residency_ggml_adapter> create(
            const llama_kv_residency_ggml_adapter_config & config,
            llama_kv_residency_ggml_status & status) noexcept;
    ~llama_kv_residency_ggml_adapter();

    llama_kv_residency_ggml_adapter(
            const llama_kv_residency_ggml_adapter &) = delete;
    llama_kv_residency_ggml_adapter & operator=(
            const llama_kv_residency_ggml_adapter &) = delete;

    llama_kv_residency_pool_backend pool_backend() noexcept;
    ggml_backend_t backend() const noexcept { return backend_; }
    ggml_backend_dev_t device() const noexcept { return device_; }
    uint32_t slot_capacity() const noexcept { return slot_capacity_; }
    uint64_t bytes_per_slot() const noexcept { return bytes_per_slot_; }
    uint32_t mapped_slots() const noexcept;
    uint32_t pending_events() const noexcept;
    void drain() noexcept;

private:
    struct pending_event;
    explicit llama_kv_residency_ggml_adapter(
            const llama_kv_residency_ggml_adapter_config & config) noexcept;

    ggml_backend_t backend_ = nullptr;
    ggml_backend_dev_t device_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_context * tensor_context_ = nullptr;
    uint32_t slot_capacity_ = 0;
    uint64_t bytes_per_slot_ = 0;
    bool force_synchronous_ = false;
    ggml_tensor * storage_tensor_ = nullptr;
    std::vector<ggml_tensor *> slot_tensors_;
    std::vector<uint8_t> mapped_;
    std::vector<pending_event> pending_;

    bool initialize_slots() noexcept;
    void release_slots() noexcept;
    bool issue(
            llama_kv_residency_transfer_direction direction,
            const llama_kv_residency_completion & completion,
            uint64_t device_offset, void * host, size_t size,
            uint64_t ticket, bool asynchronous) noexcept;
    bool complete(uint64_t ticket) noexcept;
    void cancel(uint64_t ticket) noexcept;

    static bool reserve_slots(
            void * context, uint32_t slots, uint64_t bytes_per_slot) noexcept;
    static void release_slots(
            void * context, uint32_t slots, uint64_t bytes_per_slot) noexcept;
    static bool map_slot(void * context, uint32_t slot) noexcept;
    static bool drop_slot(void * context, uint32_t slot) noexcept;
    static bool issue_copy(
            void * context, llama_kv_residency_transfer_direction direction,
            const llama_kv_residency_completion & completion,
            uint64_t device_offset, void * host, size_t size,
            uint64_t ticket, bool asynchronous) noexcept;
    static bool complete_copy(void * context, uint64_t ticket) noexcept;
    static void cancel_copy(void * context, uint64_t ticket) noexcept;
    static ggml_tensor * slot_tensor(void * context, uint32_t slot) noexcept;
};

struct llama_kv_residency_catalog_reservation {
    void * context = nullptr;
    bool (*reserve)(void * context, uint64_t bytes) noexcept = nullptr;
    void (*release)(void * context, uint64_t bytes) noexcept = nullptr;
};

struct llama_kv_residency_transfer_transport {
    vbr_pinned_chunk_ring * download_ring = nullptr;
    vbr_h2d_chunk_ring * upload_ring = nullptr;
    bool force_synchronous = false;

    void * context = nullptr;
    bool (*host_read)(
        void * context, uint32_t page_index, uint64_t offset,
        uint8_t * destination, size_t size) noexcept = nullptr;
    bool (*host_write)(
        void * context, uint32_t page_index, uint64_t offset,
        const uint8_t * source, size_t size) noexcept = nullptr;
    bool (*continue_transfer)(void * context) noexcept = nullptr;
    bool (*recheck)(
        void * context,
        const llama_kv_residency_completion & completion) noexcept = nullptr;
};

struct llama_kv_residency_transfer_counters {
    uint64_t queued = 0;
    uint64_t submitted = 0;
    uint64_t copied_useful_bytes = 0;
    uint64_t copied_aligned_bytes = 0;
    uint64_t waits = 0;
    uint64_t cancellations = 0;
    uint64_t stale_completions = 0;
    uint64_t event_completions = 0;
    uint64_t evictions = 0;
    uint64_t backpressure_waits = 0;
    uint64_t transfer_time_us = 0;
    uint64_t map_failures = 0;
};

struct llama_kv_residency_transfer_result {
    llama_kv_residency_pool_status status =
        llama_kv_residency_pool_status::internal_error;
    llama_kv_residency_transfer_counters counters;
};

class llama_kv_residency_pool;

// Move-only claim. It owns all pre-submit reservations and drops newly mapped
// slots on failure, so a partially submitted transfer cannot leak capacity.
class llama_kv_residency_transfer_claim {
public:
    llama_kv_residency_transfer_claim() noexcept = default;
    ~llama_kv_residency_transfer_claim();
    llama_kv_residency_transfer_claim(
        llama_kv_residency_transfer_claim && other) noexcept;
    llama_kv_residency_transfer_claim & operator=(
        llama_kv_residency_transfer_claim && other) noexcept;

    llama_kv_residency_transfer_claim(
        const llama_kv_residency_transfer_claim &) = delete;
    llama_kv_residency_transfer_claim & operator=(
        const llama_kv_residency_transfer_claim &) = delete;

    bool active() const noexcept;
    uint32_t reserved_slots() const noexcept;
    uint32_t reserved_events() const noexcept;

private:
    llama_kv_residency_pool * pool_ = nullptr;
    std::vector<uint32_t> slots_;
    uint32_t events_ = 0;
    uint64_t catalog_bytes_ = 0;
    llama_kv_residency_catalog_reservation catalog;
    bool mapped_ = false;
    bool active_ = false;

    void reset() noexcept;
    friend class llama_kv_residency_pool;
};

// Separate backend KV residency pool. It owns only physical-slot mappings and
// their bounded accounting; logical-page policy and atomic table publication
// remain in the higher-level residency transaction.
class llama_kv_residency_pool {
public:
    static std::unique_ptr<llama_kv_residency_pool> create(
        const llama_kv_residency_pool_config & config,
        const llama_kv_residency_pool_backend & backend,
        llama_kv_residency_pool_status & status) noexcept;
    ~llama_kv_residency_pool();

    llama_kv_residency_pool(const llama_kv_residency_pool &) = delete;
    llama_kv_residency_pool & operator=(const llama_kv_residency_pool &) = delete;

    uint32_t slot_capacity() const noexcept;
    uint64_t bytes_per_slot() const noexcept;
    uint32_t mapped_slots() const noexcept;
    uint64_t resident_bytes() const noexcept;
    uint32_t pending_events() const noexcept;

    llama_kv_residency_pool_status reserve(
        const llama_kv_residency_transfer_plan & plan,
        uint64_t staging_capacity,
        const llama_kv_residency_catalog_reservation & catalog,
        llama_kv_residency_transfer_claim & output) noexcept;

    llama_kv_residency_pool_status map_reserved(
        llama_kv_residency_transfer_claim & claim,
        const llama_kv_residency_transfer_plan & plan,
        bool host_valid) noexcept;
    void rollback(llama_kv_residency_transfer_claim & claim) noexcept;
    void commit(llama_kv_residency_transfer_claim & claim) noexcept;

    llama_kv_residency_pool_status drop_logical_page(
        const llama_kv_page_id & page,
        bool & dropped) noexcept;
    llama_kv_residency_pool_status mark_dirty(
        const llama_kv_page_id & page, bool dirty) noexcept;
    bool find_logical_page(
        const llama_kv_page_id & page, uint32_t & slot) const noexcept;

private:
    struct slot;
    struct impl;
    explicit llama_kv_residency_pool(
        std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;

    void release_claim(llama_kv_residency_transfer_claim & claim) noexcept;
    friend class llama_kv_residency_transfer_claim;
};

llama_kv_residency_transfer_result llama_kv_residency_execute_transfer(
    llama_kv_residency_pool & pool,
    const llama_kv_residency_transfer_plan & plan,
    llama_kv_residency_transfer_claim & claim,
    const llama_kv_residency_pool_backend & backend,
    const llama_kv_residency_transfer_transport & transport) noexcept;
