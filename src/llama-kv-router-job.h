#pragma once

#include "llama-kv-prefetch.h"
#include "llama-kv-router-ranking.h"
#include "llama-vbr-artifact-capture.h"
#include "ggml-backend.h"

#include <array>
#include <memory>
#include <vector>

struct llama_kv_router_query_identity {
    int32_t sequence_id = -1;
    uint64_t session_generation = 0;
    uint64_t turn_id = 0;
    uint64_t query_generation = 0;
    uint64_t rollback_generation = 0;
    uint64_t table_epoch = 0;
    uint64_t content_generation = 0;
};

struct llama_kv_router_exact_page_record {
    llama_kv_prefetch_candidate candidate;
    uint64_t descriptor_content_version = 0;
    uint64_t descriptor_summary_version = 0;
    uint32_t compact_layer = UINT32_MAX;
    uint32_t model_layer = UINT32_MAX;
    uint32_t stream = UINT32_MAX;
    uint32_t physical_slot = UINT32_MAX;
    uint32_t valid_rows = 0;
    uint32_t key_unit_id = UINT32_MAX;
    uint64_t first_absolute_row = 0;
    uint64_t descriptor_page_generation = 0;
    uint64_t descriptor_content_generation = 0;
    uint64_t descriptor_summary_generation = 0;
};

enum class llama_kv_router_finalize_reason : uint8_t {
    none = 0,
    inactive_query,
    terminal_failure,
    identity_sequence,
    identity_session,
    identity_turn,
    identity_query,
    identity_rollback,
    identity_table,
    identity_content,
    layer_completion,
    provenance,
    duplicate_logical_page,
    invalid_probability,
    descriptor_generation,
    descriptor_version,
    layer_mass_sum,
    cold_not_in_shortlist,
};

// First failed invariant only; bounded to one record and safe to retain after
// cancellation. It contains identities and scalar scores, never prompt/key data.
struct llama_kv_router_finalize_diagnostic {
    struct record_snapshot {
        llama_kv_page_id identity;
        uint32_t compact_layer = UINT32_MAX;
        uint32_t model_layer = UINT32_MAX;
        uint32_t stream = UINT32_MAX;
        uint32_t physical_slot = UINT32_MAX;
        uint32_t valid_rows = 0;
        uint32_t key_unit_id = UINT32_MAX;
        uint64_t first_absolute_row = 0;
        uint64_t descriptor_page_generation = 0;
        uint64_t descriptor_content_generation = 0;
        uint64_t descriptor_summary_generation = 0;
        uint64_t content_version = 0;
        uint64_t summary_version = 0;
        uint32_t shortlist_match = 0;
        uint32_t cold = 0;
        float peak_probability = 0.0f;
        float mean_probability = 0.0f;
    };
    struct layer_snapshot {
        uint32_t compact_layer = UINT32_MAX;
        uint32_t model_layer = UINT32_MAX;
        uint32_t page_tokens = 0;
        uint32_t query_heads = 0;
        uint32_t kv_heads = 0;
        uint32_t key_dimension = 0;
        uint32_t resident_streams = 0;
        uint64_t probe_capture_generation = 0;
        int64_t resident_key_ne[3] = {};
        int64_t validity[9] = {};
        uint32_t validity_read = 0;
        double mean_mass_sum = 0.0;
        uint64_t state_valid_channel_mask[4] = {};
        uint64_t state_has_mass_channel_mask[4] = {};
        uint32_t state_nonzero_count = 0;
        uint32_t state_finite_count = 0;
        uint32_t state_snapshot_read = 0;
        uint32_t rerank_eligible_pages = 0;
        uint32_t mass_eligible_pages = 0;
        uint32_t valid_probe_count = 0;
        uint32_t probe_finite_count = 0;
        uint32_t probe_nonzero_count = 0;
        uint32_t probe_nonfinite_count = 0;
        uint32_t probe_snapshot_read = 0;
    };
    llama_kv_router_finalize_reason reason = llama_kv_router_finalize_reason::none;
    uint32_t compact_layer = UINT32_MAX;
    uint32_t model_layer = UINT32_MAX;
    uint32_t logical_page = UINT32_MAX;
    uint32_t record_index = UINT32_MAX;
    llama_kv_router_query_identity expected;
    llama_kv_router_query_identity actual;
    uint64_t expected_page_generation = 0;
    uint64_t actual_page_generation = 0;
    uint64_t expected_content_version = 0;
    uint64_t actual_content_version = 0;
    uint64_t expected_summary_version = 0;
    uint64_t actual_summary_version = 0;
    uint8_t expected_provenance = 0;
    uint8_t actual_provenance = 0;
    float peak_probability = 0.0f;
    float mean_probability = 0.0f;
    double measured_layer_sum = 0.0;
    uint32_t record_count = 0;
    uint32_t layer_count = 0;
    record_snapshot records[64];
    layer_snapshot layers[64];
};

enum class llama_kv_router_plan_error : uint8_t {
    none = 0,
    not_configured,
    stale_query,
    invalid_capture,
    stale_page,
    missing_key,
    invalid_geometry,
};

struct llama_kv_router_owned_page_descriptor {
    llama_kv_prefetch_candidate candidate;
    uint32_t valid_rows = 0;
    uint32_t stream = UINT32_MAX;
    uint32_t physical_slot = UINT32_MAX;
    uint64_t first_absolute_row = 0;
    uint64_t page_generation = 0;
    uint64_t content_generation = 0;
    uint64_t summary_generation = 0;
    uint32_t key_unit_id = UINT32_MAX;
    llama_kv_rerank_stage_source key_source;
};

struct llama_kv_router_layer_plan {
    llama_kv_router_query_identity identity;
    uint32_t compact_layer = UINT32_MAX;
    uint32_t model_layer = UINT32_MAX;
    uint64_t probe_capture_generation = 0;
    ggml_tensor * probes = nullptr;          // borrowed from the query cache
    ggml_tensor * validity = nullptr;        // borrowed from the query cache
    ggml_tensor * resident_keys = nullptr;   // borrowed from the cache layer
    uint32_t page_tokens = 0;
    uint32_t query_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t key_dimension = 0;
    float attention_scale = 0.0f;
    float logit_softcap = 0.0f;
    std::vector<llama_kv_router_owned_page_descriptor> pages;
};

enum class llama_kv_router_execution_status : uint8_t {
    pending = 0,
    ready,
    stale,
    failed,
    cancelled,
};

// Per-query production call gate. It owns the terminal batch and makes the
// final-user executor start idempotent across repeated commit polls.
class llama_kv_router_query_executor {
public:
    using execute_fn = llama_kv_router_execution_status (*)(void * context,
            const llama_kv_router_query_identity & identity,
            const std::vector<llama_kv_router_layer_plan> & plans,
            std::vector<llama_kv_router_exact_page_record> & records) noexcept;
    llama_kv_router_execution_status execute_once(
            const llama_kv_router_query_identity & identity, bool provisional_phase,
            const std::vector<llama_kv_router_layer_plan> & plans,
            std::vector<llama_kv_router_exact_page_record> & records,
            void * context, execute_fn execute) noexcept;
    void cancel() noexcept;

private:
    bool started_ = false;
    llama_kv_router_query_identity identity_;
    llama_kv_router_execution_status status_ = llama_kv_router_execution_status::pending;
    std::vector<llama_kv_router_exact_page_record> records_;
};

// The cache and its focused fixtures use this same final-user boundary. The
// cache supplies live plan preparation and terminal adoption callbacks; tests
// can bind real captured tensors and canonical key readers without building a
// model-dependent llama_kv_cache.
struct llama_kv_router_cache_adapter {
    using prepare_fn = bool (*)(void * context, int32_t sequence_id, uint64_t turn_id,
            std::vector<llama_kv_router_layer_plan> & plans,
            llama_kv_router_plan_error & error) noexcept;
    using terminal_fn = void (*)(void * context,
            llama_kv_router_execution_status status,
            const std::vector<llama_kv_router_exact_page_record> & records) noexcept;
    struct result {
        llama_kv_router_execution_status status = llama_kv_router_execution_status::pending;
        llama_kv_router_plan_error plan_error = llama_kv_router_plan_error::none;
        uint32_t layer = UINT32_MAX;
        std::vector<llama_kv_router_exact_page_record> records;
    };
    static result execute(llama_kv_router_query_executor & gate,
            int32_t sequence_id, uint64_t turn_id,
            void * prepare_context, prepare_fn prepare,
            void * execute_context, llama_kv_router_query_executor::execute_fn execute,
            void * terminal_context, terminal_fn terminal) noexcept;
};

bool llama_kv_router_layer_plan_metadata_valid(
        const llama_kv_router_layer_plan & plan) noexcept;

class llama_kv_router_key_reader : public std::enable_shared_from_this<llama_kv_router_key_reader> {
public:
    using current_fn = bool (*)(void * context, const llama_kv_page_id & identity,
            uint64_t content_version, uint64_t summary_version) noexcept;
    static std::shared_ptr<llama_kv_router_key_reader> create(
            const llama_kv_page_id & identity, uint64_t content_version,
            uint64_t summary_version, const vbr_selected_page_unit_descriptor & unit,
            uint32_t valid_rows, void * current_context, current_fn is_current) noexcept;
    llama_kv_rerank_stage_source source() noexcept;
    const llama_kv_page_id & identity() const noexcept { return identity_; }
    const vbr_selected_page_unit_descriptor & unit() const noexcept { return unit_; }

private:
    llama_kv_router_key_reader() = default;
    static bool recheck(void * context, uint64_t content_version) noexcept;
    static bool read(void * context, uint64_t offset, void * dst, size_t bytes) noexcept;
    llama_kv_page_id identity_;
    uint64_t content_version_ = 0;
    uint64_t summary_version_ = 0;
    uint32_t valid_rows_ = 0;
    vbr_selected_page_unit_descriptor unit_;
    void * current_context_ = nullptr;
    current_fn is_current_ = nullptr;
};

// Only a terminal owner result can turn coarse candidates into exact_mass.
// The result owns both the candidate records and the descriptor versions that
// authenticated them, so graph/mailbox reuse cannot rewrite provenance.
class llama_kv_router_completed_owner_result {
private:
    static bool make_exact_rerank_candidates(
            const llama_kv_router_query_identity & identity,
            bool terminal_success,
            const std::vector<llama_kv_router_exact_page_record> & records,
            std::vector<llama_kv_prefetch_candidate> & output,
            llama_kv_router_finalize_diagnostic & diagnostic) noexcept;

    llama_kv_router_completed_owner_result() = default;
    friend class llama_kv_router_job;
};

// Persistent cache-owned storage and event lifetime for probe reranking.
// Graph construction/enqueue policy is supplied by the production owner.
class llama_kv_router_job {
public:
    struct allocation_ledger {
        uint64_t device_requested = 0;
        uint64_t device_realized = 0;
        uint64_t pinned_requested = 0;
        uint64_t pinned_realized = 0;
        uint64_t pageable_bytes = 0;
        uint64_t alignment = 0;
    };
    struct execution_ledger {
        uint32_t resident_rerank_graphs = 0;
        uint32_t cold_reader_graphs = 0;
        uint32_t reader_events = 0;
        uint32_t mass_graphs = 0;
        uint32_t output_records = 0;
        uint64_t total_us = 0;
        uint64_t cold_source_read_us = 0;
        uint64_t reader_event_wait_us = 0;
        uint64_t key_h2d_bytes = 0;
        uint64_t full_page_h2d_bytes = 0;
        uint64_t mass_readback_bytes = 0;
        uint64_t scratch_allocated_bytes = 0;
    };

    static std::unique_ptr<llama_kv_router_job> create(
            ggml_backend_t backend, ggml_backend_buffer_type_t pinned_host_buft,
            size_t slot_bytes, llama_kv_rerank_stage_backend stage_backend,
            ggml_type encoded_type = GGML_TYPE_I8,
            uint32_t encoded_row_elements = 0) noexcept;
    ~llama_kv_router_job();
    llama_kv_router_job(const llama_kv_router_job &) = delete;
    llama_kv_router_job & operator=(const llama_kv_router_job &) = delete;

    llama_kv_rerank_stage_status submit(
            const llama_kv_rerank_stage_source & source, uint64_t byte_offset,
            uint32_t rows, const llama_kv_rerank_chunk_task & task,
            uint32_t & slot, uint64_t * ticket = nullptr) noexcept;
    uint32_t poll(llama_kv_rerank_stage_ticket * output,
            uint32_t capacity) noexcept;
    bool retain_coarse_shortlist(const llama_kv_router_query_identity & identity,
            uint32_t layer,
            const std::vector<llama_kv_prefetch_candidate> & candidates) noexcept;
    bool complete_query(const llama_kv_router_query_identity & identity,
            bool terminal_success,
            const std::vector<llama_kv_router_exact_page_record> & records) noexcept;
    // Complete a fresh query when the cache owner has proved that it contains
    // no eligible historical pages. Every layer receives an explicit empty
    // coarse batch before the terminal empty result is accepted.
    bool complete_empty_query(const llama_kv_router_query_identity & identity,
            uint32_t layer_count) noexcept;
    llama_kv_router_execution_status execute_query(
            const llama_kv_router_query_identity & identity,
            const std::vector<llama_kv_router_layer_plan> & layer_plans,
            std::vector<llama_kv_router_exact_page_record> & records) noexcept;
    const std::vector<llama_kv_prefetch_candidate> & exact_candidates() const noexcept {
        return exact_candidates_;
    }
    const llama_kv_router_owner_state & owner_state() const noexcept { return owner_state_; }
    void abort_query() noexcept;
    void cancel() noexcept;
    bool failed() const noexcept { return failed_; }
    const allocation_ledger & allocations() const noexcept { return ledger_; }
    const execution_ledger & execution() const noexcept { return execution_; }
    const llama_kv_router_finalize_diagnostic & finalization_diagnostic() const noexcept {
        return finalization_diagnostic_;
    }
    size_t slot_bytes() const noexcept { return slot_bytes_; }
    ggml_tensor * encoded_key_slot(uint32_t slot) const noexcept;

private:
    llama_kv_router_job() = default;
    bool initialize(ggml_backend_t backend,
            ggml_backend_buffer_type_t pinned_host_buft, size_t slot_bytes,
            llama_kv_rerank_stage_backend stage_backend, ggml_type encoded_type,
            uint32_t encoded_row_elements) noexcept;

    llama_kv_rerank_stage_ring ring_;
    ggml_backend_t backend_ = nullptr; // borrowed; owner outlives this job
    ggml_context * device_ctx_ = nullptr;
    ggml_backend_buffer_t device_buffer_ = nullptr;
    ggml_backend_buffer_t pinned_buffer_ = nullptr;
    std::array<ggml_tensor *, LLAMA_KV_RERANK_STAGE_SLOTS> device_slots_{};
    size_t slot_bytes_ = 0;
    allocation_ledger ledger_;
    execution_ledger execution_;
    bool failed_ = false;
    llama_kv_router_query_identity query_identity_;
    llama_kv_router_owner_state owner_state_;
    std::vector<std::vector<llama_kv_prefetch_candidate>> coarse_shortlists_;
    std::vector<bool> coarse_shortlist_completed_;
    std::vector<llama_kv_prefetch_candidate> exact_candidates_;
    std::vector<llama_kv_router_exact_page_record> completed_records_;
    uint64_t completed_turn_id_ = 0;
    llama_kv_router_finalize_diagnostic finalization_diagnostic_;
    bool query_active_ = false;
    static bool enqueue_copy(void * context, uint32_t slot, const void * host,
            void * device, size_t bytes, const llama_kv_rerank_chunk_task & task,
            uint64_t * event) noexcept;
    static llama_kv_prefetch_poll poll_event(void * context, uint64_t event) noexcept;
    static void cancel_event(void * context, uint64_t event) noexcept;
    static void release_event(void * context, uint64_t event) noexcept;
};
