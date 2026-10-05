#pragma once

#include "llama-kv-prefetch.h"
#include "llama-kv-router-ranking.h"
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
            std::vector<llama_kv_prefetch_candidate> & output) noexcept;

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
    const std::vector<llama_kv_prefetch_candidate> & exact_candidates() const noexcept {
        return exact_candidates_;
    }
    const llama_kv_router_owner_state & owner_state() const noexcept { return owner_state_; }
    void abort_query() noexcept;
    void cancel() noexcept;
    bool failed() const noexcept { return failed_; }
    const allocation_ledger & allocations() const noexcept { return ledger_; }
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
    bool failed_ = false;
    llama_kv_router_query_identity query_identity_;
    llama_kv_router_owner_state owner_state_;
    std::vector<std::vector<llama_kv_prefetch_candidate>> coarse_shortlists_;
    std::vector<llama_kv_prefetch_candidate> exact_candidates_;
    bool query_active_ = false;
    static bool enqueue_copy(void * context, uint32_t slot, const void * host,
            void * device, size_t bytes, const llama_kv_rerank_chunk_task & task,
            uint64_t * event) noexcept;
    static llama_kv_prefetch_poll poll_event(void * context, uint64_t event) noexcept;
    static void cancel_event(void * context, uint64_t event) noexcept;
    static void release_event(void * context, uint64_t event) noexcept;
};
