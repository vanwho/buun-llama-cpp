#pragma once

#include "llama-cache-budget.h"
#include "llama-kv-pager-config.h"
#include "llama-kv-live-policy.h"
#include "llama-kv-routing-summary.h"
#include "llama-kv-residency.h"
#include "llama-kv-residency-transfer.h"
#include "llama-vbr-artifact-catalog.h"
#include "llama-vbr-artifact-capture.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

class vbr_h2d_chunk_ring;
struct llama_model;

struct llama_kv_pager_geometry {
    uint64_t context_tokens = 0;
    uint32_t page_tokens = 256;
    uint32_t attention_layers = 0;
    uint32_t kv_heads = 0;
    uint32_t key_length = 0;
    uint32_t value_length = 0;
    uint64_t page_bytes = 0; // complete opaque K/V page across all attention layers
    // Byte offsets in the physical slab after admission. Each layer stores all
    // admitted K rows followed by all admitted V rows, so set_rows has a
    // regular row stride and the direct kernel uses the per-layer page span.
    std::vector<uint64_t> layer_k_offsets;
    std::vector<uint64_t> layer_v_offsets;
    std::vector<uint64_t> layer_k_page_bytes;
    std::vector<uint64_t> layer_v_page_bytes;
    // Model layer IDs are not necessarily compact attention ordinals in a
    // hybrid model. Keep the checked map beside the offsets.
    std::vector<uint32_t> model_layer_ids;
};

// Construct geometry from model metadata without allocating context-sized
// tensors. This is the admission-time planner input.
bool llama_kv_pager_geometry_from_model(
        const llama_model & model,
        ggml_type type_k,
        ggml_type type_v,
        bool v_trans,
        uint32_t page_tokens,
        uint64_t context_tokens,
        llama_kv_pager_geometry & output) noexcept;

struct llama_kv_pager_resources {
    llama_cache_budget_admission_input admission;
    uint64_t host_budget_bytes = 0;
    uint64_t host_metadata_bytes = 0;
    uint64_t allocator_granularity = 1;
    // Optional upper bound carried from the pre-allocation plan. It prevents
    // reconciliation from growing the pager beyond the slab already owned by
    // the cache.
    uint64_t physical_page_cap = 0;
    bool host_budget_known = false;
    bool duplicate_representation_authority = false;

    // The host copy is optional for the unit-test/observe pager, but when it is
    // enabled these are the complete production transport binding.  The ring
    // is charged once at construction and is never expanded per page.
    bool host_capture_enabled = false;
    llama_cache_budget_config host_budget;
    uint64_t host_source_namespace = 0;
    uint64_t host_topology_identity = 0;
    uint32_t host_child_id = 0;
    uint32_t host_stream_index = 0;
    std::vector<vbr_capture_lane> host_lanes;
    ggml_backend_t host_backend = nullptr;
    uint64_t host_ring_bytes = 0;
    size_t host_chunk_bytes = 0;
    vbr_selected_page_capture_limits host_capture_limits;
    llama_kv_routing_summary_config routing_summary;

    // Selective/exact target storage is constructed by the cache and borrowed
    // here. The pager must not allocate a second physical target KV backing.
    ggml_backend_buffer_t external_storage_buffer = nullptr;
    ggml_tensor * external_storage_tensor = nullptr;
};

// The cache owns the representation-specific sampler. It is called only at a
// completed page boundary, after the graph fence, and may read a bounded set
// of rotated Turbo4 K rows into the input descriptor.
struct llama_kv_pager_routing_summary_provider {
    using build_fn = bool (*) (
            void * context,
            const llama_kv_page_record & page,
            const llama_kv_routing_summary_config & config,
            llama_kv_routing_page_input & output) noexcept;

    void * context = nullptr;
    build_fn build = nullptr;
};

struct llama_kv_pager_allocation {
    void * handle = nullptr;
    uint64_t requested_bytes = 0;
    uint64_t realized_bytes = 0;
};

struct llama_kv_pager_backend {
    std::function<bool(uint64_t, llama_kv_pager_allocation &)> allocate;
    std::function<void(llama_kv_pager_allocation &)> release;
};

enum class llama_kv_pager_host_status : uint8_t {
    ok = 0,
    not_configured,
    invalid_page,
    prepare_failed,
    capture_failed,
    catalog_failed,
    accounting_failed,
    ring_unavailable,
};

const char * llama_kv_pager_host_status_name(
        llama_kv_pager_host_status status) noexcept;

// The live owner supplies exact tensor sources and a snapshot provider.  The
// pager host boundary owns the ring, immutable pageable chains, and catalog
// accounting; it never knows how a model stores or decodes Turbo4 rows.
struct llama_kv_pager_host_provider {
    using prepare_fn = bool (*) (
        void * context,
        const llama_kv_page_record & page,
        vbr_selected_page_capture_request & request,
        std::vector<vbr_selected_page_unit_source> & sources,
        vbr_selected_page_capture_snapshot_provider & snapshots) noexcept;

    void * context = nullptr;
    prepare_fn prepare = nullptr;
};

struct llama_kv_pager_host_result {
    llama_kv_pager_host_status status =
        llama_kv_pager_host_status::not_configured;
    vbr_selected_page_capture_status capture_status =
        vbr_selected_page_capture_status::invalid_argument;
    vbr_selected_page_host_status catalog_status =
        vbr_selected_page_host_status::internal_error;
    uint64_t pageable_bytes = 0;
    uint64_t metadata_bytes = 0;
    uint64_t pinned_bytes = 0;
};

class llama_kv_pager_host {
public:
    static std::unique_ptr<llama_kv_pager_host> create(
            const llama_kv_pager_resources & resources,
            llama_kv_pager_host_provider provider,
            llama_kv_pager_host_status & status) noexcept;

    ~llama_kv_pager_host();
    llama_kv_pager_host(const llama_kv_pager_host &) = delete;
    llama_kv_pager_host & operator=(const llama_kv_pager_host &) = delete;

    void set_provider(llama_kv_pager_host_provider provider) noexcept {
        provider_ = provider;
    }

    llama_kv_pager_host_result seal(
            const llama_kv_page_record & page) noexcept;
    std::vector<vbr_selected_page_host_view> pages() const noexcept;
    bool invalidate(const llama_kv_page_id & page) noexcept;
    vbr_selected_page_host_catalog_snapshot snapshot() const noexcept;
    vbr_h2d_chunk_ring * upload_ring() const noexcept { return upload_ring_.get(); }

private:
    explicit llama_kv_pager_host(
            const llama_kv_pager_resources & resources);

    llama_kv_pager_resources resources_;
    llama_kv_pager_host_provider provider_;
    llama_cache_acct_ledger ledger_;
    llama_vbr_selected_page_host_catalog catalog_;
    std::unique_ptr<vbr_pinned_chunk_ring> ring_;
    std::shared_ptr<vbr_h2d_chunk_ring> upload_ring_;
};

struct llama_kv_pager_snapshot {
    llama_kv_pager_geometry geometry;
    llama_cache_budget_admission_result admission;
    uint32_t logical_page_count = 0;
    uint32_t physical_page_count = 0;
    uint64_t physical_rows = 0;
    uint64_t physical_bytes = 0;
    uint64_t host_metadata_bytes = 0;
    uint64_t mtp_rows = 0;
    uint64_t host_budget_bytes = 0;
    uint64_t vram_budget_bytes = 0;
    uint64_t realized_bytes = 0;
    bool initialized = false;
};

enum class llama_kv_pager_status : uint8_t {
    ok = 0,
    disabled,
    invalid_geometry,
    unsupported_authority,
    missing_backend,
    host_budget,
    admission,
    allocation,
    realized_mismatch,
    overflow,
};

const char * llama_kv_pager_status_name(llama_kv_pager_status status) noexcept;

enum class llama_kv_pager_write_status : uint8_t {
    ok = 0,
    disabled,
    invalid_position,
    no_victim,
    all_pinned,
    stale_generation,
    transaction,
    overflow,
};

const char * llama_kv_pager_write_status_name(llama_kv_pager_write_status status) noexcept;

struct llama_kv_pager_write_ticket {
    int32_t sequence_id = -1;
    uint64_t sequence_generation = 0;
    uint32_t logical_page = UINT32_MAX;
    uint32_t physical_slot = UINT32_MAX;
    uint32_t physical_row = UINT32_MAX;
    uint32_t page_generation = 0;
    llama_pos position = -1;
    bool page_created = false;
    bool row_was_valid = false;
};

enum class llama_kv_pager_mutation_kind : uint8_t {
    remove = 0,
    copy,
    keep,
    shift,
    rewind,
    clear,
};

struct llama_kv_pager_mutation {
    llama_kv_pager_mutation_kind kind = llama_kv_pager_mutation_kind::remove;
    int32_t sequence_id = -1;
    int32_t destination_sequence_id = -1;
    llama_pos position_begin = 0;
    llama_pos position_end = 0;
    llama_pos shift = 0;
    uint64_t sequence_generation = 0;
    // Optional residency-table epoch captured by the caller. Zero retains the
    // legacy unchecked form used by observers; runtime mutations bind this to
    // the table snapshot they are about to edit.
    uint64_t expected_epoch = 0;
    // Stage completed write-frontier pin release with the mutation.
    bool release_sequence_pins = false;
};

bool llama_kv_pager_plan(
        const llama_kv_pager_config & config,
        const llama_kv_pager_geometry & geometry,
        llama_kv_pager_resources resources,
        llama_kv_pager_snapshot & output,
        llama_kv_pager_status & status) noexcept;

class llama_kv_pager {
public:
    static std::unique_ptr<llama_kv_pager> create(
            const llama_kv_pager_config & config,
            const llama_kv_pager_geometry & geometry,
            llama_kv_pager_resources resources,
            llama_kv_pager_backend backend,
            llama_kv_pager_status & status) noexcept;

    ~llama_kv_pager();
    llama_kv_pager(const llama_kv_pager &) = delete;
    llama_kv_pager & operator=(const llama_kv_pager &) = delete;

    const llama_kv_pager_snapshot & snapshot() const noexcept { return snapshot_; }
    llama_kv_residency_snapshot residency() const noexcept { return residency_.snapshot(); }
    llama_kv_residency_snapshot residency(int32_t sequence_id) const noexcept {
        return residency_.snapshot().for_sequence(sequence_id);
    }

    // Reconcile the current resident table with authenticated canonical host
    // pages for exact attention.  The result contains at most one live record
    // per logical page; cold records have no physical slot and require the
    // exact wave upload callback.
    std::vector<llama_kv_page_record> exact_page_records(
            int32_t sequence_id) const noexcept;

    // Reserve the physical row for one logical position before graph submission. The returned
    // row is an implementation detail; callers must continue to use the logical position for
    // RoPE and masking. A page is kept pinned while it is the current partial write page.
    llama_kv_pager_write_status begin_write(
            int32_t sequence_id, uint64_t sequence_generation, llama_pos position,
            llama_kv_pager_write_ticket & ticket) noexcept;
    llama_kv_pager_write_status complete_write(
            const llama_kv_pager_write_ticket & ticket, uint32_t completed_segments,
            bool graph_succeeded) noexcept;
    llama_kv_pager_write_status cancel_write(
            const llama_kv_pager_write_ticket & ticket) noexcept;
    bool physical_row(
            int32_t sequence_id, llama_pos position, uint32_t & row) const noexcept;

    // Apply a metadata mutation as one table publication. Payload movement is deliberately
    // deferred to the residency transfer owner; this method never publishes a half mutation.
    llama_kv_pager_write_status mutate(const llama_kv_pager_mutation & mutation) noexcept;

    // A completed full-sequence reset is a lifecycle boundary.  Release the
    // partial write-frontier pin before publishing that removal so an empty
    // sequence can be reset after a short warmup/decode batch.
    void release_sequence_pins(int32_t sequence_id) noexcept;

    // Attach the cache-owned source/snapshot provider after pager admission.
    // A page becomes evictable only after seal_ready_pages() reports a
    // successful catalog publication.
    void set_host_provider(llama_kv_pager_host_provider provider) noexcept;
    void set_routing_summary_provider(
            llama_kv_pager_routing_summary_provider provider) noexcept;
    uint32_t seal_ready_pages() noexcept;
    bool invalidate_host_page(const llama_kv_page_id & page) noexcept;
    void bind_representation_identity(
            uint64_t model_identity,
            uint64_t topology_identity,
            uint64_t codec_digest,
            uint64_t codebook_digest,
            uint64_t rotation_digest,
            uint64_t meansub_digest,
            uint64_t representation_epoch) noexcept;
    const llama_kv_pager_host * host_catalog() const noexcept {
        return host_.get();
    }
    ggml_backend_t host_backend() const noexcept {
        return host_ ? resources_host_backend_ : nullptr;
    }
    uint64_t host_source_namespace() const noexcept {
        return host_ ? resources_host_source_namespace_ : 0;
    }
    uint64_t host_topology_identity() const noexcept {
        return host_ ? resources_host_topology_identity_ : 0;
    }
    uint32_t host_child_id() const noexcept {
        return host_ ? resources_host_child_id_ : UINT32_MAX;
    }
    uint32_t host_stream_index() const noexcept {
        return host_ ? resources_host_stream_index_ : UINT32_MAX;
    }

    // Cumulative counters from observed residency transactions. They remain
    // separate from the immutable page snapshot so metrics do not alter the
    // table epoch or page identities.
    const llama_kv_residency_transfer_counters & transfer_counters() const noexcept {
        return transfer_counters_;
    }
    const llama_kv_residency_transfer_counters & h2d_counters() const noexcept {
        return h2d_counters_;
    }
    const llama_kv_residency_transfer_counters & d2h_counters() const noexcept {
        return d2h_counters_;
    }
    uint64_t promotion_pages() const noexcept { return promotion_pages_; }
    uint64_t eviction_pages() const noexcept { return eviction_pages_; }

    const llama_kv_routing_summary_store & routing_summaries() const noexcept {
        return routing_summaries_;
    }
    const llama_kv_routing_summary_accounting & routing_summary_accounting() const noexcept {
        return routing_summaries_.accounting();
    }

    // The pager table remains the logical authority; these are the real
    // backend-owned slot/event doors used by the residency transfer owner.
    llama_kv_residency_pool_backend residency_backend() const noexcept {
        return residency_backend_;
    }
    vbr_h2d_chunk_ring * upload_ring() const noexcept {
        return host_ ? host_->upload_ring() : nullptr;
    }

    // Apply one complete runtime-H target through the pager-owned pool and
    // immutable table. The boundary snapshot must be from this pager.
    llama_kv_live_policy_result apply_live_policy(
            const llama_kv_live_policy_boundary & boundary,
            const llama_kv_residency_transfer_transport & transport,
            const llama_kv_residency_transaction_hooks & hooks = {}) noexcept;

    ggml_tensor * residency_storage_tensor() const noexcept {
        return residency_adapter_ ? residency_adapter_->storage_tensor() : nullptr;
    }
    uint64_t residency_bytes_per_slot() const noexcept {
        return residency_adapter_ ? residency_adapter_->bytes_per_slot_value() : 0;
    }
    uint64_t resident_bytes() const noexcept {
        return residency_pool_ ? residency_pool_->resident_bytes() : 0;
    }

private:
    struct page_state {
        llama_kv_page_record record;
        std::vector<uint8_t> valid_rows;
        uint32_t completed_segments = 0;
        bool present = false;
    };

    llama_kv_pager_write_status publish_page(page_state & page) noexcept;
    llama_kv_pager_write_status erase_page(page_state & page) noexcept;
    void reconcile_routing_summaries() noexcept;
    page_state * find_page(int32_t sequence_id, uint32_t logical_page) noexcept;
    const page_state * find_page(int32_t sequence_id, uint32_t logical_page) const noexcept;
    page_state * find_slot(uint32_t slot) noexcept;
    void release_current_pin(page_state * except) noexcept;

    llama_kv_pager() = default;
    llama_kv_pager_snapshot snapshot_;
    llama_kv_pager_backend backend_;
    std::unique_ptr<llama_kv_pager_host> host_;
    std::unique_ptr<llama_kv_residency_ggml_adapter> residency_adapter_;
    std::unique_ptr<llama_kv_residency_pool> residency_pool_;
    llama_kv_residency_pool_backend residency_backend_;
    ggml_backend_t resources_host_backend_ = nullptr;
    uint64_t resources_host_source_namespace_ = 0;
    uint64_t resources_host_topology_identity_ = 0;
    uint32_t resources_host_child_id_ = UINT32_MAX;
    uint32_t resources_host_stream_index_ = UINT32_MAX;
    llama_kv_routing_summary_config routing_summary_config_;
    llama_kv_pager_routing_summary_provider routing_summary_provider_;
    llama_kv_routing_summary_store routing_summaries_;
    llama_kv_page_id page_identity_;
    llama_kv_pager_allocation allocation_;
    bool owns_allocation_ = true;
    std::vector<page_state> pages_;
    std::vector<int32_t> slot_pages_;
    uint32_t current_page_index_ = UINT32_MAX;
    uint64_t mutation_generation_ = 1;
    llama_kv_residency_table residency_{0};
    llama_kv_residency_transfer_counters transfer_counters_;
    llama_kv_residency_transfer_counters h2d_counters_;
    llama_kv_residency_transfer_counters d2h_counters_;
    uint64_t promotion_pages_ = 0;
    uint64_t eviction_pages_ = 0;
};
