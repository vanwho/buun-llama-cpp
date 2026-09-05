#pragma once

#include "llama-kv-attention-view.h"

#include "ggml-backend.h"

#include <cstdint>
#include <vector>

// The initial selected-page operator is deliberately causal and Turbo4-only.
// A future backend may consume the same metadata directly; it must not infer
// a logical row from compact slot order.
enum class llama_kv_attention_operator_mode : uint8_t {
    off = 0,
    selective,
};

enum class llama_kv_attention_operator_status : uint8_t {
    ok = 0,
    disabled,
    invalid_argument,
    empty_table,
    invalid_page_table,
    invalid_shape,
    invalid_type,
    non_causal,
    overflow,
};

// The cache tensor type does not describe the value domain of TurboQuant
// rows.  Turbo4 cache bytes are stored after the forward FWHT, while the
// ordinary attention reference consumes original-domain K and the final V
// result is unrotated exactly once.  Keep this provenance in the operator
// metadata instead of asking a backend to infer it from ggml_type alone.
enum class llama_kv_attention_representation_domain : uint8_t {
    original = 0,
    turbo_rotated,
};

const char * llama_kv_attention_operator_status_name(
        llama_kv_attention_operator_status status) noexcept;

struct llama_kv_attention_operator_params {
    llama_kv_attention_operator_mode mode = llama_kv_attention_operator_mode::off;
    ggml_type type_k = GGML_TYPE_COUNT;
    ggml_type type_v = GGML_TYPE_COUNT;
    llama_kv_attention_representation_domain domain_k =
        llama_kv_attention_representation_domain::original;
    llama_kv_attention_representation_domain domain_v =
        llama_kv_attention_representation_domain::original;
    uint32_t page_tokens = VBR_GENERATION_PAGE_CELLS;
    uint32_t head_dim_k = 0;
    uint32_t head_dim_v = 0;
    uint32_t n_head_q = 0;
    uint32_t n_head_kv = 0;
    uint32_t n_query_tokens = 0;
    uint32_t n_batch = 0;
    bool causal = true;
    // Flattened [batch, query-token] native positions. These are retained in
    // the metadata so a backend has no reason to treat compact rows as pos.
    std::vector<llama_pos> query_positions;
};

class llama_kv_attention_operator_metadata {
public:
    llama_kv_attention_operator_metadata() = default;

    static llama_kv_attention_operator_metadata build(
            const llama_kv_attention_view & view,
            const llama_kv_attention_operator_params & params,
            llama_kv_attention_operator_status & status) noexcept;

    bool valid() const noexcept { return state_ != nullptr; }
    bool enabled() const noexcept;
    llama_kv_attention_operator_mode mode() const noexcept;
    uint64_t table_epoch() const noexcept;
    // This is the graph reuse/capture key for selected-page table contents.
    uint64_t graph_reuse_key() const noexcept { return table_epoch(); }
    // Unlike graph_reuse_key(), this also includes the ordered selected-page
    // contents. Two views may share one residency snapshot epoch while having
    // different compact page tables.
    uint64_t graph_content_key() const noexcept;
    uint32_t get_n_kv() const noexcept;

    ggml_type type_k() const noexcept;
    ggml_type type_v() const noexcept;
    llama_kv_attention_representation_domain domain_k() const noexcept;
    llama_kv_attention_representation_domain domain_v() const noexcept;
    uint32_t head_dim_k() const noexcept;
    uint32_t head_dim_v() const noexcept;
    uint32_t n_head_q() const noexcept;
    uint32_t n_head_kv() const noexcept;
    uint32_t n_query_tokens() const noexcept;
    uint32_t n_batch() const noexcept;
    bool causal() const noexcept;

    const std::vector<llama_kv_attention_view_page> & page_table() const noexcept;
    const std::vector<llama_pos> & native_positions() const noexcept;
    const std::vector<uint8_t> & native_mask() const noexcept;
    const std::vector<llama_pos> & query_positions() const noexcept;

    // Retain the immutable selected view for one submitted graph.  This is
    // intentionally the same lightweight fence used by the compact view.
    llama_kv_attention_view::graph_fence acquire_graph_fence() const noexcept;

private:
    struct state;
    explicit llama_kv_attention_operator_metadata(std::shared_ptr<const state> state) noexcept;
    std::shared_ptr<const state> state_;
};

enum class llama_kv_attention_backend_status : uint8_t {
    supported_reference = 0,
    disabled,
    invalid_metadata,
    unsupported_backend,
    unsupported_kv_type,
};

const char * llama_kv_attention_backend_status_name(
        llama_kv_attention_backend_status status) noexcept;

// Capability detection is by device type, not by reinterpretation of an
// unknown backend's tensors. CPU is the deterministic reference capability;
// CUDA and other devices remain explicitly unsupported until their operator
// loaders are implemented.
llama_kv_attention_backend_status llama_kv_attention_operator_check_backend(
        enum ggml_backend_dev_type device_type,
        const llama_kv_attention_operator_metadata & metadata) noexcept;

llama_kv_attention_backend_status llama_kv_attention_operator_check_backend(
        ggml_backend_t backend,
        const llama_kv_attention_operator_metadata & metadata) noexcept;
