#pragma once

#include "common.cuh"

#include <cstddef>
#include <cstdint>

// Device-side description of one selected Turbo4 page.  Keep this established
// CUDA-facing type distinct from the backend-neutral GGML graph descriptor so
// the raw CUDA API retains its existing C++ ABI. Both descriptors have the
// same field order and layout; the graph dispatch uses a reinterpretation only
// after validating the immutable host table.
struct ggml_cuda_fattn_turbo4_page {
    uint32_t logical_page;
    uint32_t source_physical_slot;
    uint32_t compact_row_begin;
    uint32_t row_count;
    int64_t  native_position_begin;
};

static_assert(sizeof(ggml_cuda_fattn_turbo4_page) ==
              sizeof(ggml_flash_attn_ext_paged_turbo4_page));
static_assert(offsetof(ggml_cuda_fattn_turbo4_page, native_position_begin) ==
              offsetof(ggml_flash_attn_ext_paged_turbo4_page, native_position_begin));

bool ggml_cuda_flash_attn_ext_paged_turbo4_supported(
        int device, const ggml_tensor * dst) noexcept;
void ggml_cuda_flash_attn_ext_paged_turbo4(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst) noexcept;

enum class ggml_cuda_fattn_turbo4_paged_status : uint8_t {
    ok = 0,
    invalid_argument,
    unsupported_type,
    unsupported_shape,
    invalid_page_table,
    cuda_error,
};

const char * ggml_cuda_fattn_turbo4_paged_status_name(
        ggml_cuda_fattn_turbo4_paged_status status) noexcept;

// Validate the host copy of the immutable page table before it is copied to
// device storage.  The launch API requires the same table in pages_device;
// retaining pages_host makes the fail-closed checks capture-safe and avoids
// trying to inspect device metadata synchronously during graph capture.
bool ggml_cuda_fattn_turbo4_page_table_valid(
        const ggml_cuda_fattn_turbo4_page * pages,
        size_t n_pages,
        uint32_t n_rows,
        uint32_t page_tokens = 256,
        uint32_t n_physical_pages = UINT32_MAX) noexcept;

struct ggml_cuda_fattn_turbo4_paged_params {
    // Q and output are [256, n_head_q] for the supported one-token, batch-one
    // geometry. Q is F32 and output is F32; q/output head strides are bytes.
    const float * q = nullptr;
    float * output = nullptr;
    size_t q_head_stride_bytes = 0;
    size_t output_head_stride_bytes = 0;

    ggml_type type_k = GGML_TYPE_COUNT;
    ggml_type type_v = GGML_TYPE_COUNT;
    uint32_t head_dim_k = 0;
    uint32_t head_dim_v = 0;

    // K/V are raw Turbo4 rows. A physical slot contains page_tokens rows for
    // one KV head; no selected or full-cache F16 buffer is formed.
    const char * k = nullptr;
    const char * v = nullptr;
    size_t k_row_stride_bytes = 0;
    size_t k_head_stride_bytes = 0;
    size_t k_page_stride_bytes = 0;
    size_t v_row_stride_bytes = 0;
    size_t v_head_stride_bytes = 0;
    size_t v_page_stride_bytes = 0;

    // pages_host is immutable metadata owned by the graph/page view. The
    // device pointers are graph inputs populated by the caller before capture.
    const ggml_cuda_fattn_turbo4_page * pages_host = nullptr;
    const ggml_cuda_fattn_turbo4_page * pages_device = nullptr;
    const int64_t * native_positions_device = nullptr;
    const uint8_t * native_mask_device = nullptr;
    const int64_t * query_positions_device = nullptr;
    float * page_mass = nullptr;
    size_t page_mass_head_stride_bytes = 0;
    // Optional unnormalized online-softmax output [m, l, o[head_dim_v]] per
    // query head. This is the bounded exact page-wave handoff; no K/V or
    // attention matrix is transferred to the host.
    float * partial_state = nullptr;
    size_t partial_state_head_stride_bytes = 0;

    uint32_t n_pages = 0;
    uint32_t n_physical_pages = 0;
    uint32_t n_rows = 0;
    uint32_t n_head_q = 0;
    uint32_t n_head_kv = 0;
    uint32_t n_query_tokens = 0;
    uint32_t n_batch = 0;
    uint32_t page_mass_logical_count = 0;
    float scale = 0.0f;
    bool reduce_page_mass = false;
    bool write_partial_state = false;
    bool causal = true;
};

// Correctness-first direct page-wave decode. The initial qualified geometry
// is causal, batch 1, one query token, head width 256, and GQA ratio 4. The
// output remains in the Turbo V rotated domain, matching the existing dense
// Turbo4 FA contract; the graph-level inverse WHT and mean restoration remain
// outside this backend primitive.
ggml_cuda_fattn_turbo4_paged_status ggml_cuda_flash_attn_ext_paged_turbo4(
        ggml_backend_cuda_context & ctx,
        const ggml_cuda_fattn_turbo4_paged_params & params) noexcept;

// Backend-facing overload for tests and the later graph integration. It
// rejects non-CUDA backends before touching their private context layout.
ggml_cuda_fattn_turbo4_paged_status ggml_cuda_flash_attn_ext_paged_turbo4(
        ggml_backend_t backend,
        const ggml_cuda_fattn_turbo4_paged_params & params) noexcept;
