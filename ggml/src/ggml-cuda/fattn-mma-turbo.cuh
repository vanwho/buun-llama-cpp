#pragma once

#include "turbo-tcq-alpha.cuh"
#include "fattn-paged-turbo4.cuh"

// fattn-mma-f16.cuh must be included before this header (provides flash_attn_ext_f16,
// launch_fattn, fattn_kernel_t, and the MMA helper functions).

// Fused MMA-native turbo flash attention: reads raw turbo-quantized K/V directly,
// dequants into half2 shmem tiles inside the attention loop. No intermediate fp16 buffers.
// Uses the same flash_attn_ext_f16 kernel with type_K/type_V template params.

struct ggml_cuda_fattn_mma_paged_turbo4_policy {
    static constexpr bool is_paged = true;

    const float * q;
    size_t q_head_stride;
    size_t q_query_stride;
    float * output;
    size_t output_head_stride;
    size_t output_query_stride;
    const char * k;
    const char * v;
    size_t k_row_stride;
    size_t k_head_stride;
    size_t k_page_stride;
    size_t v_row_stride;
    size_t v_head_stride;
    size_t v_page_stride;
    const ggml_cuda_fattn_turbo4_page * pages;
    uint32_t n_pages;
    uint32_t n_rows;
    const uint32_t * active_page_count;
    const uint32_t * active_row_count;
    uint32_t kv_head;
    uint32_t n_head_kv;
    uint32_t n_query_tokens;
    const int64_t * native_positions;
    const uint8_t * native_mask;
    const int64_t * query_positions;
    bool explicit_native_metadata;
    bool causal;
    const uint32_t * routing_selected_indices;
    size_t routing_selected_stride;
    const uint32_t * routing_selected_count;
    size_t routing_count_stride;
    uint32_t routing_top_k;

    __device__ __forceinline__ float2 q_value(const int query, const int head, const int k0) const {
        const char * base = (const char *) q + size_t(query) * q_query_stride + size_t(head) * q_head_stride;
        return ((const float2 *) base)[k0];
    }

    __device__ __forceinline__ const char * row_ptr(
            const char * data, const uint32_t page_index, const uint32_t row,
            const size_t row_stride, const size_t head_stride, const size_t page_stride) const {
        const ggml_cuda_fattn_turbo4_page page = pages[page_index];
        return data + size_t(page.source_physical_slot) * page_stride +
            size_t(kv_head) * head_stride + size_t(row) * row_stride;
    }

    __device__ __forceinline__ bool find_row(
            const uint32_t compact_row, uint32_t & page_index, uint32_t & row) const {
        const uint32_t page_count = active_page_count != nullptr
            ? min(*active_page_count, n_pages) : n_pages;
        for (uint32_t i = 0; i < page_count; ++i) {
            const ggml_cuda_fattn_turbo4_page page = pages[i];
            if (compact_row >= page.compact_row_begin &&
                    compact_row - page.compact_row_begin < page.row_count) {
                page_index = i;
                row = compact_row - page.compact_row_begin;
                return true;
            }
        }
        return false;
    }

    __device__ __forceinline__ bool route_selected(const uint32_t query, const uint32_t page_index) const {
        if (routing_selected_indices == nullptr || routing_selected_count == nullptr) {
            return true;
        }
        const size_t route_offset = (size_t(query) * n_head_kv + kv_head) * routing_count_stride;
        const uint32_t count = *reinterpret_cast<const uint32_t *>(
            (const char *) routing_selected_count + route_offset);
        const uint32_t * selected = (const uint32_t *) ((const char *) routing_selected_indices +
            (size_t(query) * n_head_kv + kv_head) * routing_selected_stride);
        for (uint32_t i = 0; i < min(count, routing_top_k); ++i) {
            if (selected[i] == page_index) {
                return true;
            }
        }
        return false;
    }

    __device__ __forceinline__ bool route_selected_for_tile(
            const uint32_t query_begin, const uint32_t query_count, const uint32_t page_index) const {
        if (routing_selected_indices == nullptr || routing_selected_count == nullptr) {
            return true;
        }
        for (uint32_t i = 0; i < query_count; ++i) {
            const uint32_t query = query_begin + i;
            if (query < n_query_tokens && route_selected(query, page_index)) {
                return true;
            }
        }
        return false;
    }

    __device__ __forceinline__ bool valid(const uint32_t query, const uint32_t compact_row,
            uint32_t & page_index, uint32_t & row) const {
        const uint32_t row_count = active_row_count != nullptr
            ? min(*active_row_count, n_rows) : n_rows;
        if (query >= n_query_tokens || compact_row >= row_count ||
                !find_row(compact_row, page_index, row)) {
            return false;
        }
        const ggml_cuda_fattn_turbo4_page page = pages[page_index];
        int64_t native_position = page.native_position_begin + int64_t(row);
        if (explicit_native_metadata && native_positions != nullptr) {
            native_position = native_positions[compact_row];
        }
        if (native_mask != nullptr && native_mask[compact_row] == 0) {
            return false;
        }
        return route_selected(query, page_index) &&
            (!causal || native_position <= query_positions[query]);
    }

    template<int ncols1, int nwarps, int nbatch_fa, bool oob_check>
    __device__ __forceinline__ void load_mask(
            half * tile_mask, const int jt, const int k0, const int i_sup) const {
        const int tid = threadIdx.y * ggml_cuda_get_physical_warp_size() + threadIdx.x;
        const int nthreads = nwarps * ggml_cuda_get_physical_warp_size();
        for (int index = tid; index < ncols1 * (nbatch_fa + 8); index += nthreads) {
            const int query = index / (nbatch_fa + 8);
            const int row = index % (nbatch_fa + 8);
            if (row >= nbatch_fa) {
                continue;
            }
            uint32_t page_index = 0;
            uint32_t page_row = 0;
            const bool row_valid = !oob_check || row < i_sup ? valid(
                uint32_t(jt * ncols1 + query), uint32_t(k0 + row), page_index, page_row) : false;
            tile_mask[query * (nbatch_fa + 8) + row] = row_valid
                ? __float2half(0.0f) : __float2half(-INFINITY);
        }
    }

    template<int D, int stride_tile, int nbatch_fa, int nthreads, bool oob_check>
    __device__ __forceinline__ void load_tile(
            half2 * tile, const char * data, const size_t row_stride,
            const size_t head_stride, const size_t page_stride, const int k0, const int i_sup,
            const int query_begin, const int query_count) const {
        const int tid = threadIdx.y * ggml_cuda_get_physical_warp_size() + threadIdx.x;
        for (int row = tid; row < nbatch_fa; row += nthreads) {
            if (oob_check && row >= i_sup) {
#pragma unroll
                for (int b = 0; b < D / 2; ++b) {
                    tile[row * stride_tile + b] = make_half2(0.0f, 0.0f);
                }
                continue;
            }
            uint32_t page_index = 0;
            uint32_t page_row = 0;
            if (!find_row(uint32_t(k0 + row), page_index, page_row)) {
#pragma unroll
                for (int b = 0; b < D / 2; ++b) {
                    tile[row * stride_tile + b] = make_half2(0.0f, 0.0f);
                }
                continue;
            }
            if (!route_selected_for_tile(uint32_t(query_begin), uint32_t(query_count), page_index)) {
#pragma unroll
                for (int b = 0; b < D / 2; ++b) {
                    tile[row * stride_tile + b] = make_half2(0.0f, 0.0f);
                }
                continue;
            }
            flash_attn_ext_turbo4_decode_row<D, stride_tile>(
                row_ptr(data, page_index, page_row, row_stride, head_stride, page_stride), tile, row);
        }
    }

    template<int D, int stride_tile, int nbatch_fa, int nthreads, bool oob_check>
    __device__ __forceinline__ void load_k(
            half2 * tile, const int k0, const int i_sup, const int query_begin, const int query_count) const {
        load_tile<D, stride_tile, nbatch_fa, nthreads, oob_check>(
            tile, k, k_row_stride, k_head_stride, k_page_stride, k0, i_sup, query_begin, query_count);
    }

    template<int D, int stride_tile, int nbatch_fa, int nthreads, bool oob_check>
    __device__ __forceinline__ void load_v(
            half2 * tile, const int k0, const int i_sup, const int query_begin, const int query_count) const {
        load_tile<D, stride_tile, nbatch_fa, nthreads, oob_check>(
            tile, v, v_row_stride, v_head_stride, v_page_stride, k0, i_sup, query_begin, query_count);
    }

    __device__ __forceinline__ void store_output(
            float2 *, const int query, const int head, const int k0, const float2 value) const {
        float2 * dst = (float2 *) ((char *) output + size_t(query) * output_query_stride +
            size_t(head) * output_head_stride);
        dst[k0] = value;
    }
};

template<int ncols1, int ncols2>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(256, 256, ncols1*ncols2),
                  ggml_cuda_fattn_mma_get_occupancy(256, 256, ncols1*ncols2))
static __global__ void ggml_cuda_fattn_mma_turbo4_paged_kernel(
        ggml_cuda_fattn_mma_paged_turbo4_policy policy,
        const uint32_t n_query_tokens,
        const uint32_t n_head_q,
        const uint32_t n_head_kv,
        const uint32_t n_rows,
        const float scale) {
    constexpr int ncols = ncols1 * ncols2;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nthreads = ggml_cuda_fattn_mma_get_nthreads(256, 256, ncols);
    constexpr int nwarps = nthreads / warp_size;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa_typed(
        256, 256, ncols, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);

    const uint32_t query_tile = uint32_t(blockIdx.x);
    const uint32_t kv_head = uint32_t(blockIdx.y);
    const uint32_t head_tile = uint32_t(blockIdx.z);
    if (query_tile * ncols1 >= n_query_tokens || kv_head >= n_head_kv ||
            head_tile * ncols2 >= n_head_q / n_head_kv) {
        return;
    }

    policy.kv_head = kv_head;
    const uint3 ne01 = make_uint3(n_query_tokens, n_query_tokens, n_query_tokens);
    const int gqa_ratio = int(n_head_q / n_head_kv);
    const int kb0_stop = (int(n_rows) + nbatch_fa - 1) / nbatch_fa;

    flash_attn_ext_f16_process_tile
        <256, 256, ncols1, ncols2, nwarps, false, false, false, false, false,
         GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0, ggml_cuda_fattn_mma_paged_turbo4_policy>
        (nullptr, nullptr, nullptr, nullptr, nullptr, (float2 *) policy.output, nullptr,
         scale, 1.0f, 0.0f, ne01, int(n_head_q), gqa_ratio, int(n_rows),
         1, 1, 1, 1, 1,
         int(query_tile), int(head_tile), 0, kb0_stop, policy);
}

template<int ncols1, int ncols2>
static bool ggml_cuda_flash_attn_ext_mma_turbo4_paged_case(
        ggml_backend_cuda_context & ctx,
        const ggml_cuda_fattn_turbo4_paged_params & params) {
    const int device = ctx.device;
    const int cc = ggml_cuda_info().devices[device].cc;
    constexpr int ncols = ncols1 * ncols2;
    const int nthreads = ggml_cuda_fattn_mma_get_nthreads(256, 256, ncols, cc);
    const int nwarps = nthreads / ggml_cuda_info().devices[device].warp_size;
    const int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa_typed(
        256, 256, ncols, cc, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
    const int nbatch_K2 = ggml_cuda_fattn_mma_get_nbatch_K2(256, 256, ncols, cc);
    const int nbatch_V2 = ggml_cuda_fattn_mma_get_nbatch_V2(256, 256, ncols, cc);
    const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(256, 256, ncols, cc);
    const bool q_in_reg = ggml_cuda_fattn_mma_get_Q_in_reg(256, 256, ncols, cc);
    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const size_t shared_kv = size_t(nbatch_fa) * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
    const size_t shared_q = size_t(ncols) * (256 / 2 + 4) * sizeof(half2);
    const size_t shared_mask = size_t(ncols1) * (nbatch_fa / 2 + 4) * sizeof(half2);
    const size_t shared_combine = size_t(nwarps) * cols_per_warp * (nbatch_combine + 4) * sizeof(half2);
    const size_t shared_bytes = std::max(shared_combine, q_in_reg
        ? std::max(shared_q, shared_kv + shared_mask)
        : shared_q + shared_kv + shared_mask);
    const uint32_t page_capacity = params.page_capacity != 0 ? params.page_capacity : params.n_pages;
    const uint32_t row_capacity = params.row_capacity != 0 ? params.row_capacity : params.n_rows;

    static size_t shared_limit[GGML_CUDA_MAX_DEVICES] = {};
    if (shared_bytes > shared_limit[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(
            ggml_cuda_fattn_mma_turbo4_paged_kernel<ncols1, ncols2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared_bytes)));
        shared_limit[device] = shared_bytes;
    }

    ggml_cuda_fattn_mma_paged_turbo4_policy policy = {};
    policy.q = params.q;
    policy.q_head_stride = params.q_head_stride_bytes;
    policy.q_query_stride = params.q_query_stride_bytes;
    policy.output = params.output;
    policy.output_head_stride = params.output_head_stride_bytes;
    policy.output_query_stride = params.output_query_stride_bytes;
    policy.k = params.k;
    policy.v = params.v;
    policy.k_row_stride = params.k_row_stride_bytes;
    policy.k_head_stride = params.k_head_stride_bytes;
    policy.k_page_stride = params.k_page_stride_bytes;
    policy.v_row_stride = params.v_row_stride_bytes;
    policy.v_head_stride = params.v_head_stride_bytes;
    policy.v_page_stride = params.v_page_stride_bytes;
    policy.pages = params.pages_device;
    policy.n_pages = page_capacity;
    policy.n_rows = row_capacity;
    policy.active_page_count = params.active_page_count_device;
    policy.active_row_count = params.active_row_count_device;
    policy.n_head_kv = params.n_head_kv;
    policy.n_query_tokens = params.n_query_tokens;
    policy.native_positions = params.native_positions_device;
    policy.native_mask = params.native_mask_device;
    policy.query_positions = params.query_positions_device;
    policy.explicit_native_metadata = params.explicit_native_metadata;
    policy.causal = params.causal;
    policy.routing_selected_indices = params.routing_selected_indices;
    policy.routing_selected_stride = params.routing_selected_stride_bytes;
    policy.routing_selected_count = params.routing_selected_count;
    policy.routing_count_stride = params.routing_count_stride_bytes;
    policy.routing_top_k = params.routing_top_k;

    const dim3 grid(
        (params.n_query_tokens + ncols1 - 1) / ncols1,
        params.n_head_kv,
        (params.n_head_q / params.n_head_kv + ncols2 - 1) / ncols2);
    ggml_cuda_fattn_mma_turbo4_paged_kernel<ncols1, ncols2><<<
        grid, dim3(ggml_cuda_info().devices[device].warp_size, nwarps, 1), shared_bytes, ctx.stream()>>>(
        policy, params.n_query_tokens, params.n_head_q, params.n_head_kv,
        row_capacity, params.scale);
    return cudaGetLastError() == cudaSuccess;
}

template <int DKQ, int DV, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_mma_turbo_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa_typed(DKQ, DV, ncols, cc, type_K, type_V);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);

    // Turbo forces synchronous tile loads because cp.async cannot do ALU dequant.
    // With nstages=0, tile_K and tile_V share the same shmem region (overlap).
    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    constexpr bool V_is_K_view = false; // Turbo K/V are separate tensors.

    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(nbatch_K2 + 4,  nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nbytes_shared_KV_1stage; // one-stage layout

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, false, type_K, type_V>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
#if defined(GGML_USE_HIP)
            // RDNA/ROCm: a D=256 fused kernel needs >32KB dynamic LDS. It launches fine eagerly,
            // but HIP graph CAPTURE rejects the launch unless the max-dynamic-shared attribute is
            // raised first. The static guard runs this once, on the first (eager, pre-capture)
            // launch. Non-fatal: clear any error if a given ROCm build rejects the attribute.
            (void) cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total);
            (void) cudaGetLastError();
#else
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
#endif
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, false, type_K, type_V>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
#if defined(GGML_USE_HIP)
            // RDNA/ROCm: a D=256 fused kernel needs >32KB dynamic LDS. It launches fine eagerly,
            // but HIP graph CAPTURE rejects the launch unless the max-dynamic-shared attribute is
            // raised first. The static guard runs this once, on the first (eager, pre-capture)
            // launch. Non-fatal: clear any error if a given ROCm build rejects the attribute.
            (void) cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total);
            (void) cudaGetLastError();
#else
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
#endif
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    }

    // Set TCQ constants in THIS compilation unit's __constant__ memory before kernel launch.
    // Each template instance .cu file has its own static __constant__ copies.
    turbo_vanilla_cb_load_fattn();  // TURBO_CB_T2/3/4/8 override, self-guarded per device
    if constexpr (type_K == GGML_TYPE_TURBO3_TCQ || type_V == GGML_TYPE_TURBO3_TCQ) {
        static bool cb3_loaded = false;
        if (!cb3_loaded) {
            cb3_loaded = true;
            turbo_tcq_load_kv_decode();
        }
    }
    if constexpr (type_K == GGML_TYPE_TURBO2_TCQ || type_V == GGML_TYPE_TURBO2_TCQ) {
        static bool cb2_loaded = false;
        if (!cb2_loaded) {
            cb2_loaded = true;
            turbo2_tcq_load_kv_decode();
        }
    }
    if constexpr (type_K == GGML_TYPE_TURBO1_TCQ || type_V == GGML_TYPE_TURBO1_TCQ) {
        static bool cb1_loaded = false;
        static int cb1_hot = -1; if (cb1_hot < 0) cb1_hot = getenv("TURBO1_TCQ_HOTSWAP") ? 1 : 0;
        if (!cb1_loaded || cb1_hot) {
            cb1_loaded = true;
            turbo1_tcq_load_kv_encode();
        }
    }
    if constexpr (type_K == GGML_TYPE_TURBO3_TCQ || type_K == GGML_TYPE_TURBO2_TCQ || type_K == GGML_TYPE_TURBO1_TCQ ||
                  type_V == GGML_TYPE_TURBO3_TCQ || type_V == GGML_TYPE_TURBO2_TCQ || type_V == GGML_TYPE_TURBO1_TCQ) {
        const ggml_tensor * V = dst->src[2];
        const int64_t n_kv = V->ne[1] > 0 ? V->ne[1] : 1;
        const float ln_ctx = logf((float)n_kv);

        // V alpha: context-adaptive unless env var override
        float alpha_v = 1.0f;
        static float alpha_v_static = -1.0f;
        if (alpha_v_static < 0.0f) {
            alpha_v_static = 0.0f;
            const char * s = getenv("TURBO_TCQ_DECODE_ALPHA_V");
            if (s) {
                char * end;
                float a = strtof(s, &end);
                if (end != s && a > 0.0f && a < 10.0f) alpha_v_static = a;
            }
        }
        if (alpha_v_static > 0.0f) {
            alpha_v = alpha_v_static;
        } else if constexpr (type_V == GGML_TYPE_TURBO3_TCQ) {
            alpha_v = TURBO_TCQ_ALPHA_V_T3;
        } else if constexpr (type_V == GGML_TYPE_TURBO2_TCQ) {
            alpha_v = TURBO_TCQ_ALPHA_V_T2;
        } else if constexpr (type_V == GGML_TYPE_TURBO1_TCQ) {
            alpha_v = TURBO_TCQ_ALPHA_V_T1;
        }
        // Push alpha to this TU's __constant__ ONCE per device, skipping the memcpy on repeat
        // launches. alpha_v is constant for this template instance, so re-pushing it every call is
        // redundant — and a synchronous cudaMemcpyToSymbol during CUDA/HIP graph capture is illegal
        // (ROCm: "operation would make the legacy stream depend on a capturing blocking stream").
        // The first (eager, pre-capture) launch sets it; captured launches skip. Matches the
        // static cbN_loaded codebook guards above.
        static float alpha_v_pushed[GGML_CUDA_MAX_DEVICES];
        static bool  alpha_v_have  [GGML_CUDA_MAX_DEVICES] = {};
        if (!alpha_v_have[id] || alpha_v_pushed[id] != alpha_v) {
            CUDA_CHECK(cudaMemcpyToSymbol(d_tcq_decode_alpha_v_fattn, &alpha_v, sizeof(float)));
            alpha_v_pushed[id] = alpha_v;
            alpha_v_have[id]   = true;
        }

        // K alpha: static (default 1.0, env var override)
        static float alpha_k = -1.0f;
        if (alpha_k < 0.0f) {
            alpha_k = 1.0f;
            const char * s = getenv("TURBO_TCQ_DECODE_ALPHA_K");
            if (s) {
                char * end;
                float a = strtof(s, &end);
                if (end != s && a > 0.0f && a < 10.0f) alpha_k = a;
            }
        }
        static float alpha_k_pushed[GGML_CUDA_MAX_DEVICES];
        static bool  alpha_k_have  [GGML_CUDA_MAX_DEVICES] = {};
        if (!alpha_k_have[id] || alpha_k_pushed[id] != alpha_k) {
            CUDA_CHECK(cudaMemcpyToSymbol(d_tcq_decode_alpha_k_fattn, &alpha_k, sizeof(float)));
            alpha_k_pushed[id] = alpha_k;
            alpha_k_have[id]   = true;
        }
    }

    // need_f16_K=false, need_f16_V=false: raw turbo data passes through to kernel.
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, false, true, warp_size_host);
}


#define DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, ncols1, ncols2, tK, tV)                                  \
    template void ggml_cuda_flash_attn_ext_mma_turbo_case                                            \
    <DKQ, DV, ncols1, ncols2, tK, tV>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)            \

// Matched K/V at D=128 and D=256. ncols2 ≤ 8.
#define DECL_FATTN_MMA_TURBO_CASES_ALL_NCOLS2(DKQ, DV, ncols, tK, tV)           \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, (ncols)/1, 1, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, (ncols)/2, 2, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, (ncols)/4, 4, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, (ncols)/8, 8, tK, tV); \

#define DECL_FATTN_MMA_TURBO_ALL(DKQ, DV, tK, tV) \
    DECL_FATTN_MMA_TURBO_CASES_ALL_NCOLS2(DKQ, DV,  8, tK, tV) \
    DECL_FATTN_MMA_TURBO_CASES_ALL_NCOLS2(DKQ, DV, 16, tK, tV) \
    DECL_FATTN_MMA_TURBO_CASES_ALL_NCOLS2(DKQ, DV, 32, tK, tV) \
    DECL_FATTN_MMA_TURBO_CASES_ALL_NCOLS2(DKQ, DV, 64, tK, tV) \

DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO4_0)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO4_0)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO8_0,   GGML_TYPE_TURBO8_0)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO8_0,   GGML_TYPE_TURBO8_0)
// Asymmetric "q6 sweet spot" (6.124 bpw): turbo8 K + turbo4 V, D=256 only.
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO8_0,   GGML_TYPE_TURBO4_0)
// Asymmetric pairs for the dynamic VBR degrade ladder, D=256 only. The priced degrade orders
// move K and V of a layer independently and are NOT banded — a live mixed layer can straddle up
// to 4 rungs (q27 holds K=t8:V=t3 across a multi-step cursor range; g31 holds K=t8:V=t1 across
// long ranges). Only the ADJACENT-tier subset gets fused instances (compile-time budget); wider
// straddles fall to the materialize path (measured -13-15% tg32 @ d8192 when uniform).
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO8_0)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO3_TCQ)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO4_0)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO2_TCQ)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO3_TCQ)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO1_TCQ)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO1_TCQ, GGML_TYPE_TURBO2_TCQ)
// f16<->t8 entry band: VBR enters at f16 and the FIRST band every layer crosses is f16->t8, which
// the adjacent-tier set above skipped (it started at t8). A straddled layer there (K or V still f16)
// otherwise falls to the materialize path — the penalty GROWS with context (-50% @ d64k measured),
// exactly the regime the dynamic controller churns through as context grows. f16 side needs no
// dequant and (as K) no WHT rotation, so the fused kernel is strictly cheaper than matched t8:t8.
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO8_0, GGML_TYPE_F16)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_F16,      GGML_TYPE_TURBO8_0)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO1_TCQ, GGML_TYPE_TURBO1_TCQ)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO1_TCQ, GGML_TYPE_TURBO1_TCQ)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO3_0,   GGML_TYPE_TURBO3_0)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO3_0,   GGML_TYPE_TURBO3_0)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO2_0,   GGML_TYPE_TURBO2_0)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO2_0,   GGML_TYPE_TURBO2_0)
