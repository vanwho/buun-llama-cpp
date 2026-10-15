#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-vbr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#endif

static bool set_fused_env(const char * value) {
#if defined(_WIN32)
    return _putenv_s("GGML_TURBO_MMA_FUSED", value != nullptr ? value : "") == 0;
#else
    return value != nullptr
        ? setenv("GGML_TURBO_MMA_FUSED", value, 1) == 0
        : unsetenv("GGML_TURBO_MMA_FUSED") == 0;
#endif
}

static bool expect_false(ggml_vbr_fused_turbo4_attn_v1_fn query,
                         ggml_backend_t backend,
                         const ggml_tensor * attention,
                         const char * what) {
    if (!query(backend, attention)) {
        return true;
    }
    std::fprintf(stderr, "fused Turbo4 capability accepted invalid %s metadata\n", what);
    return false;
}

int main(int argc, char ** argv) {
    const bool require_fused = argc == 2 && std::strcmp(argv[1], "--require-fused") == 0;
    if (argc > 2 || (argc == 2 && !require_fused)) {
        std::fprintf(stderr, "usage: %s [--require-fused]\n", argv[0]);
        return 2;
    }
    if (ggml_backend_cuda_get_device_count() == 0) {
        return 77;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        return 77;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    const auto query = reinterpret_cast<ggml_vbr_fused_turbo4_attn_v1_fn>(
        ggml_backend_reg_get_proc_address(reg, GGML_VBR_FUSED_TURBO4_ATTN_V1_PROC));
    if (query == nullptr) {
        std::fprintf(stderr, "CUDA backend is missing the optional fused Turbo4 capability procedure\n");
        ggml_backend_free(backend);
        return 1;
    }

    ggml_context * ctx = ggml_init({ 1024 * 1024, nullptr, true });
    if (ctx == nullptr) {
        ggml_backend_free(backend);
        return 1;
    }
    constexpr int64_t head_dim = 256;
    constexpr int64_t queries = 1;
    constexpr int64_t q_heads = 16;
    constexpr int64_t kv_heads = 4;
    constexpr int64_t rows = 256;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
        head_dim, queries, q_heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_TURBO4_0,
        head_dim, rows, kv_heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_TURBO4_0,
        head_dim, rows, kv_heads, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rows, queries);
    ggml_tensor * attention = ggml_flash_attn_ext(ctx, q, k, v, mask,
        1.0f / 16.0f, 0.0f, 0.0f);
    if (attention == nullptr || attention->op != GGML_OP_FLASH_ATTN_EXT) {
        std::fprintf(stderr, "failed to construct FLASH_ATTN_EXT capability descriptor\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    const char * old_env = std::getenv("GGML_TURBO_MMA_FUSED");
    const bool had_old_env = old_env != nullptr;
    const std::string saved_env = had_old_env ? old_env : "";
    if (!set_fused_env(nullptr)) {
        std::fprintf(stderr, "failed to clear GGML_TURBO_MMA_FUSED for capability check\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    const bool valid_supported = query(backend, attention);
    ggml_backend_buffer_type_t cuda_buft = ggml_backend_get_default_buffer_type(backend);
    const size_t output_bytes = ggml_nbytes(attention);
    const size_t fused_alloc_bytes = ggml_backend_buft_get_alloc_size(cuda_buft, attention);
    bool ok = true;
    if (valid_supported && fused_alloc_bytes != output_bytes) {
        std::fprintf(stderr, "fused Turbo4 descriptor allocation is %zu bytes, expected output-only %zu\n",
            fused_alloc_bytes, output_bytes);
        ok = false;
    }
    if (require_fused && !valid_supported) {
        std::fprintf(stderr, "this CUDA device/build does not support fused Turbo4 attention\n");
        set_fused_env(had_old_env ? saved_env.c_str() : nullptr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_tensor * saved_mask = attention->src[3];
    ok &= expect_false(query, nullptr, attention, "null backend");
    attention->src[3] = nullptr;
    ok &= expect_false(query, backend, attention, "null mask");
    attention->src[3] = saved_mask;

    ggml_tensor * short_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rows - 1, queries);
    attention->src[3] = short_mask;
    ok &= expect_false(query, backend, attention, "undersized mask");
    attention->src[3] = saved_mask;

    const ggml_type saved_k_type = k->type;
    k->type = GGML_TYPE_F16;
    ok &= expect_false(query, backend, attention, "unsupported K dtype");
    k->type = saved_k_type;

    const size_t saved_q_nb1 = q->nb[1];
    q->nb[1] += sizeof(float);
    ok &= expect_false(query, backend, attention, "invalid query stride");
    q->nb[1] = saved_q_nb1;

    if (!set_fused_env("0")) {
        std::fprintf(stderr, "failed to set GGML_TURBO_MMA_FUSED for capability check\n");
        ok = false;
    } else if (query(backend, attention)) {
        std::fprintf(stderr, "fused Turbo4 capability ignored GGML_TURBO_MMA_FUSED=0\n");
        ok = false;
    }
    const size_t fallback_alloc_bytes = ggml_backend_buft_get_alloc_size(cuda_buft, attention);
    // Turbo KV's generic VEC path may keep graph allocation output-only; its
    // fallback dequant buffers are persistent backend scratch, reserved apart
    // from this tensor's allocation size.
    if (fallback_alloc_bytes < output_bytes) {
        std::fprintf(stderr, "fallback allocation %zu is smaller than output %zu\n",
            fallback_alloc_bytes, output_bytes);
        ok = false;
    }
    if (!set_fused_env(had_old_env ? saved_env.c_str() : nullptr)) {
        std::fprintf(stderr, "failed to restore GGML_TURBO_MMA_FUSED after capability check\n");
        ok = false;
    }
    ggml_free(ctx);
    ggml_backend_free(backend);
    if (!ok) {
        return 1;
    }
    std::printf("mtp_attention_scratch_capability=pass valid_fused=%d invalid_metadata=false env_disable=false\n",
        valid_supported ? 1 : 0);
    return 0;
}
