#include "common.cuh"
#include "fattn-paged-turbo4.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

// Same predicate as the encoded Turbo4 MMA dispatch, including build/device/env.
bool ggml_backend_cuda_fused_turbo4_attn_v1(ggml_backend_t backend, const ggml_tensor * dst);

// Test/diagnostic observation for the matched contiguous Turbo4 fused-MMA path.
bool ggml_cuda_fattn_turbo4_fused_last_dispatch_was_mma() noexcept;

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// Release one backend context's persistent Q/K/V attention scratch. The backend destructor drains
// its streams before calling this, so VMM mappings and cudaMalloc buffers are no longer in flight.
void ggml_cuda_fattn_scratch_free(ggml_backend_cuda_context & ctx);
