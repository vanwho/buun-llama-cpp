#include "common.cuh"

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Upload the active InnerQ inverse scale used by the router transpose.
void turbo_innerq_update_turbo_wht_scales(const float * scale_inv);
