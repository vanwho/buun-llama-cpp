#include "turbo-wht.cuh"

// Sign arrays for FWHT rotation (from turbo-wht.h, seed=42)
static __constant__ float d_turbo_wht_s1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1};
static __constant__ float d_turbo_wht_s2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1};

// Keep the router scale in this compilation unit. CUDA device symbols with
// the same name are otherwise duplicated by nvcc and can retain zero-filled
// copies in unrelated translation units.
static __device__ float d_innerq_channel_scale_inv_router[128];
static __device__ int d_innerq_router_scale_ready;

void turbo_innerq_update_turbo_wht_scales(const float * scale_inv) {
    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);
    int ready = 1;
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_innerq_channel_scale_inv_router, scale_inv, 128 * sizeof(float));
        cudaMemcpyToSymbol(d_innerq_router_scale_ready, &ready, sizeof(ready));
    }
    cudaSetDevice(cur_device);
}

// One block per 128-element group. 128 threads per block.
static __global__ void k_turbo_wht(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t n_elements, const int direction) {

    const int64_t group = blockIdx.x;
    const int64_t offset = group * 128;
    if (offset >= n_elements) return;

    const bool router_transpose = direction == 2;
    const float * s_first  = (direction == 0 || router_transpose) ? d_turbo_wht_s1 : d_turbo_wht_s2;
    const float * s_second = (direction == 0 || router_transpose) ? d_turbo_wht_s2 : d_turbo_wht_s1;

    __shared__ float buf[128];

    // Load and apply first signs
    if (threadIdx.x < 128) {
        float value = src[offset + threadIdx.x];
        if (router_transpose && d_innerq_router_scale_ready) {
            value *= d_innerq_channel_scale_inv_router[threadIdx.x];
        }
        buf[threadIdx.x] = value * s_first[threadIdx.x];
    }
    __syncthreads();

    // Parallel FWHT butterfly: 64 threads, 7 passes
    for (int h = 1; h < 128; h *= 2) {
        if (threadIdx.x < 64) {
            int j = (threadIdx.x / h) * (2 * h) + (threadIdx.x % h);
            float a = buf[j], b = buf[j + h];
            buf[j] = a + b; buf[j + h] = a - b;
        }
        __syncthreads();
    }

    // Normalize and apply second signs, write output
    constexpr float inv_sqrt_128 = 0.08838834764831845f; // 1/sqrt(128)
    if (threadIdx.x < 128) {
        dst[offset + threadIdx.x] = buf[threadIdx.x] * inv_sqrt_128 * s_second[threadIdx.x];
    }
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const float * src_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));

    const int64_t n_elements = ggml_nelements(src0);
    const int64_t n_groups = n_elements / 128;

    k_turbo_wht<<<(int)n_groups, 128, 0, stream>>>(src_d, dst_d, n_elements, direction);
}
