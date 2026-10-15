#pragma once

/* Shared CPU-side transform for the canonical Turbo4 CUDA byte domain. */
static const float ggml_turbo_wht_s1_shared[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static const float ggml_turbo_wht_s2_shared[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

static inline void ggml_turbo_wht_transform_128(float * x, int inverse) {
    const float * first = inverse ? ggml_turbo_wht_s2_shared : ggml_turbo_wht_s1_shared;
    const float * second = inverse ? ggml_turbo_wht_s1_shared : ggml_turbo_wht_s2_shared;
    for (int i = 0; i < 128; ++i) {
        x[i] *= first[i];
    }
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j];
                const float b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < 128; ++i) {
        x[i] *= 0.08838834764831845f * second[i];
    }
}
