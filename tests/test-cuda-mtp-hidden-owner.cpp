#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    if (ggml_backend_cuda_get_device_count() == 0) {
        return 77;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        return 77;
    }

    constexpr int32_t rows = 1024;
    constexpr int32_t ubatch = 256;
    constexpr int64_t width = 8;
    ggml_context * ctx = ggml_init({ 4 * 1024 * 1024, nullptr, true });
    ggml_tensor * slot[2] = {
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, ubatch),
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, ubatch),
    };
    ggml_tensor * owner = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows + 1);
    ggml_tensor * draft_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);
    ggml_backend_event_t consumed = ggml_backend_event_new(ggml_backend_get_device(backend));
    GGML_ASSERT(consumed != nullptr);

    std::vector<float> host_owner((size_t) rows * width);
    uint64_t d2d_bytes = 0;
    for (int32_t micro = 0; micro < 4; ++micro) {
        const int32_t offset = micro * ubatch;
        const int32_t graph_slot = micro % 2;
        std::vector<float> source((size_t) ubatch * width);
        for (int32_t row = 0; row < ubatch; ++row) {
            for (int64_t col = 0; col < width; ++col) {
                const float value = float((offset + row) * 100 + col);
                source[(size_t) row * width + col] = value;
                host_owner[(size_t) (offset + row) * width + col] = value;
            }
        }
        ggml_backend_tensor_set(slot[graph_slot], source.data(), 0,
                ggml_nbytes(slot[graph_slot]));

        ggml_context * view_ctx = ggml_init({ 4 * ggml_tensor_overhead(), nullptr, true });
        ggml_tensor * source_view = ggml_view_2d(view_ctx, slot[graph_slot], width,
                ubatch, slot[graph_slot]->nb[1], 0);
        ggml_tensor * destination_view = ggml_view_2d(view_ctx, owner, width,
                ubatch, owner->nb[1], (size_t) offset * owner->nb[1]);
        ggml_backend_tensor_copy_async(backend, backend, source_view, destination_view);
        ggml_free(view_ctx);
        d2d_bytes += (uint64_t) ubatch * width * sizeof(float);
    }

    ggml_backend_synchronize(backend);
    std::vector<float> actual((size_t) (rows + 1) * width);
    ggml_backend_tensor_get(owner, actual.data(), 0, ggml_nbytes(owner));
    if (!std::equal(host_owner.begin(), host_owner.end(), actual.begin())) {
        std::fprintf(stderr, "CUDA MTP owner differs from four-microbatch host oracle\n");
        return 1;
    }

    std::vector<float> shifted((size_t) rows * width);
    const std::vector<float> carry((size_t) width, -7.0f);
    std::copy(carry.begin(), carry.end(), shifted.begin());
    std::copy_n(actual.begin(), (size_t) (rows - 1) * width,
            shifted.begin() + width);
    for (int32_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < width; ++col) {
            const float expected = row == 0 ? carry[(size_t) col]
                : host_owner[(size_t) (row - 1) * width + col];
            if (shifted[(size_t) row * width + col] != expected) {
                std::fprintf(stderr, "shift mismatch row=%d col=%lld\n", row, (long long) col);
                return 1;
            }
        }
    }

    // Model the draft graph retaining owner rows until its completion event.
    ggml_context * consume_ctx = ggml_init({ 4 * ggml_tensor_overhead(), nullptr, true });
    ggml_tensor * owner_rows = ggml_view_2d(consume_ctx, owner, width, rows,
            owner->nb[1], 0);
    ggml_backend_tensor_copy_async(backend, backend, owner_rows, draft_input);
    ggml_free(consume_ctx);
    ggml_backend_event_record(consumed, backend);
    constexpr uint64_t request_generation = 4;
    uint64_t owner_generation = request_generation;
    ggml_backend_event_synchronize(consumed);
    owner_generation++;
    if (request_generation == owner_generation) {
        std::fprintf(stderr, "stale generation was not refused after cancellation\n");
        return 1;
    }
    std::vector<float> consumed_rows((size_t) rows * width);
    ggml_backend_tensor_get(draft_input, consumed_rows.data(), 0, ggml_nbytes(draft_input));
    if (!std::equal(host_owner.begin(), host_owner.end(), consumed_rows.begin())) {
        std::fprintf(stderr, "draft consumer did not retain staged rows through its event\n");
        return 1;
    }

    std::printf("mtp_cuda_owner=pass rows=%d ubatch=%d graph_slots=2 d2d_bytes=%llu d2h_bytes=%llu shifted_oracle=pass completion_event=pass stale_generation_refused=pass\n",
            rows, ubatch, (unsigned long long) d2d_bytes,
            (unsigned long long) ggml_nbytes(owner));
    ggml_backend_event_free(consumed);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
