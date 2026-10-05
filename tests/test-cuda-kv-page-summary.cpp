#include "ggml.h"
#include "ggml-backend.h"
#include "llama-kv-router-job.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int64_t kDim = 128;
constexpr int64_t kHeads = 2;
constexpr int64_t kRows = 4 * 256;
constexpr int64_t kStreams = 3;
constexpr int64_t kPages = 4;
constexpr int64_t kPageSize = 256;

struct fixture {
    std::vector<uint8_t> k;
    std::vector<int64_t> metadata;
    std::vector<ggml_fp16_t> catalogue;
};

struct key_chunk_source {
    std::vector<uint8_t> bytes;
    uint64_t version = 1;
    static bool recheck(void * context, uint64_t version) noexcept {
        return static_cast<key_chunk_source *>(context)->version == version;
    }
    static bool read(void * context, uint64_t offset, void * dst, size_t bytes) noexcept {
        const auto & source = static_cast<key_chunk_source *>(context)->bytes;
        if (offset > source.size() || bytes > source.size() - size_t(offset)) return false;
        std::memcpy(dst, source.data() + offset, bytes);
        return true;
    }
};

std::unique_ptr<llama_kv_router_job> run_cuda_router_owner_staging(
        ggml_backend_t backend, const fixture & f, uint32_t & slot_out) {
    const size_t row_bytes = ggml_row_size(GGML_TYPE_TURBO4_0, kDim * kHeads);
    const size_t slot_bytes = row_bytes * kRows;
    const size_t chunk_bytes = row_bytes * kRows;
    const auto device = ggml_backend_get_device(backend);
    auto job = llama_kv_router_job::create(backend,
            ggml_backend_dev_host_buffer_type(device), slot_bytes, {},
            GGML_TYPE_TURBO4_0, kDim * kHeads);
    assert(job != nullptr);
    const auto & allocations = job->allocations();
    assert(allocations.pinned_requested == 2 * slot_bytes);
    assert(allocations.pinned_realized >= allocations.pinned_requested);
    assert(allocations.device_realized >= 2 * slot_bytes);
    std::fprintf(stderr,
            "router_owner_slots slot_bytes=%zu pinned_requested=%llu pinned_realized=%llu "
            "device_requested=%llu device_realized=%llu alignment=%llu\n",
            job->slot_bytes(), (unsigned long long) allocations.pinned_requested,
            (unsigned long long) allocations.pinned_realized,
            (unsigned long long) allocations.device_requested,
            (unsigned long long) allocations.device_realized,
            (unsigned long long) allocations.alignment);

    auto source_data = std::make_shared<key_chunk_source>();
    source_data->bytes.assign(f.k.begin(), f.k.begin() + chunk_bytes);
    const size_t page_offset = size_t(2 * kPageSize) * row_bytes;
    std::copy_n(source_data->bytes.begin() + page_offset,
            size_t(kPageSize) * row_bytes, source_data->bytes.begin());
    std::shared_ptr<const void> holder(source_data, source_data.get());
    llama_kv_rerank_stage_source source { holder, source_data.get(),
        &key_chunk_source::recheck, &key_chunk_source::read };
    llama_kv_rerank_chunk_task task;
    task.candidate_index = 2;
    task.compact_layer_index = 0;
    task.row_offset = 0;
    task.rows = kRows;
    task.row_bytes = uint32_t(row_bytes);
    task.query_generation = 1;
    task.content_version = 1;
    uint32_t slot = UINT32_MAX;
    assert(job->submit(source, 0, uint32_t(kRows), task, slot) ==
            llama_kv_rerank_stage_status::ok);
    llama_kv_rerank_stage_ticket completion[1];
    uint32_t count = 0;
    for (uint32_t attempt = 0; attempt < 1000000 && count == 0; ++attempt) {
        count = job->poll(completion, 1);
    }
    assert(count == 1 && completion[0].terminal == llama_kv_rerank_stage_terminal::succeeded);
    std::vector<uint8_t> copied(chunk_bytes);
    ggml_backend_tensor_get(job->encoded_key_slot(slot), copied.data(), 0, copied.size());
    assert(copied == source_data->bytes);
    assert(!job->failed());
    slot_out = slot;
    return job;
}

fixture make_fixture() {
    fixture f;
    const int64_t elements = kDim * kHeads;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_TURBO4_0, elements);
    f.k.resize(size_t(kStreams * kRows) * row_bytes);
    std::vector<float> row(size_t(elements), 0.0f);
    for (int64_t stream = 0; stream < kStreams; ++stream) {
        for (int64_t r = 0; r < kRows; ++r) {
            for (int64_t i = 0; i < elements; ++i) {
                row[size_t(i)] = std::sin(float((stream + 1) * 17 + r * 3 + i) * 0.031f) *
                    (0.4f + 0.1f * float(stream)) + 0.01f * float(i % 7);
            }
            auto * dst = reinterpret_cast<block_turbo4_0 *>(f.k.data() +
                    (size_t(stream * kRows + r) * row_bytes));
            quantize_row_turbo4_0_ref(row.data(), dst, elements);
        }
    }

    f.metadata.assign(size_t(8 * kPages), 0);
    auto set_page = [&](int page, int slot, int valid, int stream, bool update) {
        int64_t * data = f.metadata.data() + 8 * page;
        data[0] = page * kPageSize;
        data[1] = valid;
        data[2] = 7;
        data[3] = page + 1;
        data[4] = slot;
        data[5] = stream;
        data[6] = update ? 1 : 0; // authoritative ready
        data[7] = update ? 1 : 0; // dirty summary update
    };
    set_page(0, 0, 256, 0, true);
    set_page(1, 1, 173, 1, true);
    set_page(2, 2, 129, 0, true);
    set_page(3, 3, 256, 0, false); // copy-through cold page

    f.catalogue.resize(size_t(kDim * 3 * kHeads * kPages));
    for (int64_t page = 0; page < kPages; ++page) {
        for (int64_t head = 0; head < kHeads; ++head) {
            for (int64_t coord = 0; coord < kDim; ++coord) {
                const size_t base = size_t(coord + kDim * (3 * (head + kHeads * page)));
                const float seed = 20.0f + float(page * 3 + head) + float(coord) * 0.01f;
                f.catalogue[base] = ggml_fp32_to_fp16(seed);
                f.catalogue[base + kDim] = ggml_fp32_to_fp16(seed + 1.0f);
                f.catalogue[base + 2 * kDim] = ggml_fp32_to_fp16(seed + 0.5f);
            }
        }
    }
    return f;
}

std::vector<ggml_fp16_t> reference_summary(const fixture & f) {
    const int64_t elements = kDim * kHeads;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_TURBO4_0, elements);
    std::vector<ggml_fp16_t> result = f.catalogue;
    std::vector<float> decoded(size_t(elements), 0.0f);
    for (int64_t page = 0; page < kPages; ++page) {
        const int64_t * data = f.metadata.data() + 8 * page;
        if (data[6] == 0 || data[7] == 0) continue;
        std::vector<float> minimum(size_t(elements), INFINITY);
        std::vector<float> maximum(size_t(elements), -INFINITY);
        std::vector<double> sum(size_t(elements), 0.0);
        for (int64_t row = 0; row < data[1]; ++row) {
            const auto * source = reinterpret_cast<const block_turbo4_0 *>(f.k.data() +
                    size_t((data[5] * kRows + data[4] * kPageSize + row) * row_bytes));
            dequantize_row_turbo4_0(source, decoded.data(), elements);
            for (int64_t i = 0; i < elements; ++i) {
                minimum[size_t(i)] = std::min(minimum[size_t(i)], decoded[size_t(i)]);
                maximum[size_t(i)] = std::max(maximum[size_t(i)], decoded[size_t(i)]);
                sum[size_t(i)] += decoded[size_t(i)];
            }
        }
        for (int64_t head = 0; head < kHeads; ++head) {
            for (int64_t coord = 0; coord < kDim; ++coord) {
                const size_t i = size_t(head * kDim + coord);
                const size_t base = size_t(coord + kDim * (3 * (head + kHeads * page)));
                result[base] = ggml_fp32_to_fp16(std::nextafter(minimum[i], -INFINITY));
                result[base + kDim] = ggml_fp32_to_fp16(std::nextafter(maximum[i], INFINITY));
                result[base + 2 * kDim] = ggml_fp32_to_fp16(float(sum[i] / data[1]));
            }
        }
    }
    return result;
}

std::vector<ggml_fp16_t> run_summary(ggml_backend_t backend, const fixture & f) {
    ggml_init_params params = { 64 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, kDim * kHeads, kRows, kStreams);
    ggml_tensor * metadata = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 8, kPages);
    ggml_tensor * catalogue = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kDim, 3, kHeads, kPages);
    ggml_tensor * summary = ggml_kv_page_summary(ctx, k, metadata, catalogue, kPageSize);
    ggml_set_output(summary);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, summary);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        return {};
    }
    ggml_backend_tensor_set(k, f.k.data(), 0, f.k.size());
    ggml_backend_tensor_set(metadata, f.metadata.data(), 0, f.metadata.size() * sizeof(int64_t));
    ggml_backend_tensor_set(catalogue, f.catalogue.data(), 0, f.catalogue.size() * sizeof(ggml_fp16_t));
    const auto update_begin = std::chrono::steady_clock::now();
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    const double update_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - update_begin).count();
    std::fprintf(stderr, "summary_update backend=%s time_us=%.3f catalogue_bytes=%zu\n",
            ggml_backend_name(backend), update_us, f.catalogue.size() * sizeof(ggml_fp16_t));

    std::vector<ggml_fp16_t> result(f.catalogue.size());
    ggml_backend_tensor_get(summary, result.data(), 0, result.size() * sizeof(result[0]));
    const auto expected = reference_summary(f);
    for (size_t i = 0; i < result.size(); ++i) {
        assert(std::fabs(ggml_fp16_to_fp32(result[i]) - ggml_fp16_to_fp32(expected[i])) < 0.002f);
    }

    // Replay with no dirty pages: every output slice is copied from the
    // catalogue input, including the cold page that was not decoded.
    std::vector<int64_t> replay_metadata = f.metadata;
    for (int64_t page = 0; page < kPages; ++page) replay_metadata[size_t(8 * page + 6)] = 0;
    ggml_backend_tensor_set(metadata, replay_metadata.data(), 0,
            replay_metadata.size() * sizeof(int64_t));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(summary, result.data(), 0, result.size() * sizeof(result[0]));
    assert(result == f.catalogue);

    // Dirtying one logical page changes only that page's output slice.
    std::vector<ggml_fp16_t> changed_catalogue = f.catalogue;
    for (int64_t coord = 0; coord < kDim; ++coord) {
        const size_t base = size_t(coord + kDim * (3 * (0 + kHeads * 1)));
        changed_catalogue[base] = ggml_fp32_to_fp16(99.0f);
        changed_catalogue[base + kDim] = ggml_fp32_to_fp16(100.0f);
    }
    replay_metadata[8 * 1 + 6] = 1;
    replay_metadata[8 * 1 + 7] = 1;
    ggml_backend_tensor_set(catalogue, changed_catalogue.data(), 0,
            changed_catalogue.size() * sizeof(changed_catalogue[0]));
    ggml_backend_tensor_set(metadata, replay_metadata.data(), 0,
            replay_metadata.size() * sizeof(int64_t));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(summary, result.data(), 0, result.size() * sizeof(result[0]));
    for (int64_t page = 0; page < kPages; ++page) {
        if (page == 1) continue;
        for (int64_t head = 0; head < kHeads; ++head) {
            for (int64_t coord = 0; coord < kDim; ++coord) {
                const size_t base = size_t(coord + kDim * (3 * (head + kHeads * page)));
                assert(result[base] == f.catalogue[base]);
                assert(result[base + kDim] == f.catalogue[base + kDim]);
                assert(result[base + 2 * kDim] == f.catalogue[base + 2 * kDim]);
            }
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return expected;
}

std::vector<float> run_rerank(ggml_backend_t backend, const fixture & f,
        int64_t streams, float softcap, ggml_tensor * owned_staged_keys = nullptr,
        llama_kv_router_job * owner = nullptr) {
    constexpr int64_t query_heads = 4, probes_count = 4;
    constexpr int64_t query_generation = 7;
    const int64_t page_versions[kPages] = {14, 14, 22, 22};
    ggml_init_params params = { 64 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);
    ggml_tensor * probes = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kDim, query_heads, probes_count);
    ggml_tensor * resident_keys = ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, kDim * kHeads, kRows, streams);
    ggml_tensor * staged_keys = owned_staged_keys != nullptr
        ? owned_staged_keys
        : ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, kDim * kHeads, kRows, streams);
    if (owned_staged_keys != nullptr) {
        assert(streams == 1 && staged_keys->ne[0] == kDim * kHeads &&
                staged_keys->ne[1] == kRows && staged_keys->ne[2] == 1);
    }
    ggml_tensor * descriptors = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 10, kPages);
    ggml_tensor * identity = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 2, kPages + 1);
    ggml_tensor * validity = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 9);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, kPages, query_heads, probes_count);
    ggml_tensor * result = ggml_kv_page_rerank(ctx, probes, resident_keys, staged_keys, descriptors, identity,
            validity, state, 0.25f, softcap);
    ggml_tensor * mass = ggml_kv_page_mass(ctx, state, descriptors, identity, validity);
    ggml_set_output(result);
    ggml_set_output(mass);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, result);
    ggml_build_forward_expand(graph, mass);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) { ggml_free(ctx); return {}; }

    std::vector<float> q(size_t(kDim * query_heads * probes_count));
    for (int64_t p = 0; p < probes_count; ++p) for (int64_t h = 0; h < query_heads; ++h)
        for (int64_t d = 0; d < kDim; ++d)
            q[size_t(d + kDim * (h + query_heads * p))] =
                std::sin(float(11 + 3 * p + 5 * h + d) * 0.017f);
    std::vector<int64_t> desc(size_t(10 * kPages), 0);
    std::vector<int64_t> identity_values(size_t(2 * (kPages + 1)), 0);
    identity_values[0] = query_generation; identity_values[1] = 9001; // query generation, job serial
    for (int64_t page = 0; page < kPages; ++page) {
        const int64_t * old = f.metadata.data() + 8 * page;
        int64_t * d = desc.data() + 10 * page;
        d[0] = page; d[1] = old[4]; d[2] = old[1]; d[3] = old[5]; d[4] = kPageSize;
        identity_values[size_t(2 * (page + 1))] = 9;
        identity_values[size_t(2 * (page + 1) + 1)] = page_versions[page];
        d[5] = 9; d[6] = page_versions[page]; d[7] = 1; d[8] = page == 3 ? 10000 : page * 64;
        d[9] = (page == 1 || page == 2) ? 1 : 0;
    }
    // Logical page zero is deliberately mapped to a different physical slot.
    desc[1] = 2;
    // Deliberately stale the last descriptor; it must produce empty mass.
    desc[10 * 1 + 6] = 15; // expected 14: only this page is stale
    int64_t probe_state[9] = { query_generation, 300, 299, 298, 296, 1, 1, 1, 1 };
    std::vector<float> initial(size_t(2 * kPages * query_heads * probes_count), 0.0f);
    for (size_t i = 0; i < initial.size(); i += 2) initial[i] = -INFINITY;
    ggml_backend_tensor_set(probes, q.data(), 0, q.size() * sizeof(float));
    ggml_backend_tensor_set(resident_keys, f.k.data(), 0, ggml_nbytes(resident_keys));
    if (owned_staged_keys == nullptr) {
        std::vector<uint8_t> staged_data = f.k;
        const size_t row_bytes = ggml_row_size(GGML_TYPE_TURBO4_0, kDim * kHeads);
        std::copy_n(f.k.begin() + size_t(2 * kPageSize) * row_bytes,
                size_t(kPageSize) * row_bytes, staged_data.begin());
        ggml_backend_tensor_set(staged_keys, staged_data.data(), 0, ggml_nbytes(staged_keys));
    }
    ggml_backend_tensor_set(descriptors, desc.data(), 0, desc.size() * sizeof(int64_t));
    ggml_backend_tensor_set(identity, identity_values.data(), 0, identity_values.size() * sizeof(int64_t));
    ggml_backend_tensor_set(validity, probe_state, 0, sizeof(probe_state));
    ggml_backend_tensor_set(state, initial.data(), 0, initial.size() * sizeof(float));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> output(size_t(kPages * query_heads * probes_count));
    ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
    std::vector<ggml_kv_page_rank_record> mass_output(kPages);
    ggml_backend_tensor_get(mass, mass_output.data(), 0, ggml_nbytes(mass));
    if (owner != nullptr) {
        llama_kv_router_query_identity owner_identity;
        owner_identity.sequence_id = 0;
        owner_identity.session_generation = 1;
        owner_identity.turn_id = 7;
        owner_identity.query_generation = query_generation;
        owner_identity.rollback_generation = 1;
        owner_identity.table_epoch = 1;
        owner_identity.content_generation = 1;
        std::vector<llama_kv_router_exact_page_record> exact_records;
        llama_kv_prefetch_candidate coarse;
        for (int64_t i = 0; i < kPages; ++i) {
            const auto & mass_record = mass_output[size_t(i)];
            if (mass_record.validity_flags != 1 || mass_record.logical_page < 0) continue;
            const uint32_t page = uint32_t(mass_record.logical_page);
            llama_kv_prefetch_candidate candidate;
            candidate.identity.sequence_id = owner_identity.sequence_id;
            candidate.identity.session_generation = owner_identity.session_generation;
            candidate.identity.sequence_generation = 1;
            candidate.identity.logical_page = page;
            candidate.identity.page_generation = page + 1;
            candidate.identity.representation_epoch = 1;
            candidate.identity.position_begin = int64_t(page) * kPageSize;
            candidate.identity.position_end = candidate.identity.position_begin + kPageSize;
            candidate.attention_layer = 0;
            candidate.generation = query_generation;
            candidate.table_epoch = owner_identity.table_epoch;
            candidate.rollback_generation = owner_identity.rollback_generation;
            candidate.speculation_generation = candidate.identity.sequence_generation;
            candidate.content_version = candidate.summary_version = page_versions[page];
            candidate.provenance = llama_kv_prefetch_candidate::score_kind::probe_softmax;
            candidate.peak_probability = mass_record.peak_probability;
            candidate.mean_probability = mass_record.mean_probability;
            candidate.cold = page == 2;
            if (candidate.cold) {
                coarse = candidate;
                coarse.peak_probability = coarse.mean_probability = 0.5f;
            }
            llama_kv_router_exact_page_record exact;
            exact.candidate = candidate;
            exact.descriptor_content_version = page_versions[page];
            exact.descriptor_summary_version = page_versions[page];
            exact_records.push_back(exact);
        }
        assert(coarse.cold);
        coarse.provenance = llama_kv_prefetch_candidate::score_kind::probe_softmax;
        assert(owner->retain_coarse_shortlist(owner_identity, 0, { coarse }));
        assert(owner->complete_query(owner_identity, true, exact_records));
        assert(owner->exact_candidates().size() == exact_records.size());
    }
    assert(output.size() == size_t(result->ne[0] * result->ne[1] * result->ne[2]));
    for (int64_t h = 0; h < query_heads; ++h) for (int64_t p = 0; p < probes_count; ++p) {
        assert(std::isfinite(output[size_t(0 + kPages * h + kPages * query_heads * p)]));
        assert(output[size_t(1 + kPages * h + kPages * query_heads * p)] == -INFINITY);
        assert(std::isfinite(output[size_t(2 + kPages * h + kPages * query_heads * p)]));
    }

    // Independent represented-key scalar reference: decode each encoded row,
    // apply GQA mapping and causal masking, then reduce token logits by LSE.
    std::vector<float> decoded(size_t(kDim * kHeads));
    std::vector<double> reference_logs(output.size(), -INFINITY);
    for (int64_t page = 0; page < kPages; ++page) {
        const int64_t * d = desc.data() + 10 * page;
        const int64_t * expected_id = identity_values.data() + 2 * (page + 1);
        if (d[7] == 0 || d[5] != expected_id[0] || d[6] != expected_id[1]) continue;
        for (int64_t head = 0; head < query_heads; ++head) for (int64_t p = 0; p < probes_count; ++p) {
            const size_t out_index = size_t(page + kPages * (head + query_heads * p));
            double maximum = -INFINITY, sum = 0.0;
            if (probe_state[5 + p] != 0) for (int64_t row = 0; row < d[2]; ++row) {
                if (d[8] + row > probe_state[1 + p]) continue;
                const size_t key_row = size_t(d[3] * kRows + d[1] * d[4] + row);
                const size_t row_bytes = ggml_row_size(GGML_TYPE_TURBO4_0, kDim * kHeads);
                dequantize_row_turbo4_0(reinterpret_cast<const block_turbo4_0 *>(
                        f.k.data() + key_row * row_bytes), decoded.data(), kDim * kHeads);
                const int64_t kv_head = head / (query_heads / kHeads);
                double dot = 0.0;
                for (int64_t coord = 0; coord < kDim; ++coord) {
                    dot += double(q[size_t(coord + kDim * (head + query_heads * p))]) *
                        decoded[size_t(kv_head * kDim + coord)];
                }
                const double scaled = dot * 0.25;
                const double logit = softcap > 0.0f ? double(softcap) * std::tanh(scaled / double(softcap)) : scaled;
                if (logit > maximum) { sum = sum * std::exp(maximum - logit) + 1.0; maximum = logit; }
                else sum += std::exp(logit - maximum);
            }
            const float expected = sum == 0.0 ? -INFINITY : float(maximum + std::log(sum));
            reference_logs[out_index] = expected;
            if (std::isinf(expected)) assert(output[out_index] == expected);
            else assert(std::fabs(output[out_index] - expected) < 1e-4f);
        }
    }

    std::vector<double> reference_probabilities(output.size(), 0.0);
    for (int64_t h = 0; h < query_heads; ++h) for (int64_t p = 0; p < probes_count; ++p) {
        double maximum = -INFINITY;
        for (int64_t page = 0; page < kPages; ++page)
            maximum = std::max(maximum, reference_logs[size_t(page + kPages * (h + query_heads * p))]);
        if (!std::isfinite(maximum)) continue;
        double denom = 0.0;
        for (int64_t page = 0; page < kPages; ++page) {
            const double value = reference_logs[size_t(page + kPages * (h + query_heads * p))];
            if (std::isfinite(value)) denom += std::exp(value - maximum);
        }
        for (int64_t page = 0; page < kPages; ++page) {
            const double value = reference_logs[size_t(page + kPages * (h + query_heads * p))];
            if (std::isfinite(value)) reference_probabilities[size_t(page + kPages * (h + query_heads * p))] = std::exp(value - maximum) / denom;
        }
    }
    for (int64_t page = 0; page < kPages; ++page) {
        double peak = 0.0, total = 0.0; int channels = 0;
        for (int64_t p = 0; p < probes_count; ++p) for (int64_t h = 0; h < query_heads; ++h) {
            const double value = reference_probabilities[size_t(page + kPages * (h + query_heads * p))];
            if (value > 0.0) { peak = std::max(peak, value); total += value; ++channels; }
        }
        if (channels == 0) assert(mass_output[size_t(page)].logical_page == -1);
        else {
            assert(mass_output[size_t(page)].logical_page == page);
            assert(std::fabs(mass_output[size_t(page)].peak_probability - peak) < 2e-4);
            assert(std::fabs(mass_output[size_t(page)].mean_probability - total / channels) < 2e-4);
            assert(mass_output[size_t(page)].mean_probability <= mass_output[size_t(page)].peak_probability);
        }
    }

    // The same encoded page split at a staging boundary must combine through
    // the caller-owned (m,s) state to the exact unsplit log mass.
    std::vector<int64_t> split_desc = desc;
    for (int64_t page = 1; page < kPages; ++page) split_desc[size_t(10 * page + 7)] = 0;
    split_desc[2] = 128; split_desc[4] = 128; split_desc[1] = 0; split_desc[9] = 1;
    std::fill(initial.begin(), initial.end(), 0.0f);
    for (size_t i = 0; i < initial.size(); i += 2) initial[i] = -INFINITY;
    ggml_backend_tensor_set(state, initial.data(), 0, initial.size() * sizeof(float));
    ggml_backend_tensor_set(descriptors, split_desc.data(), 0, split_desc.size() * sizeof(int64_t));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    split_desc[1] = 1;
    split_desc[8] += 128;
    ggml_backend_tensor_set(descriptors, split_desc.data(), 0, split_desc.size() * sizeof(int64_t));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> split_output(output.size());
    ggml_backend_tensor_get(result, split_output.data(), 0, split_output.size() * sizeof(float));
    for (int64_t h = 0; h < query_heads; ++h) for (int64_t p = 0; p < probes_count; ++p) {
        const size_t index = size_t(kPages * (h + query_heads * p));
        assert(std::fabs(split_output[index] - output[index]) < 1e-4f);
    }
    std::vector<float> state_after_chunks(initial.size());
    ggml_backend_tensor_get(state, state_after_chunks.data(), 0, state_after_chunks.size() * sizeof(float));
    split_desc[6] = 15; // stale chunk version must not merge into the initialized state
    ggml_backend_tensor_set(descriptors, split_desc.data(), 0, split_desc.size() * sizeof(int64_t));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(result, split_output.data(), 0, split_output.size() * sizeof(float));
    for (int64_t h = 0; h < query_heads; ++h) for (int64_t p = 0; p < probes_count; ++p)
        assert(split_output[size_t(kPages * (h + query_heads * p))] == -INFINITY);
    std::vector<float> state_after_stale(initial.size());
    ggml_backend_tensor_get(state, state_after_stale.data(), 0, state_after_stale.size() * sizeof(float));
    assert(state_after_stale == state_after_chunks);
    probe_state[0] = query_generation + 1;
    ggml_backend_tensor_set(validity, probe_state, 0, sizeof(probe_state));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
    ggml_backend_tensor_get(mass, mass_output.data(), 0, ggml_nbytes(mass));
    for (float value : output) assert(value == -INFINITY);
    for (const auto & record : mass_output) assert(record.logical_page == -1 && record.validity_flags == 0);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return output;
}

} // namespace

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_cpu_init();
    assert(cpu != nullptr);
    const fixture f = make_fixture();
    const auto cpu_result = run_summary(cpu, f);
    const auto cpu_rerank_1stream = run_rerank(cpu, f, 1, 1.3f);
    const auto cpu_rerank_3streams = run_rerank(cpu, f, 3, 1.3f);
    ggml_backend_free(cpu);
    if (argc > 1 && std::string(argv[1]) == "--cpu-only") {
        assert(!cpu_rerank_1stream.empty() && !cpu_rerank_3streams.empty());
        std::puts("CPU KV page summary and Turbo4 rerank graph passed");
        return 0;
    }

    ggml_backend_dev_t cuda_device = ggml_backend_dev_by_name("CUDA0");
    if (cuda_device == nullptr) {
        std::fprintf(stderr, "CUDA0 unavailable; CPU summary reference/replay passed\n");
        return 77;
    }
    ggml_backend_t cuda = ggml_backend_dev_init(cuda_device, nullptr);
    assert(cuda != nullptr);
    uint32_t owner_slot = UINT32_MAX;
    auto router_job = run_cuda_router_owner_staging(cuda, f, owner_slot);
    const auto cuda_result = run_summary(cuda, f);
    const auto cuda_rerank_1stream = run_rerank(cuda, f, 1, 1.3f,
            router_job->encoded_key_slot(owner_slot), router_job.get());
    const auto cuda_rerank_3streams = run_rerank(cuda, f, 3, 1.3f);
    router_job->cancel();
    router_job.reset();
    ggml_backend_free(cuda);
    if (cuda_result.empty()) {
        std::fprintf(stderr, "CUDA0 could not allocate the summary fixture\n");
        return 77;
    }
    assert(cuda_result.size() == cpu_result.size());
    assert(cuda_rerank_1stream.size() == cpu_rerank_1stream.size());
    assert(cuda_rerank_3streams.size() == cpu_rerank_3streams.size());
    for (size_t i = 0; i < cuda_rerank_1stream.size(); ++i) {
        if (std::isinf(cpu_rerank_1stream[i])) assert(cuda_rerank_1stream[i] == cpu_rerank_1stream[i]);
        else assert(std::fabs(cuda_rerank_1stream[i] - cpu_rerank_1stream[i]) < 0.02f);
        if (std::isinf(cpu_rerank_3streams[i])) assert(cuda_rerank_3streams[i] == cpu_rerank_3streams[i]);
        else assert(std::fabs(cuda_rerank_3streams[i] - cpu_rerank_3streams[i]) < 0.02f);
    }
    for (size_t i = 0; i < cuda_result.size(); ++i) {
        assert(std::fabs(ggml_fp16_to_fp32(cuda_result[i]) -
                ggml_fp16_to_fp32(cpu_result[i])) < 0.002f);
    }
    std::puts("CUDA KV page summary, GQA rerank, mixed identity, softcap and exact-pool mass passed (2 KV heads, streams 1/3; probability tolerance 2e-4, log-mass tolerance 1e-4)");
    return 0;
}
