#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-quants.h"
#include "llama-kv-live-policy.h"
#include "llama-kv-pager.h"
#include "llama-kv-prefetch.h"
#include "speculative.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>

#undef NDEBUG

namespace {

constexpr uint64_t seed = UINT64_C(0x49060001);
constexpr uint32_t pages = 8;
constexpr uint32_t page_tokens = 256;
constexpr uint32_t final_query_row = 2;
constexpr uint32_t layers = 16;
constexpr uint32_t kv_heads = 2;
constexpr uint32_t q_heads = 4;
constexpr uint32_t head_dim = 128;
constexpr uint32_t cold_capacity = 2;
constexpr uint32_t hot_capacity = 3;
constexpr uint32_t retrieval_capacity = 2;
constexpr uint64_t source_namespace = UINT64_C(0x49060002);

static uint64_t mix(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static llama_kv_page_id page_id(uint32_t logical) {
    llama_kv_page_id id;
    id.session_generation = 1;
    id.sequence_id = 0;
    id.sequence_generation = 1;
    id.logical_page = logical;
    id.page_generation = 100 + logical;
    id.representation_epoch = 7;
    id.model_identity = 49;
    id.topology_identity = 16;
    id.codec_digest = 4;
    id.codebook_digest = 5;
    id.rotation_digest = 6;
    id.meansub_digest = 7;
    id.position_begin = llama_pos(logical * page_tokens);
    id.position_end = id.position_begin + page_tokens;
    return id;
}

struct promotion_fixture {
    static constexpr uint32_t unit_count = layers * 2;

    uint64_t row_bytes_head = ggml_row_size(GGML_TYPE_TURBO4_0, head_dim);
    uint64_t row_bytes_unit = ggml_row_size(GGML_TYPE_TURBO4_0, head_dim * kv_heads);
    std::vector<std::vector<std::vector<uint8_t>>> bytes;
    std::vector<std::vector<std::vector<uint8_t>>> capture_bytes;
    std::mutex capture_mutex;
    std::vector<std::pair<uint32_t, uint64_t>> capture_versions;
    std::vector<vbr_selected_page_unit_source> sources;
    vbr_selected_page_capture_snapshot snapshot;

    static bool read(const void * context, uint64_t offset, uint8_t * dst,
                     size_t size) noexcept {
        const auto * value = static_cast<const std::vector<uint8_t> *>(context);
        if (value == nullptr || offset > value->size() || size > value->size() - offset) {
            return false;
        }
        std::memcpy(dst, value->data() + offset, size);
        return true;
    }

    static bool acquire(void * context, const vbr_selected_page_capture_request &,
                        vbr_selected_page_capture_snapshot & output) noexcept {
        output = static_cast<promotion_fixture *>(context)->snapshot;
        return true;
    }

    static bool recheck(void *, const vbr_selected_page_capture_snapshot &) noexcept {
        return true;
    }

    static void release(void *, const vbr_selected_page_capture_snapshot &) noexcept {}

    static bool prepare(void * context, const llama_kv_page_record & page,
                        vbr_selected_page_capture_request & request,
                        std::vector<vbr_selected_page_unit_source> & output,
                        vbr_selected_page_capture_snapshot_provider & provider) noexcept {
        auto & self = *static_cast<promotion_fixture *>(context);
        if (page.id.logical_page >= self.bytes.size()) return false;
        try {
            std::lock_guard<std::mutex> lock(self.capture_mutex);
            self.capture_versions.emplace_back(page.id.logical_page, page.content_version);
        } catch (...) {
            return false;
        }
        request = {};
        request.source_namespace = source_namespace;
        request.child_id = 0;
        request.stream_index = 0;
        request.expected_unit_generations.resize(unit_count);
        request.pages.push_back({});
        request.pages[0].identity = page.id;
        request.pages[0].positions.resize(page.valid_length);
        request.pages[0].physical_cells.resize(page.valid_length);
        for (uint32_t row = 0; row < page.valid_length; ++row) {
            request.pages[0].positions[row] = page.id.position_begin + row;
            request.pages[0].physical_cells[row] =
                page.physical_slot * page_tokens + row;
        }
        request.pages[0].tail = page.valid_length != page_tokens;
        output.clear();
        output.reserve(unit_count);
        for (uint32_t unit = 0; unit < unit_count; ++unit) {
            request.required_unit_ids.push_back(unit);
            request.expected_unit_generations[unit] = self.snapshot.units[unit].generation;
            vbr_selected_page_unit_source source;
            source.logical_unit_id = unit;
            source.row_count = page.valid_length;
            source.row_bytes = self.row_bytes_unit;
            source.source_identity = UINT64_C(0x49070000) + unit;
            auto & captured = self.capture_bytes[page.id.logical_page][unit];
            captured.assign(size_t(hot_capacity) * page_tokens * self.row_bytes_unit, 0);
            std::memcpy(captured.data() + size_t(page.physical_slot) * page_tokens * self.row_bytes_unit,
                self.bytes[page.id.logical_page][unit].data(),
                self.bytes[page.id.logical_page][unit].size());
            source.row_count = hot_capacity * page_tokens;
            source.source.size = captured.size();
            source.source.context = &captured;
            source.source.read = read;
            output.push_back(source);
            vbr_capture_projected_shard_source projected;
            projected.shard_index = 0;
            projected.row_count = hot_capacity * page_tokens;
            projected.row_bytes = self.row_bytes_unit;
            projected.source_identity = source.source_identity;
            projected.source = source.source;
            assert(vbr_capture_projected_shard_topology(
                { projected }, self.snapshot.units[unit].shard_count,
                self.snapshot.units[unit].shard_topology_digest));
        }
        self.snapshot.pages = { page.id };
        provider = { &self, acquire, recheck, release };
        return true;
    }

    void initialize() {
        bytes.resize(pages, std::vector<std::vector<uint8_t>>(unit_count));
        capture_bytes.resize(pages, std::vector<std::vector<uint8_t>>(unit_count));
        for (uint32_t logical = 0; logical < pages; ++logical) {
            for (uint32_t layer = 0; layer < layers; ++layer) {
                for (uint32_t side = 0; side < 2; ++side) {
                    const uint32_t unit = layer * 2 + side;
                    auto & dst = bytes[logical][unit];
                    dst.resize(page_tokens * row_bytes_unit);
                    for (uint32_t row = 0; row < page_tokens; ++row) {
                        std::array<float, head_dim * kv_heads> values{};
                        for (uint32_t head = 0; head < kv_heads; ++head) {
                            for (uint32_t d = 0; d < head_dim; ++d) {
                                const uint64_t r = mix(seed + uint64_t(logical) * 7919 +
                                    uint64_t(layer) * 313 + uint64_t(side) * 97 +
                                    uint64_t(head) * 29 + uint64_t(row) * 17 + d);
                                float value = float(int32_t(r & 31) - 15) / 15.0f;
                                // Generated page one is deliberately aligned
                                // with the positive transformed query. Ring
                                // wrap later spills it to the cold catalogue.
                                if (logical == 1 && side == 0) value += 2.0f;
                                values[head * head_dim + d] = value;
                            }
                        }
                        quantize_row_turbo4_0_ref(values.data(),
                            reinterpret_cast<block_turbo4_0 *>(dst.data() + row * row_bytes_unit),
                            head_dim * kv_heads);
                    }
                    capture_bytes[logical][unit].resize(
                        size_t(hot_capacity) * page_tokens * row_bytes_unit);
                }
            }
        }

        snapshot = {};
        snapshot.source_namespace = source_namespace;
        snapshot.child_id = 0;
        snapshot.stream_index = 0;
        for (uint32_t unit = 0; unit < unit_count; ++unit) {
            vbr_capture_projected_shard_source projected;
            projected.shard_index = 0;
            projected.row_count = hot_capacity * page_tokens;
            projected.row_bytes = row_bytes_unit;
            projected.source_identity = UINT64_C(0x49070000) + unit;
            projected.source.context = &capture_bytes[0][unit];
            projected.source.size = capture_bytes[0][unit].size();
            projected.source.read = read;

            vbr_capture_unit_snapshot unit_snapshot;
            unit_snapshot.source_namespace = source_namespace;
            unit_snapshot.child_id = 0;
            unit_snapshot.logical_unit_id = unit;
            unit_snapshot.lineage_uuid = { UINT64_C(0x49060000) + unit, 1 };
            unit_snapshot.controller_generation = 1;
            unit_snapshot.generation.repr_gen = 1;
            unit_snapshot.generation.current_type = GGML_TYPE_TURBO4_0;
            unit_snapshot.generation.last_source_type = GGML_TYPE_TURBO4_0;
            unit_snapshot.generation.domain = vbr_repr_domain::full;
            assert(vbr_capture_projected_shard_topology(
                { projected }, unit_snapshot.shard_count,
                unit_snapshot.shard_topology_digest));
            snapshot.units.push_back(unit_snapshot);

            vbr_artifact_unit_descriptor descriptor;
            descriptor.child_id = 0;
            descriptor.logical_unit_id = unit;
            descriptor.lineage_uuid = unit_snapshot.lineage_uuid;
            descriptor.repr_gen = 1;
            descriptor.current_type = GGML_TYPE_TURBO4_0;
            descriptor.last_source_type = GGML_TYPE_TURBO4_0;
            descriptor.representation.kind = vbr_artifact_representation_kind::approximate;
            descriptor.representation.codec_id = 4;
            descriptor.representation.codec_version = 1;
            descriptor.representation.reference_digest.fill(1);
            descriptor.side = (unit & 1) ? vbr_artifact_side::value : vbr_artifact_side::key;
            descriptor.layout = vbr_artifact_layout::row_major;
            descriptor.n_stream = 1;
            descriptor.wm_cells = page_tokens;
            descriptor.codebook_digest.fill(2);
            descriptor.rotation_digest.fill(3);
            descriptor.meansub_digest.fill(4);
            descriptor.row_codec_version = 1;
            vbr_artifact_shard_descriptor shard;
            shard.row_count = hot_capacity * page_tokens;
            shard.column_count = 1;
            shard.row_bytes = row_bytes_unit;
            shard.payload_bytes = capture_bytes[0][unit].size();
            descriptor.shards.push_back(shard);
            snapshot.unit_descriptors.push_back(std::move(descriptor));
        }
    }
};

static bool build_summary(void * context, const llama_kv_page_record & page,
                          const llama_kv_routing_summary_config & config,
                          llama_kv_routing_page_input & output) noexcept {
    auto & fixture = *static_cast<promotion_fixture *>(context);
    if (page.id.logical_page >= fixture.bytes.size() || config.vector_dim != head_dim) return false;
    const auto & source = fixture.bytes[page.id.logical_page][0];
    output = {};
    output.id = page.id;
    if (page.valid_length == 0) return false;
    output.row_indices = { 0, page.valid_length / 3,
        (page.valid_length * 2) / 3, page.valid_length - 1 };
    output.rotated_k_rows.resize(output.row_indices.size() * config.vector_dim);
    std::vector<float> decoded(head_dim * kv_heads);
    for (size_t i = 0; i < output.row_indices.size(); ++i) {
        const auto * row = reinterpret_cast<const block_turbo4_0 *>(
            source.data() + output.row_indices[i] * fixture.row_bytes_unit);
        dequantize_row_turbo4_0(row, decoded.data(), head_dim * kv_heads);
        std::copy(decoded.begin(), decoded.begin() + head_dim,
                  output.rotated_k_rows.begin() + i * config.vector_dim);
    }
    output.source_bytes = output.rotated_k_rows.size() * sizeof(float);
    return true;
}

static llama_kv_pager_resources pager_resources(uint64_t page_bytes,
                                                ggml_backend_t backend,
                                                ggml_backend_dev_t device) {
    llama_kv_pager_resources result;
    // Admission also charges the fixed Turbo4 scratch and transfer floor;
    // leave one page of explicit headroom so the requested H=3 is realized.
    result.admission.capacity_bytes = page_bytes * (hot_capacity + 1);
    result.admission.target_page_bytes = page_bytes;
    result.admission.turbo4_scratch_bytes = 64;
    result.admission.mtp_present = false;
    result.host_budget_known = true;
    result.host_budget_bytes = 64u * 1024u * 1024u;
    result.allocator_granularity = 256;
    result.host_capture_enabled = true;
    result.host_source_namespace = source_namespace;
    result.host_child_id = 0;
    result.host_stream_index = 0;
    result.host_backend = backend;
    result.host_lanes = { { device, backend, true } };
    result.host_ring_bytes = 2u * 64u * 1024u;
    result.host_chunk_bytes = 64u * 1024u;
    result.host_budget.host.pageable_cap = result.host_budget_bytes;
    result.host_budget.host.pageable_state = llama_cache_budget_capacity_state::known;
    result.host_budget.host.pinned_cap = result.host_ring_bytes;
    result.host_budget.host.pinned_state = llama_cache_budget_capacity_state::known;
    result.host_budget.host.total_cap = result.host_budget_bytes;
    result.host_budget.host.total_state = llama_cache_budget_capacity_state::known;
    result.routing_summary.vector_dim = head_dim;
    result.routing_summary.representative_count = 4;
    return result;
}

struct dense_result {
    std::vector<float> values;
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

static dense_result run_dense_fa(ggml_backend_t backend,
                                 const std::vector<float> & q_host,
                                 const std::vector<uint8_t> & k_host,
                                 const std::vector<uint8_t> & v_host,
                                 uint32_t rows) {
    ggml_context * ctx = ggml_init({ 64u * 1024u * 1024u, nullptr, true });
    assert(ctx != nullptr);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 3, q_heads);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, head_dim, rows, kv_heads);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, head_dim, rows, kv_heads);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, nullptr,
        1.0f / std::sqrt(float(head_dim)), 0.0f, 0.0f);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);
    std::vector<float> q_layout(q_host.size());
    for (uint32_t query = 0; query < 3; ++query) {
        for (uint32_t head = 0; head < q_heads; ++head) {
            std::memcpy(q_layout.data() + (size_t(head) * 3 + query) * head_dim,
                q_host.data() + (size_t(query) * q_heads + head) * head_dim,
                head_dim * sizeof(float));
        }
    }
    ggml_backend_tensor_set(q, q_layout.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, k_host.data(), 0, k_host.size());
    ggml_backend_tensor_set(v, v_host.data(), 0, v_host.size());
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    dense_result result;
    result.values.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.values.data(), 0, ggml_nbytes(out));
    result.context = ctx;
    result.buffer = buffer;
    return result;
}

static void free_dense(dense_result & result) {
    ggml_backend_buffer_free(result.buffer);
    ggml_free(result.context);
    result = {};
}

static std::vector<uint8_t> pack_pages(const promotion_fixture & fixture,
                                       const std::vector<llama_kv_page_id> & ids,
                                       uint32_t layer, uint32_t side) {
    const uint64_t head_bytes = fixture.row_bytes_head;
    const uint64_t unit_bytes = fixture.row_bytes_unit;
    const uint32_t rows = uint32_t(ids.size()) * page_tokens;
    std::vector<uint8_t> result(size_t(kv_heads) * rows * head_bytes);
    for (uint32_t head = 0; head < kv_heads; ++head) {
        for (uint32_t p = 0; p < ids.size(); ++p) {
            const auto & source = fixture.bytes[ids[p].logical_page][layer * 2 + side];
            for (uint32_t row = 0; row < page_tokens; ++row) {
                std::memcpy(result.data() + (size_t(head) * rows + p * page_tokens + row) * head_bytes,
                    source.data() + row * unit_bytes + head * head_bytes, head_bytes);
            }
        }
    }
    return result;
}

static std::vector<uint8_t> read_device_pages(ggml_tensor * storage,
        const llama_kv_pager_snapshot & snapshot,
        const std::vector<llama_kv_page_record> & records,
        uint32_t layer, uint32_t side, uint32_t rows) {
    const uint64_t head_bytes = ggml_row_size(GGML_TYPE_TURBO4_0, head_dim);
    const uint64_t unit_bytes = ggml_row_size(GGML_TYPE_TURBO4_0, head_dim * kv_heads);
    const uint64_t page_bytes = side == 0 ? snapshot.geometry.layer_k_page_bytes[layer]
                                          : snapshot.geometry.layer_v_page_bytes[layer];
    const uint64_t layer_offset = side == 0 ? snapshot.geometry.layer_k_offsets[layer]
                                             : snapshot.geometry.layer_v_offsets[layer];
    const uint64_t output_page_rows = page_tokens;
    std::vector<uint8_t> result(size_t(kv_heads) * rows * head_bytes);
    std::vector<uint8_t> page(page_bytes);
    for (uint32_t p = 0; p < records.size(); ++p) {
        ggml_backend_tensor_get(storage, page.data(),
            layer_offset + uint64_t(records[p].physical_slot) * page_bytes, page.size());
        for (uint32_t head = 0; head < kv_heads; ++head) {
            for (uint32_t row = 0; row < page_tokens; ++row) {
                std::memcpy(result.data() + (size_t(head) * rows + p * page_tokens + row) * head_bytes,
                    page.data() + row * unit_bytes + head * head_bytes, head_bytes);
            }
        }
    }
    (void) output_page_rows;
    return result;
}

static llama_kv_residency_transfer_page promotion_transfer_page(
        const llama_kv_page_record & page, const llama_kv_pager_snapshot & snapshot,
        uint64_t epoch) {
    llama_kv_residency_transfer_page transfer;
    transfer.page = page.id;
    transfer.table_epoch = epoch;
    transfer.physical_slot = page.physical_slot;
    transfer.layer = UINT32_MAX;
    transfer.content_version = page.content_version;
    transfer.valid_length = page.valid_length;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        for (uint32_t side = 0; side < 2; ++side) {
            const uint64_t bytes = side == 0 ? snapshot.geometry.layer_k_page_bytes[layer]
                                             : snapshot.geometry.layer_v_page_bytes[layer];
            const uint64_t offset = (uint64_t(layer) * 2 + side) * bytes;
            transfer.runs.push_back({ UINT32_MAX, 0, layer, uint8_t(side), 0,
                page.valid_length, bytes / page_tokens,
                offset, uint64_t(page.physical_slot) * bytes });
        }
    }
    return transfer;
}

static int run_proof() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name("CUDA0");
    if (device == nullptr) return 77;
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) return 77;

    promotion_fixture fixture;
    fixture.initialize();
    const uint64_t layer_page_bytes = page_tokens * fixture.row_bytes_unit;
    llama_kv_pager_geometry geometry;
    // Leave one logical page after the staged query so the regression can
    // reserve a packet that straddles the partial query page's boundary.
    geometry.context_tokens = (pages + 1) * page_tokens;
    geometry.page_tokens = page_tokens;
    geometry.attention_layers = layers;
    geometry.kv_heads = kv_heads;
    geometry.key_length = head_dim;
    geometry.value_length = head_dim;
    geometry.page_bytes = uint64_t(layers) * 2 * layer_page_bytes;
    geometry.layer_k_page_bytes.assign(layers, layer_page_bytes);
    geometry.layer_v_page_bytes.assign(layers, layer_page_bytes);
    geometry.layer_k_offsets.assign(layers, 0);
    geometry.layer_v_offsets.assign(layers, 0);
    geometry.model_layer_ids.resize(layers);
    std::iota(geometry.model_layer_ids.begin(), geometry.model_layer_ids.end(), 0);

    llama_kv_pager_config config;
    config.mode = llama_kv_pager_mode::selective;
    config.page_size = page_tokens;
    config.hot_pages.automatic = false;
    config.hot_pages.value = hot_capacity;
    config.retrieval_pages.automatic = false;
    config.retrieval_pages.value = retrieval_capacity;
    llama_kv_pager_resources resources = pager_resources(geometry.page_bytes, backend, device);
    llama_kv_pager_snapshot planned;
    llama_kv_pager_status status;
    if (!llama_kv_pager_plan(config, geometry, resources, planned, status)) {
        std::fprintf(stderr, "pager plan failed status=%d page_bytes=%llu capacity=%llu\n",
            int(status), (unsigned long long) geometry.page_bytes,
            (unsigned long long) resources.admission.capacity_bytes);
        std::fprintf(stderr, "admission refusal=%d accepted=%d admitted=%llu charge=%llu usable=%llu\n",
            int(planned.admission.refusal), int(planned.admission.accepted),
            (unsigned long long) planned.admission.admitted_pages,
            (unsigned long long) planned.admission.charged_bytes,
            (unsigned long long) planned.admission.usable_device_bytes);
        return 1;
    }
    if (status != llama_kv_pager_status::ok || planned.physical_page_count != hot_capacity) {
        std::fprintf(stderr, "unexpected admission status=%d physical_pages=%u\n",
            int(status), planned.physical_page_count);
        return 1;
    }

    ggml_context * storage_context = ggml_init({ ggml_tensor_overhead(), nullptr, true });
    assert(storage_context != nullptr);
    ggml_tensor * storage = ggml_new_tensor_1d(storage_context, GGML_TYPE_I8, planned.physical_bytes);
    ggml_backend_buffer_t storage_buffer = ggml_backend_alloc_ctx_tensors(storage_context, backend);
    assert(storage_buffer != nullptr);
    ggml_context * draft_context = ggml_init({ ggml_tensor_overhead(), nullptr, true });
    assert(draft_context != nullptr);
    ggml_tensor * draft_storage = ggml_new_tensor_3d(draft_context, GGML_TYPE_TURBO4_0,
        head_dim * kv_heads, pages * page_tokens, layers * 2);
    ggml_backend_buffer_t draft_buffer = ggml_backend_alloc_ctx_tensors(draft_context, backend);
    assert(draft_buffer != nullptr && draft_buffer != storage_buffer);
    std::vector<uint8_t> draft_reference(size_t(pages) * page_tokens * fixture.row_bytes_unit *
        promotion_fixture::unit_count);
    for (uint32_t unit = 0; unit < promotion_fixture::unit_count; ++unit) {
        for (uint32_t logical = 0; logical < pages; ++logical) {
            std::memcpy(draft_reference.data() +
                    (size_t(unit) * pages + logical) * page_tokens * fixture.row_bytes_unit,
                fixture.bytes[logical][unit].data(), page_tokens * fixture.row_bytes_unit);
        }
    }
    ggml_backend_tensor_set(draft_storage, draft_reference.data(), 0, ggml_nbytes(draft_storage));
    resources.external_storage_buffer = storage_buffer;
    resources.external_storage_tensor = storage;
    auto pager = llama_kv_pager::create(config, geometry, resources, {}, status);
    assert(pager && status == llama_kv_pager_status::ok);
    pager->bind_representation_identity(49, 16, 4, 5, 6, 7, 4);
    pager->set_host_provider({ &fixture, promotion_fixture::prepare });
    pager->set_routing_summary_provider({ &fixture, build_summary });

    assert(planned.generation_pages == 1);
    auto write_cuda_page = [&](uint32_t logical, uint32_t slot) {
        for (uint32_t layer = 0; layer < layers; ++layer) {
            for (uint32_t side = 0; side < 2; ++side) {
                const uint64_t page_bytes = side == 0 ? planned.geometry.layer_k_page_bytes[layer]
                                                       : planned.geometry.layer_v_page_bytes[layer];
                const uint64_t offset = side == 0 ? planned.geometry.layer_k_offsets[layer]
                                                   : planned.geometry.layer_v_offsets[layer];
                ggml_backend_tensor_set(storage, fixture.bytes[logical][layer * 2 + side].data(),
                    offset + uint64_t(slot) * page_bytes, page_bytes);
            }
        }
    };
    auto write_page_metadata = [&](uint32_t logical, bool restore) {
        const auto id = page_id(logical);
        uint32_t slot = UINT32_MAX;
        if (restore) {
            for (uint32_t row = 0; row < page_tokens; ++row) {
                llama_kv_pager_write_ticket ticket;
                assert(pager->begin_restore_page(id, id.position_begin + row, ticket) ==
                       llama_kv_pager_write_status::ok);
                if (slot == UINT32_MAX) {
                    slot = ticket.physical_slot;
                    write_cuda_page(logical, slot);
                } else {
                    assert(slot == ticket.physical_slot);
                }
                assert(pager->complete_write(ticket, layers * 2, true) ==
                       llama_kv_pager_write_status::ok);
            }
            return slot;
        }

        // Exercise production batch admission with one- and three-row
        // packets throughout the multi-wrap CUDA generation fixture.
        uint32_t row = 0;
        uint32_t packet_index = 0;
        while (row < page_tokens) {
            const uint32_t packet_size = packet_index++ % 2 == 0 ? 1 : 3;
            std::vector<llama_pos> positions;
            for (uint32_t i = 0; i < packet_size && row + i < page_tokens; ++i) {
                positions.push_back(llama_pos(logical * page_tokens + row + i));
            }
            std::vector<llama_kv_pager_write_ticket> tickets;
            assert(pager->begin_write_batch(0, 1, positions, tickets) ==
                   llama_kv_pager_write_status::ok);
            assert(tickets.size() == positions.size());
            for (const auto & ticket : tickets) {
                if (slot == UINT32_MAX) {
                    slot = ticket.physical_slot;
                    write_cuda_page(logical, slot);
                } else {
                    assert(slot == ticket.physical_slot);
                }
                assert(pager->complete_write(ticket, layers * 2, true) ==
                       llama_kv_pager_write_status::ok);
            }
            row += uint32_t(positions.size());
        }
        return slot;
    };

    // Freeze one historical page, then generate seven pages through the
    // bounded target ring. The accepted frontier is resolved before the
    // rejected final row is removed from page 7.
    (void) write_page_metadata(0, true);
    pager->release_sequence_pins(0);
    assert(pager->seal_ready_pages() == 1);
    const auto history_records = pager->exact_page_records(0);
    const auto history_it = std::find_if(history_records.begin(), history_records.end(),
        [](const auto & record) { return record.id.logical_page == 0; });
    assert(history_it != history_records.end() && history_it->host_valid);
    const uint64_t turn_id = 97;
    assert(pager->transition_turn(0, turn_id, 0,
        llama_kv_pager_turn_phase::query_provisional, 255, 256, history_it->id, 0) ==
        llama_kv_pager_turn_status::ok);
    assert(pager->transition_turn(0, turn_id, 0,
        llama_kv_pager_turn_phase::retrieval_commit, 255, 256, history_it->id, 1,
        { { history_it->id, history_it->content_version } }) == llama_kv_pager_turn_status::ok);
    assert(pager->transition_turn(0, turn_id, 1,
        llama_kv_pager_turn_phase::generating, 255, 256, history_it->id, 1) ==
        llama_kv_pager_turn_status::ok);
    for (uint32_t logical = 1; logical < pages; ++logical) {
        (void) write_page_metadata(logical, false);
        (void) pager->seal_ready_pages();
    }

    const auto rejected_frontier = common_speculative_rollback_frontier_resolve(2045, 2, 1);
    assert(rejected_frontier.valid() && rejected_frontier.accepted_token_count == 2047 &&
        rejected_frontier.rejected_suffix_begin == 2047 &&
        rejected_frontier.rejected_suffix_end == 2048 &&
        rejected_frontier.rejected_draft_tokens == 1);
    const int64_t full_l_draft_frontier = rejected_frontier.accepted_token_count;
    assert(full_l_draft_frontier == 2047);
    const auto full_generated = pager->exact_page_records(0);
    const auto rejected_page = std::find_if(full_generated.begin(), full_generated.end(),
        [](const auto & record) { return record.id.logical_page == 7; });
    assert(rejected_page != full_generated.end() && rejected_page->valid_length == page_tokens);
    const auto rejected_identity = rejected_page->id;
    const auto rejected_version = rejected_page->content_version;
    (void) pager->seal_ready_pages();
    const uint64_t rollback_epoch = pager->residency().epoch();
    assert(pager->mutate({ llama_kv_pager_mutation_kind::remove, 0, -1,
        llama_pos(rejected_frontier.rejected_suffix_begin), -1, 0, 0, rollback_epoch, true }) ==
        llama_kv_pager_write_status::ok);
    const auto accepted_turn = pager->turn_state(0);
    assert(accepted_turn.selected_history.size() == 1 &&
        accepted_turn.selected_history[0].identity == history_it->id &&
        accepted_turn.selected_history[0].content_version == history_it->content_version);
    const auto accepted_page = pager->exact_page_records(0);
    const auto accepted_page_it = std::find_if(accepted_page.begin(), accepted_page.end(),
        [](const auto & record) { return record.id.logical_page == 7; });
    assert(accepted_page_it != accepted_page.end() && accepted_page_it->valid_length == 255 &&
        accepted_page_it->id != rejected_identity &&
        accepted_page_it->content_version > rejected_version && !accepted_page_it->host_valid);
    vbr_selected_page_host_view rejected_host_page;
    assert(!pager->host_catalog()->find_page(rejected_identity, rejected_host_page));
    assert(pager->clear_turn_state(0, turn_id, 1) == llama_kv_pager_turn_status::ok);
    assert(pager->host_inflight_pages() == 0);
    const auto final_records = pager->exact_page_records(0);
    assert(final_records.size() == pages);
    const auto sealed_accepted = std::find_if(final_records.begin(), final_records.end(),
        [](const auto & record) { return record.id.logical_page == 7; });
    assert(sealed_accepted != final_records.end() && sealed_accepted->valid_length == 255 &&
        sealed_accepted->host_valid && sealed_accepted->content_version > rejected_version);
    vbr_selected_page_host_view accepted_host_page;
    assert(pager->host_catalog()->find_page(sealed_accepted->id, accepted_host_page));
    assert(accepted_host_page.page.positions.size() == 255 &&
        accepted_host_page.page.positions.back() == rejected_frontier.rejected_suffix_begin - 1 &&
        std::none_of(accepted_host_page.page.positions.begin(),
            accepted_host_page.page.positions.end(), [&](llama_pos position) {
                return position >= rejected_frontier.rejected_suffix_begin;
            }));
    assert(accepted_host_page.page.units.size() == promotion_fixture::unit_count &&
        std::all_of(accepted_host_page.page.units.begin(), accepted_host_page.page.units.end(),
            [](const auto & unit) { return unit.valid_rows == 255; }));
    std::vector<uint8_t> draft_after(draft_reference.size());
    ggml_backend_tensor_get(draft_storage, draft_after.data(), 0, ggml_nbytes(draft_storage));
    assert(draft_after == draft_reference);
    {
        std::lock_guard<std::mutex> lock(fixture.capture_mutex);
        const auto captures_for = [&](uint32_t logical, uint64_t version) {
            return std::count(fixture.capture_versions.begin(), fixture.capture_versions.end(),
                std::make_pair(logical, version));
        };
        assert(captures_for(7, rejected_version) == 1);
        assert(captures_for(7, sealed_accepted->content_version) == 1);
        for (const auto & record : final_records) {
            if (!record.host_valid) continue;
            assert(captures_for(record.id.logical_page, record.content_version) == 1);
        }
    }
    std::cout << "generation_seal_and_rejection=pass rejected_identity_invalidated=1 "
        << "accepted_length=" << sealed_accepted->valid_length
        << " content_version=" << sealed_accepted->content_version
        << " capture_once_per_version=1 duplicate_d2h_avoided=1\n";
    std::cout << "async_ring_seal_fence_and_version=pass rejected_version_invalidated=1 "
        << "accepted_version_resealed=1 completion_drained=1 host_roundtrip=verified\n";
    const auto & records = final_records;

    // The selector consumes transformed Q and bounds decoded from the exact
    // packed K rows above.  Membership is derived from the live inventory.
    ggml_context * selector_context = ggml_init({ 8u * 1024u * 1024u, nullptr, true });
    assert(selector_context != nullptr);
    ggml_tensor * q = ggml_new_tensor_3d(selector_context, GGML_TYPE_F32,
        head_dim, q_heads, 3);
    ggml_tensor * transformed = ggml_turbo_wht(selector_context, ggml_cont(selector_context, q), 2);
    ggml_tensor * bounds = ggml_new_tensor_4d(selector_context, GGML_TYPE_F16,
        head_dim, 3, kv_heads, pages);
    ggml_tensor * metadata = ggml_new_tensor_2d(selector_context, GGML_TYPE_I64, 4, pages);
    ggml_tensor * membership = ggml_new_tensor_1d(selector_context, GGML_TYPE_I32, pages);
    ggml_tensor * query = ggml_new_tensor_1d(selector_context, GGML_TYPE_I64, 4);
    ggml_tensor * selected = ggml_kv_page_select(selector_context, transformed, bounds,
        metadata, membership, query, 2, cold_capacity, page_tokens, final_query_row, 1, 0);
    const int32_t routed_query_row = selected->op_params[3];
    assert(routed_query_row == int32_t(final_query_row));
    ggml_set_output(selected);
    ggml_cgraph * graph = ggml_new_graph_custom(selector_context, 64, false);
    ggml_build_forward_expand(graph, selected);
    ggml_backend_buffer_t selector_buffer = ggml_backend_alloc_ctx_tensors(selector_context, backend);
    assert(selector_buffer != nullptr);

    std::vector<float> q_host(size_t(3) * q_heads * head_dim, 0.0f);
    for (uint32_t query_row = 0; query_row < 3; ++query_row) {
        for (uint32_t head = 0; head < q_heads; ++head) {
            for (uint32_t d = 0; d < head_dim; ++d) {
                q_host[(size_t(query_row) * q_heads + head) * head_dim + d] =
                    query_row == 0 ? (-3.0f - float((d + head) & 3) * .1f) :
                    query_row == 2 ? (1.0f + float((d + head) & 3) * .1f) : 0.0f;
            }
        }
    }
    std::vector<float> q_layout(q_host.size());
    for (uint32_t row = 0; row < 3; ++row) for (uint32_t head = 0; head < q_heads; ++head)
        std::memcpy(q_layout.data() + (size_t(row) * q_heads + head) * head_dim,
            q_host.data() + (size_t(row) * q_heads + head) * head_dim,
            head_dim * sizeof(float));

    std::vector<ggml_fp16_t> bound_data(size_t(head_dim) * 2 * kv_heads * pages);
    for (uint32_t logical = 0; logical < pages; ++logical) {
        const auto record = std::find_if(records.begin(), records.end(),
            [&](const auto & value) { return value.id.logical_page == logical; });
        assert(record != records.end());
        const auto & source = fixture.bytes[logical][0];
        std::vector<float> decoded(head_dim * kv_heads);
        std::vector<float> lo(head_dim * kv_heads, std::numeric_limits<float>::infinity());
        std::vector<float> hi(head_dim * kv_heads, -std::numeric_limits<float>::infinity());
        for (uint32_t row = 0; row < record->valid_length; ++row) {
            dequantize_row_turbo4_0(reinterpret_cast<const block_turbo4_0 *>(
                source.data() + row * fixture.row_bytes_unit), decoded.data(), head_dim * kv_heads);
            for (uint32_t d = 0; d < decoded.size(); ++d) {
                lo[d] = std::min(lo[d], decoded[d]);
                hi[d] = std::max(hi[d], decoded[d]);
            }
        }
        for (uint32_t head = 0; head < kv_heads; ++head) for (uint32_t d = 0; d < head_dim; ++d) {
            const size_t base = size_t(d + head_dim * (2 * (head + kv_heads * logical)));
            bound_data[base] = ggml_fp32_to_fp16(lo[head * head_dim + d]);
            bound_data[base + head_dim] = ggml_fp32_to_fp16(hi[head * head_dim + d]);
        }
    }
    std::vector<int64_t> metadata_data(size_t(4) * pages);
    std::vector<int32_t> membership_data(pages, 0);
    uint64_t snapshot_generation = 0;
    for (const auto & record : records) snapshot_generation = std::max<uint64_t>(
        snapshot_generation, record.id.page_generation);
    for (uint32_t logical = 0; logical < pages; ++logical) {
        const auto record = std::find_if(records.begin(), records.end(),
            [&](const auto & value) { return value.id.logical_page == logical; });
        assert(record != records.end());
        metadata_data[4 * logical + 0] = record->id.position_begin;
        metadata_data[4 * logical + 1] = record->valid_length;
        metadata_data[4 * logical + 2] = record->id.sequence_generation;
        metadata_data[4 * logical + 3] = record->id.page_generation;
        membership_data[logical] = record != records.end() && record->physical_slot != UINT32_MAX;
    }
    const int64_t final_query_position = rejected_frontier.accepted_token_count;
    const std::array<int64_t, 4> query_data = {
        final_query_position, 1, int64_t(snapshot_generation), 1 };
    assert(final_query_position > 0);
    ggml_backend_tensor_set(q, q_layout.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(bounds, bound_data.data(), 0, ggml_nbytes(bounds));
    ggml_backend_tensor_set(metadata, metadata_data.data(), 0, ggml_nbytes(metadata));
    ggml_backend_tensor_set(membership, membership_data.data(), 0, ggml_nbytes(membership));
    ggml_backend_tensor_set(query, query_data.data(), 0, ggml_nbytes(query));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> transformed_host(q_host.size());
    ggml_backend_tensor_get(transformed, transformed_host.data(), 0, ggml_nbytes(transformed));
    std::vector<int32_t> selector_output(q_heads == 4 ? 4 : 4, -1);
    ggml_backend_tensor_get(selected, selector_output.data(), 0, ggml_nbytes(selected));
    const int32_t winner_index = selector_output[2];
    assert(winner_index >= 0 && winner_index < int32_t(pages));
    const uint32_t winner_logical = uint32_t(winner_index);
    assert(winner_logical > 0 && final_query_position >
        int64_t(winner_logical * page_tokens));
    assert(std::find(membership_data.begin(), membership_data.end(), 0) != membership_data.end());
    assert(membership_data[winner_logical] == 0);

    auto score = [&](uint32_t logical) {
        float best = -std::numeric_limits<float>::infinity();
        for (uint32_t head = 0; head < kv_heads; ++head) {
            for (uint32_t qhead = head * (q_heads / kv_heads);
                 qhead < (head + 1) * (q_heads / kv_heads); ++qhead) {
                float value = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    const float qi = transformed_host[(size_t(2) * q_heads + qhead) * head_dim + d];
                    const size_t base = size_t(d + head_dim * (2 * (head + kv_heads * logical)));
                    const float bound = ggml_fp16_to_fp32(qi >= 0 ? bound_data[base + head_dim] : bound_data[base]);
                    value += qi * bound;
                }
                best = std::max(best, value);
            }
        }
        return best;
    };
    float second = -std::numeric_limits<float>::infinity();
    for (uint32_t logical = 0; logical < pages; ++logical) {
        if (logical == winner_logical || membership_data[logical] != 0) continue;
        second = std::max(second, score(logical));
    }
    const float margin = score(winner_logical) - second;
    assert(std::isfinite(margin) && margin > 0.05f);

    // The positive mailbox path is event-gated. The event is released only
    // after the CUDA graph has completed and the selector output was acquired.
    auto & mailbox = pager->prefetch_candidate_mailbox();
    struct event_state { bool complete = false; } event;
    mailbox.set_backend({ &event,
        [](void * value, uint64_t) noexcept {
            return static_cast<event_state *>(value)->complete
                ? llama_kv_prefetch_mailbox_poll::completed
                : llama_kv_prefetch_mailbox_poll::pending;
        }, nullptr, nullptr });
    uint32_t mailbox_slot = UINT32_MAX;
    llama_kv_prefetch_candidate * candidates = nullptr;
    assert(mailbox.acquire(mailbox_slot, candidates) == llama_kv_prefetch_mailbox_status::ok);
    uint32_t candidate_count = 0;
    for (uint32_t rank = 0; rank < selector_output.size(); ++rank) {
        const int32_t index = selector_output[rank];
        if (index < 0) continue;
        const auto it = std::find_if(records.begin(), records.end(),
            [&](const auto & record) { return record.id.logical_page == uint32_t(index); });
        assert(it != records.end());
        auto & candidate = candidates[candidate_count++];
        candidate.identity = it->id;
        candidate.attention_layer = 0;
        candidate.generation = snapshot_generation;
        candidate.table_epoch = pager->residency().epoch();
        candidate.score = score(uint32_t(index));
        candidate.requested_bytes = it->valid_length * fixture.row_bytes_unit * 2 * layers;
        candidate.content_version = it->content_version;
        candidate.summary_version = pager->routing_summary_content_version(it->id);
        candidate.speculation_generation = 1;
        candidate.selector_rank = rank;
        candidate.query_position = final_query_position;
        candidate.cold = membership_data[index] == 0;
        candidate.rollback_generation = it->id.page_generation;
    }
    assert(candidate_count > 0);
    assert(mailbox.publish_pending(mailbox_slot, candidate_count, snapshot_generation, 4906) ==
           llama_kv_prefetch_mailbox_status::ok);
    // A pending result cannot replace the old valid mapping.
    assert(mailbox.pending_slots() == 1 && mailbox.ready_slots() == 0);
    assert(mailbox.poll(snapshot_generation, pager->residency().epoch()) ==
           llama_kv_prefetch_mailbox_status::ok);
    event.complete = true;
    assert(mailbox.poll(snapshot_generation, pager->residency().epoch()) ==
           llama_kv_prefetch_mailbox_status::ok);
    std::vector<llama_kv_prefetch_candidate> ready;
    assert(mailbox.take_ready(ready) == candidate_count);
    assert(std::any_of(ready.begin(), ready.end(), [&](const auto & value) {
        return value.identity.logical_page == winner_logical && value.cold;
    }));

    const auto cold_it = std::find_if(records.begin(), records.end(),
        [&](const auto & record) { return record.id.logical_page == winner_logical; });
    assert(cold_it != records.end() && cold_it->physical_slot == UINT32_MAX &&
        cold_it->id.logical_page > 0 && cold_it->valid_length == page_tokens);
    const auto committed_candidate = std::find_if(ready.begin(), ready.end(),
        [&](const auto & candidate) {
            return candidate.identity.logical_page == winner_logical && candidate.cold;
        });
    assert(committed_candidate != ready.end());
    std::vector<llama_kv_live_policy_page> live_pages;
    for (const auto & record : records) {
        llama_kv_live_policy_page live;
        live.record = record;
        live.attention_layer = 0;
        live.recent = record.id.logical_page == pages - 1;
        live.current = record.id.logical_page == pages - 1;
        live.recency = live.current ? 100 : (pages - record.id.logical_page);
        live.age = pages - record.id.logical_page;
        live.attention_observed = true;
        live.attention_sample_count = 1;
        live.attention_ema_q = record.id.logical_page == winner_logical ? 1000000 :
            (record.physical_slot != UINT32_MAX ? 1000 : 1);
        live_pages.push_back(live);
    }
    llama_kv_live_policy_boundary boundary;
    boundary.snapshot = pager->residency();
    boundary.hot_capacity = hot_capacity;
    boundary.logical_page_count = pages;
    boundary.pages = live_pages;
    boundary.has_write_page = true;
    boundary.write_page = sealed_accepted->id;
    boundary.previous_target.clear();
    for (const auto & record : records) if (record.physical_slot != UINT32_MAX)
        boundary.previous_target.push_back(record.id);
    boundary.retrieval.status = llama_kv_routing_retrieval_status::ok;
    boundary.retrieval.table_epoch = boundary.snapshot.epoch();
    boundary.retrieval.query_generation = committed_candidate->generation;
    boundary.retrieval.model_identity = cold_it->id.model_identity;
    boundary.retrieval.topology_identity = cold_it->id.topology_identity;
    boundary.retrieval.representation_epoch = cold_it->id.representation_epoch;
    boundary.retrieval.session_generation = cold_it->id.session_generation;
    boundary.retrieval.sequence_generation = cold_it->id.sequence_generation;
    boundary.retrieval.sequence_id = cold_it->id.sequence_id;
    boundary.retrieval.position = committed_candidate->query_position;
    // The selected current page is a resident hit. The two historical cold
    // pages below are the only identities that need H2D.
    boundary.retrieval.selected.push_back({ sealed_accepted->id,
        llama_kv_routing_retrieval_reason::recent, score(sealed_accepted->id.logical_page),
        true, false, 1 });
    boundary.retrieval.selected.push_back({ cold_it->id,
        llama_kv_routing_retrieval_reason::summary, score(winner_logical), true, false, 1 });
    boundary.query_commit.enabled = true;
    const auto prior_turn = pager->turn_state(0);
    boundary.query_commit.turn_id = std::max<uint64_t>(prior_turn.turn_id + 1,
        committed_candidate->generation);
    boundary.query_commit.retrieval_epoch = prior_turn.retrieval_epoch;
    boundary.query_commit.query_generation = committed_candidate->generation;
    boundary.query_commit.table_epoch = boundary.snapshot.epoch();
    boundary.query_commit.query_position = committed_candidate->query_position;
    boundary.query_commit.rollback_generation = std::max_element(records.begin(), records.end(),
        [](const auto & lhs, const auto & rhs) {
            return lhs.id.page_generation < rhs.id.page_generation;
        })->id.page_generation;
    boundary.query_commit.model_identity = cold_it->id.model_identity;
    boundary.query_commit.session_generation = cold_it->id.session_generation;
    boundary.query_commit.sequence_generation = cold_it->id.sequence_generation;
    boundary.query_commit.representation_epoch = cold_it->id.representation_epoch;
    boundary.query_commit.sequence_id = cold_it->id.sequence_id;
    boundary.query_commit.retrieval_budget = hot_capacity - 1;
    boundary.query_commit.generation_budget = 1;
    boundary.query_commit.selected.push_back({ sealed_accepted->id,
        sealed_accepted->content_version, { 0 } });
    boundary.query_commit.selected.push_back({ cold_it->id, cold_it->content_version, { 0 } });
    boundary.policy = llama_kv_policy_release_defaults(hot_capacity);
    boundary.transaction.max_h2d_pages = 2;
    boundary.transaction.staging_capacity = pager->upload_ring()->capacity_bytes();
    // Use the same prepared query target the pager will publish so the upload
    // and immutable mapping have one slot assignment.
    std::vector<llama_kv_page_record> committed_target;
    assert(llama_kv_live_policy_prepare_query_target(boundary, committed_target));
    assert(boundary.snapshot.pages().size() == hot_capacity);
    assert(committed_target.size() == 2);
    const auto resident_hit = std::find_if(committed_target.begin(), committed_target.end(),
        [&](const auto & record) { return record.id == sealed_accepted->id; });
    assert(resident_hit != committed_target.end() &&
        resident_hit->physical_slot == sealed_accepted->physical_slot);
    const auto committed_cold = std::find_if(committed_target.begin(), committed_target.end(),
        [&](const auto & record) { return record.id == cold_it->id; });
    assert(committed_cold != committed_target.end());
    const uint32_t free_slot = committed_cold->physical_slot;
    assert(free_slot < hot_capacity);
    llama_kv_residency_transfer_plan promotion_batch;
    assert(llama_kv_residency_build_transfer_plan(
        llama_kv_residency_transfer_direction::h2d_promotion,
        { promotion_transfer_page(*committed_cold, pager->snapshot(), boundary.snapshot.epoch()) },
        256, {}, promotion_batch));
    boundary.transaction.transfers.push_back(std::move(promotion_batch));
    const auto old_snapshot = pager->residency();
    const auto prior_mapping_intact = [&]() {
        const auto now = pager->residency();
        if (now.epoch() != old_snapshot.epoch() ||
                now.pages().size() != old_snapshot.pages().size()) return false;
        for (size_t i = 0; i < now.pages().size(); ++i) {
            if (now.pages()[i].id != old_snapshot.pages()[i].id ||
                    now.pages()[i].physical_slot != old_snapshot.pages()[i].physical_slot) return false;
        }
        return true;
    };

    auto stale_selection = boundary;
    const auto stale_hit = std::find_if(stale_selection.pages.begin(),
        stale_selection.pages.end(), [&](const auto & page) {
            return page.record.id == sealed_accepted->id;
        });
    assert(stale_hit != stale_selection.pages.end());
    ++stale_hit->record.content_version;
    ++stale_selection.query_commit.selected[0].content_version;
    std::vector<llama_kv_page_record> stale_target;
    assert(!llama_kv_live_policy_prepare_query_target(stale_selection, stale_target));
    const auto stale_selection_result = pager->apply_live_policy(stale_selection);
    assert(stale_selection_result.status ==
        llama_kv_live_policy_status::query_capacity_refused &&
        !stale_selection_result.published && prior_mapping_intact());

    auto stale_query_event = boundary;
    ++stale_query_event.query_commit.query_generation;
    const auto stale_query_event_result = pager->apply_live_policy(stale_query_event);
    assert(stale_query_event_result.status ==
        llama_kv_live_policy_status::query_capacity_refused &&
        !stale_query_event_result.published && prior_mapping_intact());

    // Pinning every logical page forces mandatory capacity overflow, so no
    // victim can be selected for the cold retrieval candidate.
    auto all_pinned = boundary;
    for (auto & page : all_pinned.pages) page.application_pin = true;
    const auto pinned_result = pager->apply_live_policy(all_pinned);
    assert(pinned_result.status == llama_kv_live_policy_status::all_pinned ||
           pinned_result.status == llama_kv_live_policy_status::mandatory_overflow ||
           pinned_result.status == llama_kv_live_policy_status::query_capacity_refused);
    assert(!pinned_result.published && prior_mapping_intact());

    // An invalid transfer plan is rejected before publication and likewise
    // leaves the old target graph's table authoritative.
    auto rejected_transfer = boundary;
    rejected_transfer.transaction.transfers.front().runs.clear();
    const auto rejected_result = pager->apply_live_policy(rejected_transfer);
    if (rejected_result.status != llama_kv_live_policy_status::transaction_failed) {
        std::fprintf(stderr, "invalid committed transfer status=%d tx=%d phase=%d published=%d\n",
            int(rejected_result.status), int(rejected_result.transaction.status),
            int(rejected_result.transaction.failed_phase), int(rejected_result.published));
    }
    assert(rejected_result.status == llama_kv_live_policy_status::transaction_failed);
    assert(!rejected_result.published && prior_mapping_intact());

    // A transfer execution failure after reservation still leaves the old
    // mapping usable and does not publish either cold page.
    auto failed_upload = boundary;
    failed_upload.transaction.transfers.front().runs.front().host_offset = UINT64_MAX;
    const auto failed_upload_result = pager->apply_live_policy(failed_upload);
    assert(failed_upload_result.status == llama_kv_live_policy_status::transaction_failed &&
        !failed_upload_result.published && prior_mapping_intact());

    const auto promotion = pager->apply_live_policy(boundary);
    if (!(promotion.status == llama_kv_live_policy_status::committed && promotion.published)) {
        std::fprintf(stderr, "promotion failed status=%d published=%d tx=%d phase=%d target=%zu\n",
            int(promotion.status), int(promotion.published), int(promotion.transaction.status),
            int(promotion.transaction.failed_phase), promotion.target_pages.size());
        return 1;
    }
    assert(promotion.status == llama_kv_live_policy_status::committed && promotion.published);
    assert(promotion.transaction.h2d_counters.copied_useful_bytes > 0);
    assert(promotion.transaction.h2d_counters.queued > 0);
    assert(promotion.target_pages.size() == 2 &&
        promotion.transaction.loaded_pages == 1 &&
        promotion.transaction.h2d_counters.copied_useful_bytes ==
            promotion_batch.useful_bytes &&
        promotion.transaction.d2h_counters.copied_useful_bytes == 0);
    assert(pager->residency().pages().size() == 2 &&
        promotion.transaction.dropped_pages >= 1);
    for (const auto & old : boundary.snapshot.pages()) {
        if (std::none_of(promotion.target_pages.begin(), promotion.target_pages.end(),
                [&](const auto & kept) { return kept.id == old.id; })) {
            assert(std::none_of(pager->residency().pages().begin(),
                    pager->residency().pages().end(), [&](const auto & resident) {
                return resident.id == old.id;
            }));
        }
    }
    const auto published_hit = std::find_if(promotion.target_pages.begin(),
        promotion.target_pages.end(), [&](const auto & record) {
            return record.id == sealed_accepted->id;
        });
    assert(published_hit != promotion.target_pages.end() &&
        published_hit->physical_slot == sealed_accepted->physical_slot);
    std::cout << "selection_diff_async_batch_and_publication=pass selected=2 resident_hits=1 "
        << "cold_misses=1 loaded=1 transfer_bytes="
        << promotion.transaction.h2d_counters.copied_useful_bytes
        << " historical_d2h_bytes="
        << promotion.transaction.d2h_counters.copied_useful_bytes
        << " stale_resident_version_rejected=1 stale_query_event_rejected=1 "
        << "transfer_failure_rollback=1 atomic_publish=1\n";
    const auto evicted_clean = std::find_if(promotion.decisions.begin(),
        promotion.decisions.end(), [](const auto & decision) {
            return decision.victim;
        });
    assert(evicted_clean != promotion.decisions.end());
    vbr_selected_page_host_view evicted_host;
    assert(pager->host_catalog()->find_page(evicted_clean->id, evicted_host));
    assert(!evicted_host.page.units.empty() && evicted_host.page.units[0].bytes);
    std::vector<uint8_t> retained_host_bytes(
        evicted_host.page.units[0].bytes->size());
    assert(evicted_host.page.units[0].bytes->read(
        0, retained_host_bytes.data(), retained_host_bytes.size()));
    assert(retained_host_bytes == fixture.bytes[evicted_clean->id.logical_page][0]);
    assert(std::any_of(promotion.decisions.begin(), promotion.decisions.end(),
        [&](const auto & decision) {
            return decision.victim && decision.id.logical_page != winner_logical;
        }));
    assert(std::any_of(promotion.target_pages.begin(), promotion.target_pages.end(),
        [&](const auto & record) { return record.id == cold_it->id; }));
    assert(pager->residency().epoch() != old_snapshot.epoch());

    // Exercise the post-publication query rewind against the CUDA promoted
    // mapping. A range edit at the query frontier may advance the mutable
    // page-table epoch, but it must retain the selected historical identity
    // and its authenticated content version.
    const auto query_frontier = std::max_element(records.begin(), records.end(),
        [](const auto & lhs, const auto & rhs) {
            return lhs.id.position_end < rhs.id.position_end;
        })->id;
    assert(committed_candidate->query_position <=
        uint64_t(std::numeric_limits<llama_pos>::max()));
    const llama_pos query_position =
        static_cast<llama_pos>(committed_candidate->query_position);
    assert(pager->transition_turn(0, boundary.query_commit.turn_id,
        boundary.query_commit.retrieval_epoch,
        llama_kv_pager_turn_phase::query_provisional,
        query_position, query_position + 1, query_frontier, 0) ==
        llama_kv_pager_turn_status::ok);
    assert(pager->transition_turn(0, boundary.query_commit.turn_id,
        boundary.query_commit.retrieval_epoch,
        llama_kv_pager_turn_phase::retrieval_commit,
        query_position, query_position + 1, query_frontier,
        boundary.query_commit.retrieval_epoch,
        { { cold_it->id, cold_it->content_version } }) ==
        llama_kv_pager_turn_status::ok);
    assert(pager->transition_turn(0, boundary.query_commit.turn_id,
        boundary.query_commit.retrieval_epoch + 1,
        llama_kv_pager_turn_phase::query_replay,
        query_position, query_position + 1, query_frontier,
        boundary.query_commit.retrieval_epoch) == llama_kv_pager_turn_status::ok);
    assert(pager->transition_turn(0, boundary.query_commit.turn_id,
        boundary.query_commit.retrieval_epoch + 1,
        llama_kv_pager_turn_phase::generating,
        query_position, query_position + 1, query_frontier,
        boundary.query_commit.retrieval_epoch) == llama_kv_pager_turn_status::ok);
    std::vector<llama_pos> next_packet = { query_position - 1, query_position,
        query_position + 1 };
    std::vector<llama_kv_pager_write_ticket> packet_tickets;
    const auto before_packet = pager->residency();
    std::fprintf(stderr, "query_commit_packet_ownership_before=");
    for (const auto & record : before_packet.pages()) {
        std::fprintf(stderr, "[%u@%u:pin%u:host%d:dirty%d]",
            record.id.logical_page, record.physical_slot, record.pin_count,
            int(record.host_valid), int(record.dirty));
    }
    std::fprintf(stderr, " capacity=%u logical=%zu\n", hot_capacity,
        before_packet.pages().size());
    const auto packet_status = pager->begin_write_batch(0,
        boundary.query_commit.sequence_generation, next_packet, packet_tickets);
    std::fprintf(stderr, "query_commit_packet_status=%s tickets=%zu\n",
        llama_kv_pager_write_status_name(packet_status), packet_tickets.size());
    assert(packet_status == llama_kv_pager_write_status::ok);
    assert(packet_tickets.size() == next_packet.size());
    for (auto it = packet_tickets.rbegin(); it != packet_tickets.rend(); ++it) {
        assert(pager->cancel_write(*it) == llama_kv_pager_write_status::ok);
    }
    const auto after_packet = pager->residency();
    assert(after_packet.pages().size() == before_packet.pages().size());
    for (size_t i = 0; i < before_packet.pages().size(); ++i) {
        assert(after_packet.pages()[i].id == before_packet.pages()[i].id &&
            after_packet.pages()[i].physical_slot == before_packet.pages()[i].physical_slot);
    }
    llama_kv_pager_write_ticket historical_ticket;
    assert(pager->begin_write(0, boundary.query_commit.sequence_generation,
        llama_pos(cold_it->id.position_begin), historical_ticket) !=
        llama_kv_pager_write_status::ok);
    std::cout << "query_commit_guarantees_generation_headroom=pass full_before="
        << hot_capacity << " committed=" << pager->residency().pages().size()
        << " next_packet_pages=2 boundary_page_writable=1 frozen_history=1\n";
    const uint64_t rewind_epoch = pager->residency().epoch();
    assert(pager->mutate({ llama_kv_pager_mutation_kind::remove, 0, -1,
        query_position, -1, 0, 0, rewind_epoch }) ==
        llama_kv_pager_write_status::ok);
    const auto retained_turn = pager->turn_state(0);
    assert(retained_turn.phase == llama_kv_pager_turn_phase::generating);
    assert(retained_turn.selected_history.size() == 1 &&
        retained_turn.selected_history[0].identity == cold_it->id &&
        retained_turn.selected_history[0].content_version == cold_it->content_version);
    const auto after_rewind = pager->residency();
    assert(std::any_of(after_rewind.pages().begin(), after_rewind.pages().end(),
        [&](const auto & page) { return page.id == cold_it->id; }));

    uint64_t host_checksum = 0, device_checksum = 0, generation = 0, content = 0;
    uint32_t physical_slot = UINT32_MAX;
    assert(pager->test_page_checksums(winner_logical, host_checksum, device_checksum,
        physical_slot, generation, content));
    assert(host_checksum == device_checksum && physical_slot == free_slot &&
        content == cold_it->content_version);
    std::vector<llama_kv_page_record> target = promotion.target_pages;
    std::sort(target.begin(), target.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.id.logical_page < rhs.id.logical_page;
    });
    std::vector<llama_kv_page_id> target_ids;
    for (const auto & record : target) target_ids.push_back(record.id);
    const uint32_t target_rows = uint32_t(target_ids.size()) * page_tokens;
    auto k_reference = pack_pages(fixture, target_ids, 0, 0);
    auto v_reference = pack_pages(fixture, target_ids, 0, 1);
    auto k_device = read_device_pages(storage, pager->snapshot(), target, 0, 0, target_rows);
    auto v_device = read_device_pages(storage, pager->snapshot(), target, 0, 1, target_rows);
    assert(k_device == k_reference && v_device == v_reference);
    auto reference_output = run_dense_fa(backend, q_host, k_reference, v_reference, target_rows);
    auto consumed_output = run_dense_fa(backend, q_host, k_device, v_device, target_rows);
    assert(reference_output.values.size() == consumed_output.values.size());
    float max_error = 0.0f;
    for (size_t i = 0; i < reference_output.values.size(); ++i)
        max_error = std::max(max_error, std::abs(reference_output.values[i] - consumed_output.values[i]));
    assert(max_error < 2e-3f);
    auto perturbed = v_reference;
    const auto winner_target = std::find_if(target.begin(), target.end(),
        [&](const auto & record) { return record.id == cold_it->id; });
    assert(winner_target != target.end());
    const uint32_t winner_order = uint32_t(winner_target - target.begin());
    for (uint32_t row = 0; row < page_tokens; ++row)
        perturbed[(size_t(0) * target_rows + winner_order * page_tokens + row) * fixture.row_bytes_head] ^= 0x0f;
    auto perturbed_output = run_dense_fa(backend, q_host, k_reference, perturbed, target_rows);
    float delta = 0.0f;
    for (size_t i = 0; i < perturbed_output.values.size(); ++i)
        delta = std::max(delta, std::abs(perturbed_output.values[i] - reference_output.values[i]));
    assert(delta > 1e-5f);
    free_dense(perturbed_output);
    free_dense(consumed_output);
    free_dense(reference_output);

    // Diagnostic mode confirms resident inventory remains selectable while
    // cold-page intervals are unusable, without reading query tensors back.
    const auto unbounded = std::find(membership_data.begin(), membership_data.end(), 0);
    assert(unbounded != membership_data.end());
    const uint32_t unbounded_page = uint32_t(unbounded - membership_data.begin());
    for (uint32_t logical = 0; logical < pages; ++logical) {
        if (membership_data[logical] != 0) continue;
        for (uint32_t head = 0; head < kv_heads; ++head) {
            for (uint32_t d = 0; d < head_dim; ++d) {
                const size_t base = size_t(d + head_dim * 2 * (head + kv_heads * logical));
                bound_data[base] = ggml_fp32_to_fp16(std::numeric_limits<float>::quiet_NaN());
                bound_data[base + head_dim] = ggml_fp32_to_fp16(std::numeric_limits<float>::quiet_NaN());
            }
        }
    }
    ggml_backend_tensor_set(bounds, bound_data.data(), 0, ggml_nbytes(bounds));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::fill(selector_output.begin(), selector_output.end(), -1);
    ggml_backend_tensor_get(selected, selector_output.data(), 0, ggml_nbytes(selected));
    assert(selector_output[0] >= 0 && selector_output[0] < int32_t(pages));
    assert(membership_data[selector_output[0]] != 0);

    // Half precision catalogue intervals can conservatively overflow to
    // +/-infinity for unusually large finite K values. Such an interval has
    // an unbounded upper score, so it must remain rankable instead of
    // turning every cold candidate into the selector's -1 sentinel.
    for (uint32_t head = 0; head < kv_heads; ++head) {
        for (uint32_t d = 0; d < head_dim; ++d) {
            const size_t base = size_t(d + head_dim * 2 * (head + kv_heads * unbounded_page));
            bound_data[base] = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
            bound_data[base + head_dim] = ggml_fp32_to_fp16(std::numeric_limits<float>::infinity());
        }
    }
    ggml_backend_tensor_set(bounds, bound_data.data(), 0, ggml_nbytes(bounds));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::fill(selector_output.begin(), selector_output.end(), -1);
    ggml_backend_tensor_get(selected, selector_output.data(), 0, ggml_nbytes(selected));
    assert(std::find(selector_output.begin() + 2, selector_output.end(),
            int32_t(unbounded_page)) != selector_output.end());

    std::fprintf(stdout,
        "cuda_deterministic_selector_transaction_integration=pass mature_fa_consumption_parity=pass "
        "seed=0x%llx winner_logical=%u margin=%.6f copy_bytes=%llu "
        "published_epoch=%llu owner_generation=%llu checksum=0x%llx "
        "reference_tolerance=0.002 perturb_delta=%.6f n_q=1,2,3 prefill=pass "
        "query_checkpoint_restore_and_mapping=pass\n",
        (unsigned long long) seed, winner_logical, margin,
        (unsigned long long) promotion.transaction.h2d_counters.copied_useful_bytes,
        (unsigned long long) promotion.published_epoch,
        (unsigned long long) generation, (unsigned long long) host_checksum, delta);

    std::fprintf(stdout, "generation_ring_cuda_integration=pass generated_pages=%u "
        "target_capacity=%u generation_capacity=%u rejected_suffix=1 "
        "cold_generated_winner=%u separate_full_l_draft=1 "
        "clean_eviction_host_bytes_retained=1\n",
        pages - 1, hot_capacity, planned.generation_pages, winner_logical);

    ggml_backend_buffer_free(selector_buffer);
    ggml_free(selector_context);
    ggml_backend_buffer_free(draft_buffer);
    ggml_free(draft_context);
    ggml_backend_buffer_free(storage_buffer);
    ggml_free(storage_context);
    ggml_backend_free(backend);
    return 0;
}

} // namespace

int main() {
    assert(llama_kv_pager_selector_q_shape_gate(3, head_dim, q_heads, 2, 1, 3) ==
        llama_kv_pager_selector_gate::query_rows_mismatch);
    assert(llama_kv_pager_selector_q_shape_gate(3, head_dim, q_heads, 3, 1, 3) ==
        llama_kv_pager_selector_gate::callback_matched);
    assert(std::strcmp(llama_kv_pager_selector_gate_name(
        llama_kv_pager_selector_gate::query_rows_mismatch), "query_rows_mismatch") == 0);
    return run_proof();
}
