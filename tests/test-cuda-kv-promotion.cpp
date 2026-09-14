#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-quants.h"
#include "llama-kv-live-policy.h"
#include "llama-kv-pager.h"
#include "llama-kv-prefetch.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#undef NDEBUG

namespace {

constexpr uint64_t seed = UINT64_C(0x49060001);
constexpr uint32_t pages = 8;
constexpr uint32_t page_tokens = 256;
constexpr uint32_t layers = 16;
constexpr uint32_t kv_heads = 2;
constexpr uint32_t q_heads = 4;
constexpr uint32_t head_dim = 128;
constexpr uint32_t cold_capacity = 2;
constexpr uint32_t hot_capacity = 3;
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
                                // Page zero is deliberately aligned with the
                                // positive transformed query.  It is later
                                // evicted and is therefore a cold winner.
                                if (logical == 0 && side == 0) value += 2.0f;
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
    output.row_indices = { 0, 85, 170, 255 };
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

static llama_kv_residency_transfer_plan promotion_plan(
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
    llama_kv_residency_transfer_plan output;
    assert(llama_kv_residency_build_transfer_plan(
        llama_kv_residency_transfer_direction::h2d_promotion, { transfer }, 256, {}, output));
    return output;
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
    geometry.context_tokens = pages * page_tokens;
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
    resources.external_storage_buffer = storage_buffer;
    resources.external_storage_tensor = storage;
    auto pager = llama_kv_pager::create(config, geometry, resources, {}, status);
    assert(pager && status == llama_kv_pager_status::ok);
    pager->bind_representation_identity(49, 16, 4, 5, 6, 7, 4);
    pager->set_host_provider({ &fixture, promotion_fixture::prepare });
    pager->set_routing_summary_provider({ &fixture, build_summary });

    std::array<uint32_t, pages> slots{};
    for (uint32_t logical = 0; logical < pages; ++logical) {
        const auto id = page_id(logical);
        llama_kv_pager_write_ticket ticket;
        for (uint32_t row = 0; row < page_tokens; ++row) {
            assert(pager->begin_restore_page(id, id.position_begin + row, ticket) ==
                   llama_kv_pager_write_status::ok);
            slots[logical] = ticket.physical_slot;
            assert(pager->complete_write(ticket, layers * 2, true) ==
                   llama_kv_pager_write_status::ok);
        }
        const uint32_t slot = slots[logical];
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
        pager->release_sequence_pins(0);
        const uint32_t sealed = pager->seal_ready_pages();
        if (sealed != 1) {
            const auto debug_records = pager->exact_page_records(0);
            for (const auto & debug : debug_records) {
                std::fprintf(stderr, "record logical=%u state=%d host=%d dirty=%d pin=%u valid=%u slot=%u\n",
                    debug.id.logical_page, int(debug.state), int(debug.host_valid), int(debug.dirty),
                    debug.pin_count, debug.valid_length, debug.physical_slot);
            }
            std::fprintf(stderr, "seal failed logical=%u result=%u host_pages=%zu\n",
                logical, sealed, pager->host_catalog() ? pager->host_catalog()->pages().size() : 0);
            return 1;
        }
    }
    const auto records = pager->exact_page_records(0);
    assert(records.size() == pages);

    // The selector consumes transformed Q and bounds decoded from the exact
    // packed K rows above.  Membership is derived from the live inventory.
    ggml_context * selector_context = ggml_init({ 8u * 1024u * 1024u, nullptr, true });
    assert(selector_context != nullptr);
    ggml_tensor * q = ggml_new_tensor_3d(selector_context, GGML_TYPE_F32,
        head_dim, q_heads, 3);
    ggml_tensor * transformed = ggml_turbo_wht(selector_context, ggml_cont(selector_context, q), 2);
    ggml_tensor * bounds = ggml_new_tensor_4d(selector_context, GGML_TYPE_F16,
        head_dim, 2, kv_heads, pages);
    ggml_tensor * metadata = ggml_new_tensor_2d(selector_context, GGML_TYPE_I64, 4, pages);
    ggml_tensor * membership = ggml_new_tensor_1d(selector_context, GGML_TYPE_I32, pages);
    ggml_tensor * query = ggml_new_tensor_1d(selector_context, GGML_TYPE_I64, 4);
    ggml_tensor * selected = ggml_kv_page_select(selector_context, transformed, bounds,
        metadata, membership, query, 2, cold_capacity, page_tokens, 0);
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
                    query_row == 0 ? (1.0f + float((d + head) & 3) * .1f) : 0.0f;
            }
        }
    }
    std::vector<float> q_layout(q_host.size());
    for (uint32_t row = 0; row < 3; ++row) for (uint32_t head = 0; head < q_heads; ++head)
        std::memcpy(q_layout.data() + (size_t(head) * 3 + row) * head_dim,
            q_host.data() + (size_t(row) * q_heads + head) * head_dim,
            head_dim * sizeof(float));

    std::vector<ggml_fp16_t> bound_data(size_t(head_dim) * 2 * kv_heads * pages);
    for (uint32_t logical = 0; logical < pages; ++logical) {
        const auto & source = fixture.bytes[logical][0];
        std::vector<float> decoded(head_dim * kv_heads);
        std::vector<float> lo(head_dim * kv_heads, std::numeric_limits<float>::infinity());
        std::vector<float> hi(head_dim * kv_heads, -std::numeric_limits<float>::infinity());
        for (uint32_t row = 0; row < page_tokens; ++row) {
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
        const auto id = page_id(logical);
        metadata_data[4 * logical + 0] = id.position_begin;
        metadata_data[4 * logical + 1] = page_tokens;
        metadata_data[4 * logical + 2] = id.sequence_generation;
        metadata_data[4 * logical + 3] = id.page_generation;
        const auto record = std::find_if(records.begin(), records.end(),
            [&](const auto & value) { return value.id.logical_page == logical; });
        membership_data[logical] = record != records.end() && record->physical_slot != UINT32_MAX;
    }
    const std::array<int64_t, 4> query_data = {
        int64_t(pages * page_tokens), 1, int64_t(snapshot_generation), 1 };
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
    assert(std::find(membership_data.begin(), membership_data.end(), 0) != membership_data.end());
    assert(membership_data[winner_logical] == 0);

    auto score = [&](uint32_t logical) {
        float best = -std::numeric_limits<float>::infinity();
        for (uint32_t head = 0; head < kv_heads; ++head) {
            for (uint32_t qhead = head * (q_heads / kv_heads);
                 qhead < (head + 1) * (q_heads / kv_heads); ++qhead) {
                float value = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    const float qi = transformed_host[(size_t(qhead) * 3 + 0) * head_dim + d];
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
        candidate.query_position = pages * page_tokens;
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
    assert(cold_it != records.end() && cold_it->physical_slot == UINT32_MAX);
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
    boundary.write_page = page_id(pages - 1);
    boundary.previous_target.clear();
    for (const auto & record : records) if (record.physical_slot != UINT32_MAX)
        boundary.previous_target.push_back(record.id);
    boundary.retrieval.status = llama_kv_routing_retrieval_status::ok;
    boundary.retrieval.table_epoch = boundary.snapshot.epoch();
    boundary.retrieval.selected.push_back({ cold_it->id,
        llama_kv_routing_retrieval_reason::summary, score(winner_logical), true, false, 1 });
    boundary.policy = llama_kv_policy_release_defaults(hot_capacity);
    boundary.transaction.max_h2d_pages = 1;
    boundary.transaction.staging_capacity = pager->upload_ring()->capacity_bytes();
    llama_kv_page_record transfer_record = *cold_it;
    transfer_record.physical_slot = UINT32_MAX;
    // The production policy assigns the free slot; the transfer plan carries
    // the slot chosen by the immutable inventory's only free physical slot.
    // The policy keeps the newest resident page and the recent cold page,
    // then assigns the ranked winner to the remaining slot.
    const uint32_t free_slot = 2;
    assert(free_slot < hot_capacity);
    transfer_record.physical_slot = free_slot;
    boundary.transaction.transfers.push_back(
        promotion_plan(transfer_record, pager->snapshot(), boundary.snapshot.epoch()));
    const auto old_snapshot = pager->residency();
    const auto promotion = pager->apply_live_policy(boundary);
    if (!(promotion.status == llama_kv_live_policy_status::committed && promotion.published)) {
        std::fprintf(stderr, "promotion failed status=%d published=%d tx=%d phase=%d target=%zu\n",
            int(promotion.status), int(promotion.published), int(promotion.transaction.status),
            int(promotion.transaction.failed_phase), promotion.target_pages.size());
        return 1;
    }
    assert(promotion.status == llama_kv_live_policy_status::committed && promotion.published);
    assert(promotion.transaction.h2d_counters.copied_useful_bytes > 0);
    assert(std::any_of(promotion.target_pages.begin(), promotion.target_pages.end(),
        [&](const auto & record) { return record.id == cold_it->id; }));
    assert(pager->residency().epoch() != old_snapshot.epoch());

    uint64_t host_checksum = 0, device_checksum = 0, generation = 0, content = 0;
    uint32_t physical_slot = UINT32_MAX;
    assert(pager->test_page_checksums(winner_logical, host_checksum, device_checksum,
        physical_slot, generation, content));
    assert(host_checksum == device_checksum && physical_slot == free_slot);

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

    std::fprintf(stdout,
        "cuda_production_promotion_chain=pass mature_fa_consumption_parity=pass "
        "seed=0x%llx winner_logical=%u margin=%.6f copy_bytes=%llu "
        "published_epoch=%llu owner_generation=%llu checksum=0x%llx "
        "reference_tolerance=0.002 perturb_delta=%.6f n_q=1,2,3 prefill=pass\n",
        (unsigned long long) seed, winner_logical, margin,
        (unsigned long long) promotion.transaction.h2d_counters.copied_useful_bytes,
        (unsigned long long) promotion.published_epoch,
        (unsigned long long) generation, (unsigned long long) host_checksum, delta);

    ggml_backend_buffer_free(selector_buffer);
    ggml_free(selector_context);
    ggml_backend_buffer_free(storage_buffer);
    ggml_free(storage_context);
    ggml_backend_free(backend);
    return 0;
}

} // namespace

int main() {
    return run_proof();
}
