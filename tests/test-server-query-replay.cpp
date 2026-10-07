#include "server-context.h"

#include "common.h"
#include "llama-context.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama-kv-residency.h"
#include "llama-kv-router-job.h"
#include "llama-kv-pager.h"
#include "speculative.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

struct options {
    std::string model;
    std::string output;
    std::string router = "legacy";
    bool fresh_only = false;
    uint32_t batch = 1024;
    uint32_t ubatch = 256;
    uint32_t context = 8192;
    uint32_t hot_tokens = 4096;
    bool owner_only = false;
    bool delayed_turn_only = false;
    bool generation_parity = false;
    bool question_layer_parity = false;
    uint32_t verify_width = 3;
};

struct question_tensor_image {
    ggml_type type = GGML_TYPE_COUNT;
    std::array<int64_t, GGML_MAX_DIMS> ne{};
    std::array<size_t, GGML_MAX_DIMS> nb{};
    std::vector<uint8_t> bytes;
};

struct question_tensor_capture {
    static constexpr size_t max_bytes = 24 * 1024 * 1024;
    bool enabled = false;
    bool overflow = false;
    size_t bytes_used = 0;
    std::map<std::string, question_tensor_image> tensors;

    void begin() {
        overflow = false;
        bytes_used = 0;
        tensors.clear();
        enabled = true;
    }

    void end() { enabled = false; }
};

bool question_tensor_name(const char * name) {
    if (name == nullptr) return false;
    static const std::array<const char *, 17> names = {
        "attn_norm-0", "linear_attn_qkv_mixed-0", "conv_input-0",
        "conv_output_raw-0", "state_predelta-0", "l_out-0",
        "attn_norm-1", "l_out-1", "attn_norm-2", "l_out-2",
        "attn_norm-3", "l_out-3", "Qcur-3", "Kcur-3", "Vcur-3",
        "kqv_out_direct-3", "attn_norm-4",
    };
    if (std::find(names.begin(), names.end(), name) != names.end()) return true;
    return std::strcmp(name, "l_out-4") == 0;
}

bool capture_question_tensor(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * capture = static_cast<question_tensor_capture *>(user_data);
    if (capture == nullptr || !capture->enabled || tensor == nullptr ||
            !question_tensor_name(tensor->name)) return false;
    if (ask) {
        if (!ggml_is_contiguous(tensor) ||
                (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16)) return false;
        const size_t bytes = ggml_nbytes(tensor);
        if (bytes == 0 || bytes > question_tensor_capture::max_bytes - capture->bytes_used) {
            capture->overflow = true;
            return false;
        }
        return true;
    }

    if (!ggml_is_contiguous(tensor) ||
            (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16)) return true;
    const size_t bytes = ggml_nbytes(tensor);
    if (bytes == 0 || bytes > question_tensor_capture::max_bytes - capture->bytes_used ||
            capture->tensors.find(tensor->name) != capture->tensors.end()) {
        capture->overflow = true;
        return true;
    }
    question_tensor_image image;
    image.type = tensor->type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        image.ne[size_t(i)] = tensor->ne[i];
        image.nb[size_t(i)] = tensor->nb[i];
    }
    image.bytes.resize(bytes);
    if (ggml_backend_buffer_is_host(tensor->buffer)) {
        std::memcpy(image.bytes.data(), tensor->data, bytes);
    } else {
        ggml_backend_tensor_get(tensor, image.bytes.data(), 0, bytes);
    }
    capture->bytes_used += bytes;
    capture->tensors.emplace(tensor->name, std::move(image));
    return true;
}

bool number(const char * raw, uint32_t & value) {
    if (!raw || !*raw || *raw == '-') return false;
    char * end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (end == raw || *end || parsed == 0 || parsed > UINT32_MAX) return false;
    value = uint32_t(parsed);
    return true;
}

bool parse_options(int argc, char ** argv, options & out) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) out.model = argv[++i];
        else if (arg == "--router" && i + 1 < argc) out.router = argv[++i];
        else if (arg == "--fresh-only") out.fresh_only = true;
        else if (arg == "--B" && i + 1 < argc && number(argv[++i], out.batch)) {}
        else if (arg == "--U" && i + 1 < argc && number(argv[++i], out.ubatch)) {}
        else if (arg == "--L" && i + 1 < argc && number(argv[++i], out.context)) {}
        else if (arg == "--H" && i + 1 < argc && number(argv[++i], out.hot_tokens)) {}
        else if (arg == "--owner-only") out.owner_only = true;
        else if (arg == "--delayed-turn-only") out.delayed_turn_only = true;
        else if (arg == "--generation-parity") out.generation_parity = true;
        else if (arg == "--question-layer-parity") out.question_layer_parity = true;
        else if (arg == "--verify-width3") out.verify_width = 3;
        else if (arg == "--verify-width12") out.verify_width = 12;
        else if (arg == "--output" && i + 1 < argc) out.output = argv[++i];
        else return false;
    }
    return !out.model.empty() && (out.router == "legacy" || out.router == "probe-rerank") &&
        out.ubatch <= out.batch &&
        out.context > out.hot_tokens && out.hot_tokens >= 512 &&
        (!out.question_layer_parity || out.generation_parity) &&
        (out.verify_width == 3 || out.verify_width == 12) &&
        out.hot_tokens % VBR_GENERATION_PAGE_CELLS == 0;
}

bool decode_one(llama_context * ctx, common_speculative * spec,
        llama_token token, llama_pos pos, llama_seq_id sequence_id = 0) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
        llama_batch_free(batch);
        return false;
    }
    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = pos;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = sequence_id;
    batch.logits[0] = true;
    const int rc = llama_decode(ctx, batch);
    if (rc == 0) {
        llama_synchronize(ctx);
        if (spec) common_speculative_process(spec, batch);
    }
    llama_batch_free(batch);
    return rc == 0;
}

bool decode_prefix(llama_context * ctx, common_speculative * spec,
        const llama_tokens & tokens, llama_seq_id sequence_id = 0) {
    const uint32_t count = uint32_t(tokens.size());
    uint32_t offset = 0;
    while (offset < count) {
        const uint32_t n = std::min<uint32_t>(256, count - offset);
        llama_batch batch = llama_batch_init(int32_t(n), 0, 1);
        if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            return false;
        }
        batch.n_tokens = int32_t(n);
        for (uint32_t i = 0; i < n; ++i) {
            batch.token[i] = tokens[offset + i];
            batch.pos[i] = llama_pos(offset + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = sequence_id;
            batch.logits[i] = offset + i + 1 == count;
        }
        const int rc = llama_decode(ctx, batch);
        if (rc == 0) {
            llama_synchronize(ctx);
            if (spec) common_speculative_process(spec, batch);
        }
        llama_batch_free(batch);
        if (rc != 0) return false;
        offset += n;
    }
    return true;
}

bool decode_prefix(llama_context * ctx, common_speculative * spec,
        uint32_t count, llama_seq_id sequence_id = 0) {
    return decode_prefix(ctx, spec, llama_tokens(count, llama_token(1)), sequence_id);
}

bool seq_image(llama_context * ctx, std::vector<uint8_t> & out) {
    const size_t size = llama_state_seq_get_size_ext(
            ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    out.resize(size);
    return size != 0 && llama_state_seq_get_data_ext(
            ctx, out.data(), size, 0,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size;
}

bool seq_range_image(llama_context * ctx, std::vector<uint8_t> & out,
        llama_pos begin, llama_pos end) {
    const size_t size = llama_state_seq_get_size_range(ctx, 0, begin, end);
    out.resize(size);
    return size != 0 && llama_state_seq_get_data_range(
            ctx, out.data(), size, 0, begin, end) == size;
}

struct attention_page_image {
    uint32_t logical_page = 0;
    llama_pos position_begin = -1;
    uint32_t span = 0;
    uint32_t physical_slot = UINT32_MAX;
    uint32_t page_generation = 0;
    uint64_t content_version = 0;
    uint32_t page_tokens = 0;
    uint32_t physical_page_count = 0;
    uint64_t slot_bytes = 0;
    std::string representation;
    std::vector<uint8_t> bytes;
    struct layer_slice {
        uint32_t model_layer = UINT32_MAX;
        uint64_t k_source_offset = 0;
        uint64_t v_source_offset = 0;
        uint64_t k_row_bytes = 0;
        uint64_t v_row_bytes = 0;
        uint64_t k_page_bytes = 0;
        uint64_t v_page_bytes = 0;
        uint64_t k_bytes = 0;
        uint64_t v_bytes = 0;
        uint64_t k_output_offset = 0;
        uint64_t v_output_offset = 0;
    };
    std::vector<layer_slice> layer_slices;
};

bool target_attention_prefix_page_images(llama_context * ctx,
        const llama_kv_pager_metrics_snapshot & metrics, llama_pos end,
        const llama_kv_pager_geometry & geometry, uint32_t physical_page_count,
        std::vector<attention_page_image> & output) {
    if (ctx == nullptr || end <= 0 || metrics.page_tokens == 0 ||
            geometry.page_tokens != metrics.page_tokens || geometry.page_bytes == 0 ||
            geometry.attention_layers == 0 ||
            geometry.layer_k_offsets.size() != geometry.attention_layers ||
            geometry.layer_v_offsets.size() != geometry.attention_layers ||
            geometry.layer_k_page_bytes.size() != geometry.attention_layers ||
            geometry.layer_v_page_bytes.size() != geometry.attention_layers ||
            geometry.model_layer_ids.size() != geometry.attention_layers) return false;
    auto * memory = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
    auto * cache = memory != nullptr ? memory->get_mem_attn() : nullptr;
    ggml_tensor * storage = cache != nullptr ? cache->pager_storage_tensor() : nullptr;
    if (storage == nullptr || storage->type != GGML_TYPE_I8) return false;
    llama_kv_pager_geometry cache_geometry;
    if (!cache->pager_geometry(metrics.page_tokens, cache_geometry) ||
            cache_geometry.attention_layers != geometry.attention_layers ||
            cache_geometry.model_layer_ids != geometry.model_layer_ids ||
            cache_geometry.layer_k_page_bytes != geometry.layer_k_page_bytes ||
            cache_geometry.layer_v_page_bytes != geometry.layer_v_page_bytes ||
            cache_geometry.page_bytes != geometry.page_bytes) return false;
    const uint64_t storage_bytes = ggml_nbytes(storage);
    if (physical_page_count == 0 ||
            uint64_t(physical_page_count) > UINT64_MAX / geometry.page_bytes ||
            storage_bytes != uint64_t(physical_page_count) * geometry.page_bytes) return false;

    llama_synchronize(ctx);
    output.clear();
    for (const auto & page : metrics.page_inventory) {
        if (page.id.sequence_id != 0 || page.physical_slot == UINT32_MAX ||
                page.valid_length == 0 || page.physical_slot >= physical_page_count) continue;
        const llama_pos begin = page.id.position_begin >= 0
            ? page.id.position_begin
            : llama_pos(uint64_t(page.id.logical_page) * metrics.page_tokens);
        if (begin < 0 || begin >= end) continue;
        const uint64_t native_page_begin =
            uint64_t(page.id.logical_page) * geometry.page_tokens;
        if (uint64_t(begin) != native_page_begin ||
                page.valid_length > geometry.page_tokens) return false;
        const uint32_t span = uint32_t(std::min<uint64_t>(
            page.valid_length, uint64_t(end - begin)));
        if (span == 0) continue;
        if (span > geometry.page_tokens) return false;

        attention_page_image image;
        image.logical_page = page.id.logical_page;
        image.position_begin = begin;
        image.span = span;
        image.physical_slot = page.physical_slot;
        image.page_generation = page.id.page_generation;
        image.content_version = page.content_version;
        image.representation = "pager_physical_kv";
        image.page_tokens = geometry.page_tokens;
        image.physical_page_count = physical_page_count;
        image.slot_bytes = geometry.page_bytes;
        const uint64_t row_in_page = uint64_t(begin) - native_page_begin;
        for (uint32_t layer = 0; layer < geometry.attention_layers; ++layer) {
            const uint64_t k_page_bytes = geometry.layer_k_page_bytes[layer];
            const uint64_t v_page_bytes = geometry.layer_v_page_bytes[layer];
            if (k_page_bytes == 0 || v_page_bytes == 0 ||
                    k_page_bytes % geometry.page_tokens != 0 ||
                    v_page_bytes % geometry.page_tokens != 0) return false;
            const uint64_t k_row_bytes = k_page_bytes / geometry.page_tokens;
            const uint64_t v_row_bytes = v_page_bytes / geometry.page_tokens;
            if (uint64_t(physical_page_count) > UINT64_MAX / k_page_bytes ||
                    uint64_t(physical_page_count) > UINT64_MAX / v_page_bytes) return false;
            const uint64_t k_layer_bytes = uint64_t(physical_page_count) * k_page_bytes;
            const uint64_t v_layer_bytes = uint64_t(physical_page_count) * v_page_bytes;
            const uint64_t k_layer_offset = geometry.layer_k_offsets[layer];
            const uint64_t v_layer_offset = geometry.layer_v_offsets[layer];
            if (k_layer_offset > storage_bytes ||
                    k_layer_bytes > storage_bytes - k_layer_offset ||
                    v_layer_offset > storage_bytes ||
                    v_layer_bytes > storage_bytes - v_layer_offset) return false;
            if (k_row_bytes == 0 || v_row_bytes == 0 ||
                    uint64_t(span) > UINT64_MAX / k_row_bytes ||
                    uint64_t(span) > UINT64_MAX / v_row_bytes ||
                    row_in_page > geometry.page_tokens ||
                    span > geometry.page_tokens - row_in_page) return false;
            const uint64_t k_bytes = uint64_t(span) * k_row_bytes;
            const uint64_t v_bytes = uint64_t(span) * v_row_bytes;
            if (uint64_t(page.physical_slot) > UINT64_MAX / k_page_bytes ||
                    uint64_t(page.physical_slot) > UINT64_MAX / v_page_bytes) return false;
            const uint64_t k_slot_offset = uint64_t(page.physical_slot) * k_page_bytes;
            const uint64_t v_slot_offset = uint64_t(page.physical_slot) * v_page_bytes;
            if (k_layer_offset > UINT64_MAX - k_slot_offset ||
                    v_layer_offset > UINT64_MAX - v_slot_offset) return false;
            const uint64_t k_base = k_layer_offset + k_slot_offset;
            const uint64_t v_base = v_layer_offset + v_slot_offset;
            if (k_base > UINT64_MAX - row_in_page * k_row_bytes ||
                    v_base > UINT64_MAX - row_in_page * v_row_bytes) return false;
            const uint64_t k_offset = k_base + row_in_page * k_row_bytes;
            const uint64_t v_offset = v_base + row_in_page * v_row_bytes;
            if (k_offset < k_layer_offset || k_offset > k_layer_offset + k_layer_bytes ||
                    k_bytes > k_layer_offset + k_layer_bytes - k_offset ||
                    v_offset < v_layer_offset || v_offset > v_layer_offset + v_layer_bytes ||
                    v_bytes > v_layer_offset + v_layer_bytes - v_offset ||
                    k_offset > storage_bytes || k_bytes > storage_bytes - k_offset ||
                    v_offset > storage_bytes || v_bytes > storage_bytes - v_offset ||
                    image.bytes.size() > SIZE_MAX - k_bytes ||
                    image.bytes.size() + size_t(k_bytes) > SIZE_MAX - v_bytes) return false;

            attention_page_image::layer_slice slice;
            slice.model_layer = geometry.model_layer_ids[layer];
            slice.k_source_offset = k_offset;
            slice.v_source_offset = v_offset;
            slice.k_row_bytes = k_row_bytes;
            slice.v_row_bytes = v_row_bytes;
            slice.k_page_bytes = k_page_bytes;
            slice.v_page_bytes = v_page_bytes;
            slice.k_bytes = k_bytes;
            slice.v_bytes = v_bytes;
            slice.k_output_offset = image.bytes.size();
            image.bytes.resize(image.bytes.size() + size_t(k_bytes) + size_t(v_bytes));
            ggml_backend_tensor_get(storage,
                    image.bytes.data() + slice.k_output_offset, size_t(k_offset), size_t(k_bytes));
            slice.v_output_offset = image.bytes.size() - size_t(v_bytes);
            ggml_backend_tensor_get(storage,
                    image.bytes.data() + slice.v_output_offset, size_t(v_offset), size_t(v_bytes));
            image.layer_slices.push_back(slice);
        }
        output.push_back(std::move(image));
    }
    std::sort(output.begin(), output.end(), [](const auto & a, const auto & b) {
        if (a.position_begin != b.position_begin) return a.position_begin < b.position_begin;
        return a.logical_page < b.logical_page;
    });
    return !output.empty();
}

bool draft_attention_prefix_page_images(llama_context * ctx,
        const std::vector<attention_page_image> & target_pages,
        std::vector<attention_page_image> & output) {
    if (ctx == nullptr || target_pages.empty()) return false;
    output.clear();
    for (const auto & target_page : target_pages) {
        attention_page_image image;
        image.logical_page = target_page.logical_page;
        image.position_begin = target_page.position_begin;
        image.span = target_page.span;
        image.representation = "native_range_state";
        image.page_tokens = target_page.page_tokens;
        const llama_pos image_end = image.position_begin + llama_pos(image.span);
        const size_t size = llama_state_seq_get_size_range(
                ctx, 0, image.position_begin, image_end);
        if (size == 0) return false;
        image.bytes.resize(size);
        if (llama_state_seq_get_data_range(ctx, image.bytes.data(), size, 0,
                image.position_begin, image_end) != size) return false;
        output.push_back(std::move(image));
    }
    return output.size() == target_pages.size();
}

struct observation {
    std::vector<float> logits;
    std::vector<float> hidden;
    std::vector<uint8_t> recurrent;
    std::vector<uint8_t> draft_recurrent;
    std::vector<uint8_t> encoded_kv;
    std::vector<uint8_t> draft_encoded_kv;
    std::vector<uint8_t> carry;
    llama_pos target_frontier = -1;
    llama_pos draft_frontier = -1;
};

struct generation_rows {
    std::vector<llama_pos> positions;
    std::vector<std::vector<float>> logits;
    std::vector<std::vector<float>> hidden;
    std::string first_attention_visibility;
    std::vector<std::string> verification_visibility;
};

struct generation_question_snapshot {
    std::vector<uint8_t> pre_target_partial;
    std::vector<uint8_t> pre_draft_partial;
    std::vector<attention_page_image> pre_target_attention_pages;
    std::vector<attention_page_image> pre_draft_attention_pages;
    std::vector<uint8_t> pre_carry;
    std::vector<float> post_logits;
    std::vector<float> post_hidden;
    std::string post_attention_visibility;
    std::map<std::string, question_tensor_image> question_tensors;
    size_t question_tensor_bytes = 0;
    bool question_tensor_overflow = false;
    llama_pos pre_target_frontier = -1;
    llama_pos pre_draft_frontier = -1;
};

struct question_tensor_comparison {
    bool shape_equal = false;
    bool layout_equal = false;
    bool type_equal = false;
    bool bytes_equal = false;
    size_t first_byte_diff = SIZE_MAX;
    size_t first_element_diff = SIZE_MAX;
    double max_abs_error = 0.0;
};

question_tensor_comparison compare_question_tensors(
        const question_tensor_image & a, const question_tensor_image & b) {
    question_tensor_comparison result;
    result.type_equal = a.type == b.type;
    result.shape_equal = a.ne == b.ne;
    result.layout_equal = a.nb == b.nb;
    result.bytes_equal = a.bytes == b.bytes;
    const size_t common_bytes = std::min(a.bytes.size(), b.bytes.size());
    for (size_t i = 0; i < common_bytes; ++i) {
        if (a.bytes[i] != b.bytes[i]) {
            result.first_byte_diff = i;
            break;
        }
    }
    if (!result.shape_equal || !result.type_equal || a.bytes.size() != b.bytes.size()) {
        return result;
    }
    const size_t element_size = a.type == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t);
    if (element_size == 0 || a.bytes.size() % element_size != 0) return result;
    const size_t count = a.bytes.size() / element_size;
    for (size_t i = 0; i < count; ++i) {
        float av = 0.0f, bv = 0.0f;
        if (a.type == GGML_TYPE_F32) {
            std::memcpy(&av, a.bytes.data() + i * element_size, sizeof(av));
            std::memcpy(&bv, b.bytes.data() + i * element_size, sizeof(bv));
        } else {
            ggml_fp16_t ah, bh;
            std::memcpy(&ah, a.bytes.data() + i * element_size, sizeof(ah));
            std::memcpy(&bh, b.bytes.data() + i * element_size, sizeof(bh));
            av = ggml_fp16_to_fp32(ah);
            bv = ggml_fp16_to_fp32(bh);
        }
        const double error = std::fabs(double(av) - double(bv));
        if (i != 0 && result.first_element_diff == SIZE_MAX &&
                std::memcmp(a.bytes.data() + i * element_size,
                    b.bytes.data() + i * element_size, element_size) != 0) {
            result.first_element_diff = i;
        } else if (i == 0 && result.first_element_diff == SIZE_MAX &&
                std::memcmp(a.bytes.data(), b.bytes.data(), element_size) != 0) {
            result.first_element_diff = 0;
        }
        if (!std::isfinite(error)) result.max_abs_error = std::numeric_limits<double>::infinity();
        else result.max_abs_error = std::max(result.max_abs_error, error);
    }
    return result;
}

bool collect_generation_row(llama_context * target, int32_t row,
        llama_pos position, generation_rows & output) {
    const uint32_t n_vocab = llama_vocab_n_tokens(
            llama_model_get_vocab(&target->get_model()));
    const float * logits = llama_get_logits_ith(target, row);
    if (!logits || n_vocab == 0) return false;
    output.positions.push_back(position);
    output.logits.emplace_back(logits, logits + n_vocab);
    const float * hidden = llama_get_embeddings_ith(target, row);
    if (hidden) {
        const uint32_t n_embd = llama_model_n_embd(&target->get_model());
        output.hidden.emplace_back(hidden, hidden + n_embd);
    } else {
        output.hidden.emplace_back();
    }
    const size_t nonfinite = std::count_if(output.logits.back().begin(),
            output.logits.back().end(), [](float value) { return !std::isfinite(value); });
    if (nonfinite != 0) {
        std::fprintf(stderr, "generation parity row has nonfinite logits row=%d pos=%lld count=%zu/%zu\n",
            row, static_cast<long long>(position), nonfinite, output.logits.back().size());
    }
    return nonfinite == 0;
}

bool decode_generation_batch(llama_context * target, common_speculative * spec,
        const std::vector<llama_token> & tokens, llama_pos position,
        uint32_t verify_width, generation_rows & output) {
    if (tokens.empty()) return false;
    if (verify_width != 3 && verify_width != 12) return false;
    size_t offset = 0;
    while (offset < tokens.size()) {
        const size_t count = std::min<size_t>(verify_width, tokens.size() - offset);
        llama_batch batch = llama_batch_init(int32_t(count), 0, 1);
        if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            return false;
        }
        batch.n_tokens = int32_t(count);
        for (size_t i = 0; i < count; ++i) {
            batch.token[i] = tokens[offset + i];
            batch.pos[i] = position + llama_pos(offset + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
        }
        target->set_kv_attention_mtp_verification(count > 1);
        const int rc = llama_decode(target, batch);
        target->set_kv_attention_mtp_verification(false);
        if (rc != 0) {
            llama_batch_free(batch);
            return false;
        }
        llama_synchronize(target);
        const std::string visibility = target->mtp_attention_visibility_snapshot_json();
        if (output.first_attention_visibility.empty()) {
            output.first_attention_visibility = visibility;
        }
        output.verification_visibility.push_back(visibility);
        bool collected = true;
        for (size_t i = 0; collected && i < count; ++i) {
            collected = collect_generation_row(target, int32_t(i),
                    position + llama_pos(offset + i), output);
        }
        const bool processed = collected && common_speculative_process(spec, batch);
        llama_batch_free(batch);
        if (!collected || !processed) return false;
        offset += count;
    }
    return true;
}

bool decode_generation_scalar(llama_context * target, common_speculative * spec,
        llama_token token, llama_pos position, generation_rows & output) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
        llama_batch_free(batch);
        return false;
    }
    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = position;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;
    const int rc = llama_decode(target, batch);
    if (rc != 0) {
        llama_batch_free(batch);
        return false;
    }
    llama_synchronize(target);
    if (output.first_attention_visibility.empty()) {
        output.first_attention_visibility = target->mtp_attention_visibility_snapshot_json();
    }
    const bool collected = collect_generation_row(target, -1, position, output);
    const bool processed = collected && common_speculative_process(spec, batch);
    llama_batch_free(batch);
    return collected && processed;
}

bool observe(llama_context * target, llama_context * draft,
        common_speculative * spec, llama_pos query_begin, observation & out) {
    const uint32_t n_vocab = llama_vocab_n_tokens(
            llama_model_get_vocab(&target->get_model()));
    const float * logits = llama_get_logits_ith(target, -1);
    if (!logits || n_vocab == 0) return false;
    out.logits.assign(logits, logits + n_vocab);
    const float * hidden = llama_get_embeddings_ith(target, -1);
    if (hidden) {
        const uint32_t n_embd = llama_model_n_embd(&target->get_model());
        out.hidden.assign(hidden, hidden + n_embd);
    }
    if (!seq_image(target, out.recurrent) ||
            (draft && !seq_image(draft, out.draft_recurrent)) ||
            !seq_range_image(target, out.encoded_kv, query_begin,
                llama_memory_seq_pos_max(llama_get_memory(target), 0) + 1) ||
            (draft && !seq_range_image(draft, out.draft_encoded_kv, query_begin,
                llama_memory_seq_pos_max(llama_get_memory(draft), 0) + 1))) return false;
    if (spec && !common_speculative_get_state(spec, 0, out.carry)) return false;
    out.target_frontier = llama_memory_seq_pos_max(llama_get_memory(target), 0);
    out.draft_frontier = draft
        ? llama_memory_seq_pos_max(llama_get_memory(draft), 0) : -1;
    return std::all_of(out.logits.begin(), out.logits.end(),
            [](float value) { return std::isfinite(value); });
}

bool close_vectors(const std::vector<float> & a, const std::vector<float> & b,
        float tolerance) {
    if (a.empty() || a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
                std::abs(a[i] - b[i]) > tolerance) return false;
    }
    return true;
}

size_t byte_difference_count(const std::vector<uint8_t> & a,
        const std::vector<uint8_t> & b, size_t & first_difference) {
    first_difference = std::numeric_limits<size_t>::max();
    if (a.size() != b.size()) {
        first_difference = std::min(a.size(), b.size());
        return std::max(a.size(), b.size()) - std::min(a.size(), b.size());
    }
    size_t count = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            if (first_difference == std::numeric_limits<size_t>::max()) first_difference = i;
            ++count;
        }
    }
    return count;
}

bool normalize_compact_page_generations(std::vector<uint8_t> & bytes) {
    auto read_u32 = [&](size_t offset, uint32_t & value) {
        if (offset > bytes.size() || bytes.size() - offset < sizeof(value)) return false;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        return true;
    };
    // The range header is followed by the memory's range state. Find and
    // validate the compact-pager layer marker before interpreting page rows.
    for (size_t marker = 32; marker + sizeof(uint32_t) <= bytes.size(); ++marker) {
        uint32_t marked_layers = 0;
        if (!read_u32(marker, marked_layers) || !(marked_layers & 0x80000000u)) continue;
        const uint32_t layers = marked_layers & 0x7fffffffu;
        if (layers == 0 || layers > 256) continue;
        const size_t page_count_offset = marker + sizeof(uint32_t) + size_t(layers) * 24;
        uint32_t page_count = 0;
        if (!read_u32(page_count_offset, page_count) || page_count == 0 || page_count > 16) continue;
        size_t page = page_count_offset + sizeof(uint32_t);
        bool valid = true;
        std::vector<size_t> generations;
        for (uint32_t i = 0; i < page_count; ++i) {
            constexpr size_t fixed = sizeof(int32_t) + sizeof(uint32_t) +
                sizeof(llama_kv_page_id) + sizeof(uint32_t);
            if (page > bytes.size() || bytes.size() - page < fixed) { valid = false; break; }
            uint32_t rows = 0;
            const size_t rows_offset = page + sizeof(int32_t) + sizeof(uint32_t) +
                sizeof(llama_kv_page_id);
            if (!read_u32(rows_offset, rows) || rows == 0 ||
                    rows > (bytes.size() - rows_offset - sizeof(uint32_t)) / 12) {
                valid = false;
                break;
            }
            generations.push_back(page + sizeof(int32_t) + sizeof(uint32_t) +
                offsetof(llama_kv_page_id, page_generation));
            page = rows_offset + sizeof(uint32_t) + size_t(rows) * 12;
        }
        if (!valid) continue;
        for (const size_t offset : generations) {
            std::fill(bytes.begin() + offset, bytes.begin() + offset + sizeof(uint32_t), 0);
        }
        return true;
    }
    return false;
}

bool same_history(const std::vector<llama_kv_pager_selected_history> & a,
        const std::vector<llama_kv_pager_selected_history> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const auto & x = a[i];
        const auto & y = b[i];
        if (x.identity != y.identity || x.content_version != y.content_version) return false;
    }
    return true;
}

bool run_mtp_cycle(llama_context * target, llama_context * draft,
        common_speculative * spec,
        llama_pos next_pos, llama_tokens & prompt, uint32_t & proposals,
        uint32_t & accepted, llama_pos & target_frontier,
        llama_pos & draft_frontier) {
    const uint32_t n_vocab = uint32_t(llama_vocab_n_tokens(
            llama_model_get_vocab(&target->get_model())));
    llama_token sampled = LLAMA_TOKEN_NULL;
    const float * logits = llama_get_logits_ith(target, -1);
    if (!logits || n_vocab == 0) return false;
    float best = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < n_vocab; ++i) {
        if (std::isfinite(logits[i]) && logits[i] > best) {
            best = logits[i];
            sampled = llama_token(i);
        }
    }
    if (sampled == LLAMA_TOKEN_NULL) return false;

    auto & dp = common_speculative_get_draft_params(spec, 0);
    std::vector<llama_token> proposed;
    dp.drafting = true;
    dp.n_max = 2;
    dp.pos0 = next_pos;
    dp.id_last = sampled;
    dp.prompt = &prompt;
    dp.result = &proposed;
    common_speculative_draft(spec);
    proposals = uint32_t(proposed.size());
    if (proposed.empty() || proposed.size() > UINT16_MAX) return false;

    const size_t count = proposed.size() + 1;
    llama_batch verify = llama_batch_init(int32_t(count), 0, 1);
    if (!verify.token || !verify.pos || !verify.n_seq_id || !verify.seq_id || !verify.logits) {
        llama_batch_free(verify);
        return false;
    }
    verify.n_tokens = int32_t(count);
    verify.token[0] = sampled;
    verify.pos[0] = next_pos;
    verify.n_seq_id[0] = 1;
    verify.seq_id[0][0] = 0;
    verify.logits[0] = true;
    for (size_t i = 0; i < proposed.size(); ++i) {
        verify.token[i + 1] = proposed[i];
        verify.pos[i + 1] = next_pos + llama_pos(i + 1);
        verify.n_seq_id[i + 1] = 1;
        verify.seq_id[i + 1][0] = 0;
        verify.logits[i + 1] = true;
    }
    target->set_kv_attention_mtp_verification(true);
    const int rc = llama_decode(target, verify);
    target->set_kv_attention_mtp_verification(false);
    if (rc != 0) {
        llama_batch_free(verify);
        return false;
    }
    llama_synchronize(target);
    const bool processed = common_speculative_process(spec, verify);
    std::vector<llama_token> target_argmax(count, LLAMA_TOKEN_NULL);
    bool finite = processed;
    for (size_t row = 0; finite && row < count; ++row) {
        const float * row_logits = llama_get_logits_ith(target, int32_t(row));
        float row_best = -std::numeric_limits<float>::infinity();
        for (uint32_t i = 0; row_logits && i < n_vocab; ++i) {
            if (std::isfinite(row_logits[i]) && row_logits[i] > row_best) {
                row_best = row_logits[i];
                target_argmax[row] = llama_token(i);
            }
        }
        finite = target_argmax[row] != LLAMA_TOKEN_NULL;
    }
    accepted = 0;
    while (finite && accepted < proposed.size() &&
            target_argmax[accepted] == proposed[accepted]) ++accepted;
    const auto frontier = common_speculative_rollback_frontier_resolve(
            next_pos, proposed.size(), accepted);
    // Match the real server boundary: acceptance selects the committed
    // hidden carry before target/draft rollback. Paired rollback must retain
    // that carry, not invalidate it as an unpaired checkpoint restore.
    if (finite && frontier.valid()) {
        common_speculative_accept(spec, 0, uint16_t(accepted));
    }
    const bool rollback = finite && frontier.valid() &&
        llama_memory_seq_rm_transient(llama_get_memory(target), 0,
            llama_pos(frontier.accepted_token_count), -1) &&
        common_speculative_rollback_dft(spec, 0,
            llama_pos(frontier.accepted_token_count), uint16_t(accepted));
    if (rollback) {
        llama_synchronize(target);
        llama_synchronize(draft);
        target_frontier = llama_memory_seq_pos_max(llama_get_memory(target), 0);
        draft_frontier = llama_memory_seq_pos_max(llama_get_memory(draft), 0);
    }
    llama_batch_free(verify);
    return rollback;
}

bool run(const options & opts) {
    question_tensor_capture layer_capture;
    common_params params;
    params.model.path = opts.model;
    params.n_ctx = int32_t(opts.context);
    params.n_batch = int32_t(opts.batch);
    params.n_ubatch = int32_t(opts.ubatch);
    params.n_gpu_layers = -1;
    params.fit_params = false;
    params.cpuparams.n_threads = 16;
    params.cpuparams.n_threads_explicit = true;
    params.cpuparams_batch.n_threads = 16;
    params.cpuparams_batch.n_threads_explicit = true;
    params.n_parallel = 1;
    params.n_sequences = 1;
    params.n_predict = 0;
    params.embedding = true;
    if (opts.question_layer_parity) {
        params.cb_eval = capture_question_tensor;
        params.cb_eval_user_data = &layer_capture;
    }
    params.cache_type_k = GGML_TYPE_TURBO4_0;
    params.cache_type_v = GGML_TYPE_TURBO4_0;
    params.reset_vbr_runtime_state();
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.kv_pager.mode = llama_kv_pager_mode::selective;
    params.kv_pager.router = opts.router == "probe-rerank"
        ? llama_kv_router_mode::probe_rerank : llama_kv_router_mode::legacy;
    params.kv_pager.page_size = VBR_GENERATION_PAGE_CELLS;
    params.kv_pager.router_top_k = 32;
    params.kv_pager.router_explore = 16;
    params.kv_pager.hot_pages.automatic = false;
    params.kv_pager.hot_pages.value = opts.hot_tokens / VBR_GENERATION_PAGE_CELLS;
    params.kv_pager.pin_recent.automatic = false;
    params.kv_pager.pin_recent.value = 0;
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max = 2;
    params.speculative.draft.cache_type_k = GGML_TYPE_TURBO4_0;
    params.speculative.draft.cache_type_v = GGML_TYPE_TURBO4_0;
    params.speculative.draft.kv_device = common_speculative_draft_kv_device::GPU;

    common_init_result_ptr init;
    try {
        init = common_init_from_params(params);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "model init failed: %s\n", e.what());
        return false;
    }
    llama_model * model = init->model();
    llama_context * target = init->context();
    if (!model || !target) return false;
    target->set_kv_attention_native_mtp(true);

    common_params draft_params = common_base_params_to_speculative(params);
    common_speculative_init_result_ptr draft_init =
        common_speculative_init_from_params(draft_params, model, target);
    if (!draft_init || !draft_init->context()) return false;
    params.speculative.draft.ctx_tgt = target;
    params.speculative.draft.ctx_dft = draft_init->context();
    params.speculative.draft.ctx_mtp = draft_init->context_mtp();
    common_speculative_ptr spec(common_speculative_init(params.speculative, 1));
    if (!spec) return false;
    llama_context * draft = draft_init->context();

    // Keep this regression on a genuinely fresh owner. The integrated replay
    // fixture below deliberately populates and mutates the same one-sequence
    // hybrid cache, so run the fresh startup/next-turn case in isolation.
    if (opts.fresh_only) {
        const bool cleared_target = llama_memory_seq_rm(
            llama_get_memory(target), 0, -1, -1);
        const bool cleared_draft = llama_memory_seq_rm(
            llama_get_memory(draft), 0, -1, -1);
        llama_synchronize(target);
        llama_synchronize(draft);
        const uint32_t fresh_count = 32;
        llama_tokens fresh_prompt(fresh_count, llama_token(1));
        const auto empty_metrics = target->get_kv_pager_metrics(draft);
        const bool empty_at_start = std::none_of(
            empty_metrics.page_inventory.begin(), empty_metrics.page_inventory.end(),
            [](const auto & page) { return page.id.sequence_id == 0; });
        const uint64_t fresh_turn = 1020;
        target->begin_kv_pager_turn(0, fresh_turn, 0, fresh_count);
        common_speculative_begin(spec.get(), 0, fresh_prompt);
        const bool fresh_decoded = cleared_target && cleared_draft && empty_at_start &&
            decode_prefix(target, spec.get(), fresh_count);
        bool fresh_changed = true;
        uint64_t fresh_generation = 0;
        std::vector<llama_kv_pager_selected_history> fresh_history;
        const bool fresh_committed = fresh_decoded &&
            target->commit_kv_pager_query(0, fresh_turn,
                &fresh_changed, &fresh_generation) && !fresh_changed &&
            target->get_kv_pager_history_for_test(0, fresh_history) &&
            fresh_history.empty() && target->freeze_kv_pager_history(
                0, fresh_turn, nullptr);
        const auto fresh_metrics = target->get_kv_pager_metrics(draft);
        const bool current_span_visible = std::any_of(
            fresh_metrics.page_inventory.begin(), fresh_metrics.page_inventory.end(),
            [&](const auto & page) {
                const int64_t begin = page.id.position_begin >= 0
                    ? page.id.position_begin
                    : int64_t(page.id.logical_page) * VBR_GENERATION_PAGE_CELLS;
                return page.id.sequence_id == 0 && page.physical_slot != UINT32_MAX &&
                    page.valid_length != 0 && begin <= fresh_count - 1 &&
                    begin + page.valid_length > fresh_count - 1;
            });
        uint32_t fresh_proposals = 0, fresh_accepted = 0;
        llama_pos fresh_target_frontier = -1, fresh_draft_frontier = -1;
        const bool fresh_mtp_carry = fresh_committed && current_span_visible &&
            run_mtp_cycle(target, draft, spec.get(), fresh_count, fresh_prompt,
                fresh_proposals, fresh_accepted,
                fresh_target_frontier, fresh_draft_frontier) &&
            fresh_proposals > 0 && fresh_target_frontier == fresh_draft_frontier;
        target->end_kv_pager_turn(0, fresh_turn);

        const llama_pos next_user_begin = std::max(
            llama_memory_seq_pos_max(llama_get_memory(target), 0),
            llama_memory_seq_pos_max(llama_get_memory(draft), 0)) + 1;
        const uint64_t next_user_turn = 1021;
        target->begin_kv_pager_turn(0, next_user_turn,
            next_user_begin, next_user_begin + 1);
        const auto next_owner = target->get_kv_pager_owner_for_test();
        const auto next_state = next_owner != nullptr
            ? next_owner->turn_state(0) : llama_kv_pager_turn_state{};
        const bool next_user_passed = next_owner != nullptr &&
            next_state.turn_id == next_user_turn &&
            next_state.phase == llama_kv_pager_turn_phase::query_provisional &&
            next_state.query_start == next_user_begin &&
            next_state.query_end == next_user_begin + 1 &&
            fresh_target_frontier == fresh_draft_frontier &&
            next_user_begin == std::max(
                llama_memory_seq_pos_max(llama_get_memory(target), 0),
                llama_memory_seq_pos_max(llama_get_memory(draft), 0)) + 1;
        target->end_kv_pager_turn(0, next_user_turn);
        const bool passed = fresh_committed && current_span_visible &&
            fresh_mtp_carry && next_user_passed;
        std::ofstream out;
        std::ostream & report = opts.output.empty() ? std::cout : (out.open(opts.output), out);
        report << "{\"proof\":\"fresh_target_draft_route_sanity_and_observed_repair\","
            << "\"passed\":" << (passed ? "true" : "false")
            << ",\"empty_at_start\":" << (empty_at_start ? "true" : "false")
            << ",\"empty_history_owner_passed\":" << (fresh_committed ? "true" : "false")
            << ",\"current_span_visible\":" << (current_span_visible ? "true" : "false")
            << ",\"mtp_carry_passed\":" << (fresh_mtp_carry ? "true" : "false")
            << ",\"next_user_turn_passed\":" << (next_user_passed ? "true" : "false")
            << ",\"next_user_begin\":" << next_user_begin
            << ",\"fresh_proposals\":" << fresh_proposals << "}\n";
        return passed;
    }

    if (opts.delayed_turn_only) {
        const bool seed_decoded = decode_prefix(target, nullptr, 512);
        llama_memory_clear(llama_get_memory(target), true);
        llama_memory_clear(llama_get_memory(draft_init->context()), true);
        const llama_seq_id sequence_id = 0;
        const auto before = target->get_kv_pager_metrics(draft_init->context());
        const bool initially_empty = seed_decoded && std::none_of(before.page_inventory.begin(),
                before.page_inventory.end(), [&](const auto & page) {
            return page.id.sequence_id == sequence_id;
        });
        const uint64_t turn_id = 1014;
        const llama_pos query_begin = 512;
        target->begin_kv_pager_turn(sequence_id, turn_id, query_begin, query_begin + 1);
        const int initial_phase = target->get_kv_pager_turn_phase_for_test(sequence_id);
        const bool prefix_decoded = initially_empty && initial_phase ==
                int(llama_kv_pager_turn_phase::idle) &&
            decode_prefix(target, nullptr, uint32_t(query_begin));
        const auto after_prefix = target->get_kv_pager_metrics(draft_init->context());
        const bool inventory_populated = std::any_of(after_prefix.page_inventory.begin(),
                after_prefix.page_inventory.end(), [&](const auto & page) {
            return page.id.sequence_id == sequence_id;
        });
        const bool query_decoded = prefix_decoded && inventory_populated &&
            decode_one(target, nullptr, 8, query_begin);
        const bool turn_opened = query_decoded && target->get_kv_pager_turn_phase_for_test(
                sequence_id) == int(llama_kv_pager_turn_phase::query_provisional);
        target->end_kv_pager_turn(sequence_id, turn_id);
        std::ostream * report = &std::cout;
        std::ofstream output;
        if (!opts.output.empty()) {
            output.open(opts.output);
            if (!output) return false;
            report = &output;
        }
        *report << "{\"proof\":\"delayed_empty_inventory_turn_open\",\"passed\":"
            << (turn_opened ? "true" : "false")
            << ",\"seed_decoded_for_catalogue_clear\":"
            << (seed_decoded ? "true" : "false")
            << ",\"initially_empty\":" << (initially_empty ? "true" : "false")
            << ",\"initial_phase\":" << initial_phase
            << ",\"inventory_populated_after_prefix\":"
            << (inventory_populated ? "true" : "false")
            << ",\"prefix_decoded\":" << (prefix_decoded ? "true" : "false")
            << ",\"query_decoded\":" << (query_decoded ? "true" : "false")
            << ",\"turn_opened_before_query_graph\":"
            << (turn_opened ? "true" : "false") << "}\n";
        return turn_opened;
    }

    // Leave one complete historical page beyond the configured hot capacity
    // so probe-rerank exercises an authenticated host-cold candidate.
    std::vector<llama_token> question_tokens = opts.generation_parity
        ? common_tokenize(target, "In Python, how should a sorted merge fill its preallocated output?", false, true)
        : std::vector<llama_token>{ llama_token(2) };
    std::vector<llama_token> continuation_tokens = opts.generation_parity
        ? common_tokenize(target,
            " Python merge uses two pointers and writes each output slot once;"
            " preserve sorted order, skip duplicates, and return the completed list.",
            false, true)
        : std::vector<llama_token>{ llama_token(3), llama_token(4) };
    if (opts.generation_parity && continuation_tokens.size() < 12) return false;
    if (opts.generation_parity) continuation_tokens.resize(12);
    const uint32_t prefix_count = opts.hot_tokens + VBR_GENERATION_PAGE_CELLS -
        (opts.generation_parity ? 8u : 0u);
    const llama_pos query_begin = llama_pos(prefix_count);
    const llama_pos query_end = query_begin + llama_pos(question_tokens.size());
    llama_tokens prefix(prefix_count, llama_token(1));
    // Match the real query token against the oldest page so the admitted
    // coarse shortlist includes host-cold keys as well as resident pages.
    std::fill(prefix.begin(), prefix.begin() + VBR_GENERATION_PAGE_CELLS,
            llama_token(2));
    common_speculative_begin(spec.get(), 0, prefix);
    const uint64_t owner_turn_id = 1009;
    target->begin_kv_pager_turn(0, owner_turn_id, 0, query_begin);
    const auto pager_owner_before_reserve = target->get_kv_pager_owner_for_test();
    const auto catalogue_before_reserve = target->get_kv_pager_metrics(draft);
    const auto catalogue_matches = [](const auto & a, const auto & b) {
        if (a.page_inventory.size() != b.page_inventory.size()) return false;
        for (size_t i = 0; i < a.page_inventory.size(); ++i) {
            const auto & x = a.page_inventory[i];
            const auto & y = b.page_inventory[i];
            if (x.id != y.id || x.content_version != y.content_version ||
                    x.physical_slot != y.physical_slot || x.valid_length != y.valid_length) {
                return false;
            }
        }
        return true;
    };
    bool reservation_owner_survived = pager_owner_before_reserve != nullptr &&
        target->get_kv_pager_turn_phase_for_test(0) == 1;
    for (int reserve = 0; reserve < 2 && reservation_owner_survived; ++reserve) {
        target->request_graph_reserve_for_test();
        const auto catalogue_after_reserve = target->get_kv_pager_metrics(draft);
        reservation_owner_survived = target->get_kv_pager_owner_for_test() == pager_owner_before_reserve &&
            target->get_kv_pager_turn_phase_for_test(0) == 1 &&
            catalogue_matches(catalogue_before_reserve, catalogue_after_reserve);
    }
    if (!decode_prefix(target, spec.get(), prefix)) {
        std::fprintf(stderr, "prefix decode failed\n");
        return false;
    }
    const auto catalogue_after_prefix = target->get_kv_pager_metrics(draft);
    reservation_owner_survived = reservation_owner_survived &&
        target->get_kv_pager_owner_for_test() == pager_owner_before_reserve &&
        target->get_kv_pager_turn_phase_for_test(0) == 1 &&
        !catalogue_after_prefix.page_inventory.empty();
    target->end_kv_pager_turn(0, owner_turn_id);
    if (opts.owner_only) {
        std::ostream * report = &std::cout;
        std::ofstream out;
        if (!opts.output.empty()) {
            out.open(opts.output);
            report = &out;
        }
        *report << "{\"proof\":\"stable_pager_owner_repeated_graph_reserve\","
                << "\"passed\":" << (reservation_owner_survived ? "true" : "false")
                << ",\"reserve_count\":2,\"catalogue_pages_after_prefix\":"
                << catalogue_after_prefix.page_inventory.size() << "}\n";
        return reservation_owner_survived;
    }
    const uint64_t turn_id = 1010;
    bool cold_catalogue_authoritative = false;
    target->begin_kv_pager_turn(0, turn_id, query_begin, query_end);
    std::vector<llama_kv_pager_selected_history> final_history;
    std::vector<llama_kv_pager_selected_history> committed_history;
    observation replay, control;
    bool selection_installed = false, replay_map_exact = false, control_map_exact = false;
    bool router_execution_observed = false;
    uint32_t router_stage = 0, resident_rerank_graphs = 0, cold_reader_graphs = 0;
    uint32_t reader_events = 0, mass_graphs = 0, router_output_records = 0;
    generation_rows scalar_generation_rows, batched_generation_rows;
    std::vector<generation_question_snapshot> generation_question_snapshots;
    const auto write_generation_failure_visibility = [&](const char * stage,
            size_t row, llama_pos position) {
        if (opts.output.empty()) return;
        std::ofstream visibility(opts.output + ".failure-visibility.json");
        if (!visibility) return;
        visibility << "{\"stage\":\"" << stage << "\",\"row\":" << row
            << ",\"position\":" << position << ",\"snapshot\":"
            << target->mtp_attention_visibility_snapshot_json() << "}\n";
    };
    const auto decode_question = [&]() {
        if (!opts.generation_parity) {
            return decode_one(target, spec.get(), question_tokens.front(), query_begin);
        }
        generation_question_snapshot snapshot;
        snapshot.pre_target_frontier = llama_memory_seq_pos_max(
                llama_get_memory(target), 0);
        snapshot.pre_draft_frontier = llama_memory_seq_pos_max(
                llama_get_memory(draft), 0);
        const auto * pager_owner = target->get_kv_pager_owner_for_test();
        if (pager_owner == nullptr) return false;
        const auto & pager_snapshot = pager_owner->snapshot();
        const auto question_pager_metrics = target->get_kv_pager_metrics(draft);
        if (!seq_image(target, snapshot.pre_target_partial) ||
                !seq_image(draft, snapshot.pre_draft_partial) ||
                !target_attention_prefix_page_images(target,
                    question_pager_metrics, query_begin, pager_snapshot.geometry,
                    pager_snapshot.physical_page_count,
                    snapshot.pre_target_attention_pages) ||
                !draft_attention_prefix_page_images(draft,
                    snapshot.pre_target_attention_pages,
                    snapshot.pre_draft_attention_pages) ||
                !common_speculative_get_state(spec.get(), 0, snapshot.pre_carry)) {
            std::fprintf(stderr, "generation parity could not capture pre-question state\n");
            return false;
        }
        llama_batch batch = llama_batch_init(int32_t(question_tokens.size()), 0, 1);
        if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            return false;
        }
        batch.n_tokens = int32_t(question_tokens.size());
        for (size_t i = 0; i < question_tokens.size(); ++i) {
            batch.token[i] = question_tokens[i];
            batch.pos[i] = query_begin + llama_pos(i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
        }
        if (opts.question_layer_parity) layer_capture.begin();
        const int rc = llama_decode(target, batch);
        if (rc == 0) {
            llama_synchronize(target);
            if (opts.question_layer_parity) {
                layer_capture.end();
                snapshot.question_tensors = layer_capture.tensors;
                snapshot.question_tensor_bytes = layer_capture.bytes_used;
                snapshot.question_tensor_overflow = layer_capture.overflow;
            }
            const uint32_t n_vocab = llama_vocab_n_tokens(
                    llama_model_get_vocab(&target->get_model()));
            for (size_t i = 0; i < question_tokens.size(); ++i) {
                const float * logits = llama_get_logits_ith(target, int32_t(i));
                const size_t nonfinite = logits == nullptr ? n_vocab :
                    std::count_if(logits, logits + n_vocab,
                        [](float value) { return !std::isfinite(value); });
                if (nonfinite != 0) {
                    write_generation_failure_visibility("question-batch", i,
                            query_begin + llama_pos(i));
                    std::fprintf(stderr,
                        "generation parity question row has nonfinite logits row=%zu pos=%lld token=%d count=%zu/%u\n",
                        i, static_cast<long long>(query_begin + llama_pos(i)),
                        question_tokens[i], nonfinite, n_vocab);
                    llama_batch_free(batch);
                    return false;
                }
            }
            snapshot.post_attention_visibility =
                target->mtp_attention_visibility_snapshot_json();
            snapshot.post_logits.reserve(size_t(question_tokens.size()) * n_vocab);
            snapshot.post_hidden.reserve(size_t(question_tokens.size()) *
                    llama_model_n_embd(&target->get_model()));
            for (size_t i = 0; i < question_tokens.size(); ++i) {
                const float * logits = llama_get_logits_ith(target, int32_t(i));
                const float * hidden = llama_get_embeddings_ith(target, int32_t(i));
                if (logits == nullptr || hidden == nullptr) {
                    std::fprintf(stderr,
                        "generation parity question snapshot missing output row=%zu\n", i);
                    llama_batch_free(batch);
                    return false;
                }
                snapshot.post_logits.insert(snapshot.post_logits.end(), logits, logits + n_vocab);
                snapshot.post_hidden.insert(snapshot.post_hidden.end(), hidden,
                        hidden + llama_model_n_embd(&target->get_model()));
            }
            common_speculative_process(spec.get(), batch);
        }
        if (opts.question_layer_parity) layer_capture.end();
        llama_batch_free(batch);
        if (rc == 0) generation_question_snapshots.push_back(std::move(snapshot));
        return rc == 0;
    };
    const auto decode_query_and_suffix = [&](observation & output, bool batched,
            generation_rows & rows) {
        if (!decode_question()) {
            std::fprintf(stderr, "generation parity question decode failed begin=%lld\n",
                    static_cast<long long>(query_begin));
            return false;
        }
        const llama_pos continuation_begin = query_begin + llama_pos(question_tokens.size());
        if (opts.generation_parity) {
            if (batched) {
                if (!decode_generation_batch(target, spec.get(), continuation_tokens,
                        continuation_begin, opts.verify_width, rows)) return false;
            } else {
                for (size_t i = 0; i < continuation_tokens.size(); ++i) {
                    const llama_pos position = continuation_begin + llama_pos(i);
                    if (!decode_generation_scalar(target, spec.get(),
                            continuation_tokens[i], position, rows)) {
                        std::fprintf(stderr, "generation parity scalar decode/capture failed row=%zu pos=%lld token=%d\n",
                            i, static_cast<long long>(position), continuation_tokens[i]);
                        return false;
                    }
                }
            }
        } else {
            for (size_t i = 0; i < continuation_tokens.size(); ++i) {
                if (!decode_one(target, spec.get(), continuation_tokens[i],
                        continuation_begin + llama_pos(i))) return false;
            }
        }
        const bool observed = observe(target, draft, spec.get(), query_begin, output);
        if (!observed) std::fprintf(stderr, "generation parity observation failed batched=%d begin=%lld\n",
            batched ? 1 : 0, static_cast<long long>(query_begin));
        return observed;
    };
    const auto fixture = server_query_checkpoint_replay_for_test(
            target, draft, spec.get(), 0, turn_id, turn_id,
            query_begin, prefix_count, true,
            [&]() {
                return decode_question();
            },
            [&](bool & changed, uint64_t & generation) {
                if (opts.router == "probe-rerank") {
                    if (!target->commit_kv_pager_query(0, turn_id, &changed, &generation) ||
                            !target->get_kv_pager_history_for_test(0, committed_history)) {
                        return false;
                    }
                    final_history = committed_history;
                    selection_installed = !final_history.empty();
                    if (!selection_installed) return false;
                    // Exercise the production transaction, not only the
                    // residency repair helper. Removed GPU pages remain
                    // canonical cold objects and cannot keep a reused slot.
                    const auto * pager = target->get_kv_pager_owner_for_test();
                    if (pager == nullptr) return false;
                    const auto snapshot = pager->residency(0);
                    size_t cold_count = 0;
                    const auto normalized = [&](const auto & records) {
                        for (const auto & record : records) {
                            const bool resident = std::any_of(snapshot.pages().begin(),
                                    snapshot.pages().end(), [&](const auto & current) {
                                return current.id == record.id;
                            });
                            if (resident) continue;
                            ++cold_count;
                            if (record.physical_slot != UINT32_MAX ||
                                    record.state != llama_kv_page_state::host_clean ||
                                    record.dirty || record.pin_count != 0 ||
                                    record.consumer_events != 0) return false;
                        }
                        return true;
                    };
                    cold_catalogue_authoritative = normalized(pager->exact_page_records(0)) &&
                        normalized(target->get_kv_pager_metrics(draft).page_inventory) &&
                        cold_count > 0;
                    if (!cold_catalogue_authoritative) return false;
                    router_execution_observed =
                        target->get_kv_pager_router_execution_for_test(router_stage,
                            resident_rerank_graphs, cold_reader_graphs, reader_events,
                            mass_graphs, router_output_records) &&
                        router_stage == uint32_t(llama_kv_router_owner_stage::ready) &&
                        resident_rerank_graphs > 0 && cold_reader_graphs > 0 &&
                        reader_events > 0 && mass_graphs > 0 && router_output_records > 0;
                    if (!router_execution_observed) return false;
                    bool repeated_changed = true;
                    uint64_t repeated_generation = 0;
                    return target->commit_kv_pager_query(0, turn_id,
                            &repeated_changed, &repeated_generation) &&
                        !repeated_changed && repeated_generation == generation;
                }
                const auto metrics = target->get_kv_pager_metrics(draft);
                for (const auto & page : metrics.page_inventory) {
                    if (page.id.sequence_id != 0 ||
                            page.id.position_end >= query_begin ||
                            page.physical_slot == UINT32_MAX ||
                            page.valid_length == 0 || page.content_version == 0) continue;
                    final_history.push_back({page.id, page.content_version});
                    break;
                }
                selection_installed = !final_history.empty() &&
                    target->set_kv_pager_history_for_test(
                        0, query_begin, final_history);
                if (!selection_installed) return false;
                if (!target->commit_kv_pager_query(0, turn_id, &changed, &generation)) {
                    return false;
                }
                bool repeated_changed = true;
                uint64_t repeated_generation = 0;
                if (!target->commit_kv_pager_query(0, turn_id,
                        &repeated_changed, &repeated_generation) ||
                        repeated_changed || repeated_generation != generation) {
                    return false;
                }
                return target->get_kv_pager_history_for_test(0, committed_history) &&
                    same_history(final_history, committed_history);
            },
            [&]() {
                if (!decode_query_and_suffix(replay, false, scalar_generation_rows)) return false;
                std::vector<llama_kv_pager_selected_history> replay_history;
                replay_map_exact = target->get_kv_pager_history_for_test(0, replay_history) &&
                    same_history(committed_history, replay_history);
                return replay_map_exact;
            },
            [&]() {
                std::vector<llama_kv_pager_selected_history> before;
                control_map_exact = target->get_kv_pager_history_for_test(0, before) &&
                    same_history(committed_history, before);
                return control_map_exact && decode_query_and_suffix(control,
                    opts.generation_parity, batched_generation_rows);
            });

    target->end_kv_pager_turn(0, turn_id);
    const uint64_t unchanged_turn = 1013;

    const llama_pos cancel1_begin = llama_memory_seq_pos_max(llama_get_memory(target), 0) + 1;
    const uint64_t cancel1_turn = 1011;
    target->begin_kv_pager_turn(0, cancel1_turn, cancel1_begin, cancel1_begin + 1);
    const int cancel1_begin_phase = target->get_kv_pager_turn_phase_for_test(0);
    const auto cancel1 = server_query_checkpoint_cancel_for_test(
            target, draft, spec.get(), 0, cancel1_turn, cancel1_turn,
            cancel1_begin, cancel1_begin, true,
            [&]() { return decode_one(target, spec.get(), 5, cancel1_begin); }, {},
            [&]() { return decode_one(target, spec.get(), 5, cancel1_begin); });
    target->end_kv_pager_turn(0, cancel1_turn);
    const int cancel1_end_phase = target->get_kv_pager_turn_phase_for_test(0);

    const llama_pos cancel2_begin = llama_memory_seq_pos_max(llama_get_memory(target), 0) + 1;
    const uint64_t cancel2_turn = 1012;
    target->begin_kv_pager_turn(0, cancel2_turn, cancel2_begin, cancel2_begin + 1);
    const int cancel2_begin_phase = target->get_kv_pager_turn_phase_for_test(0);
    std::vector<llama_kv_pager_selected_history> published_history;
    bool alternate_installed = false;
    const auto cancel2 = server_query_checkpoint_cancel_for_test(
            target, draft, spec.get(), 0, cancel2_turn, cancel2_turn,
            cancel2_begin, cancel2_begin, true,
            [&]() { return decode_one(target, spec.get(), 6, cancel2_begin); },
            [&](bool & changed, uint64_t & generation) {
                std::vector<llama_kv_pager_selected_history> before;
                if (!target->get_kv_pager_history_for_test(0, before)) return false;
                auto alternate_history = before;
                // Give the real commit path a distinct authenticated map to
                // replace, even when every currently resident historical
                // page already belongs to the just-published selection.
                if (alternate_history.size() > 1) alternate_history.pop_back();
                alternate_installed = alternate_history.size() != before.size() &&
                    target->set_kv_pager_history_for_test(
                        0, cancel2_begin, alternate_history);
                if (!alternate_installed || !target->commit_kv_pager_query(
                        0, cancel2_turn, &changed, &generation)) return false;
                return changed && target->get_kv_pager_history_for_test(
                    0, published_history);
            },
            [&]() {
                std::vector<llama_kv_pager_selected_history> after_restore;
                return target->get_kv_pager_history_for_test(0, after_restore) &&
                    same_history(published_history, after_restore) &&
                    decode_one(target, spec.get(), 6, cancel2_begin);
            });
    target->end_kv_pager_turn(0, cancel2_turn);

    const llama_pos unchanged_begin = llama_memory_seq_pos_max(
            llama_get_memory(target), 0) + 1;
    target->begin_kv_pager_turn(0, unchanged_turn, unchanged_begin, unchanged_begin + 1);
    std::vector<llama_kv_pager_selected_history> unchanged_history;
    bool unchanged_replay_called = false;
    const auto unchanged = server_query_checkpoint_replay_for_test(
            target, draft, spec.get(), 0, unchanged_turn, unchanged_turn,
            unchanged_begin, unchanged_begin, true,
            [&]() { return decode_one(target, spec.get(), 6, unchanged_begin); },
            [&](bool & changed, uint64_t & generation) {
                std::vector<llama_kv_pager_selected_history> before;
                if (!target->get_kv_pager_history_for_test(0, before) ||
                    !target->set_kv_pager_history_for_test(
                        0, unchanged_begin, before) ||
                    !target->get_kv_pager_history_for_test(0, unchanged_history)) return false;
                changed = !same_history(before, unchanged_history);
                generation = unchanged_turn;
                return !changed;
            },
            [&]() { unchanged_replay_called = true; return true; },
            []() { return true; });
    target->end_kv_pager_turn(0, unchanged_turn);
    const int unchanged_end_phase = target->get_kv_pager_turn_phase_for_test(0);

    uint32_t mtp_proposals = 0, mtp_accepted = 0;
    const llama_pos mtp_begin = llama_memory_seq_pos_max(llama_get_memory(target), 0) + 1;
    llama_pos mtp_target_frontier = -1, mtp_draft_frontier = -1;
    const bool mtp_cycle = unchanged.captured && unchanged.provisional_decode_succeeded &&
        run_mtp_cycle(target, draft, spec.get(), mtp_begin, prefix,
            mtp_proposals, mtp_accepted, mtp_target_frontier, mtp_draft_frontier);

    auto replay_kv_canonical = replay.encoded_kv;
    auto control_kv_canonical = control.encoded_kv;
    auto replay_draft_kv_canonical = replay.draft_encoded_kv;
    auto control_draft_kv_canonical = control.draft_encoded_kv;
    const bool target_kv_canonicalized =
        normalize_compact_page_generations(replay_kv_canonical) &&
        normalize_compact_page_generations(control_kv_canonical);
    const bool draft_kv_canonicalized = replay.draft_encoded_kv == control.draft_encoded_kv ||
        (normalize_compact_page_generations(replay_draft_kv_canonical) &&
         normalize_compact_page_generations(control_draft_kv_canonical));
    const bool target_kv_equal = replay.encoded_kv == control.encoded_kv ||
        (target_kv_canonicalized && replay_kv_canonical == control_kv_canonical);
    const bool draft_kv_equal = replay.draft_encoded_kv == control.draft_encoded_kv ||
        (draft_kv_canonicalized &&
         replay_draft_kv_canonical == control_draft_kv_canonical);

    const llama_pos expected_generation_frontier = query_begin +
        llama_pos(question_tokens.size() + continuation_tokens.size() - 1);
    bool generation_rows_match = !opts.generation_parity;
    float max_generation_logit_error = 0.0f;
    float max_generation_hidden_error = 0.0f;
    if (opts.generation_parity &&
            scalar_generation_rows.positions == batched_generation_rows.positions &&
            scalar_generation_rows.logits.size() == continuation_tokens.size() &&
            batched_generation_rows.logits.size() == continuation_tokens.size() &&
            scalar_generation_rows.hidden.size() == continuation_tokens.size() &&
            batched_generation_rows.hidden.size() == continuation_tokens.size()) {
        generation_rows_match = true;
        for (size_t row = 0; row < continuation_tokens.size(); ++row) {
            const auto & scalar_logits = scalar_generation_rows.logits[row];
            const auto & batch_logits = batched_generation_rows.logits[row];
            if (scalar_logits.size() != batch_logits.size()) {
                generation_rows_match = false;
                break;
            }
            for (size_t i = 0; i < scalar_logits.size(); ++i) {
                const float error = std::fabs(scalar_logits[i] - batch_logits[i]);
                if (!std::isfinite(error)) generation_rows_match = false;
                else max_generation_logit_error = std::max(max_generation_logit_error, error);
            }
            const auto & scalar_hidden = scalar_generation_rows.hidden[row];
            const auto & batch_hidden = batched_generation_rows.hidden[row];
            if (scalar_hidden.size() != batch_hidden.size()) {
                generation_rows_match = false;
                break;
            }
            for (size_t i = 0; i < scalar_hidden.size(); ++i) {
                const float error = std::fabs(scalar_hidden[i] - batch_hidden[i]);
                if (!std::isfinite(error)) generation_rows_match = false;
                else max_generation_hidden_error = std::max(max_generation_hidden_error, error);
            }
            generation_rows_match = generation_rows_match &&
                close_vectors(scalar_logits, batch_logits, 0.08f) &&
                close_vectors(scalar_hidden, batch_hidden, 0.02f);
        }
    }

    bool parity = reservation_owner_survived && fixture.captured && fixture.provisional_decode_succeeded &&
        fixture.status == server_query_replay_transition_status::replay &&
        fixture.history_changed && fixture.restored && fixture.replay_decode_succeeded &&
        fixture.control_restored && fixture.control_decode_succeeded && selection_installed &&
        !final_history.empty() && !committed_history.empty() && replay_map_exact &&
        (opts.router != "probe-rerank" ||
            (router_execution_observed && cold_catalogue_authoritative)) &&
        control_map_exact && mtp_cycle && mtp_proposals > 0 &&
        cancel1.captured && cancel1.provisional_decode_succeeded && cancel1.restored &&
        cancel1.recovery_decode_succeeded && cancel2.captured &&
        cancel2.provisional_decode_succeeded && cancel2.publication_succeeded &&
        cancel2.history_changed && cancel2.restored &&
        cancel2.recovery_decode_succeeded && alternate_installed &&
        unchanged.captured && unchanged.provisional_decode_succeeded &&
        unchanged.status == server_query_replay_transition_status::unchanged &&
        !unchanged_replay_called &&
        replay.target_frontier == expected_generation_frontier &&
        replay.draft_frontier == expected_generation_frontier &&
        replay.target_frontier == control.target_frontier &&
        replay.draft_frontier == control.draft_frontier &&
        mtp_target_frontier == mtp_begin + llama_pos(mtp_accepted) &&
        mtp_draft_frontier == mtp_begin + llama_pos(mtp_accepted) &&
        replay.recurrent == control.recurrent &&
        replay.draft_recurrent == control.draft_recurrent && replay.carry == control.carry &&
        target_kv_equal && draft_kv_equal && generation_rows_match &&
        close_vectors(replay.logits, control.logits, 0.08f) &&
        !replay.hidden.empty() && close_vectors(replay.hidden, control.hidden, 0.02f);

    size_t target_kv_first_diff = 0, draft_kv_first_diff = 0;
    const size_t target_kv_diff_bytes = byte_difference_count(
            replay.encoded_kv, control.encoded_kv, target_kv_first_diff);
    const size_t draft_kv_diff_bytes = byte_difference_count(
            replay.draft_encoded_kv, control.draft_encoded_kv, draft_kv_first_diff);

    bool generation_row_artifacts_written = !opts.generation_parity;
    if (opts.generation_parity && !opts.output.empty()) {
        const auto flatten = [](const std::vector<std::vector<float>> & rows) {
            std::vector<float> values;
            size_t count = 0;
            for (const auto & row : rows) count += row.size();
            values.reserve(count);
            for (const auto & row : rows) values.insert(values.end(), row.begin(), row.end());
            return values;
        };
        const auto write_floats = [](const std::string & path,
                const std::vector<float> & values) {
            std::ofstream file(path, std::ios::binary);
            if (!file) return false;
            file.write(reinterpret_cast<const char *>(values.data()),
                    std::streamsize(values.size() * sizeof(float)));
            return bool(file);
        };
        const auto write_bytes = [](const std::string & path,
                const std::vector<uint8_t> & values) {
            std::ofstream file(path, std::ios::binary);
            if (!file) return false;
            file.write(reinterpret_cast<const char *>(values.data()),
                    std::streamsize(values.size()));
            return bool(file);
        };
        const auto write_attention_pages = [&](const std::string & stem,
                const char * role, const std::vector<attention_page_image> & pages) {
            for (const auto & page : pages) {
                const std::string path = stem + "-pre-" + role + "-attention-page-" +
                    std::to_string(page.logical_page) + "-" +
                    std::to_string(page.position_begin) + "-" +
                    std::to_string(page.span) + ".bin";
                if (!write_bytes(path, page.bytes)) return false;
            }
            return !pages.empty();
        };
        const auto scalar_logits = flatten(scalar_generation_rows.logits);
        const auto batch_logits = flatten(batched_generation_rows.logits);
        const auto scalar_hidden = flatten(scalar_generation_rows.hidden);
        const auto batch_hidden = flatten(batched_generation_rows.hidden);
        const std::string sidecar_path = opts.output + ".generation-parity.json";
        const std::string scalar_logits_path = opts.output + ".scalar-logits.bin";
        const std::string batch_logits_path = opts.output + ".batch-logits.bin";
        const std::string scalar_hidden_path = opts.output + ".scalar-hidden.bin";
        const std::string batch_hidden_path = opts.output + ".batch-hidden.bin";
        bool question_snapshots_written = generation_question_snapshots.size() >= 3;
        const char * question_branch_names[] = {"provisional", "replay", "control"};
        bool replay_control_pre_target_exact = false;
        bool replay_control_pre_draft_exact = false;
        bool replay_control_target_attention_pages_exact = false;
        bool replay_control_target_attention_pages_bytes_equal = false;
        bool replay_control_draft_attention_pages_exact = false;
        bool replay_control_draft_attention_pages_bytes_equal = false;
        bool provisional_replay_target_attention_pages_exact = false;
        bool provisional_replay_target_attention_pages_bytes_equal = false;
        bool provisional_replay_draft_attention_pages_exact = false;
        bool provisional_replay_draft_attention_pages_bytes_equal = false;
        bool replay_control_pre_carry_exact = false;
        float replay_control_question_logit_max_error = 0.0f;
        float replay_control_question_hidden_max_error = 0.0f;
        for (size_t i = 0; question_snapshots_written && i < 3; ++i) {
            const auto & snapshot = generation_question_snapshots[i];
            const std::string stem = opts.output + "." + question_branch_names[i] + "-question";
            question_snapshots_written =
                write_bytes(stem + "-pre-target-partial.bin", snapshot.pre_target_partial) &&
                write_bytes(stem + "-pre-draft-partial.bin", snapshot.pre_draft_partial) &&
                write_attention_pages(stem, "target", snapshot.pre_target_attention_pages) &&
                write_attention_pages(stem, "draft", snapshot.pre_draft_attention_pages) &&
                write_bytes(stem + "-pre-carry.bin", snapshot.pre_carry) &&
                write_floats(stem + "-post-logits.bin", snapshot.post_logits) &&
                write_floats(stem + "-post-hidden.bin", snapshot.post_hidden);
        }
        if (generation_question_snapshots.size() >= 3) {
            const auto & replay_question = generation_question_snapshots[1];
            const auto & control_question = generation_question_snapshots[2];
            replay_control_pre_target_exact = replay_question.pre_target_partial ==
                control_question.pre_target_partial;
            replay_control_pre_draft_exact = replay_question.pre_draft_partial ==
                control_question.pre_draft_partial;
            const auto attention_page_keys_equal = [](const auto & a, const auto & b) {
                if (a.size() != b.size() || a.empty()) return false;
                for (size_t i = 0; i < a.size(); ++i) {
                    if (a[i].logical_page != b[i].logical_page ||
                            a[i].position_begin != b[i].position_begin ||
                            a[i].span != b[i].span ||
                            a[i].representation != b[i].representation) return false;
                }
                return true;
            };
            const auto attention_page_bytes_equal = [&](const auto & a, const auto & b) {
                if (!attention_page_keys_equal(a, b)) return false;
                for (size_t i = 0; i < a.size(); ++i) {
                    if (a[i].bytes != b[i].bytes) return false;
                }
                return true;
            };
            replay_control_target_attention_pages_exact = attention_page_keys_equal(
                replay_question.pre_target_attention_pages,
                control_question.pre_target_attention_pages);
            replay_control_target_attention_pages_bytes_equal = attention_page_bytes_equal(
                replay_question.pre_target_attention_pages,
                control_question.pre_target_attention_pages);
            replay_control_draft_attention_pages_exact = attention_page_keys_equal(
                replay_question.pre_draft_attention_pages,
                control_question.pre_draft_attention_pages);
            replay_control_draft_attention_pages_bytes_equal = attention_page_bytes_equal(
                replay_question.pre_draft_attention_pages,
                control_question.pre_draft_attention_pages);
            const auto & provisional_question = generation_question_snapshots[0];
            provisional_replay_target_attention_pages_exact = attention_page_keys_equal(
                provisional_question.pre_target_attention_pages,
                replay_question.pre_target_attention_pages);
            provisional_replay_target_attention_pages_bytes_equal = attention_page_bytes_equal(
                provisional_question.pre_target_attention_pages,
                replay_question.pre_target_attention_pages);
            provisional_replay_draft_attention_pages_exact = attention_page_keys_equal(
                provisional_question.pre_draft_attention_pages,
                replay_question.pre_draft_attention_pages);
            provisional_replay_draft_attention_pages_bytes_equal = attention_page_bytes_equal(
                provisional_question.pre_draft_attention_pages,
                replay_question.pre_draft_attention_pages);
            replay_control_pre_carry_exact = replay_question.pre_carry ==
                control_question.pre_carry;
            const auto max_error = [](const std::vector<float> & a,
                    const std::vector<float> & b) {
                if (a.size() != b.size() || a.empty()) return std::numeric_limits<float>::infinity();
                float maximum = 0.0f;
                for (size_t i = 0; i < a.size(); ++i) {
                    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
                        return std::numeric_limits<float>::infinity();
                    }
                    maximum = std::max(maximum, std::fabs(a[i] - b[i]));
                }
                return maximum;
            };
            replay_control_question_logit_max_error = max_error(
                    replay_question.post_logits, control_question.post_logits);
            replay_control_question_hidden_max_error = max_error(
                    replay_question.post_hidden, control_question.post_hidden);
        }
        std::ofstream sidecar(sidecar_path);
        generation_row_artifacts_written = sidecar &&
            question_snapshots_written &&
            write_floats(scalar_logits_path, scalar_logits) &&
            write_floats(batch_logits_path, batch_logits) &&
            write_floats(scalar_hidden_path, scalar_hidden) &&
            write_floats(batch_hidden_path, batch_hidden);
        if (generation_row_artifacts_written) {
            sidecar << "{\"proof\":\"teacher_forced_scalar_batched_generation_parity\","
                << "\"passed\":" << (generation_rows_match ? "true" : "false")
                << ",\"question_token_ids\":[";
            for (size_t i = 0; i < question_tokens.size(); ++i) {
                if (i) sidecar << ',';
                sidecar << question_tokens[i];
            }
            sidecar << "],\"continuation_token_ids\":[";
            for (size_t i = 0; i < continuation_tokens.size(); ++i) {
                if (i) sidecar << ',';
                sidecar << continuation_tokens[i];
            }
            sidecar << "],\"max_logit_error\":" << max_generation_logit_error
                << ",\"max_hidden_error\":" << max_generation_hidden_error
                << ",\"target_encoded_kv_equal\":" << (target_kv_equal ? "true" : "false")
                << ",\"draft_encoded_kv_equal\":" << (draft_kv_equal ? "true" : "false")
                << ",\"target_recurrent_equal\":"
                << (replay.recurrent == control.recurrent ? "true" : "false")
                << ",\"draft_recurrent_equal\":"
                << (replay.draft_recurrent == control.draft_recurrent ? "true" : "false")
                << ",\"mtp_carry_equal\":" << (replay.carry == control.carry ? "true" : "false")
                << ",\"target_frontier\":" << replay.target_frontier
                << ",\"draft_frontier\":" << replay.draft_frontier
                << ",\"verify_width\":" << opts.verify_width
                << ",\"replay_control_pre_target_partial_exact\":"
                << (replay_control_pre_target_exact ? "true" : "false")
                << ",\"replay_control_pre_draft_partial_exact\":"
                << (replay_control_pre_draft_exact ? "true" : "false")
                << ",\"replay_control_target_attention_page_keys_equal\":"
                << (replay_control_target_attention_pages_exact ? "true" : "false")
                << ",\"replay_control_target_attention_page_bytes_equal\":"
                << (replay_control_target_attention_pages_bytes_equal ? "true" : "false")
                << ",\"replay_control_draft_attention_page_keys_equal\":"
                << (replay_control_draft_attention_pages_exact ? "true" : "false")
                << ",\"replay_control_draft_attention_page_bytes_equal\":"
                << (replay_control_draft_attention_pages_bytes_equal ? "true" : "false")
                << ",\"provisional_replay_target_attention_page_keys_equal\":"
                << (provisional_replay_target_attention_pages_exact ? "true" : "false")
                << ",\"provisional_replay_target_attention_page_bytes_equal\":"
                << (provisional_replay_target_attention_pages_bytes_equal ? "true" : "false")
                << ",\"provisional_replay_draft_attention_page_keys_equal\":"
                << (provisional_replay_draft_attention_pages_exact ? "true" : "false")
                << ",\"provisional_replay_draft_attention_page_bytes_equal\":"
                << (provisional_replay_draft_attention_pages_bytes_equal ? "true" : "false")
                << ",\"replay_control_pre_carry_exact\":"
                << (replay_control_pre_carry_exact ? "true" : "false")
                << ",\"replay_control_question_logit_max_error\":"
                << replay_control_question_logit_max_error
                << ",\"replay_control_question_hidden_max_error\":"
                << replay_control_question_hidden_max_error
                << ",\"question_layer_capture_enabled\":"
                << (opts.question_layer_parity ? "true" : "false")
                << ",\"question_snapshots\":[";
            for (size_t i = 0; i < generation_question_snapshots.size(); ++i) {
                if (i) sidecar << ',';
                const auto & snapshot = generation_question_snapshots[i];
                sidecar << "{\"branch\":\""
                    << (i < 3 ? question_branch_names[i] : "extra")
                    << "\",\"pre_target_frontier\":" << snapshot.pre_target_frontier
                    << ",\"pre_draft_frontier\":" << snapshot.pre_draft_frontier
                    << ",\"pre_target_partial_bytes\":" << snapshot.pre_target_partial.size()
                    << ",\"pre_draft_partial_bytes\":" << snapshot.pre_draft_partial.size()
                    << ",\"pre_carry_bytes\":" << snapshot.pre_carry.size()
                    << ",\"question_tensor_bytes\":" << snapshot.question_tensor_bytes
                    << ",\"question_tensor_overflow\":"
                    << (snapshot.question_tensor_overflow ? "true" : "false")
                    << ",\"post_logits_floats\":" << snapshot.post_logits.size()
                    << ",\"post_hidden_floats\":" << snapshot.post_hidden.size()
                    << ",\"pre_target_attention_pages\":[";
                for (size_t page = 0; page < snapshot.pre_target_attention_pages.size(); ++page) {
                    if (page) sidecar << ',';
                    const auto & image = snapshot.pre_target_attention_pages[page];
                    sidecar << "{\"logical_page\":" << image.logical_page
                        << ",\"position_begin\":" << image.position_begin
                        << ",\"span\":" << image.span
                        << ",\"representation\":\"" << image.representation << "\""
                        << ",\"page_tokens\":" << image.page_tokens
                        << ",\"physical_page_count\":" << image.physical_page_count
                        << ",\"slot_bytes\":" << image.slot_bytes
                        << ",\"physical_slot\":" << image.physical_slot
                        << ",\"page_generation\":" << image.page_generation
                        << ",\"content_version\":" << image.content_version
                        << ",\"bytes\":" << image.bytes.size()
                        << ",\"layer_slices\":[";
                    for (size_t layer = 0; layer < image.layer_slices.size(); ++layer) {
                        if (layer) sidecar << ',';
                        const auto & slice = image.layer_slices[layer];
                        sidecar << "{\"model_layer\":" << slice.model_layer
                            << ",\"k_source_offset\":" << slice.k_source_offset
                            << ",\"v_source_offset\":" << slice.v_source_offset
                            << ",\"k_row_bytes\":" << slice.k_row_bytes
                            << ",\"v_row_bytes\":" << slice.v_row_bytes
                            << ",\"k_page_bytes\":" << slice.k_page_bytes
                            << ",\"v_page_bytes\":" << slice.v_page_bytes
                            << ",\"k_bytes\":" << slice.k_bytes
                            << ",\"v_bytes\":" << slice.v_bytes
                            << ",\"k_output_offset\":" << slice.k_output_offset
                            << ",\"v_output_offset\":" << slice.v_output_offset << '}';
                    }
                    sidecar << "]}";
                }
                sidecar << "],\"pre_draft_attention_pages\":[";
                for (size_t page = 0; page < snapshot.pre_draft_attention_pages.size(); ++page) {
                    if (page) sidecar << ',';
                    const auto & image = snapshot.pre_draft_attention_pages[page];
                    sidecar << "{\"logical_page\":" << image.logical_page
                        << ",\"position_begin\":" << image.position_begin
                        << ",\"span\":" << image.span
                        << ",\"representation\":\"" << image.representation << "\""
                        << ",\"page_tokens\":" << image.page_tokens
                        << ",\"physical_slot\":" << image.physical_slot
                        << ",\"page_generation\":" << image.page_generation
                        << ",\"content_version\":" << image.content_version
                        << ",\"bytes\":" << image.bytes.size() << '}';
                }
                sidecar << "],\"actual_visibility\":"
                    << snapshot.post_attention_visibility
                    << ",\"question_tensors\":[";
                size_t tensor_index = 0;
                for (const auto & entry : snapshot.question_tensors) {
                    if (tensor_index++) sidecar << ',';
                    sidecar << "{\"name\":\"" << entry.first
                        << "\",\"type\":" << int(entry.second.type)
                        << ",\"ne\":[";
                    for (size_t dim = 0; dim < entry.second.ne.size(); ++dim) {
                        if (dim) sidecar << ',';
                        sidecar << entry.second.ne[dim];
                    }
                    sidecar << "],\"nb\":[";
                    for (size_t dim = 0; dim < entry.second.nb.size(); ++dim) {
                        if (dim) sidecar << ',';
                        sidecar << entry.second.nb[dim];
                    }
                    sidecar << "],\"bytes\":" << entry.second.bytes.size() << '}';
                }
                sidecar << "]}";
            }
            sidecar << "],\"replay_control_question_tensor_comparisons\":[";
            if (generation_question_snapshots.size() >= 3) {
                const auto & replay_tensors = generation_question_snapshots[1].question_tensors;
                const auto & control_tensors = generation_question_snapshots[2].question_tensors;
                size_t compared = 0;
                for (const auto & entry : replay_tensors) {
                    const auto control_it = control_tensors.find(entry.first);
                    if (control_it == control_tensors.end()) continue;
                    if (compared++) sidecar << ',';
                    const auto diff = compare_question_tensors(entry.second, control_it->second);
                    sidecar << "{\"name\":\"" << entry.first
                        << "\",\"shape_equal\":" << (diff.shape_equal ? "true" : "false")
                        << ",\"layout_equal\":" << (diff.layout_equal ? "true" : "false")
                        << ",\"type_equal\":" << (diff.type_equal ? "true" : "false")
                        << ",\"bytes_equal\":" << (diff.bytes_equal ? "true" : "false")
                        << ",\"first_byte_diff\":"
                        << (diff.first_byte_diff == SIZE_MAX ? -1 : int64_t(diff.first_byte_diff))
                        << ",\"first_element_diff\":"
                        << (diff.first_element_diff == SIZE_MAX ? -1 : int64_t(diff.first_element_diff))
                        << ",\"max_abs_error\":" << diff.max_abs_error << '}';
                }
            }
            sidecar << "],\"scalar_first_actual_visibility\":"
                << (scalar_generation_rows.first_attention_visibility.empty()
                    ? "null" : scalar_generation_rows.first_attention_visibility)
                << ",\"verification_group_actual_visibility\":[";
            for (size_t i = 0; i < batched_generation_rows.verification_visibility.size(); ++i) {
                if (i) sidecar << ',';
                sidecar << batched_generation_rows.verification_visibility[i];
            }
            sidecar << "],\"rows\":[";
            for (size_t i = 0; i < scalar_generation_rows.positions.size(); ++i) {
                if (i) sidecar << ',';
                const llama_pos position = scalar_generation_rows.positions[i];
                sidecar << "{\"row\":" << i << ",\"position\":" << position
                    << ",\"expected_causal_input_span\":\"question_begin..query_position; inspect actual_visibility above\","
                    << "\"attended_history_pages\":[";
                for (size_t page = 0; page < final_history.size(); ++page) {
                    if (page) sidecar << ',';
                    sidecar << final_history[page].identity.logical_page;
                }
                sidecar << "],\"expected_input_positions\":[";
                for (llama_pos input_position = query_begin; input_position <= position;
                        ++input_position) {
                    if (input_position != query_begin) sidecar << ',';
                    sidecar << input_position;
                }
                sidecar << "]}";
            }
            sidecar << "],\"scalar_logits\":\"" << scalar_logits_path
                << "\",\"batch_logits\":\"" << batch_logits_path
                << "\",\"scalar_hidden\":\"" << scalar_hidden_path
                << "\",\"batch_hidden\":\"" << batch_hidden_path << "\"}\n";
            generation_row_artifacts_written = bool(sidecar);
        }
    }

    bool encoded_kv_blobs_written = true;
    if (!opts.output.empty()) {
        const auto write_blob = [](const std::string & path, const std::vector<uint8_t> & bytes) {
            std::ofstream blob(path, std::ios::binary);
            blob.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
            return bool(blob);
        };
        encoded_kv_blobs_written =
            write_blob(opts.output + ".replay-kv.bin", replay.encoded_kv) &&
            write_blob(opts.output + ".control-kv.bin", control.encoded_kv) &&
            write_blob(opts.output + ".replay-draft-kv.bin", replay.draft_encoded_kv) &&
            write_blob(opts.output + ".control-draft-kv.bin", control.draft_encoded_kv);
    }
    parity = parity && encoded_kv_blobs_written && generation_row_artifacts_written;

    std::ofstream out;
    std::ostream & report = opts.output.empty() ? std::cout : (out.open(opts.output), out);
    report << "{\"proof\":\"integrated_server_query_replay_one_pass_parity\","
        << "\"passed\":" << (parity ? "true" : "false")
        << ",\"router_mode\":\"" << opts.router << "\""
        << ",\"router_execution_observed\":"
        << (router_execution_observed ? "true" : "false")
        << ",\"router_owner_stage\":" << router_stage
        << ",\"resident_rerank_graphs\":" << resident_rerank_graphs
        << ",\"cold_reader_graphs\":" << cold_reader_graphs
        << ",\"reader_events\":" << reader_events
        << ",\"mass_graphs\":" << mass_graphs
        << ",\"router_output_records\":" << router_output_records
        << ",\"cold_catalogue_authoritative\":"
        << (cold_catalogue_authoritative ? "true" : "false")
        << ",\"reservation_owner_survived\":"
        << (reservation_owner_survived ? "true" : "false")
        << ",\"captured\":" << (fixture.captured ? "true" : "false")
        << ",\"selection_installed\":" << (selection_installed ? "true" : "false")
        << ",\"history_changed\":" << (fixture.history_changed ? "true" : "false")
        << ",\"history_generation\":" << fixture.history_generation
        << ",\"restored\":" << (fixture.restored ? "true" : "false")
        << ",\"replay_decode_succeeded\":"
        << (fixture.replay_decode_succeeded ? "true" : "false")
        << ",\"control_restored\":" << (fixture.control_restored ? "true" : "false")
        << ",\"final_history_pages\":" << final_history.size()
        << ",\"committed_history_pages\":" << committed_history.size()
        << ",\"replay_map_exact\":" << (replay_map_exact ? "true" : "false")
        << ",\"control_map_exact\":" << (control_map_exact ? "true" : "false")
        << ",\"mtp_next_cycle\":" << (mtp_cycle ? "true" : "false")
        << ",\"mtp_proposals\":" << mtp_proposals
        << ",\"mtp_accepted\":" << mtp_accepted
        << ",\"mtp_committed_frontier\":"
        << mtp_begin + llama_pos(mtp_accepted)
        << ",\"mtp_target_frontier\":" << mtp_target_frontier
        << ",\"mtp_draft_frontier\":" << mtp_draft_frontier
        << ",\"cancel_after_provisional_recovered\":"
        << (cancel1.recovery_decode_succeeded ? "true" : "false")
        << ",\"cancel_after_publication_recovered\":"
        << (cancel2.recovery_decode_succeeded ? "true" : "false")
        << ",\"cancel_publication_changed\":"
        << (cancel2.history_changed ? "true" : "false")
        << ",\"unchanged_map_skipped_replay\":"
        << (!unchanged_replay_called && unchanged.status ==
            server_query_replay_transition_status::unchanged ? "true" : "false")
        << ",\"unchanged_status\":" << int(unchanged.status)
        << ",\"unchanged_history_changed\":"
        << (unchanged.history_changed ? "true" : "false")
        << ",\"unchanged_captured\":" << (unchanged.captured ? "true" : "false")
        << ",\"unchanged_provisional_decode\":"
        << (unchanged.provisional_decode_succeeded ? "true" : "false")
        << ",\"unchanged_end_phase\":" << unchanged_end_phase
        << ",\"cancel1_begin_phase\":" << cancel1_begin_phase
        << ",\"cancel1_end_phase\":" << cancel1_end_phase
        << ",\"cancel2_begin_phase\":" << cancel2_begin_phase
        << ",\"cancel2_publication_succeeded\":"
        << (cancel2.publication_succeeded ? "true" : "false")
        << ",\"target_frontier_replay\":" << replay.target_frontier
        << ",\"target_frontier_control\":" << control.target_frontier
        << ",\"draft_frontier_replay\":" << replay.draft_frontier
        << ",\"draft_frontier_control\":" << control.draft_frontier
        << ",\"hidden_available\":" << (!replay.hidden.empty() ? "true" : "false")
        << ",\"encoded_kv_bytes\":" << replay.encoded_kv.size()
        << ",\"draft_encoded_kv_bytes\":" << replay.draft_encoded_kv.size()
        << ",\"encoded_kv_equal\":"
        << (replay.encoded_kv == control.encoded_kv ? "true" : "false")
        << ",\"draft_encoded_kv_equal\":"
        << (replay.draft_encoded_kv == control.draft_encoded_kv ? "true" : "false")
        << ",\"encoded_kv_blobs_written\":"
        << (encoded_kv_blobs_written ? "true" : "false")
        << ",\"encoded_kv_canonical_equal\":"
        << (target_kv_equal ? "true" : "false")
        << ",\"draft_encoded_kv_canonical_equal\":"
        << (draft_kv_equal ? "true" : "false")
        << ",\"target_encoded_kv_diff_bytes\":" << target_kv_diff_bytes
        << ",\"target_encoded_kv_first_diff\":"
        << (target_kv_first_diff == std::numeric_limits<size_t>::max() ? -1 : int64_t(target_kv_first_diff))
        << ",\"draft_encoded_kv_diff_bytes\":" << draft_kv_diff_bytes
        << ",\"teacher_forced_generation_parity\":"
        << (generation_rows_match ? "true" : "false")
        << ",\"generation_max_logit_error\":" << max_generation_logit_error
        << ",\"generation_max_hidden_error\":" << max_generation_hidden_error
        << ",\"generation_row_artifacts_written\":"
        << (generation_row_artifacts_written ? "true" : "false")
        << "}\n";
    return parity;
}

} // namespace

int main(int argc, char ** argv) {
    options opts;
    if (!parse_options(argc, argv, opts)) {
        std::fprintf(stderr,
                "usage: %s --model MODEL.gguf [--router legacy|probe-rerank] [--fresh-only] [--generation-parity] [--B 1024] [--U 256] [--L 8192] [--H 4096] [--owner-only] [--output FILE]\n",
                argv[0]);
        return 2;
    }
    llama_backend_init();
    const bool passed = run(opts);
    llama_backend_free();
    return passed ? 0 : 1;
}
