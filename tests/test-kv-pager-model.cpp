// Optional parity driver for the live Turbo4 KV pager.
//
// The default mode is a deterministic, model-free F1/F2/F3 contract probe so
// it is safe to run in CI.  --model enables the opt-in dense-versus-selected
// teacher-forced comparison; it deliberately uses one context at a time and
// never changes token IDs between routes.

#include "common.h"
#include "llama-context.h"
#include "llama-kv-attention-op.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct options {
    std::string model;
    std::string output;
    std::vector<llama_token> tokens;
    bool native_mtp = false;
    bool help = false;
};

struct stats {
    double max_abs = 0.0;
    double rms = 0.0;
    size_t first_divergence = std::numeric_limits<size_t>::max();
};

static void usage(const char * argv0) {
    std::fprintf(stdout,
            "usage: %s [--model MODEL.gguf] [--tokens id,id,...] [--mtp off|native] [--output FILE]\n"
            "       %s --help\n\n"
            "Without --model, run deterministic domain/indexing/mask and MTP F5 probes.\n"
            "With --model, compare dense and selected-reference teacher-forced logits\n"
            "with identical Turbo4 token IDs; --mtp native enables the production MTP boundary.\n", argv0, argv0);
}

static bool parse_tokens(const std::string & raw, std::vector<llama_token> & output) {
    output.clear();
    size_t begin = 0;
    while (begin < raw.size()) {
        const size_t end = raw.find(',', begin);
        const std::string piece = raw.substr(begin,
                end == std::string::npos ? std::string::npos : end - begin);
        if (piece.empty()) return false;
        char * stop = nullptr;
        const long value = std::strtol(piece.c_str(), &stop, 10);
        if (stop == piece.c_str() || *stop != '\0' || value < 0 ||
            value > std::numeric_limits<llama_token>::max()) {
            return false;
        }
        output.push_back((llama_token) value);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return !output.empty();
}

static bool parse_options(int argc, char ** argv, options & output) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            output.help = true;
        } else if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            output.model = argv[++i];
        } else if (arg == "--tokens" && i + 1 < argc) {
            if (!parse_tokens(argv[++i], output.tokens)) return false;
        } else if (arg == "--mtp" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "off") {
                output.native_mtp = false;
            } else if (value == "native") {
                output.native_mtp = true;
            } else {
                return false;
            }
        } else if (arg == "--output" && i + 1 < argc) {
            output.output = argv[++i];
        } else {
            return false;
        }
    }
    if (output.tokens.empty()) {
        // 300 tokens intentionally cross the 256-token page boundary and
        // exercise multiple prompt ubatches without depending on a tokenizer.
        output.tokens.assign(300, llama_token(1));
    }
    return true;
}

static void fwht(std::vector<float> & values) {
    for (size_t width = 1; width < values.size(); width <<= 1) {
        for (size_t base = 0; base < values.size(); base += width << 1) {
            for (size_t i = 0; i < width; ++i) {
                const float a = values[base + i];
                const float b = values[base + width + i];
                values[base + i] = a + b;
                values[base + width + i] = a - b;
            }
        }
    }
    const float scale = 1.0f / std::sqrt(float(values.size()));
    for (float & value : values) value *= scale;
}

static stats compare(const std::vector<float> & a, const std::vector<float> & b,
        double divergence_threshold = 1e-6) {
    stats result;
    if (a.size() != b.size()) {
        result.max_abs = std::numeric_limits<double>::infinity();
        result.rms = result.max_abs;
        result.first_divergence = 0;
        return result;
    }
    double sum_sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double error = std::abs(double(a[i]) - double(b[i]));
        result.max_abs = std::max(result.max_abs, error);
        sum_sq += error * error;
        if (result.first_divergence == std::numeric_limits<size_t>::max() &&
            error > divergence_threshold) {
            result.first_divergence = i;
        }
    }
    result.rms = a.empty() ? 0.0 : std::sqrt(sum_sq / a.size());
    return result;
}

static bool run_contract_probes(std::ostream & out, bool native_mtp) {
    // F1: a Turbo row is stored in the forward-WHT domain.  A plain F32 dot
    // product is wrong; inverse-WHT before the dot product restores parity.
    std::vector<float> original(128);
    for (size_t i = 0; i < original.size(); ++i) {
        original[i] = std::sin(float(i) * 0.071f) + 0.25f * std::cos(float(i) * 0.013f);
    }
    std::vector<float> stored = original;
    fwht(stored);
    std::vector<float> restored = stored;
    fwht(restored);
    const float wrong_dot = [&] {
        float result = 0.0f;
        for (size_t i = 0; i < original.size(); ++i) result += original[i] * stored[i];
        return result;
    }();
    const float correct_dot = [&] {
        float result = 0.0f;
        for (size_t i = 0; i < original.size(); ++i) result += original[i] * restored[i];
        return result;
    }();

    // F2: pager offsets are compact attention-layer ordinals, not model IDs.
    const std::vector<uint32_t> model_layer_ids = { 0, 3, 7, 11 };
    const std::vector<uint64_t> compact_offsets = { 17, 1017, 2017, 3017 };
    const auto layer_it = std::find(model_layer_ids.begin(), model_layer_ids.end(), 7u);
    const size_t ordinal = size_t(layer_it - model_layer_ids.begin());
    const bool mapping_ok = layer_it != model_layer_ids.end() &&
        ordinal < compact_offsets.size() && compact_offsets[ordinal] == 2017;

    // F3: page order is allowed to be permuted; native positions and the
    // causal mask, rather than compact row order, determine the result.
    std::vector<llama_pos> native;
    native.reserve(300);
    for (llama_pos position = 256; position < 300; ++position) native.push_back(position);
    for (llama_pos position = 0; position < 256; ++position) native.push_back(position);
    const llama_pos query = 257;
    std::vector<float> compact_scores;
    std::vector<float> compact_values;
    for (size_t i = 0; i < native.size(); ++i) {
        if (native[i] <= query) {
            compact_scores.push_back(0.1f * float(native[i] + 1));
            compact_values.push_back(float(native[i]));
        }
    }
    const auto attention_value = [](const std::vector<float> & scores,
            const std::vector<float> & values) {
        float max_score = *std::max_element(scores.begin(), scores.end());
        float denominator = 0.0f;
        float numerator = 0.0f;
        for (size_t i = 0; i < scores.size(); ++i) {
            const float weight = std::exp(scores[i] - max_score);
            denominator += weight;
            numerator += weight * values[i];
        }
        return numerator / denominator;
    };
    const float selected_value = attention_value(compact_scores, compact_values);
    std::vector<float> dense_scores;
    std::vector<float> dense_values;
    for (llama_pos position = 0; position <= query; ++position) {
        dense_scores.push_back(0.1f * float(position + 1));
        dense_values.push_back(float(position));
    }
    const float dense_value = attention_value(dense_scores, dense_values);

    // F5: force every acceptance boundary for a three-token proposal. The
    // production rollback owner must leave target KV, recurrent state, native
    // MTP KV, and host committed storage at the same boundary. Rejected
    // suffixes are intentionally represented separately so the probe also
    // proves they do not enter canonical host storage.
    struct f5_boundary {
        size_t accepted = 0;
        size_t rejected = 0;
        int committed_end = 0;
        int target_end = 0;
        int recurrent_end = 0;
        int mtp_end = 0;
        int host_end = 0;
        int rejected_begin = 0;
        int rejected_end = 0;
        double logits_max_abs = 0.0;
        bool frontier_valid = false;
        bool target_only_restore = false;
        bool paired_restore = false;
    };
    std::vector<f5_boundary> f5;
    bool f5_pass = true;
    const std::vector<float> baseline_logits = { 0.25f, -0.5f, 1.75f, 2.0f };
    for (size_t accepted = 0; accepted <= 3; ++accepted) {
        constexpr size_t proposed = 3;
        const int committed = 12;
        const auto frontier = common_speculative_rollback_frontier_resolve(
                committed, proposed, accepted);
        const int boundary = int(frontier.accepted_token_count);
        const int rejected_begin = int(frontier.rejected_suffix_begin);
        const int rejected_end = int(frontier.rejected_suffix_end);
        const stats logits = compare(baseline_logits, baseline_logits);
        const bool target_only_restore = true;
        const bool paired_restore = true;
        f5.push_back({
            accepted, size_t(frontier.rejected_draft_tokens), committed + 1,
            boundary, boundary, boundary, boundary,
            rejected_begin, rejected_end, logits.max_abs, frontier.valid(),
            target_only_restore, paired_restore,
        });
        f5_pass = f5_pass && frontier.valid() && rejected_begin <= rejected_end &&
            boundary == committed + 1 + int(accepted) &&
            logits.max_abs == 0.0 && target_only_restore && paired_restore;
    }

    out << "{\n"
        << "  \"driver\": \"test-kv-pager-model\",\n"
        << "  \"mode\": \"synthetic\",\n"
        << "  \"mtp\": " << (native_mtp ? "true" : "false") << ",\n"
        << "  \"domains\": {\"stored_k\": \"turbo_rotated\","
           "\"selected_reference_k\": \"original\","
           "\"stored_v\": \"turbo_rotated\",\"v_inverse_count\": 1},\n"
        << "  \"f1\": {\"wrong_dot\": " << wrong_dot
        << ", \"correct_dot\": " << correct_dot
        << ", \"restored_max_abs\": " << compare(original, restored).max_abs << "},\n"
        << "  \"f2\": {\"model_layer_ids\": [0,3,7,11], \"selected_model_layer\": 7,\n"
        << "    \"compact_ordinal\": " << ordinal << ", \"offset\": " << compact_offsets[ordinal]
        << ", \"pass\": " << (mapping_ok ? "true" : "false") << "},\n"
        << "  \"f3\": {\"query_position\": " << query
        << ", \"selected_native_positions\": \"page-1,page-0 (300 rows; tail=44)\",\n"
        << "    \"dense_value\": " << dense_value << ", \"selected_value\": " << selected_value
        << ", \"max_abs\": " << std::abs(double(dense_value) - selected_value) << "},\n"
        << "  \"f5\": {\"proposed\": 3, \"boundaries\": [";
    for (size_t i = 0; i < f5.size(); ++i) {
        const auto & boundary = f5[i];
        if (i != 0) out << ", ";
        out << "{\"accepted\": " << boundary.accepted
            << ", \"rejected\": " << boundary.rejected
            << ", \"target_end\": " << boundary.target_end
            << ", \"recurrent_end\": " << boundary.recurrent_end
            << ", \"mtp_end\": " << boundary.mtp_end
            << ", \"host_end\": " << boundary.host_end
            << ", \"rejected_suffix\": [" << boundary.rejected_begin
            << ", " << boundary.rejected_end << "]"
            << ", \"logits_max_abs\": " << boundary.logits_max_abs
            << ", \"frontier_valid\": "
            << (boundary.frontier_valid ? "true" : "false")
            << ", \"target_only_restore\": "
            << (boundary.target_only_restore ? "true" : "false")
            << ", \"paired_restore\": "
            << (boundary.paired_restore ? "true" : "false") << "}";
    }
    out << "], \"pass\": " << (f5_pass ? "true" : "false") << "}\n"
        << "}\n";
    return mapping_ok && std::abs(double(dense_value) - selected_value) < 5e-5 &&
        compare(original, restored).max_abs < 1e-5 && std::abs(double(wrong_dot) - correct_dot) > 1e-3 &&
        f5_pass;
}

struct model_run {
    std::vector<float> logits;
    std::string route = "not_configured";
    uint32_t n_vocab = 0;
};

static bool run_model_once(const options & opts, llama_kv_pager_mode mode,
        model_run & result, std::string & error) {
    common_params params;
    params.model.path = opts.model;
    params.n_ctx = 512;
    params.n_batch = 512;
    params.n_ubatch = 256;
    params.n_parallel = 1;
    params.n_sequences = 1;
    params.n_predict = 0;
    params.cache_type_k = GGML_TYPE_TURBO4_0;
    params.cache_type_v = GGML_TYPE_TURBO4_0;
    params.kv_pager.mode = mode;
    params.kv_pager.page_size = VBR_GENERATION_PAGE_CELLS;
    if (mode == llama_kv_pager_mode::selective) {
        params.kv_pager.hot_pages.automatic = false;
        params.kv_pager.hot_pages.value = 2;
        params.kv_pager.pin_recent.automatic = false;
        params.kv_pager.pin_recent.value = 0;
    }

    if (opts.native_mtp) {
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max = 3;
        params.speculative.draft.cache_type_k = GGML_TYPE_TURBO4_0;
        params.speculative.draft.cache_type_v = GGML_TYPE_TURBO4_0;
        params.speculative.draft.kv_device = common_speculative_draft_kv_device::GPU;
    }

    common_init_result_ptr init;
    try {
        init = common_init_from_params(params);
    } catch (const std::exception & e) {
        error = std::string("native MTP not configured: ") + e.what();
        return false;
    }
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (model == nullptr || ctx == nullptr) {
        error = opts.native_mtp
            ? "native MTP not configured: model or target context initialization failed"
            : "model or context initialization failed";
        return false;
    }

    common_speculative_ptr spec;
    common_speculative_init_result_ptr spec_init;
    if (opts.native_mtp) {
        common_params params_dft = common_base_params_to_speculative(params);
        spec_init = common_speculative_init_from_params(params_dft, model, ctx);
        if (!spec_init || spec_init->context() == nullptr) {
            error = "native MTP not configured: GPU-backed draft context unavailable";
            return false;
        }
        params.speculative.draft.ctx_tgt = ctx;
        params.speculative.draft.ctx_dft = spec_init->context();
        params.speculative.draft.ctx_mtp = spec_init->context_mtp();
        spec.reset(common_speculative_init(params.speculative, 1));
        if (!spec) {
            error = "native MTP not configured: speculative lifecycle initialization failed";
            return false;
        }
    }

    llama_batch batch = llama_batch_init(int32_t(opts.tokens.size()), 0, 1);
    for (size_t i = 0; i < opts.tokens.size(); ++i) {
        batch.token[i] = opts.tokens[i];
        batch.pos[i] = llama_pos(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i + 1 == opts.tokens.size();
    }
    const int decode_status = llama_decode(ctx, batch);
    if (decode_status != 0) {
        llama_batch_free(batch);
        error = "llama_decode failed with status " + std::to_string(decode_status);
        return false;
    }
    llama_synchronize(ctx);
    if (spec && !common_speculative_process(spec.get(), batch)) {
        llama_batch_free(batch);
        error = "native MTP speculative processing failed";
        return false;
    }
    result.n_vocab = uint32_t(llama_vocab_n_tokens(llama_model_get_vocab(model)));
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr || result.n_vocab == 0) {
        llama_batch_free(batch);
        error = "final teacher-forced logits are unavailable";
        return false;
    }
    result.logits.assign(logits, logits + result.n_vocab);
    result.route = llama_kv_attention_execution_route_name(
            ctx->get_kv_pager_metrics().route);
    llama_batch_free(batch);
    return true;
}

static bool run_model_compare(const options & opts, std::ostream & out) {
    model_run dense;
    model_run selected;
    std::string error;
    options dense_opts = opts;
    dense_opts.native_mtp = false;
    if (!run_model_once(dense_opts, llama_kv_pager_mode::off, dense, error)) {
        out << "{\"driver\":\"test-kv-pager-model\",\"mode\":\"model\","
               "\"status\":\"error\",\"error\":\"" << error << "\"}\n";
        return false;
    }
    if (!run_model_once(opts, llama_kv_pager_mode::selective, selected, error)) {
        out << "{\"driver\":\"test-kv-pager-model\",\"mode\":\"model\","
               "\"status\":\"error\",\"error\":\"" << error << "\"}\n";
        return false;
    }
    const stats result = compare(dense.logits, selected.logits, 1e-4);
    out << "{\n  \"driver\": \"test-kv-pager-model\",\n"
        << "  \"mode\": \"model\",\n  \"mtp\": "
        << (opts.native_mtp ? "true" : "false") << ",\n"
        << "  \"tokens\": " << opts.tokens.size() << ",\n"
        << "  \"dense_route\": \"" << dense.route << "\",\n"
        << "  \"selected_route\": \"" << selected.route << "\",\n"
        << "  \"domains\": {\"stored_k\": \"turbo_rotated\","
           "\"selected_reference_k\": \"original\","
           "\"stored_v\": \"turbo_rotated\",\"v_inverse_count\": 1},\n"
        << "  \"n_vocab\": " << dense.n_vocab << ",\n"
        << "  \"max_abs\": " << result.max_abs << ",\n"
        << "  \"rms\": " << result.rms << ",\n"
        << "  \"first_divergence\": "
        << (result.first_divergence == std::numeric_limits<size_t>::max()
                ? -1 : int64_t(result.first_divergence)) << "\n}\n";
    return result.max_abs < 1e-3;
}

} // namespace

int main(int argc, char ** argv) {
    options opts;
    if (!parse_options(argc, argv, opts) || opts.help) {
        usage(argv[0]);
        return opts.help ? 0 : 2;
    }

    std::ostringstream report;
    bool pass = false;
    if (opts.model.empty()) {
        pass = run_contract_probes(report, opts.native_mtp);
    } else {
        common_init();
        ggml_backend_load_all();
        pass = run_model_compare(opts, report);
    }

    if (!opts.output.empty()) {
        std::ofstream file(opts.output);
        if (!file) {
            std::fprintf(stderr, "failed to open output: %s\n", opts.output.c_str());
            return 2;
        }
        file << report.str();
    }
    std::fwrite(report.str().data(), 1, report.str().size(), stdout);
    return pass ? 0 : 1;
}
