#include "server-context.h"

#include "common.h"
#include "llama-context.h"
#include "llama-kv-residency.h"
#include "llama-kv-router-job.h"
#include "llama-kv-pager.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
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
};

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
        else if (arg == "--output" && i + 1 < argc) out.output = argv[++i];
        else return false;
    }
    return !out.model.empty() && (out.router == "legacy" || out.router == "probe-rerank") &&
        out.ubatch <= out.batch &&
        out.context > out.hot_tokens && out.hot_tokens >= 512 &&
        out.hot_tokens % VBR_GENERATION_PAGE_CELLS == 0;
}

bool decode_one(llama_context * ctx, common_speculative * spec,
        llama_token token, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
        llama_batch_free(batch);
        return false;
    }
    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = pos;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
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
        uint32_t count) {
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
            batch.token[i] = 1;
            batch.pos[i] = llama_pos(offset + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
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
    const bool rollback = finite && frontier.valid() &&
        llama_memory_seq_rm_transient(llama_get_memory(target), 0,
            llama_pos(frontier.accepted_token_count), -1) &&
        common_speculative_rollback_dft(spec, 0,
            llama_pos(frontier.accepted_token_count), uint16_t(accepted));
    if (rollback) {
        common_speculative_accept(spec, 0, uint16_t(accepted));
        llama_synchronize(target);
        llama_synchronize(draft);
        target_frontier = llama_memory_seq_pos_max(llama_get_memory(target), 0);
        draft_frontier = llama_memory_seq_pos_max(llama_get_memory(draft), 0);
    }
    llama_batch_free(verify);
    return rollback;
}

bool run(const options & opts) {
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

    // Fill beyond the 16-page target admission so this single real-model path
    // exercises resident rerank and host-cold key readers together while
    // leaving ample room under --L for the query and MTP verification rows.
    const uint32_t prefix_count = 4608;
    const llama_pos query_begin = llama_pos(prefix_count);
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
    if (!decode_prefix(target, spec.get(), prefix_count)) {
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
    target->begin_kv_pager_turn(0, turn_id, query_begin, query_begin + 1);
    std::vector<llama_kv_pager_selected_history> final_history;
    std::vector<llama_kv_pager_selected_history> committed_history;
    observation replay, control;
    bool selection_installed = false, replay_map_exact = false, control_map_exact = false;
    bool router_execution_observed = false;
    uint32_t router_stage = 0, resident_rerank_graphs = 0, cold_reader_graphs = 0;
    uint32_t reader_events = 0, mass_graphs = 0, router_output_records = 0;
    const auto decode_query_and_suffix = [&](observation & output) {
        if (!decode_one(target, spec.get(), 2, query_begin) ||
                !observe(target, draft, spec.get(), query_begin, output)) return false;
        if (!decode_one(target, spec.get(), 3, query_begin + 1) ||
                !decode_one(target, spec.get(), 4, query_begin + 2)) return false;
        return observe(target, draft, spec.get(), query_begin, output);
    };
    const auto fixture = server_query_checkpoint_replay_for_test(
            target, draft, spec.get(), 0, turn_id, turn_id,
            query_begin, prefix_count, true,
            [&]() { return decode_one(target, spec.get(), 2, query_begin); },
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
                if (!decode_query_and_suffix(replay)) return false;
                std::vector<llama_kv_pager_selected_history> replay_history;
                replay_map_exact = target->get_kv_pager_history_for_test(0, replay_history) &&
                    same_history(committed_history, replay_history);
                return replay_map_exact;
            },
            [&]() {
                std::vector<llama_kv_pager_selected_history> before;
                control_map_exact = target->get_kv_pager_history_for_test(0, before) &&
                    same_history(committed_history, before);
                return control_map_exact && decode_query_and_suffix(control);
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

    const bool parity = reservation_owner_survived && fixture.captured && fixture.provisional_decode_succeeded &&
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
        replay.target_frontier == query_begin + 2 &&
        replay.draft_frontier == query_begin + 2 &&
        replay.target_frontier == control.target_frontier &&
        replay.draft_frontier == control.draft_frontier &&
        mtp_target_frontier == mtp_begin + llama_pos(mtp_accepted) &&
        mtp_draft_frontier == mtp_begin + llama_pos(mtp_accepted) &&
        replay.recurrent == control.recurrent &&
        replay.draft_recurrent == control.draft_recurrent && replay.carry == control.carry &&
        target_kv_equal && draft_kv_equal &&
        close_vectors(replay.logits, control.logits, 0.08f) &&
        !replay.hidden.empty() && close_vectors(replay.hidden, control.hidden, 0.02f);

    size_t target_kv_first_diff = 0, draft_kv_first_diff = 0;
    const size_t target_kv_diff_bytes = byte_difference_count(
            replay.encoded_kv, control.encoded_kv, target_kv_first_diff);
    const size_t draft_kv_diff_bytes = byte_difference_count(
            replay.draft_encoded_kv, control.draft_encoded_kv, draft_kv_first_diff);

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
        << "}\n";
    return parity;
}

} // namespace

int main(int argc, char ** argv) {
    options opts;
    if (!parse_options(argc, argv, opts)) {
        std::fprintf(stderr,
                "usage: %s --model MODEL.gguf [--router legacy|probe-rerank] [--fresh-only] [--B 1024] [--U 256] [--L 8192] [--H 4096] [--owner-only] [--output FILE]\n",
                argv[0]);
        return 2;
    }
    llama_backend_init();
    const bool passed = run(opts);
    llama_backend_free();
    return passed ? 0 : 1;
}
