#include "llama-kv-attention-exact.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>
#include <utility>

namespace {

bool finite_or_neg_inf(float value) noexcept {
    return std::isfinite(value) || value == -INFINITY;
}

bool state_valid(const llama_kv_attention_online_state & state, uint32_t & dim) noexcept {
    dim = uint32_t(state.weighted_value.size());
    if (dim == 0 || !finite_or_neg_inf(state.max_logit) ||
        !std::isfinite(state.sum_exp) || state.sum_exp < 0.0f) {
        return false;
    }
    if (state.max_logit == -INFINITY) {
        if (state.sum_exp != 0.0f) return false;
        for (const float value : state.weighted_value) if (value != 0.0f) return false;
        return true;
    }
    if (state.sum_exp <= 0.0f) return false;
    for (const float value : state.weighted_value) if (!std::isfinite(value)) return false;
    return true;
}

bool add_u64(uint64_t a, uint64_t b, uint64_t & output) noexcept {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    output = a + b;
    return true;
}

bool resident_state(llama_kv_page_state state) noexcept {
    return state == llama_kv_page_state::filling_gpu ||
           state == llama_kv_page_state::gpu_host_clean ||
           state == llama_kv_page_state::gpu_dirty;
}

const llama_kv_page_record * find_page(
        const llama_kv_residency_snapshot & snapshot, uint32_t logical_page) noexcept {
    for (const auto & page : snapshot.pages()) {
        if (page.id.logical_page == logical_page) return &page;
    }
    return nullptr;
}

} // namespace

const char * llama_kv_attention_exact_status_name(
        llama_kv_attention_exact_status status) noexcept {
    switch (status) {
        case llama_kv_attention_exact_status::ok:                return "ok";
        case llama_kv_attention_exact_status::invalid_argument:  return "invalid_argument";
        case llama_kv_attention_exact_status::empty_partition:   return "empty_partition";
        case llama_kv_attention_exact_status::invalid_state:     return "invalid_state";
        case llama_kv_attention_exact_status::invalid_page:      return "invalid_page";
        case llama_kv_attention_exact_status::duplicate_page:   return "duplicate_page";
        case llama_kv_attention_exact_status::missing_page:      return "missing_page";
        case llama_kv_attention_exact_status::stale_page:       return "stale_page";
        case llama_kv_attention_exact_status::invalid_wave:     return "invalid_wave";
        case llama_kv_attention_exact_status::overflow:          return "overflow";
        case llama_kv_attention_exact_status::not_configured:   return "not_configured";
    }
    return "invalid";
}

bool llama_kv_attention_online_state_valid(
        const llama_kv_attention_online_state & state) noexcept {
    uint32_t dimension = 0;
    return state_valid(state, dimension);
}

llama_kv_attention_exact_status llama_kv_attention_online_state_from_rows(
        const float * logits,
        const float * values,
        uint32_t n_rows,
        uint32_t value_dim,
        const uint8_t * valid_mask,
        llama_kv_attention_online_state & output) noexcept {
    if (logits == nullptr || values == nullptr || n_rows == 0 || value_dim == 0) {
        return llama_kv_attention_exact_status::invalid_argument;
    }
    try {
        output.max_logit = -INFINITY;
        output.sum_exp = 0.0f;
        output.weighted_value.assign(value_dim, 0.0f);

        for (uint32_t row = 0; row < n_rows; ++row) {
            if (valid_mask != nullptr && valid_mask[row] == 0) continue;
            if (!std::isfinite(logits[row])) return llama_kv_attention_exact_status::invalid_argument;
            output.max_logit = std::max(output.max_logit, logits[row]);
        }
        if (output.max_logit == -INFINITY) {
            return llama_kv_attention_exact_status::empty_partition;
        }

        for (uint32_t row = 0; row < n_rows; ++row) {
            if (valid_mask != nullptr && valid_mask[row] == 0) continue;
            const float weight = std::exp(logits[row] - output.max_logit);
            output.sum_exp += weight;
            for (uint32_t d = 0; d < value_dim; ++d) {
                output.weighted_value[d] += weight * values[size_t(row) * value_dim + d];
            }
        }
        if (!std::isfinite(output.sum_exp) || output.sum_exp <= 0.0f) {
            output = {};
            return llama_kv_attention_exact_status::invalid_state;
        }
        for (const float value : output.weighted_value) {
            if (!std::isfinite(value)) {
                output = {};
                return llama_kv_attention_exact_status::invalid_argument;
            }
        }
        return llama_kv_attention_exact_status::ok;
    } catch (const std::bad_alloc &) {
        output = {};
        return llama_kv_attention_exact_status::overflow;
    }
}

llama_kv_attention_exact_status llama_kv_attention_online_state_merge(
        const llama_kv_attention_online_state & a,
        const llama_kv_attention_online_state & b,
        llama_kv_attention_online_state & output) noexcept {
    uint32_t dim_a = 0;
    uint32_t dim_b = 0;
    if (!state_valid(a, dim_a) || !state_valid(b, dim_b) || dim_a != dim_b) {
        return llama_kv_attention_exact_status::invalid_state;
    }
    if (a.empty()) {
        if (&output != &b) output = b;
        return llama_kv_attention_exact_status::ok;
    }
    if (b.empty()) {
        if (&output != &a) output = a;
        return llama_kv_attention_exact_status::ok;
    }
    try {
        const float max_logit = std::max(a.max_logit, b.max_logit);
        const float scale_a = std::exp(a.max_logit - max_logit);
        const float scale_b = std::exp(b.max_logit - max_logit);
        output.max_logit = max_logit;
        output.sum_exp = a.sum_exp * scale_a + b.sum_exp * scale_b;
        output.weighted_value.resize(dim_a);
        for (uint32_t d = 0; d < dim_a; ++d) {
            output.weighted_value[d] = a.weighted_value[d] * scale_a +
                b.weighted_value[d] * scale_b;
        }
        if (!std::isfinite(output.sum_exp) || output.sum_exp <= 0.0f) {
            return llama_kv_attention_exact_status::invalid_state;
        }
        for (const float value : output.weighted_value) {
            if (!std::isfinite(value)) return llama_kv_attention_exact_status::invalid_state;
        }
        return llama_kv_attention_exact_status::ok;
    } catch (const std::bad_alloc &) {
        return llama_kv_attention_exact_status::overflow;
    }
}

llama_kv_attention_exact_status llama_kv_attention_online_state_normalize(
        const llama_kv_attention_online_state & state,
        float * output,
        uint32_t value_dim) noexcept {
    if (output == nullptr || value_dim == 0) {
        return llama_kv_attention_exact_status::invalid_argument;
    }
    uint32_t state_dim = 0;
    if (!state_valid(state, state_dim) || state_dim != value_dim) {
        return llama_kv_attention_exact_status::invalid_state;
    }
    if (state.empty()) {
        std::fill(output, output + value_dim, 0.0f);
        return llama_kv_attention_exact_status::empty_partition;
    }
    for (uint32_t d = 0; d < value_dim; ++d) output[d] = state.weighted_value[d] / state.sum_exp;
    return llama_kv_attention_exact_status::ok;
}

llama_kv_attention_exact_wave_plan llama_kv_attention_exact_wave_plan::build(
        const std::vector<llama_kv_attention_exact_page> & pages,
        const llama_kv_residency_snapshot & resident_snapshot,
        const llama_kv_attention_exact_config & config,
        llama_kv_attention_exact_status & status) noexcept {
    llama_kv_attention_exact_wave_plan output;
    status = llama_kv_attention_exact_status::invalid_argument;
    if (pages.empty() || pages.size() > 1024 || config.pages_per_wave == 0 ||
        config.staging_slots == 0 ||
        (config.schedule == llama_kv_attention_exact_schedule::double_buffered &&
            config.staging_slots < 2)) {
        return output;
    }
    uint32_t largest_logical_page = 0;
    for (const auto & page : pages) {
        if (page.id.logical_page == UINT32_MAX) return output;
        largest_logical_page = std::max(largest_logical_page, page.id.logical_page);
    }
    if (largest_logical_page == UINT32_MAX) return output;
    const uint32_t page_count = config.logical_page_count == 0
        ? largest_logical_page + 1 : config.logical_page_count;
    if (page_count == 0 || page_count > 1024 || pages.size() > page_count ||
        config.pages_per_wave == 0 || config.staging_slots == 0 ||
        (config.schedule == llama_kv_attention_exact_schedule::double_buffered &&
            config.staging_slots < 2)) {
        return output;
    }
    if (resident_snapshot.epoch() == 0) {
        status = llama_kv_attention_exact_status::stale_page;
        return output;
    }

    try {
        std::vector<llama_kv_attention_exact_page> sorted = pages;
        std::sort(sorted.begin(), sorted.end(), [](const auto & a, const auto & b) {
            return a.id.logical_page < b.id.logical_page;
        });
        output.visited_.assign(page_count, 0);
        output.expected_.assign(page_count, 0);
        output.ledger_.logical_page_count = page_count;

        for (size_t i = 0; i < sorted.size(); ++i) {
            const auto & page = sorted[i];
            if (page.id.logical_page >= page_count) {
                status = llama_kv_attention_exact_status::invalid_page;
                return output;
            }
            if (i != 0 && sorted[i - 1].id.logical_page == page.id.logical_page) {
                status = llama_kv_attention_exact_status::duplicate_page;
                return output;
            }
            if (!llama_kv_page_id_valid(page.id, page.id.logical_page + 1 == page_count)) {
                status = llama_kv_attention_exact_status::invalid_page;
                return output;
            }
            const uint64_t page_length = uint64_t(page.id.position_end) - page.id.position_begin;
            if (page.valid_tokens != page_length || page.valid_tokens == 0) {
                status = llama_kv_attention_exact_status::invalid_page;
                return output;
            }
            if (page.resident) {
                const auto * resident = find_page(
                        resident_snapshot, page.id.logical_page);
                if (resident == nullptr || resident->id != page.id ||
                    resident->physical_slot != page.physical_slot ||
                    !resident_state(resident->state)) {
                    ++output.ledger_.stale_pages;
                    status = llama_kv_attention_exact_status::stale_page;
                    return output;
                }
                if (page.physical_slot == UINT32_MAX || page.physical_slot >= resident_snapshot.slot_capacity()) {
                    status = llama_kv_attention_exact_status::invalid_page;
                    return output;
                }
            } else if (!page.host_valid || page.physical_slot != UINT32_MAX) {
                status = llama_kv_attention_exact_status::invalid_page;
                return output;
            }
            output.expected_[page.id.logical_page] = 1;
            if (!add_u64(output.ledger_.valid_tokens, page.valid_tokens, output.ledger_.valid_tokens)) {
                status = llama_kv_attention_exact_status::overflow;
                return output;
            }
            if (page.resident) ++output.ledger_.resident_pages;
            else ++output.ledger_.cold_pages;
        }

        // Hot pages are made one deterministic prefix of the plan. Cold pages
        // are then batched by logical order; this makes output independent of
        // selected physical slots and of wave size.
        std::vector<llama_kv_attention_exact_page> ordered;
        ordered.reserve(sorted.size());
        for (const auto & page : sorted) if (page.resident) ordered.push_back(page);
        const uint32_t hot_begin = uint32_t(ordered.size());
        for (const auto & page : sorted) if (!page.resident) ordered.push_back(page);
        output.schedule_ = config.schedule;
        output.staging_slots_ = config.staging_slots;

        auto append_wave = [&](uint32_t index, uint32_t begin, uint32_t end, bool cold) {
            llama_kv_attention_exact_wave wave;
            wave.index = index;
            wave.contains_cold_pages = cold;
            wave.staging_slot = cold ? index % config.staging_slots : UINT32_MAX;
            for (uint32_t i = begin; i < end; ++i) {
                const auto & page = ordered[i];
                wave.pages.push_back({ page.id.logical_page, page.valid_tokens,
                    page.physical_slot, !page.resident,
                    page.id.position_begin, page.id.position_end });
            }
            output.waves_.push_back(std::move(wave));
        };

        uint32_t wave_index = 0;
        if (hot_begin != 0) append_wave(wave_index++, 0, hot_begin, false);
        for (uint32_t begin = hot_begin; begin < ordered.size(); begin += config.pages_per_wave) {
            const uint32_t end = std::min(uint32_t(ordered.size()),
                    begin + config.pages_per_wave);
            append_wave(wave_index++, begin, end, true);
        }
        output.ledger_.waves = output.waves_.size();
        output.ledger_.peak_staging_pages = 0;
        for (const auto & wave : output.waves_) {
            if (wave.contains_cold_pages) {
                output.ledger_.peak_staging_pages = std::max<uint64_t>(
                    output.ledger_.peak_staging_pages, wave.pages.size());
                if (config.page_bytes != 0 && wave.pages.size() >
                        std::numeric_limits<uint64_t>::max() / config.page_bytes) {
                    status = llama_kv_attention_exact_status::overflow;
                    return output;
                }
                const uint64_t bytes = uint64_t(wave.pages.size()) * config.page_bytes;
                if (!add_u64(output.ledger_.h2d_useful_bytes, bytes, output.ledger_.h2d_useful_bytes) ||
                    !add_u64(output.ledger_.h2d_aligned_bytes, bytes, output.ledger_.h2d_aligned_bytes)) {
                    status = llama_kv_attention_exact_status::overflow;
                    return output;
                }
            }
        }
        output.valid_ = true;
        status = llama_kv_attention_exact_status::ok;
        return output;
    } catch (const std::bad_alloc &) {
        status = llama_kv_attention_exact_status::overflow;
        return {};
    }
}

llama_kv_attention_exact_status llama_kv_attention_exact_wave_plan::record_visit(
        uint32_t logical_page) noexcept {
    if (!valid_) return llama_kv_attention_exact_status::invalid_argument;
    if (logical_page >= visited_.size() || expected_[logical_page] == 0) {
        return llama_kv_attention_exact_status::invalid_page;
    }
    if (visited_[logical_page] != 0) {
        ++ledger_.duplicate_pages;
        return llama_kv_attention_exact_status::duplicate_page;
    }
    visited_[logical_page] = 1;
    ++ledger_.pages_visited;
    return llama_kv_attention_exact_status::ok;
}

llama_kv_attention_exact_status llama_kv_attention_exact_wave_plan::finish() noexcept {
    if (!valid_) return llama_kv_attention_exact_status::invalid_argument;
    for (size_t i = 0; i < visited_.size(); ++i) {
        if (expected_[i] != 0 && visited_[i] == 0) {
            ++ledger_.missing_pages;
            return llama_kv_attention_exact_status::missing_page;
        }
    }
    return llama_kv_attention_exact_status::ok;
}

llama_kv_attention_exact_status llama_kv_attention_exact_executor::execute(
        llama_kv_attention_exact_wave_plan & plan,
        const llama_kv_attention_exact_backend & backend,
        llama_kv_attention_online_state & output) noexcept {
    if (!plan.valid() || backend.compute_wave == nullptr) {
        return llama_kv_attention_exact_status::invalid_argument;
    }
    if (plan.waves().empty()) return llama_kv_attention_exact_status::invalid_wave;

    output = {};
    bool have_output = false;
    bool next_wave_preuploaded = false;
    const auto record_transfer_time = [&](std::chrono::steady_clock::time_point start) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start).count();
        if (elapsed > 0) {
            const uint64_t value = uint64_t(elapsed);
            plan.ledger_.h2d_transfer_time_us = value >
                UINT64_MAX - plan.ledger_.h2d_transfer_time_us
                ? UINT64_MAX : plan.ledger_.h2d_transfer_time_us + value;
        }
    };
    for (size_t wave_index = 0; wave_index < plan.waves().size(); ++wave_index) {
        const auto & wave = plan.waves()[wave_index];
        if (wave.pages.empty() || (wave.contains_cold_pages &&
                wave.staging_slot >= plan.staging_slots_) || (wave.contains_cold_pages &&
                (backend.upload_cold_wave == nullptr || backend.wait_wave == nullptr))) {
            return llama_kv_attention_exact_status::not_configured;
        }
        if (wave.contains_cold_pages) {
            if (!next_wave_preuploaded) {
                const bool asynchronous = plan.schedule_ ==
                    llama_kv_attention_exact_schedule::double_buffered;
                const auto transfer_start = std::chrono::steady_clock::now();
                if (!backend.upload_cold_wave(backend.context, wave, wave.staging_slot, asynchronous)) {
                    return llama_kv_attention_exact_status::not_configured;
                }
                record_transfer_time(transfer_start);
            }
            next_wave_preuploaded = false;
            ++plan.ledger_.waits;
            const auto wait_start = std::chrono::steady_clock::now();
            if (!backend.wait_wave(backend.context, wave)) {
                return llama_kv_attention_exact_status::not_configured;
            }
            record_transfer_time(wait_start);
        }
        llama_kv_attention_online_state partial;
        if (!backend.compute_wave(backend.context, wave, partial)) {
            return llama_kv_attention_exact_status::not_configured;
        }
        if (!llama_kv_attention_online_state_valid(partial)) {
            return llama_kv_attention_exact_status::invalid_state;
        }
        for (const auto & page : wave.pages) {
            const auto status = plan.record_visit(page.logical_page);
            if (status != llama_kv_attention_exact_status::ok) return status;
        }
        if (!have_output) {
            output = std::move(partial);
            have_output = true;
        } else {
            const auto status = llama_kv_attention_online_state_merge(output, partial, output);
            if (status != llama_kv_attention_exact_status::ok) return status;
        }

        if (plan.schedule_ == llama_kv_attention_exact_schedule::double_buffered &&
            wave_index + 1 < plan.waves().size() &&
            plan.waves()[wave_index + 1].contains_cold_pages) {
            const auto & next = plan.waves()[wave_index + 1];
            const auto transfer_start = std::chrono::steady_clock::now();
            if (!backend.upload_cold_wave(backend.context, next, next.staging_slot, true)) {
                return llama_kv_attention_exact_status::not_configured;
            }
            record_transfer_time(transfer_start);
            next_wave_preuploaded = true;
        }
    }
    if (!have_output) return llama_kv_attention_exact_status::empty_partition;
    return plan.finish();
}
