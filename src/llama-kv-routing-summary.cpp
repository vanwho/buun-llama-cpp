#include "llama-kv-routing-summary.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>

namespace {
bool add(uint64_t a, uint64_t b, uint64_t & out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    out = a + b;
    return true;
}

bool mul(uint64_t a, uint64_t b, uint64_t & out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}

bool valid_state(llama_kv_page_state state) {
    return state != llama_kv_page_state::absent && state != llama_kv_page_state::invalid;
}

bool score_order(const llama_kv_routing_page_score & a, const llama_kv_routing_page_score & b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.logical_page != b.logical_page) return a.logical_page < b.logical_page;
    return a.page_generation < b.page_generation;
}
}

const char * llama_kv_routing_summary_status_name(
        llama_kv_routing_summary_status status) noexcept {
    switch (status) {
        case llama_kv_routing_summary_status::ok: return "ok";
        case llama_kv_routing_summary_status::invalid_argument: return "invalid_argument";
        case llama_kv_routing_summary_status::duplicate_page: return "duplicate_page";
        case llama_kv_routing_summary_status::missing_page: return "missing_page";
        case llama_kv_routing_summary_status::invalid_page: return "invalid_page";
        case llama_kv_routing_summary_status::stale_summary: return "stale_summary";
        case llama_kv_routing_summary_status::insufficient_budget: return "insufficient_budget";
        case llama_kv_routing_summary_status::overflow: return "overflow";
    }
    return "invalid";
}

llama_kv_routing_summary_store llama_kv_routing_summary_store::build(
        const llama_kv_residency_snapshot & snapshot,
        const std::vector<llama_kv_routing_page_input> & inputs,
        const llama_kv_routing_summary_config & config,
        llama_kv_routing_summary_status & status) noexcept {
    status = llama_kv_routing_summary_status::invalid_argument;
    llama_kv_routing_summary_store result;
    if (snapshot.epoch() == 0 || config.representative_count < 4 ||
        config.representative_count > 8 || config.vector_dim == 0 ||
        config.allocation_granularity == 0) return result;

    try {
        result.snapshot_epoch_ = snapshot.epoch();
        result.representative_count_ = config.representative_count;
        result.vector_dim_ = config.vector_dim;
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = i + 1; j < inputs.size(); ++j) {
                if (inputs[i].id.logical_page == inputs[j].id.logical_page) {
                    status = llama_kv_routing_summary_status::duplicate_page;
                    return {};
                }
            }
        }
        for (const auto & record : snapshot.pages()) {
            const bool tail = record.state == llama_kv_page_state::filling_gpu;
            if (!valid_state(record.state) || !llama_kv_page_id_valid(record.id, tail)) continue;
            const auto it = std::find_if(inputs.begin(), inputs.end(), [&](const auto & input) {
                return input.id == record.id;
            });
            if (it == inputs.end()) {
                status = llama_kv_routing_summary_status::missing_page;
                return {};
            }
            const uint64_t row_count = uint64_t(record.id.position_end - record.id.position_begin);
            uint64_t expected = 0;
            if (!mul(row_count, config.vector_dim, expected) || it->rotated_k_rows.size() != expected) {
                status = llama_kv_routing_summary_status::invalid_page;
                return {};
            }
            for (const auto & existing : result.pages_) {
                if (existing.id.logical_page == record.id.logical_page) {
                    status = llama_kv_routing_summary_status::duplicate_page;
                    return {};
                }
            }
            page summary;
            summary.id = record.id;
            summary.vectors.resize(size_t(config.representative_count) * config.vector_dim);
            for (uint32_t representative = 0; representative < config.representative_count; ++representative) {
                const uint64_t row = (uint64_t(representative) * (row_count - 1)) /
                                     (config.representative_count - 1);
                std::copy_n(it->rotated_k_rows.begin() + row * config.vector_dim,
                            config.vector_dim,
                            summary.vectors.begin() + size_t(representative) * config.vector_dim);
            }
            result.pages_.push_back(std::move(summary));
        }
        if (result.pages_.empty()) {
            status = llama_kv_routing_summary_status::invalid_page;
            return {};
        }

        uint64_t vectors = 0, payload = 0, metadata = 0, logical = 0, charged = 0;
        if (!mul(result.pages_.size(), config.representative_count, vectors) ||
            !mul(vectors, config.vector_dim, vectors) || !mul(vectors, sizeof(float), payload) ||
            !mul(result.pages_.size(), sizeof(llama_kv_page_id), metadata) ||
            !add(payload, metadata, logical)) {
            status = llama_kv_routing_summary_status::overflow;
            return {};
        }
        const uint64_t granularity = config.allocation_granularity;
        const uint64_t remainder = logical % granularity;
        if (!add(logical, remainder == 0 ? 0 : granularity - remainder, charged)) {
            status = llama_kv_routing_summary_status::overflow;
            return {};
        }
        if (config.byte_budget != 0 && charged > config.byte_budget) {
            status = llama_kv_routing_summary_status::insufficient_budget;
            return {};
        }
        result.accounting_ = { result.pages_.size(), config.representative_count, config.vector_dim,
                               payload, metadata, logical, charged };
        status = llama_kv_routing_summary_status::ok;
        return result;
    } catch (const std::bad_alloc &) {
        status = llama_kv_routing_summary_status::overflow;
        return {};
    }
}

llama_kv_routing_score_result llama_kv_routing_summary_store::score(
        const llama_kv_residency_snapshot & snapshot,
        const std::vector<float> & query, uint32_t top_k) const noexcept {
    llama_kv_routing_score_result result;
    result.summary_epoch = snapshot_epoch_;
    if (!valid() || snapshot.epoch() != snapshot_epoch_ || query.size() != vector_dim_) {
        result.status = llama_kv_routing_summary_status::stale_summary;
        if (query.size() != vector_dim_) result.status = llama_kv_routing_summary_status::invalid_argument;
        return result;
    }
    // Epochs are scoped to a residency table.  Identity validation prevents
    // an epoch-1 summary from one table being accepted for epoch 1 of another,
    // and catches page seal/mutation/representation changes.
    for (const auto & summary_page : pages_) {
        const auto it = std::find_if(snapshot.pages().begin(), snapshot.pages().end(),
                [&](const auto & record) { return record.id.logical_page == summary_page.id.logical_page; });
        if (it == snapshot.pages().end() || it->id != summary_page.id) {
            result.status = llama_kv_routing_summary_status::stale_summary;
            return result;
        }
    }
    const auto start = std::chrono::steady_clock::now();
    try {
        result.top_pages.reserve(pages_.size());
        for (const auto & page : pages_) {
            float best = -std::numeric_limits<float>::infinity();
            for (uint32_t representative = 0; representative < representative_count_; ++representative) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < vector_dim_; ++d) {
                    dot += page.vectors[size_t(representative) * vector_dim_ + d] * query[d];
                }
                best = std::max(best, dot);
                ++result.comparisons;
            }
            result.top_pages.push_back({ page.id.logical_page, page.id.page_generation, best });
        }
        result.pages_scored = result.top_pages.size();
        std::sort(result.top_pages.begin(), result.top_pages.end(), score_order);
        if (top_k < result.top_pages.size()) result.top_pages.resize(top_k);
        result.status = llama_kv_routing_summary_status::ok;
    } catch (const std::bad_alloc &) {
        result.status = llama_kv_routing_summary_status::overflow;
    }
    result.latency_us = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count());
    return result;
}
