#include "llama-kv-routing-summary.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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

bool valid_form(llama_kv_routing_summary_form form) {
    return form == llama_kv_routing_summary_form::representatives ||
           form == llama_kv_routing_summary_form::centroid_upper_bound;
}

bool score_order(const llama_kv_routing_page_score & a, const llama_kv_routing_page_score & b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.logical_page != b.logical_page) return a.logical_page < b.logical_page;
    return a.page_generation < b.page_generation;
}

uint64_t hash_mix(uint64_t hash, uint64_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
        hash ^= uint8_t(value >> (i * 8));
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t hash_float(uint64_t hash, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return hash_mix(hash, bits);
}

uint64_t hash_id(uint64_t hash, const llama_kv_page_id & id) {
    hash = hash_mix(hash, id.session_generation);
    hash = hash_mix(hash, uint32_t(id.sequence_id));
    hash = hash_mix(hash, id.sequence_generation);
    hash = hash_mix(hash, id.logical_page);
    hash = hash_mix(hash, id.page_generation);
    hash = hash_mix(hash, id.representation_epoch);
    hash = hash_mix(hash, id.model_identity);
    hash = hash_mix(hash, id.topology_identity);
    hash = hash_mix(hash, id.codec_digest);
    hash = hash_mix(hash, id.codebook_digest);
    hash = hash_mix(hash, id.rotation_digest);
    hash = hash_mix(hash, id.meansub_digest);
    hash = hash_mix(hash, uint64_t(id.position_begin));
    return hash_mix(hash, uint64_t(id.position_end));
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

namespace {
bool make_vectors(const llama_kv_routing_page_input & input,
                  uint64_t row_count,
                  const llama_kv_routing_summary_config & config,
                  std::vector<float> & vectors,
                  float & radius,
                  uint64_t & source_rows) {
    if (row_count == 0 || !valid_form(config.form)) return false;
    std::vector<uint32_t> rows;
    if (input.row_indices.empty()) {
        uint64_t expected = 0;
        if (!mul(row_count, config.vector_dim, expected) || input.rotated_k_rows.size() != expected) return false;
        rows.resize(config.representative_count);
        for (uint32_t i = 0; i < config.representative_count; ++i) {
            rows[i] = uint32_t((uint64_t(i) * (row_count - 1)) /
                               (config.representative_count - 1));
        }
        source_rows = row_count;
    } else {
        if (input.row_indices.size() != config.representative_count ||
            input.rotated_k_rows.size() != size_t(config.representative_count) * config.vector_dim) return false;
        rows = input.row_indices;
        source_rows = rows.size();
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i] >= row_count || (i != 0 && rows[i - 1] > rows[i])) return false;
        }
    }
    const auto row = [&](uint32_t i) {
        return input.row_indices.empty()
            ? input.rotated_k_rows.data() + size_t(rows[i]) * config.vector_dim
            : input.rotated_k_rows.data() + size_t(i) * config.vector_dim;
    };
    if (config.form == llama_kv_routing_summary_form::representatives) {
        vectors.resize(size_t(config.representative_count) * config.vector_dim);
        for (uint32_t i = 0; i < config.representative_count; ++i) {
            std::copy_n(row(i), config.vector_dim,
                        vectors.begin() + size_t(i) * config.vector_dim);
        }
        return true;
    }
    vectors.assign(config.vector_dim, 0.0f);
    for (uint32_t i = 0; i < config.representative_count; ++i) {
        for (uint32_t d = 0; d < config.vector_dim; ++d) vectors[d] += row(i)[d];
    }
    const float inverse = 1.0f / float(config.representative_count);
    for (float & value : vectors) value *= inverse;
    radius = 0.0f;
    for (uint32_t i = 0; i < config.representative_count; ++i) {
        double distance = 0.0;
        for (uint32_t d = 0; d < config.vector_dim; ++d) {
            const double delta = double(row(i)[d]) - vectors[d];
            distance += delta * delta;
        }
        radius = std::max(radius, float(std::sqrt(distance)));
    }
    return std::isfinite(radius);
}
}

llama_kv_routing_summary_store llama_kv_routing_summary_store::build(
        const llama_kv_residency_snapshot & snapshot,
        const std::vector<llama_kv_routing_page_input> & inputs,
        const llama_kv_routing_summary_config & config,
        llama_kv_routing_summary_status & status) noexcept {
    status = llama_kv_routing_summary_status::invalid_argument;
    llama_kv_routing_summary_store result;
    const auto start = std::chrono::steady_clock::now();
    if (snapshot.epoch() == 0 || config.representative_count < 4 || config.representative_count > 8 ||
        config.vector_dim == 0 || config.allocation_granularity == 0 || !valid_form(config.form)) return result;
    try {
        result.snapshot_epoch_ = snapshot.epoch();
        result.representative_count_ = config.representative_count;
        result.vector_dim_ = config.vector_dim;
        result.layer_index_ = config.layer_index;
        result.head_index_ = config.head_index;
        result.form_ = config.form;
        result.allocation_granularity_ = config.allocation_granularity;
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = i + 1; j < inputs.size(); ++j) {
                if (inputs[i].id.logical_page == inputs[j].id.logical_page) {
                    status = llama_kv_routing_summary_status::duplicate_page;
                    return {};
                }
            }
        }
        for (const auto & input : inputs) {
            const auto it = std::find_if(snapshot.pages().begin(), snapshot.pages().end(),
                    [&](const auto & record) { return record.id == input.id; });
            if (it == snapshot.pages().end()) {
                status = llama_kv_routing_summary_status::invalid_page;
                return {};
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
            page summary;
            summary.id = record.id;
            if (!make_vectors(*it, uint64_t(record.id.position_end - record.id.position_begin),
                              config, summary.vectors, summary.radius, summary.source_rows)) {
                status = llama_kv_routing_summary_status::invalid_page;
                return {};
            }
            summary.source_bytes = it->source_bytes;
            result.pages_.push_back(std::move(summary));
        }
        if (result.pages_.empty()) {
            status = llama_kv_routing_summary_status::invalid_page;
            return {};
        }
        result.rebuild_accounting(config, start);
        if (config.byte_budget != 0 && result.accounting_.charged_bytes > config.byte_budget) {
            status = llama_kv_routing_summary_status::insufficient_budget;
            return {};
        }
        status = llama_kv_routing_summary_status::ok;
        return result;
    } catch (const std::bad_alloc &) {
        status = llama_kv_routing_summary_status::overflow;
        return {};
    }
}

llama_kv_routing_summary_store llama_kv_routing_summary_store::update_page(
        const llama_kv_residency_snapshot & snapshot,
        const llama_kv_routing_page_input & input,
        const llama_kv_routing_summary_config & config,
        llama_kv_routing_summary_status & status) const noexcept {
    status = llama_kv_routing_summary_status::invalid_argument;
    const auto start = std::chrono::steady_clock::now();
    if (snapshot.epoch() == 0 || config.representative_count < 4 || config.representative_count > 8 ||
        config.vector_dim == 0 || config.allocation_granularity == 0 || !valid_form(config.form)) return {};
    if (!pages_.empty() && (representative_count_ != config.representative_count || vector_dim_ != config.vector_dim ||
        layer_index_ != config.layer_index || head_index_ != config.head_index || form_ != config.form)) {
        status = llama_kv_routing_summary_status::stale_summary;
        return {};
    }
    try {
        const auto record = std::find_if(snapshot.pages().begin(), snapshot.pages().end(),
                [&](const auto & page) { return page.id == input.id; });
        if (record == snapshot.pages().end()) {
            status = llama_kv_routing_summary_status::stale_summary;
            return {};
        }
        const bool tail = record->state == llama_kv_page_state::filling_gpu;
        if (!valid_state(record->state) || !llama_kv_page_id_valid(record->id, tail)) {
            status = llama_kv_routing_summary_status::invalid_page;
            return {};
        }
        llama_kv_routing_summary_status reconcile_status;
        llama_kv_routing_summary_store result = reconcile(snapshot, reconcile_status);
        if (reconcile_status != llama_kv_routing_summary_status::ok) {
            status = reconcile_status;
            return {};
        }
        result.snapshot_epoch_ = snapshot.epoch();
        result.representative_count_ = config.representative_count;
        result.vector_dim_ = config.vector_dim;
        result.layer_index_ = config.layer_index;
        result.head_index_ = config.head_index;
        result.form_ = config.form;
        result.allocation_granularity_ = config.allocation_granularity;
        page summary;
        summary.id = input.id;
        if (!make_vectors(input, uint64_t(record->id.position_end - record->id.position_begin), config,
                          summary.vectors, summary.radius, summary.source_rows)) {
            status = llama_kv_routing_summary_status::invalid_page;
            return {};
        }
        summary.source_bytes = input.source_bytes;
        const auto existing = std::find_if(result.pages_.begin(), result.pages_.end(),
                [&](const auto & value) { return value.id.logical_page == input.id.logical_page; });
        if (existing == result.pages_.end()) result.pages_.push_back(std::move(summary));
        else *existing = std::move(summary);
        std::sort(result.pages_.begin(), result.pages_.end(),
                [](const auto & lhs, const auto & rhs) {
                    return lhs.id.logical_page < rhs.id.logical_page;
                });
        const uint64_t previous_build_count = accounting_.build_count;
        result.rebuild_accounting(config, start);
        result.accounting_.build_count = previous_build_count + 1;
        if (config.byte_budget != 0 && result.accounting_.charged_bytes > config.byte_budget) {
            status = llama_kv_routing_summary_status::insufficient_budget;
            return {};
        }
        status = llama_kv_routing_summary_status::ok;
        return result;
    } catch (...) {
        status = llama_kv_routing_summary_status::overflow;
        return {};
    }
}

llama_kv_routing_summary_store llama_kv_routing_summary_store::reconcile(
        const llama_kv_residency_snapshot & snapshot,
        llama_kv_routing_summary_status & status) const noexcept {
    status = llama_kv_routing_summary_status::invalid_argument;
    if (snapshot.epoch() == 0) return {};
    try {
        llama_kv_routing_summary_store result = *this;
        result.snapshot_epoch_ = snapshot.epoch();
        const uint64_t previous_build_count = accounting_.build_count;
        const uint64_t before = result.pages_.size();
        result.pages_.erase(std::remove_if(result.pages_.begin(), result.pages_.end(),
                [&](const auto & summary) {
                    const auto it = std::find_if(snapshot.pages().begin(), snapshot.pages().end(),
                            [&](const auto & page) { return page.id.logical_page == summary.id.logical_page; });
                    return it == snapshot.pages().end() || it->id != summary.id;
                }), result.pages_.end());
        result.rebuild_accounting({}, std::chrono::steady_clock::now());
        result.accounting_.invalidation_count = accounting_.invalidation_count + before - result.pages_.size();
        result.accounting_.build_count = previous_build_count;
        status = llama_kv_routing_summary_status::ok;
        return result;
    } catch (...) {
        status = llama_kv_routing_summary_status::overflow;
        return {};
    }
}

llama_kv_routing_summary_store llama_kv_routing_summary_store::invalidate_page(
        const llama_kv_residency_snapshot & snapshot,
        uint32_t logical_page,
        llama_kv_routing_summary_status & status) const noexcept {
    status = llama_kv_routing_summary_status::invalid_argument;
    if (snapshot.epoch() == 0) return {};
    try {
        llama_kv_routing_summary_store result = *this;
        result.snapshot_epoch_ = snapshot.epoch();
        const uint64_t previous_build_count = accounting_.build_count;
        const auto old_size = result.pages_.size();
        result.pages_.erase(std::remove_if(result.pages_.begin(), result.pages_.end(),
                [&](const auto & page) { return page.id.logical_page == logical_page; }),
                result.pages_.end());
        result.rebuild_accounting({}, std::chrono::steady_clock::now());
        result.accounting_.invalidation_count = accounting_.invalidation_count +
            (old_size - result.pages_.size());
        result.accounting_.build_count = previous_build_count;
        status = llama_kv_routing_summary_status::ok;
        return result;
    } catch (...) {
        status = llama_kv_routing_summary_status::overflow;
        return {};
    }
}

bool llama_kv_routing_summary_store::contains(uint32_t logical_page) const noexcept {
    return std::find_if(pages_.begin(), pages_.end(), [&](const auto & page) {
        return page.id.logical_page == logical_page;
    }) != pages_.end();
}

void llama_kv_routing_summary_store::rebuild_accounting(
        const llama_kv_routing_summary_config & config,
        std::chrono::steady_clock::time_point start) noexcept {
    uint64_t vectors = 0, payload = 0, metadata = 0, logical = 0, charged = 0;
    const uint64_t vector_count = form_ == llama_kv_routing_summary_form::representatives
        ? representative_count_ : 1;
    uint64_t radius_bytes = 0;
    if (!mul(pages_.size(), vector_count, vectors) || !mul(vectors, vector_dim_, vectors) ||
        !mul(vectors, sizeof(float), payload) ||
        (form_ == llama_kv_routing_summary_form::centroid_upper_bound &&
         !mul(pages_.size(), sizeof(float), radius_bytes)) ||
        (form_ == llama_kv_routing_summary_form::centroid_upper_bound && !add(payload, radius_bytes, payload)) ||
        !mul(pages_.size(), sizeof(llama_kv_page_id), metadata) || !add(payload, metadata, logical)) {
        accounting_ = {};
        return;
    }
    const uint64_t granularity = config.allocation_granularity != 0
        ? config.allocation_granularity : allocation_granularity_;
    const uint64_t remainder = logical % granularity;
    if (!add(logical, remainder == 0 ? 0 : granularity - remainder, charged)) charged = UINT64_MAX;
    accounting_.page_count = pages_.size();
    accounting_.representative_count = representative_count_;
    accounting_.vector_dim = vector_dim_;
    accounting_.payload_bytes = payload;
    accounting_.metadata_bytes = metadata;
    accounting_.logical_bytes = logical;
    accounting_.charged_bytes = charged;
    accounting_.source_bytes = accounting_.source_rows = 0;
    accounting_.build_time_us = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count());
    accounting_.build_count = pages_.size();
    for (const auto & page : pages_) {
        if (!add(accounting_.source_bytes, page.source_bytes, accounting_.source_bytes)) {
            accounting_.source_bytes = UINT64_MAX;
        }
        if (!add(accounting_.source_rows, page.source_rows, accounting_.source_rows)) {
            accounting_.source_rows = UINT64_MAX;
        }
    }
    uint64_t hash = 1469598103934665603ull;
    hash = hash_mix(hash, LLAMA_KV_ROUTING_SUMMARY_VERSION);
    hash = hash_mix(hash, uint32_t(form_));
    hash = hash_mix(hash, layer_index_);
    hash = hash_mix(hash, head_index_);
    for (const auto & page : pages_) {
        hash = hash_id(hash, page.id);
        for (const float value : page.vectors) hash = hash_float(hash, value);
        hash = hash_float(hash, page.radius);
    }
    accounting_.content_hash = hash == 0 ? 1 : hash;
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
    for (const auto & summary_page : pages_) {
        const auto it = std::find_if(snapshot.pages().begin(), snapshot.pages().end(),
                [&](const auto & record) { return record.id.logical_page == summary_page.id.logical_page; });
        if (it == snapshot.pages().end() || it->id != summary_page.id) {
            result.status = llama_kv_routing_summary_status::stale_summary;
            return result;
        }
    }
    // A partial write frontier is mandatory structural context and may be
    // intentionally absent from the router. Every other live page must have a
    // current summary; silently scoring a subset would make a cold page
    // permanently undiscoverable.
    for (const auto & record : snapshot.pages()) {
        const bool tail = record.state == llama_kv_page_state::filling_gpu;
        if (!valid_state(record.state) || !llama_kv_page_id_valid(record.id, tail) || tail) continue;
        const bool found = std::find_if(pages_.begin(), pages_.end(),
                [&](const auto & page) { return page.id == record.id; }) != pages_.end();
        if (!found) {
            result.status = llama_kv_routing_summary_status::stale_summary;
            return result;
        }
    }
    const auto start = std::chrono::steady_clock::now();
    try {
        result.top_pages.reserve(pages_.size());
        double query_sum = 0.0;
        for (float value : query) query_sum += double(value) * value;
        const float query_norm = float(std::sqrt(query_sum));
        for (const auto & page : pages_) {
            const uint32_t vector_count = form_ == llama_kv_routing_summary_form::representatives
                ? representative_count_ : 1;
            float best = -std::numeric_limits<float>::infinity();
            for (uint32_t representative = 0; representative < vector_count; ++representative) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < vector_dim_; ++d) {
                    dot += page.vectors[size_t(representative) * vector_dim_ + d] * query[d];
                }
                best = std::max(best, dot);
                ++result.comparisons;
            }
            const bool upper_bound = form_ == llama_kv_routing_summary_form::centroid_upper_bound;
            result.top_pages.push_back({ page.id.logical_page, page.id.page_generation,
                    upper_bound ? best + page.radius * query_norm : best, upper_bound });
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
