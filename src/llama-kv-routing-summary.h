#pragma once

#include "llama-kv-residency.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Experimental, internal-only routing representation.  The vectors are already
// in the rotated K domain and are deliberately kept separate from the KV page.
constexpr uint32_t LLAMA_KV_ROUTING_SUMMARY_VERSION = 1;

enum class llama_kv_routing_summary_status : uint8_t {
    ok = 0,
    invalid_argument,
    duplicate_page,
    missing_page,
    invalid_page,
    stale_summary,
    insufficient_budget,
    overflow,
};

const char * llama_kv_routing_summary_status_name(
        llama_kv_routing_summary_status status) noexcept;

struct llama_kv_routing_summary_config {
    uint32_t representative_count = 4;
    uint32_t vector_dim = 256;
    uint64_t byte_budget = 0;          // zero means unbounded
    uint64_t allocation_granularity = 1;
};

// rows contains rotated K rows for one page, in row-major order.  A tail may
// contain fewer than 256 rows.  The builder samples representative rows at
// deterministic evenly-spaced positions.
struct llama_kv_routing_page_input {
    llama_kv_page_id id;
    std::vector<float> rotated_k_rows;
};

struct llama_kv_routing_page_score {
    uint32_t logical_page = UINT32_MAX;
    uint32_t page_generation = 0;
    float score = 0.0f;
};

struct llama_kv_routing_score_result {
    llama_kv_routing_summary_status status = llama_kv_routing_summary_status::invalid_argument;
    uint64_t summary_epoch = 0;
    uint32_t pages_scored = 0;
    uint64_t comparisons = 0;
    uint64_t latency_us = 0;
    std::vector<llama_kv_routing_page_score> top_pages;
};

struct llama_kv_routing_summary_accounting {
    uint64_t page_count = 0;
    uint64_t representative_count = 0;
    uint64_t vector_dim = 0;
    uint64_t payload_bytes = 0;
    uint64_t metadata_bytes = 0;
    uint64_t logical_bytes = 0;
    uint64_t charged_bytes = 0;
};

class llama_kv_routing_summary_store {
public:
    llama_kv_routing_summary_store() = default;

    static llama_kv_routing_summary_store build(
            const llama_kv_residency_snapshot & snapshot,
            const std::vector<llama_kv_routing_page_input> & inputs,
            const llama_kv_routing_summary_config & config,
            llama_kv_routing_summary_status & status) noexcept;

    // Rebuilds the immutable store for a newly published snapshot.  This is
    // the page-seal/mutation update boundary; old stores remain safe to score.
    static llama_kv_routing_summary_store update(
            const llama_kv_residency_snapshot & snapshot,
            const std::vector<llama_kv_routing_page_input> & inputs,
            const llama_kv_routing_summary_config & config,
            llama_kv_routing_summary_status & status) noexcept {
        return build(snapshot, inputs, config, status);
    }

    bool valid() const noexcept { return !pages_.empty() && snapshot_epoch_ != 0; }
    uint64_t version() const noexcept { return LLAMA_KV_ROUTING_SUMMARY_VERSION; }
    uint64_t snapshot_epoch() const noexcept { return snapshot_epoch_; }
    const llama_kv_routing_summary_accounting & accounting() const noexcept { return accounting_; }

    llama_kv_routing_score_result score(
            const llama_kv_residency_snapshot & snapshot,
            const std::vector<float> & query,
            uint32_t top_k) const noexcept;

private:
    struct page {
        llama_kv_page_id id;
        std::vector<float> vectors;
    };

    uint64_t snapshot_epoch_ = 0;
    uint32_t representative_count_ = 0;
    uint32_t vector_dim_ = 0;
    std::vector<page> pages_;
    llama_kv_routing_summary_accounting accounting_;
};
