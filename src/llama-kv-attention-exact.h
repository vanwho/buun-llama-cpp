#pragma once

#include "llama-kv-residency.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// Exact attention is intentionally an internal reference seam.  A partial is
// unnormalized: weighted_value is already shifted by max_logit and is merged
// before the final division.  Keeping this state separate from an attention
// output prevents a page wave from accidentally normalizing its local mass.
struct llama_kv_attention_online_state {
    float max_logit = -INFINITY;
    float sum_exp = 0.0f;
    std::vector<float> weighted_value;

    bool empty() const noexcept { return max_logit == -INFINITY && sum_exp == 0.0f; }
};

enum class llama_kv_attention_exact_status : uint8_t {
    ok = 0,
    invalid_argument,
    empty_partition,
    invalid_state,
    invalid_page,
    duplicate_page,
    missing_page,
    stale_page,
    invalid_wave,
    overflow,
    not_configured,
};

const char * llama_kv_attention_exact_status_name(
        llama_kv_attention_exact_status status) noexcept;

bool llama_kv_attention_online_state_valid(
        const llama_kv_attention_online_state & state) noexcept;

// Build one unnormalized partition from dense logits and values.  Masked rows
// are omitted, and an entirely masked partition is represented by an empty
// state.  Values are row-major [row][value_dim].
llama_kv_attention_exact_status llama_kv_attention_online_state_from_rows(
        const float * logits,
        const float * values,
        uint32_t n_rows,
        uint32_t value_dim,
        const uint8_t * valid_mask,
        llama_kv_attention_online_state & output) noexcept;

// Stable associative merge from plan section 8.10.  `output` may alias either
// input state, which lets a wave accumulator merge in place.
llama_kv_attention_exact_status llama_kv_attention_online_state_merge(
        const llama_kv_attention_online_state & a,
        const llama_kv_attention_online_state & b,
        llama_kv_attention_online_state & output) noexcept;

// Normalize only after all page-wave states have been merged.
llama_kv_attention_exact_status llama_kv_attention_online_state_normalize(
        const llama_kv_attention_online_state & state,
        float * output,
        uint32_t value_dim) noexcept;

struct llama_kv_attention_exact_page {
    llama_kv_page_id id;
    uint32_t physical_slot = UINT32_MAX;
    uint32_t valid_tokens = 0;
    bool resident = false;
    bool host_valid = false;
};

enum class llama_kv_attention_exact_schedule : uint8_t {
    serial = 0,
    double_buffered,
};

struct llama_kv_attention_exact_config {
    // Zero derives the logical page count from pages.size().
    uint32_t logical_page_count = 0;
    uint32_t pages_per_wave = 1;
    uint32_t staging_slots = 1;
    uint64_t page_bytes = 0;
    llama_kv_attention_exact_schedule schedule =
        llama_kv_attention_exact_schedule::serial;
};

struct llama_kv_attention_exact_wave_page {
    uint32_t logical_page = UINT32_MAX;
    uint32_t valid_tokens = 0;
    uint32_t physical_slot = UINT32_MAX;
    bool cold = false;
    llama_pos native_position_begin = -1;
    llama_pos native_position_end = -1;
};

struct llama_kv_attention_exact_wave {
    uint32_t index = 0;
    uint32_t staging_slot = UINT32_MAX;
    bool contains_cold_pages = false;
    std::vector<llama_kv_attention_exact_wave_page> pages;
};

// The ledger is deliberately independent from the transfer implementation.
// It can be checked by a CUDA callback after every wave and is also useful for
// deterministic plan-only tests.
struct llama_kv_attention_exact_ledger {
    uint32_t logical_page_count = 0;
    uint64_t valid_tokens = 0;
    uint64_t resident_pages = 0;
    uint64_t cold_pages = 0;
    uint64_t waves = 0;
    uint64_t pages_visited = 0;
    uint64_t duplicate_pages = 0;
    uint64_t missing_pages = 0;
    uint64_t stale_pages = 0;
    uint64_t h2d_useful_bytes = 0;
    uint64_t h2d_aligned_bytes = 0;
    uint64_t waits = 0;
    uint64_t peak_staging_pages = 0;
};

class llama_kv_attention_exact_wave_plan {
public:
    static llama_kv_attention_exact_wave_plan build(
            const std::vector<llama_kv_attention_exact_page> & pages,
            const llama_kv_residency_snapshot & resident_snapshot,
            const llama_kv_attention_exact_config & config,
            llama_kv_attention_exact_status & status) noexcept;

    bool valid() const noexcept { return valid_; }
    uint32_t logical_page_count() const noexcept { return ledger_.logical_page_count; }
    const std::vector<llama_kv_attention_exact_wave> & waves() const noexcept { return waves_; }
    const llama_kv_attention_exact_ledger & ledger() const noexcept { return ledger_; }

    // Record one completed callback visit.  A plan's own executor uses this
    // internally; exposing it lets hardware tests reconcile the callback's
    // page coverage with the plan without copying K/V data to the host.
    llama_kv_attention_exact_status record_visit(uint32_t logical_page) noexcept;
    llama_kv_attention_exact_status finish() noexcept;

private:
    friend class llama_kv_attention_exact_executor;
    bool valid_ = false;
    llama_kv_attention_exact_schedule schedule_ =
        llama_kv_attention_exact_schedule::serial;
    uint32_t staging_slots_ = 1;
    std::vector<uint8_t> visited_;
    std::vector<llama_kv_attention_exact_wave> waves_;
    llama_kv_attention_exact_ledger ledger_;
};

struct llama_kv_attention_exact_backend {
    void * context = nullptr;

    // Upload one complete cold page wave into one bounded staging slot. The
    // callback may be asynchronous; compute_wave must observe its completion.
    bool (*upload_cold_wave)(
            void * context,
            const llama_kv_attention_exact_wave & wave,
            uint32_t staging_slot,
            bool asynchronous) noexcept = nullptr;
    bool (*wait_wave)(
            void * context,
            const llama_kv_attention_exact_wave & wave) noexcept = nullptr;
    // Produce one unnormalized (m,l,o) state for this query/head/layer and
    // all pages in the wave. No normalized output is accepted here.
    bool (*compute_wave)(
            void * context,
            const llama_kv_attention_exact_wave & wave,
            llama_kv_attention_online_state & output) noexcept = nullptr;
};

class llama_kv_attention_exact_executor {
public:
    llama_kv_attention_exact_status execute(
            llama_kv_attention_exact_wave_plan & plan,
            const llama_kv_attention_exact_backend & backend,
            llama_kv_attention_online_state & output) noexcept;
};
