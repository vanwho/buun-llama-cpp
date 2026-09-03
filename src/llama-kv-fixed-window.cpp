#include "llama-kv-fixed-window.h"

#include <algorithm>
#include <limits>
#include <new>
#include <utility>

namespace {

bool mul_u64(uint64_t a, uint64_t b, uint64_t & output) noexcept {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    output = a * b;
    return true;
}

bool add_u64(uint64_t a, uint64_t b, uint64_t & output) noexcept {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    output = a + b;
    return true;
}

bool div_exact(uint64_t numerator, uint64_t denominator,
               uint64_t & output) noexcept {
    if (denominator == 0 || numerator % denominator != 0) {
        return false;
    }
    output = numerator / denominator;
    return true;
}

} // namespace

bool llama_kv_fixed_window_derive_geometry(
        const llama_kv_fixed_window_geometry & input,
        llama_kv_fixed_window_derived_geometry & output) noexcept {
    output = {};
    if (input.page_tokens != VBR_GENERATION_PAGE_CELLS ||
        input.logical_pages == 0 || input.target_layers == 0 ||
        input.kv_heads == 0 || input.key_length == 0 ||
        input.value_length == 0 || input.key_length != input.value_length ||
        input.bits_per_value_numerator == 0 ||
        input.bits_per_value_denominator == 0) {
        return false;
    }
    uint64_t key_value_length = 0;
    uint64_t kv_values = 0;
    uint64_t values_per_token = 0;
    uint64_t bits_denominator = 0;
    uint64_t value_bits = 0;
    uint64_t row_values = 0;
    uint64_t row_bits = 0;
    uint64_t row_token_bytes = 0;
    if (!add_u64(input.key_length, input.value_length, key_value_length) ||
        !mul_u64(input.kv_heads, key_value_length,
                 kv_values) ||
        !mul_u64(input.target_layers, kv_values, values_per_token) ||
        !mul_u64(input.bits_per_value_denominator, 8, bits_denominator) ||
        !mul_u64(values_per_token, input.bits_per_value_numerator, value_bits) ||
        !div_exact(value_bits, bits_denominator, output.bytes_per_token) ||
        !mul_u64(input.kv_heads, input.key_length, row_values) ||
        !mul_u64(row_values, input.bits_per_value_numerator, row_bits) ||
        !div_exact(row_bits, bits_denominator, row_token_bytes) ||
        !mul_u64(input.page_tokens, row_token_bytes, output.row_bytes) ||
        !mul_u64(input.page_tokens, output.bytes_per_token, output.page_bytes) ||
        !mul_u64(output.page_bytes, input.logical_pages,
                 output.full_context_bytes)) {
        output = {};
        return false;
    }
    output.values_per_token = values_per_token;
    return output.page_bytes != 0 && output.row_bytes != 0;
}

bool llama_kv_fixed_window_select(
        uint32_t logical_page_count,
        uint32_t resident_capacity,
        uint32_t newest_page_count,
        const std::vector<uint32_t> & explicit_pages,
        llama_kv_fixed_window_selection & output) noexcept {
    output.logical_pages.clear();
    if (logical_page_count == 0 || resident_capacity == 0 ||
        resident_capacity > logical_page_count ||
        newest_page_count > resident_capacity) {
        return false;
    }
    try {
        output.logical_pages.reserve(resident_capacity);
        const uint32_t first_newest = logical_page_count - newest_page_count;
        for (uint32_t page = first_newest; page < logical_page_count; ++page) {
            output.logical_pages.push_back(page);
        }
        for (const uint32_t page : explicit_pages) {
            if (page >= logical_page_count) {
                output.logical_pages.clear();
                return false;
            }
            if (std::find(output.logical_pages.begin(),
                          output.logical_pages.end(), page) !=
                    output.logical_pages.end()) {
                continue;
            }
            if (output.logical_pages.size() >= resident_capacity) {
                output.logical_pages.clear();
                return false;
            }
            output.logical_pages.push_back(page);
        }
        return !output.logical_pages.empty();
    } catch (...) {
        output.logical_pages.clear();
        return false;
    }
}

const char * llama_kv_fixed_window_status_name(
        llama_kv_fixed_window_status status) noexcept {
    switch (status) {
        case llama_kv_fixed_window_status::ok: return "ok";
        case llama_kv_fixed_window_status::invalid_argument: return "invalid_argument";
        case llama_kv_fixed_window_status::geometry_mismatch: return "geometry_mismatch";
        case llama_kv_fixed_window_status::host_budget: return "host_budget";
        case llama_kv_fixed_window_status::slot_unavailable: return "slot_unavailable";
        case llama_kv_fixed_window_status::pinned: return "pinned";
        case llama_kv_fixed_window_status::host_missing: return "host_missing";
        case llama_kv_fixed_window_status::checksum_mismatch: return "checksum_mismatch";
        case llama_kv_fixed_window_status::dirty: return "dirty";
        case llama_kv_fixed_window_status::not_found: return "not_found";
        case llama_kv_fixed_window_status::short_page: return "short_page";
        case llama_kv_fixed_window_status::overflow: return "overflow";
        case llama_kv_fixed_window_status::_count: break;
    }
    return "invalid";
}

struct llama_kv_fixed_window::impl {
    llama_kv_fixed_window_config config;
    llama_kv_fixed_window_derived_geometry derived;
    llama_kv_fixed_window_ledger ledger;
    std::vector<llama_kv_fixed_window_page> pages;
};

llama_kv_fixed_window_status llama_kv_fixed_window::seal_impl(
        uint32_t logical_page,
        uint32_t valid_tokens, uint64_t checksum, uint64_t aligned_bytes,
        bool require_resident) noexcept {
    if (!impl_ || logical_page >= impl_->pages.size() || valid_tokens == 0 ||
        valid_tokens > impl_->config.geometry.page_tokens || checksum == 0) {
        return llama_kv_fixed_window_status::invalid_argument;
    }
    auto & state = *impl_;
    auto & page = state.pages[logical_page];
    if (require_resident && page.physical_slot == UINT32_MAX) {
        return llama_kv_fixed_window_status::not_found;
    }
    if (page.pinned && !require_resident) {
        return llama_kv_fixed_window_status::pinned;
    }
    uint64_t useful_bytes = 0;
    if (!mul_u64(valid_tokens, state.derived.bytes_per_token, useful_bytes) ||
        aligned_bytes < useful_bytes) {
        return llama_kv_fixed_window_status::short_page;
    }
    const uint64_t previous_host_bytes = page.host_payload_bytes;
    uint64_t next_host_bytes = state.ledger.host_payload_bytes;
    if (useful_bytes > previous_host_bytes) {
        const uint64_t delta = useful_bytes - previous_host_bytes;
        if (next_host_bytes > state.config.host_budget_bytes ||
            delta > state.config.host_budget_bytes - next_host_bytes ||
            !add_u64(next_host_bytes, delta, next_host_bytes)) {
            return llama_kv_fixed_window_status::host_budget;
        }
    } else if (previous_host_bytes > useful_bytes) {
        const uint64_t delta = previous_host_bytes - useful_bytes;
        if (delta > next_host_bytes) {
            return llama_kv_fixed_window_status::overflow;
        }
        next_host_bytes -= delta;
    }
    uint64_t next_d2h_useful = 0;
    uint64_t next_d2h_aligned = 0;
    if (!add_u64(state.ledger.d2h_seal_useful_bytes, useful_bytes,
                 next_d2h_useful) ||
        !add_u64(state.ledger.d2h_seal_aligned_bytes, aligned_bytes,
                 next_d2h_aligned)) {
        return llama_kv_fixed_window_status::overflow;
    }
    state.ledger.host_payload_bytes = next_host_bytes;
    page.host_payload_bytes = useful_bytes;
    if (!page.host_valid) {
        ++state.ledger.host_valid_pages;
    }
    if (page.pinned) {
        --state.ledger.pinned_pages;
    }
    page.valid_tokens = valid_tokens;
    page.checksum = checksum;
    page.host_valid = true;
    page.dirty = false;
    page.pinned = false;
    state.ledger.d2h_seal_useful_bytes = next_d2h_useful;
    state.ledger.d2h_seal_aligned_bytes = next_d2h_aligned;
    return llama_kv_fixed_window_status::ok;
}

llama_kv_fixed_window::llama_kv_fixed_window(
        std::unique_ptr<impl> state) noexcept
    : impl_(std::move(state)) {}

std::unique_ptr<llama_kv_fixed_window> llama_kv_fixed_window::create(
        const llama_kv_fixed_window_config & config,
        llama_kv_fixed_window_derived_geometry & derived,
        llama_kv_fixed_window_status & status) noexcept {
    derived = {};
    status = llama_kv_fixed_window_status::invalid_argument;
    try {
        if (config.resident_slot_capacity == 0 ||
            config.host_budget_bytes == 0 || config.staging_bytes == 0 ||
            !llama_kv_fixed_window_derive_geometry(config.geometry, derived) ||
            config.resident_slot_capacity > config.geometry.logical_pages) {
            status = llama_kv_fixed_window_status::geometry_mismatch;
            return nullptr;
        }
        if (config.host_budget_bytes < derived.page_bytes) {
            status = llama_kv_fixed_window_status::host_budget;
            return nullptr;
        }
        uint64_t resident_bytes = 0;
        if (!mul_u64(config.resident_slot_capacity, derived.page_bytes,
                     resident_bytes)) {
            status = llama_kv_fixed_window_status::overflow;
            return nullptr;
        }
        std::unique_ptr<impl> state(new impl);
        state->config = config;
        state->derived = derived;
        state->ledger.staging_bytes = config.staging_bytes;
        state->pages.resize(config.geometry.logical_pages);
        for (uint32_t page = 0; page < state->pages.size(); ++page) {
            state->pages[page].logical_page = page;
        }
        status = llama_kv_fixed_window_status::ok;
        return std::unique_ptr<llama_kv_fixed_window>(
            new llama_kv_fixed_window(std::move(state)));
    } catch (...) {
        derived = {};
        status = llama_kv_fixed_window_status::overflow;
        return nullptr;
    }
}

llama_kv_fixed_window::~llama_kv_fixed_window() = default;

const llama_kv_fixed_window_derived_geometry &
llama_kv_fixed_window::geometry() const noexcept {
    static const llama_kv_fixed_window_derived_geometry empty;
    return impl_ ? impl_->derived : empty;
}

const llama_kv_fixed_window_ledger &
llama_kv_fixed_window::ledger() const noexcept {
    static const llama_kv_fixed_window_ledger empty;
    return impl_ ? impl_->ledger : empty;
}

const llama_kv_fixed_window_page * llama_kv_fixed_window::find(
        uint32_t logical_page) const noexcept {
    return impl_ && logical_page < impl_->pages.size()
        ? &impl_->pages[logical_page] : nullptr;
}

llama_kv_fixed_window_status llama_kv_fixed_window::begin_fill(
        uint32_t logical_page, uint32_t physical_slot,
        uint32_t valid_tokens) noexcept {
    if (!impl_ || logical_page >= impl_->pages.size() ||
        physical_slot >= impl_->config.resident_slot_capacity ||
        valid_tokens == 0 ||
        valid_tokens > impl_->config.geometry.page_tokens) {
        return llama_kv_fixed_window_status::invalid_argument;
    }
    auto & page = impl_->pages[logical_page];
    if (page.physical_slot != UINT32_MAX || page.pinned) {
        return llama_kv_fixed_window_status::slot_unavailable;
    }
    for (const auto & other : impl_->pages) {
        if (other.physical_slot == physical_slot) {
            return llama_kv_fixed_window_status::slot_unavailable;
        }
    }
    if (page.host_valid) {
        --impl_->ledger.host_valid_pages;
    }
    page.physical_slot = physical_slot;
    page.valid_tokens = valid_tokens;
    page.checksum = 0;
    page.host_valid = false;
    page.dirty = true;
    page.pinned = true;
    ++impl_->ledger.resident_pages;
    ++impl_->ledger.pinned_pages;
    if (!add_u64(impl_->ledger.resident_bytes, impl_->derived.page_bytes,
                 impl_->ledger.resident_bytes)) {
        page = {};
        page.logical_page = logical_page;
        --impl_->ledger.resident_pages;
        --impl_->ledger.pinned_pages;
        return llama_kv_fixed_window_status::overflow;
    }
    return llama_kv_fixed_window_status::ok;
}

llama_kv_fixed_window_status llama_kv_fixed_window::seal(
        uint32_t logical_page, uint32_t valid_tokens,
        uint64_t checksum, uint64_t aligned_bytes) noexcept {
    return seal_impl(logical_page, valid_tokens, checksum, aligned_bytes, true);
}

llama_kv_fixed_window_status llama_kv_fixed_window::seal_host_only(
        uint32_t logical_page, uint32_t valid_tokens,
        uint64_t checksum, uint64_t aligned_bytes) noexcept {
    return seal_impl(logical_page, valid_tokens, checksum, aligned_bytes, false);
}

llama_kv_fixed_window_status llama_kv_fixed_window::mutate(
        uint32_t logical_page) noexcept {
    if (!impl_ || logical_page >= impl_->pages.size()) {
        return llama_kv_fixed_window_status::not_found;
    }
    auto & page = impl_->pages[logical_page];
    if (page.physical_slot == UINT32_MAX) {
        return llama_kv_fixed_window_status::not_found;
    }
    if (page.host_valid) {
        --impl_->ledger.host_valid_pages;
    }
    page.host_valid = false;
    page.dirty = true;
    page.checksum = 0;
    return llama_kv_fixed_window_status::ok;
}

llama_kv_fixed_window_status llama_kv_fixed_window::pin(
        uint32_t logical_page) noexcept {
    if (!impl_ || logical_page >= impl_->pages.size()) {
        return llama_kv_fixed_window_status::not_found;
    }
    auto & page = impl_->pages[logical_page];
    if (page.physical_slot == UINT32_MAX) {
        return llama_kv_fixed_window_status::not_found;
    }
    if (!page.pinned) {
        page.pinned = true;
        ++impl_->ledger.pinned_pages;
    }
    return llama_kv_fixed_window_status::ok;
}

llama_kv_fixed_window_status llama_kv_fixed_window::unpin(
        uint32_t logical_page) noexcept {
    if (!impl_ || logical_page >= impl_->pages.size()) {
        return llama_kv_fixed_window_status::not_found;
    }
    auto & page = impl_->pages[logical_page];
    if (!page.pinned) {
        return llama_kv_fixed_window_status::ok;
    }
    page.pinned = false;
    --impl_->ledger.pinned_pages;
    return llama_kv_fixed_window_status::ok;
}

llama_kv_fixed_window_status llama_kv_fixed_window::promote(
        uint32_t logical_page, uint32_t physical_slot,
        uint64_t checksum, uint64_t aligned_bytes) noexcept {
    if (!impl_ || logical_page >= impl_->pages.size() ||
        physical_slot >= impl_->config.resident_slot_capacity || checksum == 0) {
        return llama_kv_fixed_window_status::invalid_argument;
    }
    auto & page = impl_->pages[logical_page];
    if (!page.host_valid || page.dirty || page.host_payload_bytes == 0) {
        return llama_kv_fixed_window_status::host_missing;
    }
    if (page.checksum != checksum) {
        ++impl_->ledger.checksum_failures;
        return llama_kv_fixed_window_status::checksum_mismatch;
    }
    if (page.physical_slot != UINT32_MAX) {
        return page.physical_slot == physical_slot
            ? llama_kv_fixed_window_status::ok
            : llama_kv_fixed_window_status::slot_unavailable;
    }
    for (const auto & other : impl_->pages) {
        if (other.physical_slot == physical_slot) {
            return llama_kv_fixed_window_status::slot_unavailable;
        }
    }
    const uint64_t useful_bytes = page.host_payload_bytes;
    if (aligned_bytes < useful_bytes) {
        return llama_kv_fixed_window_status::short_page;
    }
    uint64_t next_resident_bytes = 0;
    uint64_t next_h2d_useful = 0;
    uint64_t next_h2d_aligned = 0;
    if (!add_u64(impl_->ledger.resident_bytes, impl_->derived.page_bytes,
                 next_resident_bytes) ||
        !add_u64(impl_->ledger.h2d_useful_bytes, useful_bytes,
                 next_h2d_useful) ||
        !add_u64(impl_->ledger.h2d_aligned_bytes, aligned_bytes,
                 next_h2d_aligned)) {
        return llama_kv_fixed_window_status::overflow;
    }
    page.physical_slot = physical_slot;
    ++impl_->ledger.resident_pages;
    impl_->ledger.resident_bytes = next_resident_bytes;
    impl_->ledger.h2d_useful_bytes = next_h2d_useful;
    impl_->ledger.h2d_aligned_bytes = next_h2d_aligned;
    return llama_kv_fixed_window_status::ok;
}

llama_kv_fixed_window_status llama_kv_fixed_window::evict_clean(
        uint32_t logical_page) noexcept {
    if (!impl_ || logical_page >= impl_->pages.size()) {
        return llama_kv_fixed_window_status::not_found;
    }
    auto & page = impl_->pages[logical_page];
    if (page.physical_slot == UINT32_MAX) {
        return llama_kv_fixed_window_status::not_found;
    }
    if (page.pinned) {
        return llama_kv_fixed_window_status::pinned;
    }
    if (!page.host_valid) {
        return page.dirty ? llama_kv_fixed_window_status::dirty
                          : llama_kv_fixed_window_status::host_missing;
    }
    page.physical_slot = UINT32_MAX;
    --impl_->ledger.resident_pages;
    impl_->ledger.resident_bytes -= impl_->derived.page_bytes;
    ++impl_->ledger.evictions;
    return llama_kv_fixed_window_status::ok;
}
