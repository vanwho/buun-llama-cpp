#include "llama-kv-attention-execution.h"

#include "llama-impl.h"
#include "llama-vram-demand.h"

#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace {

template<typename T>
T saturating_add(T a, T b) noexcept {
    if (b > std::numeric_limits<T>::max() - a) {
        return std::numeric_limits<T>::max();
    }
    return a + b;
}

void saturating_add_u64(uint64_t & target, uint64_t value) noexcept {
    target = saturating_add(target, value);
}

bool direct_shape(const llama_kv_attention_operator_metadata & metadata) noexcept {
    return metadata.causal() && metadata.type_k() == GGML_TYPE_TURBO4_0 &&
           metadata.type_v() == GGML_TYPE_TURBO4_0 && metadata.head_dim_k() == 256 &&
           metadata.head_dim_v() == 256 && metadata.n_query_tokens() >= 1 &&
           metadata.n_batch() == 1 && metadata.n_head_q() != 0 &&
           metadata.n_head_kv() != 0 &&
           metadata.n_head_q() % metadata.n_head_kv() == 0;
}

bool production_direct_shape(
        const llama_kv_attention_operator_metadata & metadata,
        llama_kv_attention_execution_phase phase) noexcept {
    // The CUDA direct prefill primitive is graph-sized in tiny query tiles.
    // The server may submit a larger ubatch, which must remain on the mature
    // selected consumer and be split by the normal scheduler. Decode and MTP
    // verification are already single/tiny-query shapes and keep the direct
    // multi-page promotion.
    const bool bounded_prefill = phase != llama_kv_attention_execution_phase::prefill ||
        metadata.n_query_tokens() <= 2;
    return direct_shape(metadata) && bounded_prefill &&
        (phase == llama_kv_attention_execution_phase::decode ||
         phase == llama_kv_attention_execution_phase::prefill ||
         phase == llama_kv_attention_execution_phase::mtp_verify);
}

} // namespace

uint32_t llama_kv_attention_packed_row_capacity(
        const llama_kv_attention_operator_metadata & metadata,
        uint32_t page_tokens) noexcept {
    if (!metadata.valid() || page_tokens == 0) {
        return 0;
    }
    uint64_t selected_rows = 0;
    for (const auto & page : metadata.page_table()) {
        const uint64_t page_end = uint64_t(page.compact_row_begin) + page.row_count;
        selected_rows = std::max(selected_rows, page_end);
    }
    if (selected_rows == 0 || selected_rows > UINT64_MAX - page_tokens + 1) {
        return 0;
    }
    const uint64_t pages = (selected_rows + page_tokens - 1) / page_tokens;
    const uint64_t rows = pages * page_tokens;
    return rows > UINT32_MAX ? 0 : uint32_t(rows);
}

size_t llama_kv_attention_packed_allocation_bytes(
        uint32_t row_capacity,
        size_t k_bytes_per_row,
        size_t v_bytes_per_row) noexcept {
    if (k_bytes_per_row > SIZE_MAX - v_bytes_per_row) {
        return SIZE_MAX;
    }
    const size_t bytes_per_row = k_bytes_per_row + v_bytes_per_row;
    return row_capacity > 0 && bytes_per_row > SIZE_MAX / row_capacity
        ? SIZE_MAX : size_t(row_capacity) * bytes_per_row;
}

llama_kv_attention_packed_cache::~llama_kv_attention_packed_cache() {
    for (auto & cached : entries_) {
        release_entry(cached.get());
    }
}

void llama_kv_attention_packed_cache::release_entry(entry * cached) noexcept {
    if (cached == nullptr) {
        return;
    }
    if (cached->buffer != nullptr) {
        ggml_backend_buffer_free(cached->buffer);
        cached->buffer = nullptr;
    }
    if (cached->context != nullptr) {
        ggml_free(cached->context);
        cached->context = nullptr;
    }
    cached->k = nullptr;
    cached->v = nullptr;
}

void llama_kv_attention_packed_cache::begin_graph_build() noexcept {
    abort_graph_build();
    graph_build_active_ = true;
    for (auto & cached : entries_) {
        cached->current_graph_use = false;
    }
}

void llama_kv_attention_packed_cache::abort_graph_build() noexcept {
    graph_build_active_ = false;
    for (entry * cached : graph_build_entries_) {
        if (cached != nullptr && cached->in_flight_leases == 0) {
            cached->draining = true;
        }
    }
    graph_build_entries_.clear();
    release_completed();
}

bool llama_kv_attention_packed_cache::same_domain(
        const entry & cached,
        uint32_t layer_id,
        int32_t sequence_id,
        ggml_backend_t backend) noexcept {
    return cached.layer_id == layer_id && cached.sequence_id == sequence_id &&
        cached.backend == backend &&
        cached.device == (backend != nullptr ? ggml_backend_get_device(backend) : nullptr);
}

bool llama_kv_attention_packed_cache::same_structural_key(
        const entry & cached,
        uint32_t layer_id,
        int32_t sequence_id,
        uint64_t source_lifetime_epoch,
        uint32_t row_capacity,
        ggml_tensor * source_k,
        ggml_tensor * source_v,
        ggml_backend_t backend) noexcept {
    return !cached.draining && same_domain(cached, layer_id, sequence_id, backend) &&
        cached.source_lifetime_epoch == source_lifetime_epoch &&
        cached.row_capacity == row_capacity && source_k != nullptr && source_v != nullptr &&
        cached.source_k_type == source_k->type && cached.source_v_type == source_v->type &&
        cached.source_k_ne0 == source_k->ne[0] && cached.source_k_ne1 == source_k->ne[1] &&
        cached.source_v_ne0 == source_v->ne[0] && cached.source_v_ne1 == source_v->ne[1] &&
        cached.source_k == source_k && cached.source_v == source_v &&
        cached.source_k_buffer == source_k->buffer && cached.source_v_buffer == source_v->buffer;
}

void llama_kv_attention_packed_cache::update_slots(
        entry & cached,
        const std::vector<llama_kv_attention_view_page> & pages) {
    std::vector<slot> next_slots;
    std::vector<uint64_t> next_versions;
    std::vector<dirty_interval> next_dirty;
    next_slots.reserve(pages.size());
    next_versions.reserve(pages.size());

    for (size_t i = 0; i < pages.size(); ++i) {
        const auto & page = pages[i];
        slot next;
        next.logical_page = page.logical_page;
        next.source_physical_slot = page.source_physical_slot;
        next.page_generation = page.page_generation;
        next.native_position_begin = page.native_position_begin;
        next.native_position_end = page.native_position_end;
        next.valid_rows = page.row_count;
        next.destination_row_begin = page.compact_row_begin;

        // Keep the copied version with a page identity, not with its current
        // selection index. The owner remains reusable across reorderings;
        // moving a page's destination row or changing its generation makes
        // only that slot dirty.
        for (const auto & previous : cached.slots) {
            if (previous.logical_page == page.logical_page &&
                    previous.source_physical_slot == page.source_physical_slot &&
                    previous.page_generation == page.page_generation &&
                    previous.valid_rows == page.row_count &&
                    previous.destination_row_begin == page.compact_row_begin) {
                next.copied_content_version = previous.copied_content_version;
                break;
            }
        }
        next_slots.push_back(next);
        next_versions.push_back(next.copied_content_version);
        if (next.copied_content_version == UINT64_MAX && page.row_count != 0) {
            const uint32_t begin = page.compact_row_begin;
            const uint32_t end = begin + page.row_count;
            if (!next_dirty.empty() && next_dirty.back().row_end >= begin) {
                next_dirty.back().row_end = std::max(next_dirty.back().row_end, end);
            } else {
                next_dirty.push_back({ begin, end });
            }
        }
    }

    cached.pages = pages;
    cached.slots.swap(next_slots);
    cached.dirty_intervals.swap(next_dirty);
    cached.content_versions.swap(next_versions);
}

void llama_kv_attention_packed_cache::rebuild_dirty_intervals(entry & cached) noexcept {
    cached.dirty_intervals.clear();
    for (const auto & slot : cached.slots) {
        if (slot.copied_content_version != UINT64_MAX && slot.valid_rows != 0) {
            continue;
        }
        if (slot.valid_rows == 0) {
            continue;
        }
        const uint32_t begin = slot.destination_row_begin;
        const uint32_t end = begin + slot.valid_rows;
        if (!cached.dirty_intervals.empty() &&
                cached.dirty_intervals.back().row_end >= begin) {
            cached.dirty_intervals.back().row_end =
                    std::max(cached.dirty_intervals.back().row_end, end);
        } else {
            cached.dirty_intervals.push_back({ begin, end });
        }
    }
}

bool llama_kv_attention_packed_cache::submit_graph(
        const std::vector<entry *> & owners) noexcept {
    std::vector<entry *> unique;
    try {
        unique.reserve(owners.size());
        for (entry * owner : owners) {
            if (owner == nullptr || owner->draining || owner->k == nullptr || owner->v == nullptr) {
                return false;
            }
            const bool owned = std::any_of(entries_.begin(), entries_.end(),
                    [&](const auto & cached) { return cached.get() == owner; });
            if (!owned) {
                return false;
            }
            if (std::find(unique.begin(), unique.end(), owner) == unique.end()) {
                unique.push_back(owner);
            }
        }
        if (unique.empty()) {
            return false;
        }
        for (entry * owner : unique) {
            if (owner->in_flight_leases != UINT64_MAX) {
                ++owner->in_flight_leases;
            }
        }
        // Copy the small owner list into the lease before changing the build
        // state.  Keeping `unique` intact until push_back succeeds means an
        // allocation failure cannot strand the incremented leases on a
        // moved-from temporary.
        graph_leases_.push_back(unique);
        // A submitted owner is now protected by graph_leases_; it must not be
        // reclaimed by a later build-abort cleanup.
        for (entry * owner : graph_leases_.back()) {
            graph_build_entries_.erase(std::remove(graph_build_entries_.begin(),
                    graph_build_entries_.end(), owner), graph_build_entries_.end());
        }
        graph_build_active_ = false;
        return true;
    } catch (...) {
        for (entry * owner : unique) {
            if (owner->in_flight_leases != 0) {
                --owner->in_flight_leases;
            }
        }
        return false;
    }
}

void llama_kv_attention_packed_cache::complete_one_graph() noexcept {
    if (graph_leases_.empty()) {
        return;
    }
    for (entry * owner : graph_leases_.front()) {
        if (owner != nullptr && owner->in_flight_leases != 0) {
            --owner->in_flight_leases;
        }
    }
    graph_leases_.erase(graph_leases_.begin());
}

void llama_kv_attention_packed_cache::complete_all_graphs() noexcept {
    while (!graph_leases_.empty()) {
        complete_one_graph();
    }
}

void llama_kv_attention_packed_cache::release_completed() noexcept {
    // A sequence clear or structural replacement can retire a provisional
    // owner before the build guard runs. Remove its pointer before erasing the
    // owning unique_ptr, otherwise a later abort would inspect freed storage.
    graph_build_entries_.erase(std::remove_if(graph_build_entries_.begin(),
            graph_build_entries_.end(), [](const entry * cached) {
                return cached == nullptr || cached->draining;
            }), graph_build_entries_.end());
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [](auto & cached) {
        if (!cached->draining || cached->in_flight_leases != 0) {
            return false;
        }
        release_entry(cached.get());
        return true;
    }), entries_.end());
}

void llama_kv_attention_packed_cache::clear_sequence(int32_t sequence_id) noexcept {
    for (auto & cached : entries_) {
        if (cached->sequence_id == sequence_id) {
            cached->draining = true;
        }
    }
    release_completed();
}

void llama_kv_attention_packed_cache::clear() noexcept {
    graph_build_active_ = false;
    graph_build_entries_.clear();
    for (auto & cached : entries_) {
        cached->draining = true;
    }
    release_completed();
}

llama_kv_attention_packed_cache::entry * llama_kv_attention_packed_cache::find_or_create(
        uint32_t layer_id,
        int32_t sequence_id,
        uint64_t representation_epoch,
        uint64_t source_lifetime_epoch,
        const std::vector<llama_kv_attention_view_page> & pages,
        ggml_tensor * source_k,
        ggml_tensor * source_v,
        ggml_backend_t backend,
        uint32_t row_capacity) noexcept {
    if (source_k == nullptr || source_v == nullptr || backend == nullptr || pages.empty()) {
        return nullptr;
    }

    uint64_t required_rows = 0;
    // The page vector is a selection, not an ownership-order contract.  In
    // particular, a small first request can arrive with its last selected
    // page not being the page with the greatest compact destination row.
    // Size the owner from the complete compact extent or the following graph
    // copy can address beyond the allocation.
    for (const auto & page : pages) {
        const uint64_t page_end = uint64_t(page.compact_row_begin) + page.row_count;
        if (page_end > UINT32_MAX) {
            return nullptr;
        }
        required_rows = std::max(required_rows, page_end);
    }
    if (row_capacity == 0) {
        row_capacity = uint32_t(required_rows);
    }
    if (row_capacity < required_rows || row_capacity == 0 ||
            uint64_t(row_capacity) > uint64_t(source_k->ne[2]) ||
            uint64_t(row_capacity) > uint64_t(source_v->ne[2])) {
        return nullptr;
    }

    for (const auto & cached : entries_) {
        if (!same_structural_key(*cached, layer_id, sequence_id, source_lifetime_epoch,
                row_capacity, source_k, source_v, backend) ||
                cached->k == nullptr || cached->v == nullptr) {
            continue;
        }
        try {
            update_slots(*cached, pages);
            cached->representation_epoch = representation_epoch;
            cached->current_graph_use = true;
            return cached.get();
        } catch (...) {
            return nullptr;
        }
    }

    // A structural replacement may coexist with one draining owner, but a
    // second replacement before completion would make the bounded owner
    // budget meaningless. Keep the active owner untouched on refusal.
    for (const auto & cached : entries_) {
        if (cached->draining && same_domain(*cached, layer_id, sequence_id, backend)) {
            return nullptr;
        }
    }

    try {
        auto cached = std::make_unique<entry>();
        cached->layer_id = layer_id;
        cached->sequence_id = sequence_id;
        cached->representation_epoch = representation_epoch;
        cached->source_lifetime_epoch = source_lifetime_epoch;
        cached->row_capacity = row_capacity;
        cached->backend = backend;
        cached->device = ggml_backend_get_device(backend);
        cached->source_k_type = source_k->type;
        cached->source_v_type = source_v->type;
        cached->source_k_ne0 = source_k->ne[0];
        cached->source_k_ne1 = source_k->ne[1];
        cached->source_v_ne0 = source_v->ne[0];
        cached->source_v_ne1 = source_v->ne[1];
        cached->owner_generation = next_owner_generation_ == 0
            ? UINT64_MAX : next_owner_generation_++;
        cached->source_k = source_k;
        cached->source_v = source_v;
        cached->source_k_buffer = source_k->buffer;
        cached->source_v_buffer = source_v->buffer;
        cached->current_graph_use = true;
        update_slots(*cached, pages);

        const ggml_init_params params = { 2 * ggml_tensor_overhead(), nullptr, true };
        cached->context = ggml_init(params);
        if (cached->context == nullptr) {
            return nullptr;
        }
        cached->k = ggml_new_tensor_4d(cached->context, source_k->type,
                source_k->ne[0], source_k->ne[1], row_capacity, 1);
        cached->v = ggml_new_tensor_4d(cached->context, source_v->type,
                source_v->ne[0], source_v->ne[1], row_capacity, 1);
        if (cached->k == nullptr || cached->v == nullptr) {
            return nullptr;
        }
        // Keep the source tensor's semantic name on the packed owner. CUDA
        // set_rows uses the cache name to select the Turbo mean-subtraction
        // parameters; replacing it with a generic packed-cache name makes
        // current rows disagree with the already encoded historical rows.
        ggml_set_name(cached->k, source_k->name);
        ggml_set_name(cached->v, source_v->name);
        cached->buffer = llama_vram_hold_alloc_ctx_tensors(cached->context,
                ggml_backend_get_default_buffer_type(backend));
        if (cached->buffer == nullptr) {
            return nullptr;
        }

        for (auto & previous : entries_) {
            if (!previous->draining && same_domain(*previous, layer_id, sequence_id, backend)) {
                previous->draining = true;
            }
        }
        entries_.push_back(std::move(cached));
        if (graph_build_active_) {
            try {
                graph_build_entries_.push_back(entries_.back().get());
            } catch (...) {
                entries_.pop_back();
                return nullptr;
            }
        }
        return entries_.back().get();
    } catch (...) {
        return nullptr;
    }
}

uint64_t llama_kv_attention_packed_cache::content_version(
        const entry * cached, uint32_t page_index) const noexcept {
    return cached != nullptr && page_index < cached->content_versions.size()
        ? cached->content_versions[page_index] : UINT64_MAX;
}

void llama_kv_attention_packed_cache::set_content_version(
        entry * cached, uint32_t page_index, uint64_t version) noexcept {
    if (cached != nullptr && page_index < cached->content_versions.size()) {
        cached->content_versions[page_index] = version;
        if (page_index < cached->slots.size()) {
            cached->slots[page_index].copied_content_version = version;
        }
        rebuild_dirty_intervals(*cached);
    }
}

void llama_kv_attention_packed_cache::set_content_versions(
        entry * cached,
        const std::vector<llama_kv_attention_view_page> & pages) noexcept {
    if (cached == nullptr) {
        return;
    }
    const size_t count = std::min(pages.size(), cached->content_versions.size());
    for (size_t i = 0; i < count; ++i) {
        cached->content_versions[i] = pages[i].page_generation;
        if (i < cached->slots.size()) {
            cached->slots[i].copied_content_version = pages[i].page_generation;
        }
    }
    rebuild_dirty_intervals(*cached);
}

const char * llama_kv_attention_execution_mode_name(
        llama_kv_attention_execution_mode mode) noexcept {
    switch (mode) {
        case llama_kv_attention_execution_mode::off:       return "off";
        case llama_kv_attention_execution_mode::observe:   return "observe";
        case llama_kv_attention_execution_mode::selective:return "selective";
        case llama_kv_attention_execution_mode::exact:    return "exact";
    }
    return "invalid";
}

const char * llama_kv_attention_execution_phase_name(
        llama_kv_attention_execution_phase phase) noexcept {
    switch (phase) {
        case llama_kv_attention_execution_phase::prefill: return "prefill";
        case llama_kv_attention_execution_phase::decode:  return "decode";
        case llama_kv_attention_execution_phase::mtp_verify: return "mtp_verify";
    }
    return "invalid";
}

const char * llama_kv_attention_execution_route_name(
        llama_kv_attention_execution_route route) noexcept {
    switch (route) {
        case llama_kv_attention_execution_route::dense:             return "dense";
        case llama_kv_attention_execution_route::observe:           return "observe";
        case llama_kv_attention_execution_route::selected_reference:return "selected reference";
        case llama_kv_attention_execution_route::selected_dense:   return "selected dense";
        case llama_kv_attention_execution_route::selected_packed:  return "selected packed";
        case llama_kv_attention_execution_route::selected_direct:  return "selected direct";
        case llama_kv_attention_execution_route::exact_reference:   return "exact reference";
        case llama_kv_attention_execution_route::exact_direct:      return "exact direct";
        case llama_kv_attention_execution_route::refusal:           return "refusal";
    }
    return "invalid";
}

const char * llama_kv_attention_execution_route_override_name(
        llama_kv_attention_execution_route_override route) noexcept {
    switch (route) {
        case llama_kv_attention_execution_route_override::automatic: return "auto";
        case llama_kv_attention_execution_route_override::reference:return "reference";
        case llama_kv_attention_execution_route_override::dense:     return "dense";
        case llama_kv_attention_execution_route_override::packed:    return "packed";
        case llama_kv_attention_execution_route_override::direct:    return "direct";
        case llama_kv_attention_execution_route_override::invalid:   return "invalid";
    }
    return "invalid";
}

const char * llama_kv_attention_scratch_context_role_name(
        llama_kv_attention_scratch_context_role role) noexcept {
    switch (role) {
        case llama_kv_attention_scratch_context_role::target: return "target";
        case llama_kv_attention_scratch_context_role::draft:  return "draft";
    }
    return "invalid";
}

void llama_kv_attention_execution_route_counts::record(
        llama_kv_attention_execution_route route) noexcept {
    uint64_t * counter = nullptr;
    switch (route) {
        case llama_kv_attention_execution_route::dense:              counter = &dense; break;
        case llama_kv_attention_execution_route::observe:            counter = &observe; break;
        case llama_kv_attention_execution_route::selected_reference: counter = &selected_reference; break;
        case llama_kv_attention_execution_route::selected_dense:    counter = &selected_dense; break;
        case llama_kv_attention_execution_route::selected_packed:   counter = &selected_packed; break;
        case llama_kv_attention_execution_route::selected_direct:   counter = &selected_direct; break;
        case llama_kv_attention_execution_route::exact_reference:   counter = &exact_reference; break;
        case llama_kv_attention_execution_route::exact_direct:      counter = &exact_direct; break;
        case llama_kv_attention_execution_route::refusal:            counter = &refusal; break;
    }
    if (counter != nullptr) {
        *counter = saturating_add(*counter, uint64_t(1));
    }
}

void llama_kv_attention_execution_metrics::record_wait_time_us(uint64_t elapsed_us) noexcept {
    wait_time_us = saturating_add(wait_time_us, elapsed_us);
}

void llama_kv_attention_execution_metrics::record_copy_time_us(uint64_t elapsed_us) noexcept {
    copy_time_us = saturating_add(copy_time_us, elapsed_us);
}

void llama_kv_attention_execution_metrics::record_queue_time_us(uint64_t elapsed_us) noexcept {
    queue_time_us = saturating_add(queue_time_us, elapsed_us);
}

void llama_kv_attention_execution_metrics::record_pack(
        uint64_t bytes, uint64_t elapsed_us) noexcept {
    pack_bytes = saturating_add(pack_bytes, bytes);
    pack_time_us = saturating_add(pack_time_us, elapsed_us);
    pack_epochs = saturating_add(pack_epochs, uint64_t(1));
}

const char * llama_kv_attention_execution_status_name(
        llama_kv_attention_execution_status status) noexcept {
    switch (status) {
        case llama_kv_attention_execution_status::ok:                    return "ok";
        case llama_kv_attention_execution_status::disabled:              return "disabled";
        case llama_kv_attention_execution_status::invalid_metadata:      return "invalid_metadata";
        case llama_kv_attention_execution_status::invalid_prefill_transition:return "invalid_prefill_transition";
        case llama_kv_attention_execution_status::overflow:               return "overflow";
        case llama_kv_attention_execution_status::not_configured:         return "not_configured";
    }
    return "invalid";
}

void llama_kv_attention_execution_metrics::record_exact_ledger(
        const llama_kv_attention_exact_ledger & ledger) noexcept {
    exact_refusal_reason.clear();
    exact_plan_waves = ledger.waves;
    exact_plan_pages = ledger.logical_page_count;
    exact_resident_pages = ledger.resident_pages;
    exact_cold_pages = ledger.cold_pages;
    exact_pages_visited = ledger.pages_visited;
    exact_h2d_useful_bytes = ledger.h2d_useful_bytes;
    exact_h2d_aligned_bytes = ledger.h2d_aligned_bytes;
    exact_h2d_transfer_time_us = ledger.h2d_transfer_time_us;
    exact_waits = ledger.waits;
    exact_peak_staging_pages = ledger.peak_staging_pages;
    exact_duplicate_pages = ledger.duplicate_pages;
    exact_missing_pages = ledger.missing_pages;
    exact_stale_pages = ledger.stale_pages;
    exact_faults = ledger.cold_pages;
}

void llama_kv_attention_execution_metrics::record_exact_refusal(
        const std::string & reason) noexcept {
    try {
        exact_refusal_reason = reason;
    } catch (...) {
        exact_refusal_reason.clear();
    }
}

uint64_t llama_kv_attention_scratch_request::required_rows() const noexcept {
    return saturating_add(saturating_add(std::max(resident_rows, materialized_rows()), transfer_rows), router_rows);
}

size_t llama_kv_attention_scratch_request::required_bytes() const noexcept {
    if (materialized_k_bytes_per_row != 0 || materialized_v_bytes_per_row != 0) {
        const auto multiply = [](uint64_t rows, size_t row_bytes, size_t & result) {
            if (row_bytes != 0 && rows > uint64_t(std::numeric_limits<size_t>::max()) / row_bytes) {
                return false;
            }
            result = size_t(rows) * row_bytes;
            return true;
        };
        size_t k_bytes = 0;
        size_t v_bytes = 0;
        if (!multiply(materialized_k_rows, materialized_k_bytes_per_row, k_bytes) ||
            !multiply(materialized_v_rows, materialized_v_bytes_per_row, v_bytes) ||
            k_bytes > std::numeric_limits<size_t>::max() - v_bytes) {
            return std::numeric_limits<size_t>::max();
        }
        size_t result = k_bytes + v_bytes;
        if (packed_bytes > uint64_t(std::numeric_limits<size_t>::max()) - result) {
            return std::numeric_limits<size_t>::max();
        }
        return result + size_t(packed_bytes);
    }
    const uint64_t rows = required_rows();
    const size_t row_bytes = bytes_per_row;
    size_t result = 0;
    if (row_bytes != 0 && rows > uint64_t(std::numeric_limits<size_t>::max()) / row_bytes) {
        return std::numeric_limits<size_t>::max();
    }
    result = size_t(rows) * row_bytes;
    if (packed_bytes > uint64_t(std::numeric_limits<size_t>::max()) - result) {
        return std::numeric_limits<size_t>::max();
    }
    return result + size_t(packed_bytes);
}

llama_kv_attention_execution_route llama_kv_attention_execution::planned_route(
        const llama_kv_attention_operator_metadata & metadata,
        llama_kv_attention_execution_phase phase,
        bool direct_capable,
        bool dense_capable,
        bool packed_capable) const noexcept {
    if (mode_ == llama_kv_attention_execution_mode::off) {
        return llama_kv_attention_execution_route::dense;
    }
    if (mode_ == llama_kv_attention_execution_mode::observe) {
        return llama_kv_attention_execution_route::observe;
    }
    if (mode_ == llama_kv_attention_execution_mode::exact) {
        return (phase == llama_kv_attention_execution_phase::decode ||
                phase == llama_kv_attention_execution_phase::mtp_verify) &&
               direct_capable && direct_shape(metadata)
            ? llama_kv_attention_execution_route::exact_direct
            : llama_kv_attention_execution_route::exact_reference;
    }

    if (route_override_ != llama_kv_attention_execution_route_override::automatic) {
        switch (route_override_) {
            case llama_kv_attention_execution_route_override::reference:
                return llama_kv_attention_execution_route::selected_reference;
            case llama_kv_attention_execution_route_override::dense:
                return dense_capable ? llama_kv_attention_execution_route::selected_dense
                                      : llama_kv_attention_execution_route::refusal;
            case llama_kv_attention_execution_route_override::packed:
                return packed_capable ? llama_kv_attention_execution_route::selected_packed
                                       : llama_kv_attention_execution_route::refusal;
            case llama_kv_attention_execution_route_override::direct:
                return direct_capable && production_direct_shape(metadata, phase)
                    ? llama_kv_attention_execution_route::selected_direct
                    : llama_kv_attention_execution_route::refusal;
            case llama_kv_attention_execution_route_override::automatic:
            case llama_kv_attention_execution_route_override::invalid:
                return llama_kv_attention_execution_route::refusal;
        }
    }

    // The direct paged Turbo4 consumer preserves the selected logical page
    // set without materializing a compact owner. Prefer it whenever the
    // backend and metadata satisfy the same shape contract as the explicit
    // direct diagnostic route.
    if (direct_capable && production_direct_shape(metadata, phase)) {
        return llama_kv_attention_execution_route::selected_direct;
    }

    if (dense_capable) {
        return llama_kv_attention_execution_route::selected_dense;
    }

    // The compact packed bridge remains available only through the explicit
    // diagnostic override above. Automatic dispatch must retain the
    // canonical reference consumer for non-contiguous selected views when
    // the direct and dense routes are unavailable.
    return llama_kv_attention_execution_route::selected_reference;
}

uint32_t llama_kv_attention_prefill_chunk_size(
        uint32_t configured_ubatch,
        uint32_t physical_page_count,
        uint32_t page_tokens,
        uint32_t query_tile) noexcept {
    if (configured_ubatch == 0 || physical_page_count == 0 || page_tokens == 0) {
        return configured_ubatch;
    }
    const uint64_t rows = uint64_t(physical_page_count) * page_tokens;
    const uint64_t bounded_rows = std::min<uint64_t>(rows, std::numeric_limits<uint32_t>::max());
    const uint64_t bounded_tile = query_tile == 0 ? bounded_rows : query_tile;
    return uint32_t(std::min<uint64_t>(configured_ubatch,
            std::min<uint64_t>(bounded_rows, bounded_tile)));
}

llama_kv_attention_prefill_batch_plan llama_kv_attention_prefill_batch_plan_make(
        uint32_t requested_batch,
        uint32_t physical_page_count,
        uint32_t page_tokens,
        uint32_t query_tile) noexcept {
    llama_kv_attention_prefill_batch_plan result;
    result.requested_batch = requested_batch;
    result.query_tile = query_tile == 0 ? LLAMA_KV_ATTENTION_PREFILL_QUERY_TILE : query_tile;
    if (physical_page_count == 0 || page_tokens == 0) {
        result.physical_write_capacity = requested_batch;
    } else {
        const uint64_t rows = uint64_t(physical_page_count) * page_tokens;
        result.physical_write_capacity = uint32_t(std::min<uint64_t>(
                rows, std::numeric_limits<uint32_t>::max()));
    }
    result.effective_batch = std::min(requested_batch, result.physical_write_capacity);
    if (result.effective_batch != 0) {
        result.subbatch_count = (requested_batch + result.effective_batch - 1) /
                result.effective_batch;
    }
    return result;
}

llama_kv_attention_execution_status llama_kv_attention_prefill_admission::append(
        uint32_t logical_page, uint32_t row_count) noexcept {
    if (phase_ != llama_kv_attention_execution_phase::prefill || row_count == 0 ||
        row_count > VBR_GENERATION_PAGE_CELLS || decode_ready_) {
        return llama_kv_attention_execution_status::invalid_prefill_transition;
    }

    if (pages_.empty()) {
        if (logical_page != 0) {
            return llama_kv_attention_execution_status::invalid_prefill_transition;
        }
        pages_.push_back(0);
    } else if (logical_page >= pages_.size()) {
        if (logical_page != pages_.size() || pages_.back() != VBR_GENERATION_PAGE_CELLS) {
            return llama_kv_attention_execution_status::invalid_prefill_transition;
        }
        pages_.push_back(0);
    } else if (logical_page + 1 != pages_.size()) {
        return llama_kv_attention_execution_status::invalid_prefill_transition;
    }

    if (pages_[logical_page] > VBR_GENERATION_PAGE_CELLS - row_count) {
        return llama_kv_attention_execution_status::overflow;
    }
    pages_[logical_page] += row_count;
    if (resident_rows_ > std::numeric_limits<uint32_t>::max() - row_count) {
        return llama_kv_attention_execution_status::overflow;
    }
    resident_rows_ += row_count;
    tail_finished_ = false;
    return llama_kv_attention_execution_status::ok;
}

llama_kv_attention_execution_status llama_kv_attention_prefill_admission::finish_tail() noexcept {
    if (phase_ != llama_kv_attention_execution_phase::prefill || pages_.empty() ||
        pages_.back() == 0 || pages_.back() > VBR_GENERATION_PAGE_CELLS) {
        return llama_kv_attention_execution_status::invalid_prefill_transition;
    }
    tail_finished_ = true;
    return llama_kv_attention_execution_status::ok;
}

llama_kv_attention_execution_status llama_kv_attention_prefill_admission::begin_decode() noexcept {
    if (phase_ != llama_kv_attention_execution_phase::prefill || pages_.empty() ||
        pages_.back() == 0 || (pages_.back() != VBR_GENERATION_PAGE_CELLS && !tail_finished_)) {
        return llama_kv_attention_execution_status::invalid_prefill_transition;
    }
    phase_ = llama_kv_attention_execution_phase::decode;
    decode_ready_ = true;
    return llama_kv_attention_execution_status::ok;
}

llama_kv_attention_execution::llama_kv_attention_execution(
        llama_kv_attention_execution_mode mode) noexcept : mode_(mode) {}

void llama_kv_attention_execution::set_mode(llama_kv_attention_execution_mode mode) noexcept {
    if (mode_ != mode) {
        clear();
    }
    mode_ = mode;
}

void llama_kv_attention_execution::set_route_override(const char * name) noexcept {
    auto parsed = llama_kv_attention_execution_route_override::automatic;
    if (name != nullptr && *name != '\0' && std::strcmp(name, "auto") != 0) {
        if (std::strcmp(name, "reference") == 0) {
            parsed = llama_kv_attention_execution_route_override::reference;
        } else if (std::strcmp(name, "dense") == 0) {
            parsed = llama_kv_attention_execution_route_override::dense;
        } else if (std::strcmp(name, "packed") == 0) {
            parsed = llama_kv_attention_execution_route_override::packed;
        } else if (std::strcmp(name, "direct") == 0) {
            parsed = llama_kv_attention_execution_route_override::direct;
        } else {
            parsed = llama_kv_attention_execution_route_override::invalid;
        }
    }
    if (route_override_ != parsed) {
        clear();
    }
    route_override_ = parsed;
}

bool llama_kv_attention_execution::same_graph(
        const llama_kv_attention_operator_metadata & metadata,
        llama_kv_attention_execution_phase phase,
        uint64_t representation_epoch,
        uint64_t shape_epoch,
        llama_kv_attention_execution_route route) const noexcept {
    // Direct CUDA keeps the page table, native positions, mask, and query
    // positions in graph-owned device inputs. A residency publication changes
    // the immutable view lease, but not the captured graph topology: set_input
    // patches those inputs before the next submission. Packed content
    // generations are mutable owner data; only a physical remap changes the
    // captured view/copy plan.
    const bool mutable_direct_inputs =
        route == llama_kv_attention_execution_route::selected_direct ||
        route == llama_kv_attention_execution_route::selected_packed;
    return have_graph_ && metadata.graph_layout_key() == metadata_.graph_layout_key() &&
           (mutable_direct_inputs || metadata.table_epoch() == table_epoch_) &&
           phase == phase_ && representation_epoch == representation_epoch_ &&
           shape_epoch == shape_epoch_ && route == route_ &&
           ((route != llama_kv_attention_execution_route::selected_dense &&
             route != llama_kv_attention_execution_route::selected_packed) ||
            metadata.graph_physical_key() == metadata_.graph_physical_key()) &&
           exact_graph_plan_.get() == graph_plan_.get();
}

llama_kv_attention_execution_decision llama_kv_attention_execution::prepare(
        const llama_kv_attention_operator_metadata & metadata,
        llama_kv_attention_execution_phase phase,
        uint64_t representation_epoch,
        uint64_t shape_epoch,
        bool direct_capable,
        const llama_kv_attention_scratch_request & scratch,
        const std::string & direct_reason,
        bool dense_capable,
        bool packed_capable) {
    llama_kv_attention_execution_decision result;
    result.phase = phase;
    result.representation_epoch = representation_epoch;
    result.shape_epoch = shape_epoch;
    result.scratch_rows = scratch.required_rows();
    result.scratch_bytes = scratch.required_bytes();

    if (mode_ == llama_kv_attention_execution_mode::off) {
        result.status = llama_kv_attention_execution_status::disabled;
        result.route = llama_kv_attention_execution_route::dense;
        result.reason = "feature disabled";
    } else if (mode_ == llama_kv_attention_execution_mode::observe) {
        result.status = llama_kv_attention_execution_status::ok;
        result.route = llama_kv_attention_execution_route::observe;
        result.reason = "observation preserves dense attention";
    } else if (mode_ == llama_kv_attention_execution_mode::exact) {
        result.status = llama_kv_attention_execution_status::ok;
        result.route = planned_route(metadata, phase, direct_capable,
                dense_capable, packed_capable);
        result.reason = result.route == llama_kv_attention_execution_route::exact_direct
            ? "all-page Turbo4 direct exact route"
            : "all-page online-softmax reference";
    } else if ((scratch.required_rows() == std::numeric_limits<uint64_t>::max() &&
                (scratch.resident_rows != 0 || scratch.materialized_rows() != 0 ||
                 scratch.transfer_rows != 0 || scratch.router_rows != 0)) ||
               (scratch.required_bytes() == std::numeric_limits<size_t>::max() &&
                (scratch.bytes_per_row != 0 || scratch.packed_bytes != 0 ||
                 scratch.materialized_k_bytes_per_row != 0 ||
                 scratch.materialized_v_bytes_per_row != 0))) {
        result.status = llama_kv_attention_execution_status::overflow;
        result.route = llama_kv_attention_execution_route::refusal;
        result.reason = "selected scratch reservation overflows";
    } else if (!metadata.valid() || !metadata.enabled()) {
        result.status = llama_kv_attention_execution_status::invalid_metadata;
        result.route = llama_kv_attention_execution_route::refusal;
        result.reason = "selected metadata is invalid";
    } else {
        result.route = planned_route(metadata, phase, direct_capable,
                dense_capable, packed_capable);
        if (result.route == llama_kv_attention_execution_route::refusal) {
            result.status = llama_kv_attention_execution_status::not_configured;
            result.reason = std::string("route override '") + route_override_name() +
                "' is unsupported for this selected view/phase";
            saturating_add_u64(metrics_.route_override_refused, 1);
        } else {
            result.status = llama_kv_attention_execution_status::ok;
            if (route_override_ != llama_kv_attention_execution_route_override::automatic) {
                saturating_add_u64(metrics_.route_override_accepted, 1);
            }
            result.reason = result.route == llama_kv_attention_execution_route::selected_dense
                ? "contiguous Turbo4 rows use dense Flash Attention"
                : result.route == llama_kv_attention_execution_route::selected_packed
                ? phase == llama_kv_attention_execution_phase::prefill &&
                  metadata.n_query_tokens() > 2 * LLAMA_KV_ATTENTION_PREFILL_QUERY_TILE
                    ? "prefill query batch uses cached compact packing above two 64-token tiles"
                    : "noncontiguous Turbo4 rows use cached compact packing"
                : result.route == llama_kv_attention_execution_route::selected_direct
                ? phase == llama_kv_attention_execution_phase::prefill
                    ? "qualified Turbo4 selective prefill query tile"
                    : "qualified Turbo4 decode"
                : direct_capable && !production_direct_shape(metadata, phase)
                    ? "bounded Turbo4 selected reference for unsupported direct query tile"
                    : direct_reason.empty() ? "compact selected reference" : direct_reason;
        }
        result.table_epoch = metadata.table_epoch();
    }

    result.graph_rebuild = result.status == llama_kv_attention_execution_status::ok &&
        !same_graph(metadata, phase, representation_epoch, shape_epoch, result.route);

    if (result.status == llama_kv_attention_execution_status::ok) {
        const bool rebuild = result.graph_rebuild;
        if (have_graph_ && metadata.table_epoch() != table_epoch_) {
            saturating_add_u64(metrics_.table_epoch_changes, 1);
        }
        ++metrics_.graph_submission_count;
        if (rebuild) {
            ++metrics_.graph_capture_count;
            ++metrics_.graph_rebuild_count;
        } else {
            ++metrics_.graph_replay_count;
        }
        if (result.route == llama_kv_attention_execution_route::selected_direct && rebuild) {
            // This is the exact host-to-device page-table payload written by
            // llm_graph_input_attn_kv::set_input when a graph is captured. A
            // replay retains the immutable descriptor and does not upload it
            // again. The descriptor is four uint32 fields followed by one
            // native int64 position.
            constexpr uint64_t direct_page_bytes =
                4 * sizeof(uint32_t) + sizeof(int64_t);
            const uint64_t page_count = uint64_t(metadata.page_table().size());
            saturating_add_u64(metrics_.table_upload_bytes,
                    page_count > std::numeric_limits<uint64_t>::max() / direct_page_bytes
                        ? std::numeric_limits<uint64_t>::max()
                        : page_count * direct_page_bytes);
        }
        metrics_.scratch_high_water_rows = std::max(
                metrics_.scratch_high_water_rows, result.scratch_rows);
        metrics_.scratch_high_water_bytes = std::max(
                metrics_.scratch_high_water_bytes, uint64_t(result.scratch_bytes));
        metrics_.selected_pages = std::max<uint64_t>(metrics_.selected_pages,
                metadata.page_table().size());
        metadata_ = metadata;
        route_ = result.route;
        phase_ = phase;
        table_epoch_ = metadata.table_epoch();
        representation_epoch_ = representation_epoch;
        shape_epoch_ = shape_epoch;
        graph_plan_ = exact_graph_plan_;
        have_graph_ = true;
        graph_fences_.push_back(metadata.acquire_graph_fence());
        if (result.route == llama_kv_attention_execution_route::selected_packed) {
            metrics_.packed_inflight_consumers_high_water = std::max(
                    metrics_.packed_inflight_consumers_high_water,
                    uint64_t(graph_fences_.size()));
        }
    }

    switch (phase) {
        case llama_kv_attention_execution_phase::prefill:
            metrics_.prefill_routes.record(result.route);
            break;
        case llama_kv_attention_execution_phase::decode:
            metrics_.decode_routes.record(result.route);
            break;
        case llama_kv_attention_execution_phase::mtp_verify:
            metrics_.mtp_verify_routes.record(result.route);
            break;
    }
    metrics_.selected_page_count = metadata.page_table().size();
    metrics_.selected_page_ids.clear();
    metrics_.selected_page_ids.reserve(metadata.page_table().size());
    for (const auto & page : metadata.page_table()) {
        metrics_.selected_page_ids.push_back(page.logical_page);
    }

    LLAMA_LOG_DEBUG("kv-attention: %s path (%s, table=%llu, representation=%llu, shape=%llu, scratch_rows=%llu)\n",
            llama_kv_attention_execution_route_name(result.route),
            llama_kv_attention_execution_phase_name(phase),
            (unsigned long long) result.table_epoch,
            (unsigned long long) result.representation_epoch,
            (unsigned long long) result.shape_epoch,
            (unsigned long long) result.scratch_rows);
    return result;
}

void llama_kv_attention_execution::complete_one_graph() noexcept {
    if (graph_fences_.empty()) {
        return;
    }
    graph_fences_.front().release();
    graph_fences_.erase(graph_fences_.begin());
    ++metrics_.graph_completion_count;
}

void llama_kv_attention_execution::record_descriptor_prepare_us(uint64_t elapsed_us) noexcept {
    saturating_add_u64(metrics_.descriptor_prepare_us, elapsed_us);
}

void llama_kv_attention_execution::record_kernel_us(uint64_t elapsed_us) noexcept {
    saturating_add_u64(metrics_.kernel_us, elapsed_us);
}

void llama_kv_attention_execution::record_total_token_us(uint64_t elapsed_us) noexcept {
    saturating_add_u64(metrics_.total_token_us, elapsed_us);
}

void llama_kv_attention_execution::record_wait() noexcept {
    saturating_add_u64(metrics_.waits, 1);
}

void llama_kv_attention_execution::record_wait_time_us(uint64_t elapsed_us) noexcept {
    metrics_.record_wait_time_us(elapsed_us);
}

void llama_kv_attention_execution::record_copy_time_us(uint64_t elapsed_us) noexcept {
    metrics_.record_copy_time_us(elapsed_us);
}

void llama_kv_attention_execution::record_queue_time_us(uint64_t elapsed_us) noexcept {
    metrics_.record_queue_time_us(elapsed_us);
}

void llama_kv_attention_execution::record_pack(uint64_t bytes, uint64_t elapsed_us) noexcept {
    metrics_.record_pack(bytes, elapsed_us);
}

void llama_kv_attention_execution::record_graph_construction_us(uint64_t elapsed_us) noexcept {
    metrics_.record_graph_construction_us(elapsed_us);
}

void llama_kv_attention_execution::record_effective_ubatch(uint64_t value) noexcept {
    metrics_.record_effective_ubatch(value);
}

void llama_kv_attention_execution::record_prefill_batch(
        const llama_kv_attention_prefill_batch_plan & plan) noexcept {
    metrics_.record_prefill_batch(plan);
}

void llama_kv_attention_execution::record_target_tokens(uint64_t value) noexcept {
    metrics_.record_target_tokens(value);
}

void llama_kv_attention_execution::reset_metrics() noexcept {
    if (metrics_reset_epoch_ != std::numeric_limits<uint64_t>::max()) {
        ++metrics_reset_epoch_;
    }
    metrics_ = {};
}

void llama_kv_attention_execution::clear() noexcept {
    // A route/mode change invalidates the current graph key, but it does not
    // cancel work already submitted to the scheduler.  Keep every lease until
    // the normal completion boundary so an old page table cannot be reclaimed
    // while its graph is still consuming it.
    metadata_ = {};
    route_ = llama_kv_attention_execution_route::dense;
    phase_ = llama_kv_attention_execution_phase::prefill;
    table_epoch_ = representation_epoch_ = shape_epoch_ = 0;
    have_graph_ = false;
    exact_graph_plan_.reset();
    graph_plan_.reset();
}
