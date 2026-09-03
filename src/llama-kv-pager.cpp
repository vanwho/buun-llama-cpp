#include "llama-kv-pager.h"

#include <algorithm>
#include <limits>
#include <new>

namespace {
bool mul(uint64_t a, uint64_t b, uint64_t & out) noexcept {
    if (a && b > std::numeric_limits<uint64_t>::max() / a) return false;
    out = a * b; return true;
}

bool add(uint64_t a, uint64_t b, uint64_t & out) noexcept {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    out = a + b; return true;
}
}

const char * llama_kv_pager_status_name(llama_kv_pager_status status) noexcept {
    switch (status) {
        case llama_kv_pager_status::ok: return "ok";
        case llama_kv_pager_status::disabled: return "disabled";
        case llama_kv_pager_status::invalid_geometry: return "invalid_geometry";
        case llama_kv_pager_status::unsupported_authority: return "unsupported_authority";
        case llama_kv_pager_status::missing_backend: return "missing_backend";
        case llama_kv_pager_status::host_budget: return "host_budget";
        case llama_kv_pager_status::admission: return "admission";
        case llama_kv_pager_status::allocation: return "allocation";
        case llama_kv_pager_status::realized_mismatch: return "realized_mismatch";
        case llama_kv_pager_status::overflow: return "overflow";
    }
    return "invalid";
}

const char * llama_kv_pager_write_status_name(llama_kv_pager_write_status status) noexcept {
    switch (status) {
        case llama_kv_pager_write_status::ok: return "ok";
        case llama_kv_pager_write_status::disabled: return "disabled";
        case llama_kv_pager_write_status::invalid_position: return "invalid_position";
        case llama_kv_pager_write_status::no_victim: return "no_victim";
        case llama_kv_pager_write_status::all_pinned: return "all_pinned";
        case llama_kv_pager_write_status::stale_generation: return "stale_generation";
        case llama_kv_pager_write_status::transaction: return "transaction";
        case llama_kv_pager_write_status::overflow: return "overflow";
    }
    return "invalid";
}

bool llama_kv_pager_plan(const llama_kv_pager_config & config,
        const llama_kv_pager_geometry & geometry, llama_kv_pager_resources resources,
        llama_kv_pager_snapshot & output, llama_kv_pager_status & status) noexcept {
    output = {};
    status = llama_kv_pager_status::invalid_geometry;
    if (!config.enabled()) { status = llama_kv_pager_status::disabled; return true; }
    std::string error;
    if (!config.validate(error) || geometry.context_tokens == 0 || geometry.page_tokens == 0 ||
        geometry.page_tokens != config.page_size || geometry.attention_layers == 0 ||
        geometry.kv_heads == 0 || geometry.key_length == 0 || geometry.value_length == 0 ||
        geometry.page_bytes == 0) return false;
    if (resources.duplicate_representation_authority) { status = llama_kv_pager_status::unsupported_authority; return false; }
    const uint64_t logical = (geometry.context_tokens - 1) / geometry.page_tokens + 1;
    if (logical > std::numeric_limits<uint32_t>::max()) {
        status = llama_kv_pager_status::overflow;
        return false;
    }
    if (!resources.host_budget_known || resources.host_budget_bytes == 0) { status = llama_kv_pager_status::host_budget; return false; }
    if (!mul(logical, sizeof(llama_kv_page_id), resources.host_metadata_bytes) ||
        resources.host_metadata_bytes > resources.host_budget_bytes) { status = llama_kv_pager_status::host_budget; return false; }
    auto & admission = resources.admission;
    admission.page_tokens = geometry.page_tokens;
    admission.logical_page_count = logical;
    admission.target_page_bytes = geometry.page_bytes;
    admission.user_page_cap = config.hot_pages.automatic ? 0 : config.hot_pages.value;
    admission.allocation_granularity = resources.allocator_granularity;
    const auto result = llama_cache_budget_admit(admission);
    if (result.refusal != llama_cache_budget_admission_refusal::none || result.admitted_pages == 0) {
        status = llama_kv_pager_status::admission; output.admission = result; return false;
    }
    uint64_t rows = 0, bytes = 0;
    if (!mul(result.admitted_pages, geometry.page_tokens, rows) ||
        !mul(result.admitted_pages, result.page_charge_bytes, bytes)) { status = llama_kv_pager_status::overflow; return false; }
    output.geometry = geometry;
    output.admission = result;
    output.logical_page_count = uint32_t(logical);
    output.physical_page_count = uint32_t(result.admitted_pages);
    output.physical_rows = rows;
    output.physical_bytes = bytes;
    output.host_metadata_bytes = resources.host_metadata_bytes;
    status = llama_kv_pager_status::ok;
    return true;
}

std::unique_ptr<llama_kv_pager> llama_kv_pager::create(
        const llama_kv_pager_config & config, const llama_kv_pager_geometry & geometry,
        llama_kv_pager_resources resources, llama_kv_pager_backend backend,
        llama_kv_pager_status & status) noexcept {
    status = llama_kv_pager_status::invalid_geometry;
    if (!config.enabled()) { status = llama_kv_pager_status::disabled; return nullptr; }
    if (config.mode != llama_kv_pager_mode::observe &&
        (!backend.allocate || !backend.release)) {
        status = llama_kv_pager_status::missing_backend;
        return nullptr;
    }
    auto output = std::unique_ptr<llama_kv_pager>(new (std::nothrow) llama_kv_pager);
    if (!output) { status = llama_kv_pager_status::allocation; return nullptr; }
    output->backend_ = std::move(backend);
    try {
        if (!llama_kv_pager_plan(config, geometry, resources, output->snapshot_, status)) return nullptr;
        if (config.mode == llama_kv_pager_mode::observe) {
            output->snapshot_.physical_page_count = 0;
            output->snapshot_.physical_rows = 0;
            output->snapshot_.physical_bytes = 0;
            output->snapshot_.initialized = true;
            output->pages_.reserve(output->snapshot_.logical_page_count);
            output->slot_pages_.clear();
            output->residency_ = llama_kv_residency_table(0);
            return output;
        }
        if (!output->backend_.allocate(output->snapshot_.physical_bytes, output->allocation_) ||
            output->allocation_.handle == nullptr || output->allocation_.realized_bytes != output->snapshot_.physical_bytes) {
            const bool allocated = output->allocation_.handle != nullptr;
            const bool mismatch = allocated && output->allocation_.realized_bytes != output->snapshot_.physical_bytes;
            if (allocated) output->backend_.release(output->allocation_);
            status = mismatch ? llama_kv_pager_status::realized_mismatch : llama_kv_pager_status::allocation;
            return nullptr;
        }
        output->snapshot_.realized_bytes = output->allocation_.realized_bytes;
        output->snapshot_.initialized = true;
        output->pages_.reserve(output->snapshot_.logical_page_count);
        output->slot_pages_.assign(output->snapshot_.physical_page_count, -1);
        output->residency_ = llama_kv_residency_table(output->snapshot_.physical_page_count);
        return output;
    } catch (...) {
        status = llama_kv_pager_status::allocation;
        return nullptr;
    }
}

llama_kv_pager::~llama_kv_pager() {
    if (allocation_.handle && backend_.release) backend_.release(allocation_);
}

llama_kv_pager_write_status llama_kv_pager::publish_page(page_state & page) noexcept {
    auto tx = residency_.begin();
    llama_kv_residency_status result = llama_kv_residency_status::not_found;
    for (const auto & existing : tx.pages()) {
        if (existing.id.session_generation == page.record.id.session_generation &&
            existing.id.sequence_id == page.record.id.sequence_id &&
            existing.id.sequence_generation == page.record.id.sequence_generation &&
            existing.id.logical_page == page.record.id.logical_page) {
            result = residency_.update(tx, page.record);
            break;
        }
    }
    if (result == llama_kv_residency_status::not_found) {
        result = residency_.replace(tx, page.record);
    }
    if (result != llama_kv_residency_status::ok ||
        residency_.publish(tx) != llama_kv_residency_status::ok) {
        residency_.rollback(tx);
        return llama_kv_pager_write_status::transaction;
    }
    return llama_kv_pager_write_status::ok;
}

llama_kv_pager_write_status llama_kv_pager::erase_page(page_state & page) noexcept {
    if (!page.present) return llama_kv_pager_write_status::ok;
    auto tx = residency_.begin();
    // The in-memory record may have just had its pin count or valid range changed
    // (for example while cancelling the write frontier). Bring the transaction up
    // to that record before erasing it, otherwise erase() would inspect the older
    // pinned snapshot and reject an otherwise legal removal.
    const auto update = residency_.update(tx, page.record);
    const auto result = update == llama_kv_residency_status::ok
        ? residency_.erase(tx, page.record.id)
        : update;
    if (result != llama_kv_residency_status::ok ||
        residency_.publish(tx) != llama_kv_residency_status::ok) {
        residency_.rollback(tx);
        return result == llama_kv_residency_status::pinned_slot
            ? llama_kv_pager_write_status::all_pinned
            : llama_kv_pager_write_status::transaction;
    }
    if (page.record.physical_slot < slot_pages_.size()) {
        slot_pages_[page.record.physical_slot] = -1;
    }
    page = {};
    return llama_kv_pager_write_status::ok;
}

llama_kv_pager::page_state * llama_kv_pager::find_page(
        int32_t sequence_id, uint32_t logical_page) noexcept {
    for (auto & page : pages_) {
        if (page.present && page.record.id.sequence_id == sequence_id &&
            page.record.id.logical_page == logical_page) return &page;
    }
    return nullptr;
}

const llama_kv_pager::page_state * llama_kv_pager::find_page(
        int32_t sequence_id, uint32_t logical_page) const noexcept {
    for (const auto & page : pages_) {
        if (page.present && page.record.id.sequence_id == sequence_id &&
            page.record.id.logical_page == logical_page) return &page;
    }
    return nullptr;
}

llama_kv_pager::page_state * llama_kv_pager::find_slot(uint32_t slot) noexcept {
    if (slot >= slot_pages_.size() || slot_pages_[slot] < 0 ||
        size_t(slot_pages_[slot]) >= pages_.size()) return nullptr;
    page_state & page = pages_[size_t(slot_pages_[slot])];
    return page.present ? &page : nullptr;
}

void llama_kv_pager::release_current_pin(page_state * except) noexcept {
    if (current_page_index_ == UINT32_MAX || current_page_index_ >= pages_.size()) return;
    page_state & page = pages_[current_page_index_];
    if (!page.present || &page == except || page.record.pin_count == 0) return;
    page.record.pin_count--;
    if (page.record.state == llama_kv_page_state::filling_gpu &&
        page.valid_rows.size() == snapshot_.geometry.page_tokens &&
        std::all_of(page.valid_rows.begin(), page.valid_rows.end(), [](uint8_t value) { return value != 0; })) {
        page.record.state = llama_kv_page_state::gpu_dirty;
    }
    (void) publish_page(page);
}

llama_kv_pager_write_status llama_kv_pager::begin_write(
        int32_t sequence_id, uint64_t sequence_generation, llama_pos position,
        llama_kv_pager_write_ticket & ticket) noexcept {
    ticket = {};
    if (!snapshot_.initialized || snapshot_.physical_page_count == 0) {
        return llama_kv_pager_write_status::disabled;
    }
    if (sequence_id < 0 || position < 0 || uint64_t(position) >= snapshot_.geometry.context_tokens ||
        snapshot_.geometry.page_tokens == 0) {
        return llama_kv_pager_write_status::invalid_position;
    }
    const uint64_t logical64 = uint64_t(position) / snapshot_.geometry.page_tokens;
    const uint64_t offset64 = uint64_t(position) % snapshot_.geometry.page_tokens;
    if (logical64 >= snapshot_.logical_page_count || offset64 > UINT32_MAX ||
        snapshot_.geometry.page_tokens > UINT32_MAX) {
        return llama_kv_pager_write_status::overflow;
    }
    const uint32_t logical = uint32_t(logical64);
    const uint32_t offset = uint32_t(offset64);
    const uint64_t physical = uint64_t(snapshot_.physical_page_count - 1) * snapshot_.geometry.page_tokens + offset;
    if (physical > UINT32_MAX) return llama_kv_pager_write_status::overflow;
    page_state * page = find_page(sequence_id, logical);
    if (page != nullptr && page->record.id.sequence_generation != sequence_generation) {
        return llama_kv_pager_write_status::stale_generation;
    }

    release_current_pin(page);
    bool created = false;
    if (page == nullptr) {
        uint32_t slot = UINT32_MAX;
        for (uint32_t i = 0; i < slot_pages_.size(); ++i) {
            if (slot_pages_[i] < 0) { slot = i; break; }
        }
        if (slot == UINT32_MAX) {
            for (uint32_t i = 0; i < slot_pages_.size(); ++i) {
                page_state * candidate = find_slot(i);
                if (candidate && candidate->record.pin_count == 0 && candidate->record.host_valid &&
                    (candidate->record.state == llama_kv_page_state::host_clean ||
                     candidate->record.state == llama_kv_page_state::gpu_host_clean)) {
                    if (erase_page(*candidate) != llama_kv_pager_write_status::ok) {
                        return llama_kv_pager_write_status::transaction;
                    }
                    slot = i;
                    break;
                }
            }
        }
        if (slot == UINT32_MAX) {
            bool pinned = false;
            for (const auto & candidate : pages_) {
                pinned = pinned || (candidate.present && candidate.record.pin_count != 0);
            }
            return pinned ? llama_kv_pager_write_status::all_pinned : llama_kv_pager_write_status::no_victim;
        }
        size_t page_index = 0;
        while (page_index < pages_.size() && pages_[page_index].present) ++page_index;
        if (page_index == pages_.size()) pages_.push_back({});
        page = &pages_[page_index];
        page->record = {};
        page->record.physical_slot = slot;
        page->record.id.session_generation = 1;
        page->record.id.sequence_id = sequence_id;
        page->record.id.sequence_generation = sequence_generation;
        page->record.id.logical_page = logical;
        page->record.id.page_generation = uint32_t(++mutation_generation_);
        page->record.id.position_begin = llama_pos(
                uint64_t(logical) * snapshot_.geometry.page_tokens);
        page->valid_rows.assign(snapshot_.geometry.page_tokens, 0);
        page->present = true;
        page->completed_segments = 0;
        slot_pages_[slot] = int32_t(page_index);
        created = true;
    }
    const bool row_was_valid = page->valid_rows[offset] != 0;
    page->valid_rows[offset] = 1;
    uint32_t end_row = 0;
    for (uint32_t i = 0; i < page->valid_rows.size(); ++i) {
        if (page->valid_rows[i]) end_row = i + 1;
    }
    const uint64_t end = std::min<uint64_t>(snapshot_.geometry.context_tokens,
            uint64_t(page->record.id.position_begin) + end_row);
    page->record.id.position_end = llama_pos(end);
    page->record.state = end_row == snapshot_.geometry.page_tokens
        ? llama_kv_page_state::gpu_dirty : llama_kv_page_state::filling_gpu;
    page->record.host_valid = false;
    page->record.dirty = true;
    page->record.pin_count++;
    current_page_index_ = uint32_t(page - pages_.data());
    if (publish_page(*page) != llama_kv_pager_write_status::ok) {
        page->valid_rows[offset] = row_was_valid;
        if (created) {
            slot_pages_[page->record.physical_slot] = -1;
            *page = {};
        }
        return llama_kv_pager_write_status::transaction;
    }
    ticket.sequence_id = sequence_id;
    ticket.sequence_generation = sequence_generation;
    ticket.logical_page = logical;
    ticket.physical_slot = page->record.physical_slot;
    ticket.physical_row = uint32_t(uint64_t(page->record.physical_slot) * snapshot_.geometry.page_tokens + offset);
    ticket.page_generation = page->record.id.page_generation;
    ticket.position = position;
    ticket.page_created = created;
    ticket.row_was_valid = row_was_valid;
    return llama_kv_pager_write_status::ok;
}

llama_kv_pager_write_status llama_kv_pager::complete_write(
        const llama_kv_pager_write_ticket & ticket, uint32_t completed_segments,
        bool graph_succeeded) noexcept {
    if (!snapshot_.initialized || snapshot_.physical_page_count == 0) {
        return llama_kv_pager_write_status::disabled;
    }
    page_state * page = find_page(ticket.sequence_id, ticket.logical_page);
    if (page == nullptr || page->record.physical_slot != ticket.physical_slot ||
        page->record.id.page_generation != ticket.page_generation ||
        page->record.id.sequence_generation != ticket.sequence_generation ||
        ticket.position < page->record.id.position_begin ||
        uint64_t(ticket.position) >= snapshot_.geometry.context_tokens) {
        return llama_kv_pager_write_status::stale_generation;
    }
    if (!graph_succeeded) {
        return cancel_write(ticket);
    }

    page->completed_segments = std::max(page->completed_segments, completed_segments);
    if (page->record.pin_count != 0) {
        page->record.pin_count--;
    }
    const bool full = !page->valid_rows.empty() && std::all_of(
            page->valid_rows.begin(), page->valid_rows.end(), [](uint8_t value) { return value != 0; });
    const uint64_t expected = uint64_t(snapshot_.geometry.attention_layers) * 2;
    page->record.state = full && page->completed_segments >= expected
        ? llama_kv_page_state::gpu_dirty : llama_kv_page_state::filling_gpu;
    page->record.host_valid = false;
    page->record.dirty = true;
    // The partial page is the write frontier. Keep it pinned until a later page takes over.
    const bool is_current = current_page_index_ < pages_.size() &&
        &pages_[current_page_index_] == page;
    if (!full && is_current) {
        page->record.pin_count = std::max(page->record.pin_count, 1u);
    }
    return publish_page(*page);
}

llama_kv_pager_write_status llama_kv_pager::cancel_write(
        const llama_kv_pager_write_ticket & ticket) noexcept {
    if (!snapshot_.initialized || snapshot_.physical_page_count == 0) {
        return llama_kv_pager_write_status::disabled;
    }
    page_state * page = find_page(ticket.sequence_id, ticket.logical_page);
    if (page == nullptr || page->record.physical_slot != ticket.physical_slot ||
        page->record.id.page_generation != ticket.page_generation ||
        page->record.id.sequence_generation != ticket.sequence_generation) {
        return llama_kv_pager_write_status::stale_generation;
    }
    if (page->record.pin_count != 0) {
        page->record.pin_count--;
    }
    const uint32_t offset = uint32_t(uint64_t(ticket.position) % snapshot_.geometry.page_tokens);
    if (!ticket.row_was_valid && offset < page->valid_rows.size()) {
        page->valid_rows[offset] = 0;
    }
    const bool any = std::any_of(page->valid_rows.begin(), page->valid_rows.end(),
            [](uint8_t value) { return value != 0; });
    if (ticket.page_created && !any) {
        return erase_page(*page);
    }
    uint32_t end_row = 0;
    for (uint32_t i = 0; i < page->valid_rows.size(); ++i) {
        if (page->valid_rows[i]) end_row = i + 1;
    }
    page->record.id.position_end = page->record.id.position_begin + llama_pos(end_row);
    page->record.state = end_row == page->valid_rows.size()
        ? llama_kv_page_state::gpu_dirty : llama_kv_page_state::filling_gpu;
    page->record.dirty = true;
    return publish_page(*page);
}

bool llama_kv_pager::physical_row(
        int32_t sequence_id, llama_pos position, uint32_t & row) const noexcept {
    row = UINT32_MAX;
    if (!snapshot_.initialized || snapshot_.physical_page_count == 0 || position < 0 ||
        uint64_t(position) >= snapshot_.geometry.context_tokens) return false;
    const uint32_t logical = uint32_t(uint64_t(position) / snapshot_.geometry.page_tokens);
    const uint32_t offset = uint32_t(uint64_t(position) % snapshot_.geometry.page_tokens);
    const page_state * page = find_page(sequence_id, logical);
    if (page == nullptr || offset >= page->valid_rows.size() || !page->valid_rows[offset] ||
        page->record.physical_slot == UINT32_MAX) return false;
    const uint64_t physical = uint64_t(page->record.physical_slot) * snapshot_.geometry.page_tokens + offset;
    if (physical > UINT32_MAX) return false;
    row = uint32_t(physical);
    return true;
}

llama_kv_pager_write_status llama_kv_pager::mutate(
        const llama_kv_pager_mutation & mutation) noexcept {
    if (!snapshot_.initialized || snapshot_.physical_page_count == 0) {
        return llama_kv_pager_write_status::disabled;
    }
    try {
        auto next = pages_;
        auto next_slots = slot_pages_;
        uint32_t next_current = current_page_index_;

        const auto selected = [&](const page_state & page, int32_t sequence) {
            return page.present && page.record.id.sequence_id == sequence &&
                (mutation.sequence_generation == 0 ||
                 page.record.id.sequence_generation == mutation.sequence_generation);
        };
        const auto overlaps = [&](const page_state & page) {
            const llama_pos begin = mutation.position_begin;
            const llama_pos end = mutation.position_end > begin
                ? mutation.position_end : std::numeric_limits<llama_pos>::max();
            return page.record.id.position_end > begin && page.record.id.position_begin < end;
        };
        const auto reject_pinned = [&](const auto & predicate) {
            for (const auto & page : pages_) {
                if (page.present && page.record.pin_count != 0 && predicate(page)) {
                    return true;
                }
            }
            return false;
        };
        if (mutation.kind == llama_kv_pager_mutation_kind::clear) {
            if (reject_pinned([](const page_state &) { return true; })) {
                return llama_kv_pager_write_status::all_pinned;
            }
            for (auto & page : next) page = {};
            std::fill(next_slots.begin(), next_slots.end(), -1);
            next_current = UINT32_MAX;
        } else if (mutation.kind == llama_kv_pager_mutation_kind::keep) {
            if (reject_pinned([&](const page_state & page) {
                    return page.record.id.sequence_id != mutation.sequence_id;
                })) {
                return llama_kv_pager_write_status::all_pinned;
            }
            for (auto & page : next) {
                if (page.present && page.record.id.sequence_id != mutation.sequence_id) {
                    next_slots[page.record.physical_slot] = -1;
                    page = {};
                }
            }
            if (next_current < next.size() && !next[next_current].present) next_current = UINT32_MAX;
        } else if (mutation.kind == llama_kv_pager_mutation_kind::copy) {
            if (mutation.destination_sequence_id < 0 || mutation.sequence_id < 0) {
                return llama_kv_pager_write_status::invalid_position;
            }
            if (reject_pinned([&](const page_state & page) {
                    return selected(page, mutation.sequence_id) && overlaps(page);
                })) {
                return llama_kv_pager_write_status::all_pinned;
            }
            for (uint32_t logical = 0; logical < next.size(); ++logical) {
                page_state & source = next[logical];
                if (!selected(source, mutation.sequence_id) ||
                    source.record.id.position_end <= mutation.position_begin ||
                    source.record.id.position_begin >= mutation.position_end) continue;
                if (std::any_of(next.begin(), next.end(), [&](const auto & destination) {
                    return destination.present && destination.record.id.sequence_id == mutation.destination_sequence_id &&
                        destination.record.id.logical_page == logical;
                })) {
                    return llama_kv_pager_write_status::no_victim;
                }
                uint32_t slot = UINT32_MAX;
                for (uint32_t i = 0; i < next_slots.size(); ++i) {
                    if (next_slots[i] < 0) { slot = i; break; }
                }
                if (slot == UINT32_MAX) return llama_kv_pager_write_status::no_victim;
                // One compact page index is sufficient for the supported single-sequence target;
                // a destination already occupying this logical page is a co-tenancy refusal.
                page_state copy = source;
                copy.record.physical_slot = slot;
                copy.record.id.sequence_id = mutation.destination_sequence_id;
                copy.record.id.sequence_generation = mutation.sequence_generation != 0
                    ? mutation.sequence_generation : source.record.id.sequence_generation + 1;
                copy.record.id.page_generation = uint32_t(++mutation_generation_);
                copy.record.pin_count = 0;
                copy.record.host_valid = false;
                copy.record.dirty = true;
                size_t copy_index = 0;
                while (copy_index < next.size() && next[copy_index].present) ++copy_index;
                if (copy_index == next.size()) next.push_back({});
                next[copy_index] = std::move(copy);
                next_slots[slot] = int32_t(copy_index);
            }
        } else {
            const llama_pos begin = mutation.position_begin;
            const llama_pos end = mutation.position_end > begin
                ? mutation.position_end : std::numeric_limits<llama_pos>::max();
            if (mutation.sequence_id < 0) return llama_kv_pager_write_status::invalid_position;
            if (reject_pinned([&](const page_state & page) {
                    return selected(page, mutation.sequence_id) && overlaps(page);
                })) {
                return llama_kv_pager_write_status::all_pinned;
            }
            for (auto & page : next) {
                if (!selected(page, mutation.sequence_id)) continue;
                const bool touches = mutation.kind == llama_kv_pager_mutation_kind::shift
                    ? page.record.id.position_end > begin && page.record.id.position_begin < end
                    : page.record.id.position_end > begin && page.record.id.position_begin < end;
                if (!touches) continue;
                if (mutation.kind == llama_kv_pager_mutation_kind::shift) {
                    if (mutation.shift % llama_pos(snapshot_.geometry.page_tokens) != 0) {
                        return llama_kv_pager_write_status::invalid_position;
                    }
                    const int64_t new_begin = int64_t(page.record.id.position_begin) + mutation.shift;
                    const int64_t new_end = int64_t(page.record.id.position_end) + mutation.shift;
                    if (new_begin < 0 || new_end > int64_t(snapshot_.geometry.context_tokens) ||
                        new_begin % snapshot_.geometry.page_tokens != 0) {
                        return llama_kv_pager_write_status::invalid_position;
                    }
                    const uint32_t new_logical = uint32_t(new_begin / snapshot_.geometry.page_tokens);
                    if (new_logical >= snapshot_.logical_page_count) {
                        return llama_kv_pager_write_status::invalid_position;
                    }
                    page.record.id.logical_page = new_logical;
                    page.record.id.position_begin = llama_pos(new_begin);
                    page.record.id.position_end = llama_pos(new_end);
                } else {
                    for (uint32_t row = 0; row < page.valid_rows.size(); ++row) {
                        const llama_pos pos = page.record.id.position_begin + llama_pos(row);
                        if (pos >= begin && pos < end) page.valid_rows[row] = 0;
                    }
                    uint32_t end_row = 0;
                    for (uint32_t row = 0; row < page.valid_rows.size(); ++row) {
                        if (page.valid_rows[row]) end_row = row + 1;
                    }
                    if (end_row == 0) {
                        next_slots[page.record.physical_slot] = -1;
                        page = {};
                    } else {
                        page.record.id.position_end = page.record.id.position_begin + llama_pos(end_row);
                        page.record.state = end_row == page.valid_rows.size()
                            ? llama_kv_page_state::gpu_dirty : llama_kv_page_state::filling_gpu;
                    }
                }
            }
            if (mutation.kind == llama_kv_pager_mutation_kind::shift) {
                std::fill(next_slots.begin(), next_slots.end(), -1);
                for (uint32_t i = 0; i < next.size(); ++i) {
                    if (!next[i].present) continue;
                    for (uint32_t j = i + 1; j < next.size(); ++j) {
                        if (next[j].present && next[j].record.id.logical_page == next[i].record.id.logical_page) {
                            return llama_kv_pager_write_status::no_victim;
                        }
                    }
                    if (next[i].record.id.logical_page >= snapshot_.logical_page_count) return llama_kv_pager_write_status::no_victim;
                    next_slots[next[i].record.physical_slot] = int32_t(i);
                }
            }
        }

        auto tx = residency_.begin();
        const auto old_pages = tx.pages();
        // Retain records for pages unaffected by this mutation. In particular, the
        // current partial page may remain pinned while another sequence/page is
        // removed or copied. Only changed or removed records are erased.
        for (const auto & page : old_pages) {
            const bool retained = std::any_of(next.begin(), next.end(), [&](const auto & candidate) {
                return candidate.present && candidate.record.id == page.id;
            });
            if (!retained && residency_.erase(tx, page.id) != llama_kv_residency_status::ok) {
                residency_.rollback(tx);
                return llama_kv_pager_write_status::transaction;
            }
        }
        for (const auto & page : next) {
            if (!page.present) continue;
            const bool existed = std::any_of(old_pages.begin(), old_pages.end(), [&](const auto & old) {
                return old.id == page.record.id;
            });
            const auto result = existed
                ? residency_.update(tx, page.record)
                : residency_.replace(tx, page.record);
            if (result != llama_kv_residency_status::ok) {
                residency_.rollback(tx);
                return llama_kv_pager_write_status::transaction;
            }
        }
        if (residency_.publish(tx) != llama_kv_residency_status::ok) {
            residency_.rollback(tx);
            return llama_kv_pager_write_status::transaction;
        }
        pages_ = std::move(next);
        slot_pages_ = std::move(next_slots);
        current_page_index_ = next_current;
        ++mutation_generation_;
        return llama_kv_pager_write_status::ok;
    } catch (...) {
        return llama_kv_pager_write_status::overflow;
    }
}
