#include "llama-kv-residency.h"

#include <algorithm>
#include <limits>

struct llama_kv_residency_snapshot::state {
    uint64_t epoch = 0;
    uint32_t slot_capacity = 0;
    std::vector<llama_kv_page_record> pages;
};

bool operator==(const llama_kv_page_id & a, const llama_kv_page_id & b) noexcept {
    return a.session_generation == b.session_generation && a.sequence_id == b.sequence_id &&
        a.sequence_generation == b.sequence_generation && a.logical_page == b.logical_page &&
        a.page_generation == b.page_generation && a.representation_epoch == b.representation_epoch &&
        a.model_identity == b.model_identity && a.topology_identity == b.topology_identity &&
        a.codec_digest == b.codec_digest && a.codebook_digest == b.codebook_digest &&
        a.rotation_digest == b.rotation_digest && a.meansub_digest == b.meansub_digest &&
        a.position_begin == b.position_begin && a.position_end == b.position_end;
}

bool operator!=(const llama_kv_page_id & a, const llama_kv_page_id & b) noexcept { return !(a == b); }

bool llama_kv_page_id_valid(const llama_kv_page_id & id, bool tail) noexcept {
    constexpr uint32_t page_size = VBR_GENERATION_PAGE_CELLS;
    if (id.sequence_id < 0 || id.position_begin < 0 || id.position_end <= id.position_begin ||
        id.position_begin % page_size != 0) {
        return false;
    }
    const uint64_t length = uint64_t(id.position_end) - uint64_t(id.position_begin);
    if (length > page_size || (!tail && length != page_size) || (tail && length == page_size)) {
        return false;
    }
    return id.logical_page == uint32_t(id.position_begin / page_size);
}

uint32_t llama_kv_page_count(uint32_t cells) noexcept {
    constexpr uint32_t size = VBR_GENERATION_PAGE_CELLS;
    return cells == 0 ? 0 : (cells - 1) / size + 1;
}

llama_kv_residency_snapshot::llama_kv_residency_snapshot(std::shared_ptr<const state> value) noexcept : state_(std::move(value)) {}
uint64_t llama_kv_residency_snapshot::epoch() const noexcept { return state_ ? state_->epoch : 0; }
uint32_t llama_kv_residency_snapshot::slot_capacity() const noexcept { return state_ ? state_->slot_capacity : 0; }
const std::vector<llama_kv_page_record> & llama_kv_residency_snapshot::pages() const noexcept {
    static const std::vector<llama_kv_page_record> empty;
    return state_ ? state_->pages : empty;
}

uint64_t llama_kv_residency_transaction::base_epoch() const noexcept { return base_epoch_; }
bool llama_kv_residency_transaction::active() const noexcept { return active_; }
const std::vector<llama_kv_page_record> & llama_kv_residency_transaction::pages() const noexcept { return pages_; }

llama_kv_residency_table::llama_kv_residency_table(uint32_t slots) : state_(std::make_shared<llama_kv_residency_snapshot::state>()) {
    auto value = std::make_shared<llama_kv_residency_snapshot::state>();
    value->slot_capacity = slots;
    state_ = std::move(value);
}

llama_kv_residency_snapshot llama_kv_residency_table::snapshot() const noexcept {
    return llama_kv_residency_snapshot(state_);
}

llama_kv_residency_transaction llama_kv_residency_table::begin() const noexcept {
    llama_kv_residency_transaction tx;
    tx.base_epoch_ = state_->epoch;
    tx.slot_capacity_ = state_->slot_capacity;
    tx.pages_ = state_->pages;
    tx.active_ = true;
    return tx;
}

llama_kv_residency_status llama_kv_residency_table::replace(
        llama_kv_residency_transaction & tx, const llama_kv_page_record & page) const noexcept {
    if (!tx.active_) return llama_kv_residency_status::transaction_closed;
    if (!llama_kv_page_id_valid(page.id, page.state == llama_kv_page_state::filling_gpu))
        return llama_kv_residency_status::invalid_position_range;
    if (page.physical_slot >= tx.slot_capacity_) return llama_kv_residency_status::out_of_range;
    for (const auto & existing : tx.pages_) {
        if (existing.id == page.id) return llama_kv_residency_status::duplicate_logical_page;
        if (existing.id.session_generation == page.id.session_generation &&
            existing.id.sequence_id == page.id.sequence_id &&
            existing.id.sequence_generation == page.id.sequence_generation &&
            existing.id.logical_page == page.id.logical_page)
            return llama_kv_residency_status::duplicate_logical_page;
        if (existing.physical_slot == page.physical_slot) {
            if (existing.pin_count != 0) return llama_kv_residency_status::pinned_slot;
            return llama_kv_residency_status::duplicate_physical_slot;
        }
    }
    tx.pages_.push_back(page);
    return llama_kv_residency_status::ok;
}

llama_kv_residency_status llama_kv_residency_table::erase(
        llama_kv_residency_transaction & tx, const llama_kv_page_id & id) const noexcept {
    if (!tx.active_) return llama_kv_residency_status::transaction_closed;
    const auto it = std::find_if(tx.pages_.begin(), tx.pages_.end(), [&](const auto & p) { return p.id == id; });
    if (it == tx.pages_.end()) return llama_kv_residency_status::not_found;
    if (it->pin_count != 0) return llama_kv_residency_status::pinned_slot;
    tx.pages_.erase(it);
    return llama_kv_residency_status::ok;
}

llama_kv_residency_status llama_kv_residency_table::update(
        llama_kv_residency_transaction & tx, const llama_kv_page_record & page) const noexcept {
    if (!tx.active_) return llama_kv_residency_status::transaction_closed;
    if (!llama_kv_page_id_valid(page.id, page.state == llama_kv_page_state::filling_gpu)) {
        return llama_kv_residency_status::invalid_position_range;
    }
    auto it = std::find_if(tx.pages_.begin(), tx.pages_.end(), [&](const auto & existing) {
        return existing.id.session_generation == page.id.session_generation &&
            existing.id.sequence_id == page.id.sequence_id &&
            existing.id.sequence_generation == page.id.sequence_generation &&
            existing.id.logical_page == page.id.logical_page;
    });
    if (it == tx.pages_.end()) return llama_kv_residency_status::not_found;
    if (it->physical_slot != page.physical_slot) {
        return llama_kv_residency_status::duplicate_physical_slot;
    }
    *it = page;
    return llama_kv_residency_status::ok;
}

llama_kv_residency_status llama_kv_residency_table::publish(llama_kv_residency_transaction & tx) noexcept {
    if (!tx.active_) return llama_kv_residency_status::transaction_closed;
    if (tx.base_epoch_ != state_->epoch) return llama_kv_residency_status::stale_epoch;
    auto next = std::make_shared<llama_kv_residency_snapshot::state>();
    next->epoch = state_->epoch + 1;
    next->slot_capacity = state_->slot_capacity;
    next->pages = std::move(tx.pages_);
    state_ = std::move(next);
    tx.active_ = false;
    return llama_kv_residency_status::ok;
}

void llama_kv_residency_table::rollback(llama_kv_residency_transaction & tx) const noexcept {
    tx.pages_.clear();
    tx.active_ = false;
}
