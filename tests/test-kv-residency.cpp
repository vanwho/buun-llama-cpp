#include "../src/llama-kv-residency.h"
#include "../src/llama-kv-residency-transfer.h"

#include <cassert>
#include <cstring>
#include <vector>

static llama_kv_page_id page(uint32_t index, llama_pos end = -1) {
    llama_kv_page_id id;
    id.session_generation = 7;
    id.sequence_id = 0;
    id.sequence_generation = 3;
    id.logical_page = index;
    id.page_generation = 1;
    id.representation_epoch = 2;
    id.position_begin = llama_pos(index * VBR_GENERATION_PAGE_CELLS);
    id.position_end = end < 0 ? id.position_begin + VBR_GENERATION_PAGE_CELLS : end;
    return id;
}

static void test_page_geometry() {
    assert(llama_kv_page_count(0) == 0);
    assert(llama_kv_page_count(1) == 1);
    assert(llama_kv_page_count(255) == 1);
    assert(llama_kv_page_count(256) == 1);
    assert(llama_kv_page_count(257) == 2);
    assert(llama_kv_page_count(77824) == 304);
    assert(llama_kv_page_count(262144) == 1024);
    assert(llama_kv_page_id_valid(page(0), false));
    assert(llama_kv_page_id_valid(page(1, 511), true));
    assert(!llama_kv_page_id_valid(page(1, 514), false));
    assert(!llama_kv_page_id_valid(page(1, 512), true));
}

static llama_kv_page_record resident(uint32_t logical, uint32_t slot) {
    llama_kv_page_record result;
    result.id = page(logical);
    result.physical_slot = slot;
    result.state = llama_kv_page_state::gpu_host_clean;
    result.host_valid = true;
    return result;
}

static void test_table_identity_and_snapshot() {
    llama_kv_residency_table table(304);
    auto tx = table.begin();
    assert(table.replace(tx, resident(3, 19)) == llama_kv_residency_status::ok);
    assert(table.replace(tx, resident(5, 2)) == llama_kv_residency_status::ok);
    assert(table.publish(tx) == llama_kv_residency_status::ok);
    const auto old = table.snapshot();
    assert(old.epoch() == 1 && old.pages().size() == 2);

    auto next = table.begin();
    assert(table.erase(next, page(3)) == llama_kv_residency_status::ok);
    assert(table.publish(next) == llama_kv_residency_status::ok);
    assert(old.pages().size() == 2);
    assert(table.snapshot().pages().size() == 1);
}

static void test_rejection_and_stale_publication() {
    llama_kv_residency_table table(2);
    auto first = table.begin();
    auto stale = table.begin();
    assert(table.replace(first, resident(0, 0)) == llama_kv_residency_status::ok);
    assert(table.publish(first) == llama_kv_residency_status::ok);
    assert(table.publish(stale) == llama_kv_residency_status::stale_epoch);

    auto tx = table.begin();
    assert(table.replace(tx, resident(0, 0)) == llama_kv_residency_status::duplicate_logical_page);
    assert(table.replace(tx, resident(0, 1)) == llama_kv_residency_status::duplicate_logical_page);
    auto pinned = resident(1, 0);
    pinned.pin_count = 1;
    table.rollback(tx);
    tx = table.begin();
    assert(table.erase(tx, page(0)) == llama_kv_residency_status::ok);
    assert(table.replace(tx, pinned) == llama_kv_residency_status::ok);
    assert(table.replace(tx, resident(2, 0)) == llama_kv_residency_status::pinned_slot);
    table.rollback(tx);
}

struct fake_transfer_backend {
    struct pending_copy {
        uint64_t ticket = 0;
        llama_kv_residency_transfer_direction direction =
            llama_kv_residency_transfer_direction::h2d_promotion;
        uint32_t slot = 0;
        uint64_t offset = 0;
        void * host = nullptr;
        size_t size = 0;
    };

    std::vector<std::vector<uint8_t>> slots;
    std::vector<bool> mapped;
    std::vector<pending_copy> pending;
    std::vector<uint8_t> host_bytes;
    uint64_t catalog_reserved = 0;
    uint32_t issue_calls = 0;
    uint32_t map_calls = 0;
    uint32_t drop_calls = 0;
    bool fail_issue = false;
    bool fail_complete = false;
    bool fail_catalog = false;
    bool fail_map = false;
    bool stale = false;

    static bool reserve_slots(void * opaque, uint32_t count, uint64_t bytes) noexcept {
        auto & self = *static_cast<fake_transfer_backend *>(opaque);
        try {
            self.slots.assign(count, std::vector<uint8_t>(size_t(bytes), 0));
            self.mapped.assign(count, false);
            return true;
        } catch (...) {
            return false;
        }
    }

    static void release_slots(void *, uint32_t, uint64_t) noexcept {}

    static bool map_slot(void * opaque, uint32_t slot) noexcept {
        auto & self = *static_cast<fake_transfer_backend *>(opaque);
        ++self.map_calls;
        if (self.fail_map || slot >= self.mapped.size() || self.mapped[slot]) {
            return false;
        }
        self.mapped[slot] = true;
        return true;
    }

    static bool drop_slot(void * opaque, uint32_t slot) noexcept {
        auto & self = *static_cast<fake_transfer_backend *>(opaque);
        ++self.drop_calls;
        if (slot >= self.mapped.size() || !self.mapped[slot]) return false;
        self.mapped[slot] = false;
        return true;
    }

    static bool issue(
            void * opaque, llama_kv_residency_transfer_direction direction,
            const llama_kv_residency_completion & completion,
            uint64_t offset, void * host, size_t size, uint64_t ticket,
            bool asynchronous) noexcept {
        auto & self = *static_cast<fake_transfer_backend *>(opaque);
        ++self.issue_calls;
        if (self.fail_issue || completion.physical_slot >= self.slots.size() ||
            offset > self.slots[completion.physical_slot].size() ||
            size > self.slots[completion.physical_slot].size() - offset) {
            return false;
        }
        pending_copy copy { ticket, direction, completion.physical_slot,
                            offset, host, size };
        if (asynchronous) {
            try {
                self.pending.push_back(copy);
                return true;
            } catch (...) {
                return false;
            }
        }
        return finish(self, copy);
    }

    static bool finish(
            fake_transfer_backend & self, const pending_copy & copy) noexcept {
        auto & bytes = self.slots[copy.slot];
        if (copy.direction == llama_kv_residency_transfer_direction::d2h_seal ||
            copy.direction == llama_kv_residency_transfer_direction::d2h_reseal) {
            std::memcpy(copy.host, bytes.data() + copy.offset, copy.size);
        } else {
            std::memcpy(bytes.data() + copy.offset, copy.host, copy.size);
        }
        return true;
    }

    static bool complete(void * opaque, uint64_t ticket) noexcept {
        auto & self = *static_cast<fake_transfer_backend *>(opaque);
        for (size_t i = 0; i < self.pending.size(); ++i) {
            if (self.pending[i].ticket == ticket) {
                const auto copy = self.pending[i];
                self.pending.erase(self.pending.begin() + i);
                return !self.fail_complete && finish(self, copy);
            }
        }
        return false;
    }

    static void cancel(void * opaque, uint64_t ticket) noexcept {
        auto & self = *static_cast<fake_transfer_backend *>(opaque);
        for (size_t i = 0; i < self.pending.size(); ++i) {
            if (self.pending[i].ticket == ticket) {
                self.pending.erase(self.pending.begin() + i);
                return;
            }
        }
    }

    static bool reserve_catalog(void * opaque, uint64_t bytes) noexcept {
        auto & self = *static_cast<fake_transfer_backend *>(opaque);
        if (self.fail_catalog) return false;
        self.catalog_reserved += bytes;
        return true;
    }

    static void release_catalog(void * opaque, uint64_t bytes) noexcept {
        auto & self = *static_cast<fake_transfer_backend *>(opaque);
        assert(self.catalog_reserved >= bytes);
        self.catalog_reserved -= bytes;
    }

    static bool host_read(
            void *, uint32_t, uint64_t offset, uint8_t * destination,
            size_t size) noexcept {
        for (size_t i = 0; i < size; ++i) destination[i] = uint8_t(offset + i + 1);
        return true;
    }

    static bool host_write(
            void * opaque, uint32_t, uint64_t offset,
            const uint8_t * source, size_t size) noexcept {
        auto & self = *static_cast<fake_transfer_backend *>(opaque);
        try {
            if (offset > SIZE_MAX - size) return false;
            if (self.host_bytes.size() < offset + size) {
                self.host_bytes.resize(offset + size);
            }
            std::memcpy(self.host_bytes.data() + offset, source, size);
            return true;
        } catch (...) {
            return false;
        }
    }

    static bool recheck(
            void * opaque, const llama_kv_residency_completion &) noexcept {
        return !static_cast<fake_transfer_backend *>(opaque)->stale;
    }
};

static llama_kv_residency_transfer_page transfer_page(
        uint32_t slot, uint64_t epoch = 11) {
    llama_kv_residency_transfer_page result;
    result.page = page(0);
    result.table_epoch = epoch;
    result.physical_slot = slot;
    result.runs.push_back({
        UINT32_MAX, 0, 0, 0, 0, 8, 1, 0, 0,
    });
    return result;
}

static void test_batched_transfer_pool() {
    llama_kv_residency_transfer_plan upload;
    assert(llama_kv_residency_build_transfer_plan(
        llama_kv_residency_transfer_direction::h2d_promotion,
        { transfer_page(0) }, 4, {}, upload));
    assert(upload.useful_bytes == 8 && upload.aligned_bytes == 8);
    assert(upload.runs.size() == 1 && upload.event_count == 1);

    fake_transfer_backend fake;
    llama_kv_residency_pool_backend backend {
        &fake, fake_transfer_backend::reserve_slots,
        fake_transfer_backend::release_slots, fake_transfer_backend::map_slot,
        fake_transfer_backend::drop_slot, fake_transfer_backend::issue,
        fake_transfer_backend::complete, fake_transfer_backend::cancel,
    };
    llama_kv_residency_pool_status pool_status;
    auto pool = llama_kv_residency_pool::create(
        { 2, 64, 4, 4, 1024 }, backend, pool_status);
    assert(pool && pool_status == llama_kv_residency_pool_status::ok);

    llama_kv_residency_transfer_claim claim;
    assert(pool->reserve(upload, 32, {}, claim) ==
        llama_kv_residency_pool_status::ok);
    llama_kv_residency_transfer_transport transport;
    vbr_h2d_status h2d_status;
    auto upload_ring = vbr_h2d_chunk_ring::create(
        { {} }, 128, 32, h2d_status);
    assert(upload_ring && h2d_status == vbr_h2d_status::ok);
    transport.upload_ring = upload_ring.get();
    transport.context = &fake;
    transport.host_read = fake_transfer_backend::host_read;
    transport.recheck = fake_transfer_backend::recheck;
    auto result = llama_kv_residency_execute_transfer(
        *pool, upload, claim, backend, transport);
    assert(result.status == llama_kv_residency_pool_status::ok);
    assert(result.counters.copied_useful_bytes == 8);
    assert(result.counters.event_completions == 1);
    assert(pool->mapped_slots() == 1 && pool->resident_bytes() == 64);
    assert(fake.pending.empty());

    llama_kv_residency_transfer_plan download;
    assert(llama_kv_residency_build_transfer_plan(
        llama_kv_residency_transfer_direction::d2h_seal,
        { transfer_page(0, 12) }, 4, {}, download));
    llama_kv_residency_transfer_claim download_claim;
    llama_kv_residency_catalog_reservation catalog {
        &fake, fake_transfer_backend::reserve_catalog,
        fake_transfer_backend::release_catalog,
    };
    assert(pool->reserve(download, 32, catalog, download_claim) ==
        llama_kv_residency_pool_status::ok);
    vbr_capture_stream_status d2h_status;
    auto download_ring = vbr_pinned_chunk_ring::create(
        { {} }, 128, 32, d2h_status);
    assert(download_ring && d2h_status == vbr_capture_stream_status::ok);
    transport.download_ring = download_ring.get();
    transport.host_write = fake_transfer_backend::host_write;
    result = llama_kv_residency_execute_transfer(
        *pool, download, download_claim, backend, transport);
    assert(result.status == llama_kv_residency_pool_status::ok);
    assert(result.counters.event_completions == 1);
    assert(fake.catalog_reserved == 8);
    assert(fake.host_bytes == std::vector<uint8_t>({ 1, 2, 3, 4, 5, 6, 7, 8 }));

    bool dropped = false;
    assert(pool->mark_dirty(page(0), true) ==
        llama_kv_residency_pool_status::ok);
    assert(pool->drop_logical_page(page(0), dropped) ==
        llama_kv_residency_pool_status::dirty_page && !dropped);
    assert(pool->mark_dirty(page(0), false) ==
        llama_kv_residency_pool_status::ok);
    const uint32_t issues_before_drop = fake.issue_calls;
    const uint32_t drops_before = fake.drop_calls;
    assert(pool->drop_logical_page(page(0), dropped) ==
        llama_kv_residency_pool_status::ok && dropped);
    assert(fake.drop_calls == drops_before + 1);
    assert(fake.issue_calls == issues_before_drop);
    assert(pool->mapped_slots() == 0);
}

static void test_transfer_rollback_and_stale_completion() {
    fake_transfer_backend fake;
    llama_kv_residency_pool_backend backend {
        &fake, fake_transfer_backend::reserve_slots,
        fake_transfer_backend::release_slots, fake_transfer_backend::map_slot,
        fake_transfer_backend::drop_slot, fake_transfer_backend::issue,
        fake_transfer_backend::complete, fake_transfer_backend::cancel,
    };
    llama_kv_residency_pool_status status;
    auto pool = llama_kv_residency_pool::create(
        { 2, 64, 4, 4, 1024 }, backend, status);
    assert(pool);
    vbr_h2d_status ring_status;
    auto ring = vbr_h2d_chunk_ring::create({ {} }, 128, 32, ring_status);
    assert(ring);
    llama_kv_residency_transfer_transport transport;
    transport.upload_ring = ring.get();
    transport.context = &fake;
    transport.host_read = fake_transfer_backend::host_read;
    transport.recheck = fake_transfer_backend::recheck;

    auto plan = transfer_page(1);
    llama_kv_residency_transfer_plan upload;
    assert(llama_kv_residency_build_transfer_plan(
        llama_kv_residency_transfer_direction::h2d_promotion,
        { plan }, 4, {}, upload));
    fake.fail_issue = true;
    llama_kv_residency_transfer_claim claim;
    assert(pool->reserve(upload, 32, {}, claim) ==
        llama_kv_residency_pool_status::ok);
    auto result = llama_kv_residency_execute_transfer(
        *pool, upload, claim, backend, transport);
    assert(result.status == llama_kv_residency_pool_status::transfer_failed);
    assert(pool->mapped_slots() == 0 && fake.pending.empty());

    auto d2h_page = transfer_page(0);
    llama_kv_residency_transfer_plan download;
    assert(llama_kv_residency_build_transfer_plan(
        llama_kv_residency_transfer_direction::d2h_reseal,
        { d2h_page }, 4, {}, download));
    llama_kv_residency_catalog_reservation catalog {
        &fake, fake_transfer_backend::reserve_catalog,
        fake_transfer_backend::release_catalog,
    };
    fake.fail_catalog = true;
    assert(pool->reserve(download, 32, catalog, claim) ==
        llama_kv_residency_pool_status::catalog_unavailable);
    assert(!claim.active() && pool->pending_events() == 0 &&
        fake.catalog_reserved == 0);
    fake.fail_catalog = false;

    fake.fail_map = true;
    assert(pool->reserve(upload, 32, {}, claim) ==
        llama_kv_residency_pool_status::ok);
    result = llama_kv_residency_execute_transfer(
        *pool, upload, claim, backend, transport);
    assert(result.status == llama_kv_residency_pool_status::backend_unavailable);
    assert(pool->mapped_slots() == 0 && pool->pending_events() == 0);
    fake.fail_map = false;

    fake.fail_complete = true;
    assert(pool->reserve(upload, 32, {}, claim) ==
        llama_kv_residency_pool_status::ok);
    result = llama_kv_residency_execute_transfer(
        *pool, upload, claim, backend, transport);
    assert(result.status == llama_kv_residency_pool_status::transfer_failed);
    assert(pool->mapped_slots() == 0 && pool->pending_events() == 0 &&
        fake.pending.empty());
    fake.fail_complete = false;

    fake.fail_issue = false;
    fake.stale = true;
    assert(pool->reserve(upload, 32, {}, claim) ==
        llama_kv_residency_pool_status::ok);
    result = llama_kv_residency_execute_transfer(
        *pool, upload, claim, backend, transport);
    assert(result.status == llama_kv_residency_pool_status::stale_completion);
    assert(result.counters.stale_completions == 1);
    assert(pool->mapped_slots() == 0 && fake.pending.empty());
}

int main() {
    test_page_geometry();
    test_table_identity_and_snapshot();
    test_rejection_and_stale_publication();
    test_batched_transfer_pool();
    test_transfer_rollback_and_stale_completion();
    return 0;
}
