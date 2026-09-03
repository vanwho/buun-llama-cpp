#include "../src/llama-kv-residency.h"
#include "../src/llama-kv-residency-transfer.h"
#include "../src/llama-kv-residency-transaction.h"
#include "../src/llama-kv-fixed-window.h"

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

struct fake_residency_transaction {
    llama_kv_residency_pool * pool = nullptr;
    llama_kv_residency_transaction_phase fail_phase =
        llama_kv_residency_transaction_phase::_count;
    uint32_t pins = 0;
    uint32_t unpins = 0;
    uint32_t drops = 0;
    uint32_t restores = 0;
    uint32_t retires = 0;
    bool stale = false;

    static bool phase(
            void * opaque,
            llama_kv_residency_transaction_phase value) noexcept {
        return static_cast<fake_residency_transaction *>(opaque)->fail_phase != value;
    }

    static bool pin(void * opaque, const llama_kv_page_id &) noexcept {
        ++static_cast<fake_residency_transaction *>(opaque)->pins;
        return true;
    }

    static void unpin(void * opaque, const llama_kv_page_id &) noexcept {
        ++static_cast<fake_residency_transaction *>(opaque)->unpins;
    }

    static bool drop_clean(
            void * opaque, const llama_kv_page_record & page) noexcept {
        auto & self = *static_cast<fake_residency_transaction *>(opaque);
        if (self.pool) {
            bool dropped = false;
            if (self.pool->drop_logical_page(page.id, dropped) !=
                    llama_kv_residency_pool_status::ok || !dropped) {
                return false;
            }
        }
        ++self.drops;
        return true;
    }

    static bool restore_clean(
            void * opaque, const llama_kv_page_record &) noexcept {
        ++static_cast<fake_residency_transaction *>(opaque)->restores;
        return true;
    }

    static void retire(void * opaque, const llama_kv_page_record &) noexcept {
        ++static_cast<fake_residency_transaction *>(opaque)->retires;
    }

    static bool has_clean_host(
            void *, const llama_kv_page_id &, uint64_t bytes) noexcept {
        return bytes != 0;
    }

    static bool recheck(
            void * opaque, uint64_t base_epoch,
            const std::vector<llama_kv_page_record> & desired) noexcept {
        const auto & self = *static_cast<fake_residency_transaction *>(opaque);
        return !self.stale && base_epoch != 0 && desired.size() <= 1;
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

static llama_kv_residency_transfer_page transfer_page_for(
        uint32_t logical, uint32_t slot, uint64_t epoch = 11) {
    auto result = transfer_page(slot, epoch);
    result.page = page(logical);
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

static void test_ggml_adapter_tensor_route() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, 128);
    assert(buffer != nullptr);

    llama_kv_residency_ggml_status adapter_status;
    auto adapter = llama_kv_residency_ggml_adapter::create(
            { backend, buffer, 2, 64, false }, adapter_status);
    assert(adapter && adapter_status == llama_kv_residency_ggml_status::ok);
    auto pool_backend = adapter->pool_backend();
    llama_kv_residency_pool_status pool_status;
    auto pool = llama_kv_residency_pool::create(
            { 2, 64, 4, 4, 1024 }, pool_backend, pool_status);
    assert(pool && pool_status == llama_kv_residency_pool_status::ok);
    fake_transfer_backend fake;

    vbr_h2d_status ring_status;
    auto upload_ring = vbr_h2d_chunk_ring::create(
            { {} },
            128, 32, ring_status);
    assert(upload_ring && ring_status == vbr_h2d_status::ok);
    llama_kv_residency_transfer_plan upload;
    assert(llama_kv_residency_build_transfer_plan(
            llama_kv_residency_transfer_direction::h2d_promotion,
            { transfer_page(0) }, 4, {}, upload));
    llama_kv_residency_transfer_claim claim;
    assert(pool->reserve(upload, 32, {}, claim) ==
            llama_kv_residency_pool_status::ok);
    llama_kv_residency_transfer_transport transport;
    transport.upload_ring = upload_ring.get();
    transport.context = &fake;
    transport.host_read = fake_transfer_backend::host_read;
    transport.recheck = fake_transfer_backend::recheck;
    assert(llama_kv_residency_execute_transfer(
            *pool, upload, claim, pool_backend, transport).status ==
            llama_kv_residency_pool_status::ok);
    assert(adapter->mapped_slots() == 1 && adapter->pending_events() == 0);

    const auto * base = static_cast<const uint8_t *>(
            ggml_backend_buffer_get_base(buffer));
    for (size_t i = 0; i < 8; ++i) {
        assert(base[i] == uint8_t(i + 1));
    }

    vbr_capture_stream_status download_status;
    auto download_ring = vbr_pinned_chunk_ring::create(
            { {} },
            128, 32, download_status);
    assert(download_ring && download_status == vbr_capture_stream_status::ok);
    llama_kv_residency_transfer_plan download;
    assert(llama_kv_residency_build_transfer_plan(
            llama_kv_residency_transfer_direction::d2h_seal,
            { transfer_page(0) }, 4, {}, download));
    llama_kv_residency_transfer_claim download_claim;
    llama_kv_residency_catalog_reservation catalog {
        &fake, fake_transfer_backend::reserve_catalog,
        fake_transfer_backend::release_catalog,
    };
    assert(pool->reserve(download, 32, catalog, download_claim) ==
            llama_kv_residency_pool_status::ok);
    transport.download_ring = download_ring.get();
    transport.host_write = fake_transfer_backend::host_write;
    transport.context = &fake;
    assert(llama_kv_residency_execute_transfer(
            *pool, download, download_claim, pool_backend, transport).status ==
            llama_kv_residency_pool_status::ok);
    assert(fake.host_bytes ==
            std::vector<uint8_t>({ 1, 2, 3, 4, 5, 6, 7, 8 }));

    bool dropped = false;
    assert(pool->drop_logical_page(page(0), dropped) ==
            llama_kv_residency_pool_status::ok && dropped);
    assert(adapter->mapped_slots() == 0);
    pool.reset();
    adapter.reset();
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
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

static llama_kv_residency_transaction_hooks transaction_hooks(
        fake_residency_transaction & fake) {
    llama_kv_residency_transaction_hooks hooks;
    hooks.context = &fake;
    hooks.phase = fake_residency_transaction::phase;
    hooks.pin = fake_residency_transaction::pin;
    hooks.unpin = fake_residency_transaction::unpin;
    hooks.drop_clean = fake_residency_transaction::drop_clean;
    hooks.restore_clean = fake_residency_transaction::restore_clean;
    hooks.retire = fake_residency_transaction::retire;
    hooks.has_clean_host = fake_residency_transaction::has_clean_host;
    hooks.recheck = fake_residency_transaction::recheck;
    return hooks;
}

static llama_kv_residency_transaction_result run_transaction(
        llama_kv_residency_transaction_phase fail_phase,
        fake_residency_transaction & transaction_fake,
        fake_transfer_backend & transfer_fake,
        llama_kv_residency_table & table) {
    llama_kv_residency_pool_backend backend {
        &transfer_fake, fake_transfer_backend::reserve_slots,
        fake_transfer_backend::release_slots, fake_transfer_backend::map_slot,
        fake_transfer_backend::drop_slot, fake_transfer_backend::issue,
        fake_transfer_backend::complete, fake_transfer_backend::cancel,
    };
    llama_kv_residency_pool_status pool_status;
    auto pool = llama_kv_residency_pool::create(
        { 2, 64, 4, 4, 1024 }, backend, pool_status);
    assert(pool && pool_status == llama_kv_residency_pool_status::ok);

    llama_kv_residency_transfer_plan upload;
    assert(llama_kv_residency_build_transfer_plan(
        llama_kv_residency_transfer_direction::h2d_promotion,
        { transfer_page_for(1, 1, 19) }, 4, {}, upload));
    vbr_h2d_status ring_status;
    auto ring = vbr_h2d_chunk_ring::create({ {} }, 128, 32, ring_status);
    assert(ring && ring_status == vbr_h2d_status::ok);
    llama_kv_residency_transfer_transport transport;
    transport.upload_ring = ring.get();
    transport.context = &transfer_fake;
    transport.host_read = fake_transfer_backend::host_read;
    transport.recheck = fake_transfer_backend::recheck;

    llama_kv_residency_transaction_request request;
    request.desired_pages.push_back(resident(1, 1));
    request.transfers.push_back(upload);
    request.staging_capacity = 32;
    transaction_fake.fail_phase = fail_phase;
    return llama_kv_residency_execute_transaction(
        table, *pool, request, backend, transport,
        transaction_hooks(transaction_fake));
}

static void test_residency_transaction() {
    llama_kv_residency_table table(2);
    auto initial = table.begin();
    assert(table.replace(initial, resident(0, 0)) ==
        llama_kv_residency_status::ok);
    assert(table.publish(initial) == llama_kv_residency_status::ok);
    const auto before = table.snapshot();

    fake_residency_transaction transaction_fake;
    fake_transfer_backend transfer_fake;
    auto result = run_transaction(
        llama_kv_residency_transaction_phase::_count,
        transaction_fake, transfer_fake, table);
    assert(result.status == llama_kv_residency_transaction_status::committed);
    assert(result.published && result.base_epoch == 1 &&
        result.published_epoch == 2 && result.pinned_pages == 1 &&
        result.dropped_pages == 1 && result.loaded_pages == 1);
    assert(transaction_fake.pins == 1 && transaction_fake.unpins == 1 &&
        transaction_fake.drops == 1 && transaction_fake.restores == 0 &&
        transaction_fake.retires == 1);
    assert(table.snapshot().epoch() == 2 && table.snapshot().pages().size() == 1 &&
        table.snapshot().pages()[0].id == page(1));
    assert(before.epoch() == 1 && before.pages()[0].id == page(0));

    for (uint8_t raw = uint8_t(llama_kv_residency_transaction_phase::snapshot);
         raw <= uint8_t(llama_kv_residency_transaction_phase::retire); ++raw) {
        table = llama_kv_residency_table(2);
        auto tx = table.begin();
        assert(table.replace(tx, resident(0, 0)) ==
            llama_kv_residency_status::ok);
        assert(table.publish(tx) == llama_kv_residency_status::ok);
        transaction_fake = {};
        transfer_fake = {};
        result = run_transaction(
            llama_kv_residency_transaction_phase(raw),
            transaction_fake, transfer_fake, table);
        assert(!result.published && result.rollback_complete);
        assert(table.snapshot().epoch() == 1 &&
            table.snapshot().pages()[0].id == page(0));
        assert(transaction_fake.pins == transaction_fake.unpins);
        assert(transaction_fake.drops == transaction_fake.restores);
    }

    table = llama_kv_residency_table(2);
    auto dirty_tx = table.begin();
    auto dirty = resident(0, 0);
    dirty.dirty = true;
    assert(table.replace(dirty_tx, dirty) == llama_kv_residency_status::ok);
    assert(table.publish(dirty_tx) == llama_kv_residency_status::ok);
    transaction_fake = {};
    transfer_fake = {};
    result = run_transaction(
        llama_kv_residency_transaction_phase::_count,
        transaction_fake, transfer_fake, table);
    assert(result.status == llama_kv_residency_transaction_status::dirty_victim);
    assert(table.snapshot().epoch() == 1 && transfer_fake.pending.empty());

    table = llama_kv_residency_table(2);
    auto missing_tx = table.begin();
    auto missing = resident(0, 0);
    missing.host_valid = false;
    assert(table.replace(missing_tx, missing) == llama_kv_residency_status::ok);
    assert(table.publish(missing_tx) == llama_kv_residency_status::ok);
    transaction_fake = {};
    transfer_fake = {};
    result = run_transaction(
        llama_kv_residency_transaction_phase::_count,
        transaction_fake, transfer_fake, table);
    assert(result.status == llama_kv_residency_transaction_status::missing_host_source);
    assert(table.snapshot().epoch() == 1 && transfer_fake.pending.empty());

    table = llama_kv_residency_table(2);
    auto stale_table_tx = table.begin();
    assert(table.replace(stale_table_tx, resident(0, 0)) ==
        llama_kv_residency_status::ok);
    assert(table.publish(stale_table_tx) == llama_kv_residency_status::ok);
    transaction_fake = {};
    transaction_fake.stale = true;
    transfer_fake = {};
    result = run_transaction(
        llama_kv_residency_transaction_phase::_count,
        transaction_fake, transfer_fake, table);
    assert(result.status == llama_kv_residency_transaction_status::stale_epoch);
    assert(!result.published && result.rollback_complete &&
        transaction_fake.pins == transaction_fake.unpins);

    table = llama_kv_residency_table(2);
    auto pinned_tx = table.begin();
    auto pinned = resident(0, 0);
    pinned.pin_count = 1;
    assert(table.replace(pinned_tx, pinned) == llama_kv_residency_status::ok);
    assert(table.publish(pinned_tx) == llama_kv_residency_status::ok);
    transaction_fake = {};
    transfer_fake = {};
    result = run_transaction(
        llama_kv_residency_transaction_phase::_count,
        transaction_fake, transfer_fake, table);
    assert(result.status == llama_kv_residency_transaction_status::all_pinned);
    assert(table.snapshot().epoch() == 1 && transfer_fake.pending.empty());

    table = llama_kv_residency_table(2);
    auto stale_tx = table.begin();
    assert(table.replace(stale_tx, resident(0, 0)) ==
        llama_kv_residency_status::ok);
    assert(table.publish(stale_tx) == llama_kv_residency_status::ok);
    auto concurrent = table.begin();
    assert(table.replace(concurrent, resident(0, 0)) ==
        llama_kv_residency_status::duplicate_logical_page);
    auto replacement = table.begin();
    assert(table.erase(replacement, page(0)) == llama_kv_residency_status::ok);
    assert(table.replace(replacement, resident(0, 1)) ==
        llama_kv_residency_status::ok);
    assert(table.publish(replacement) == llama_kv_residency_status::ok);
    transaction_fake = {};
    transfer_fake = {};
    result = run_transaction(
        llama_kv_residency_transaction_phase::_count,
        transaction_fake, transfer_fake, table);
    assert(result.status == llama_kv_residency_transaction_status::committed);
}

static void test_reseal_before_eviction() {
    fake_transfer_backend transfer_fake;
    llama_kv_residency_pool_backend backend {
        &transfer_fake, fake_transfer_backend::reserve_slots,
        fake_transfer_backend::release_slots, fake_transfer_backend::map_slot,
        fake_transfer_backend::drop_slot, fake_transfer_backend::issue,
        fake_transfer_backend::complete, fake_transfer_backend::cancel,
    };
    llama_kv_residency_pool_status pool_status;
    auto pool = llama_kv_residency_pool::create(
        { 2, 64, 4, 4, 1024 }, backend, pool_status);
    assert(pool && pool_status == llama_kv_residency_pool_status::ok);

    llama_kv_residency_transfer_plan upload;
    assert(llama_kv_residency_build_transfer_plan(
        llama_kv_residency_transfer_direction::h2d_promotion,
        { transfer_page(0) }, 4, {}, upload));
    vbr_h2d_status upload_status;
    auto upload_ring = vbr_h2d_chunk_ring::create(
        { {} }, 128, 32, upload_status);
    assert(upload_ring && upload_status == vbr_h2d_status::ok);
    llama_kv_residency_transfer_transport transport;
    transport.upload_ring = upload_ring.get();
    transport.context = &transfer_fake;
    transport.host_read = fake_transfer_backend::host_read;
    transport.recheck = fake_transfer_backend::recheck;
    llama_kv_residency_transfer_claim upload_claim;
    assert(pool->reserve(upload, 32, {}, upload_claim) ==
        llama_kv_residency_pool_status::ok);
    assert(llama_kv_residency_execute_transfer(
        *pool, upload, upload_claim, backend, transport).status ==
        llama_kv_residency_pool_status::ok);

    llama_kv_residency_table table(2);
    auto initial = table.begin();
    auto dirty = resident(0, 0);
    dirty.dirty = true;
    assert(table.replace(initial, dirty) == llama_kv_residency_status::ok);
    assert(table.publish(initial) == llama_kv_residency_status::ok);

    llama_kv_residency_transfer_plan reseal;
    assert(llama_kv_residency_build_transfer_plan(
        llama_kv_residency_transfer_direction::d2h_reseal,
        { transfer_page(0, 22) }, 4, {}, reseal));
    vbr_capture_stream_status download_status;
    auto download_ring = vbr_pinned_chunk_ring::create(
        { {} }, 128, 32, download_status);
    assert(download_ring && download_status == vbr_capture_stream_status::ok);
    transport.download_ring = download_ring.get();
    transport.host_write = fake_transfer_backend::host_write;
    llama_kv_residency_catalog_reservation catalog {
        &transfer_fake, fake_transfer_backend::reserve_catalog,
        fake_transfer_backend::release_catalog,
    };
    llama_kv_residency_transaction_request request;
    request.transfers.push_back(reseal);
    request.staging_capacity = 32;
    request.catalog = catalog;
    fake_residency_transaction transaction_fake;
    transaction_fake.pool = pool.get();
    const auto result = llama_kv_residency_execute_transaction(
        table, *pool, request, backend, transport,
        transaction_hooks(transaction_fake));
    assert(result.status == llama_kv_residency_transaction_status::committed);
    assert(result.published && result.dropped_pages == 1 &&
        result.loaded_pages == 0 && transaction_fake.pins == 1 &&
        transaction_fake.unpins == 1 && transaction_fake.drops == 1);
    assert(table.snapshot().epoch() == 2 && table.snapshot().pages().empty());
    assert(pool->mapped_slots() == 0 && transfer_fake.catalog_reserved == 8);
    assert(transfer_fake.host_bytes ==
        std::vector<uint8_t>({ 1, 2, 3, 4, 5, 6, 7, 8 }));
}

static void test_fixed_window_proof() {
    llama_kv_fixed_window_geometry geometry;
    geometry.logical_pages = 1024;
    llama_kv_fixed_window_derived_geometry derived;
    assert(llama_kv_fixed_window_derive_geometry(geometry, derived));
    assert(derived.values_per_token == 32768);
    assert(derived.bytes_per_token == 16896);
    assert(derived.row_bytes == 135168);
    assert(derived.page_bytes == 4325376);
    assert(derived.full_context_bytes == 4429185024ULL);

    llama_kv_fixed_window_selection selection;
    assert(llama_kv_fixed_window_select(
        geometry.logical_pages, 304, 303, { 0 }, selection));
    assert(selection.logical_pages.size() == 304);
    assert(selection.logical_pages.front() == 721 &&
        selection.logical_pages.back() == 0);
    llama_kv_fixed_window_selection invalid_selection;
    assert(!llama_kv_fixed_window_select(
        geometry.logical_pages, 304, 304, { 0 }, invalid_selection));

    llama_kv_fixed_window_status status;
    auto window = llama_kv_fixed_window::create(
        { geometry, 304, derived.full_context_bytes, 32*1024*1024 },
        derived, status);
    assert(window && status == llama_kv_fixed_window_status::ok);
    for (uint32_t logical_page = 0;
         logical_page < geometry.logical_pages; ++logical_page) {
        assert(window->seal_host_only(
            logical_page, geometry.page_tokens, logical_page + 1,
            derived.page_bytes) == llama_kv_fixed_window_status::ok);
    }
    assert(window->ledger().host_valid_pages == 1024);
    assert(window->ledger().host_payload_bytes == derived.full_context_bytes);
    assert(window->ledger().d2h_seal_useful_bytes == derived.full_context_bytes);
    assert(window->ledger().d2h_seal_aligned_bytes == derived.full_context_bytes);

    for (uint32_t slot = 0; slot < selection.logical_pages.size(); ++slot) {
        const uint32_t logical_page = selection.logical_pages[slot];
        assert(window->promote(
            logical_page, slot, logical_page + 1, derived.page_bytes) ==
            llama_kv_fixed_window_status::ok);
    }
    assert(window->ledger().resident_pages == 304);
    assert(window->ledger().resident_bytes == 304ULL * derived.page_bytes);
    assert(window->ledger().h2d_useful_bytes ==
        304ULL * derived.page_bytes);

    const uint64_t d2h_before_evict =
        window->ledger().d2h_seal_useful_bytes;
    assert(window->evict_clean(0) == llama_kv_fixed_window_status::ok);
    assert(window->ledger().resident_pages == 303 &&
        window->ledger().d2h_seal_useful_bytes == d2h_before_evict &&
        window->ledger().d2h_eviction_useful_bytes == 0 &&
        window->ledger().evictions == 1);
    assert(window->promote(0, 303, 1, derived.page_bytes) ==
        llama_kv_fixed_window_status::ok);

    assert(window->mutate(0) == llama_kv_fixed_window_status::ok);
    assert(window->evict_clean(0) == llama_kv_fixed_window_status::dirty);
    assert(window->seal(0, geometry.page_tokens, 1001, derived.page_bytes) ==
        llama_kv_fixed_window_status::ok);
    assert(window->ledger().host_valid_pages == 1024);
    assert(window->evict_clean(0) == llama_kv_fixed_window_status::ok);

    assert(window->begin_fill(500, 303, 17) == llama_kv_fixed_window_status::ok);
    assert(window->ledger().pinned_pages == 1);
    assert(window->evict_clean(500) == llama_kv_fixed_window_status::pinned);
    const uint64_t partial_bytes = 17ULL * derived.bytes_per_token;
    assert(window->seal(500, 17, 5001, derived.page_bytes) ==
        llama_kv_fixed_window_status::ok);
    assert(window->ledger().pinned_pages == 0 &&
        window->find(500)->valid_tokens == 17 &&
        window->find(500)->host_payload_bytes == partial_bytes);
    assert(window->evict_clean(500) == llama_kv_fixed_window_status::ok);

    assert(window->promote(1, 303, 9999, derived.page_bytes) ==
        llama_kv_fixed_window_status::checksum_mismatch);
    assert(window->ledger().checksum_failures == 1);
    assert(window->promote(1, 303, 2, derived.page_bytes) ==
        llama_kv_fixed_window_status::ok);
    assert(window->pin(1) == llama_kv_fixed_window_status::ok);
    assert(window->unpin(1) == llama_kv_fixed_window_status::ok);
    assert(window->evict_clean(1) == llama_kv_fixed_window_status::ok);
}

int main() {
    test_page_geometry();
    test_table_identity_and_snapshot();
    test_rejection_and_stale_publication();
    test_batched_transfer_pool();
    test_ggml_adapter_tensor_route();
    test_transfer_rollback_and_stale_completion();
    test_residency_transaction();
    test_reseal_before_eviction();
    test_fixed_window_proof();
    return 0;
}
