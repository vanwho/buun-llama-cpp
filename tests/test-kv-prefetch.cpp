#include "llama-kv-prefetch.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

struct prefetch_fake {
    struct ticket {
        uint64_t value = 0;
        bool complete = false;
    };

    std::vector<ticket> tickets;
    std::vector<uint64_t> cancelled;
    uint64_t complete_next = 0;
    uint64_t clock = 100;
    bool fail_publish = false;

    static bool submit(void * opaque, const llama_kv_prefetch_intent &,
                       uint32_t, uint64_t ticket, bool asynchronous) noexcept {
        auto & self = *static_cast<prefetch_fake *>(opaque);
        if (!asynchronous) return false;
        self.tickets.push_back({ ticket, false });
        return true;
    }

    static llama_kv_prefetch_poll poll(void * opaque, uint64_t ticket) noexcept {
        auto & self = *static_cast<prefetch_fake *>(opaque);
        for (auto & value : self.tickets) {
            if (value.value != ticket) continue;
            if (self.complete_next == ticket) {
                value.complete = true;
                self.complete_next = 0;
            }
            return value.complete ? llama_kv_prefetch_poll::completed
                                  : llama_kv_prefetch_poll::pending;
        }
        return llama_kv_prefetch_poll::stale_generation;
    }

    static void cancel(void * opaque, uint64_t ticket) noexcept {
        static_cast<prefetch_fake *>(opaque)->cancelled.push_back(ticket);
    }

    static bool publish(void * opaque, const llama_kv_prefetch_intent &) noexcept {
        return !static_cast<prefetch_fake *>(opaque)->fail_publish;
    }

    static uint64_t timestamp(void * opaque) noexcept {
        return ++static_cast<prefetch_fake *>(opaque)->clock;
    }
};

struct rerank_stage_fake {
    struct event_state { uint64_t id; bool done; };
    std::vector<event_state> events;
    uint64_t next_event = 1;
    uint32_t released = 0;
    static bool enqueue(void * opaque, uint32_t, const void * host, void * device,
            size_t bytes, uint64_t, uint64_t, uint64_t * event) noexcept {
        auto & self = *static_cast<rerank_stage_fake *>(opaque);
        std::memcpy(device, host, bytes);
        *event = self.next_event++;
        self.events.push_back({ *event, false });
        return true;
    }
    static llama_kv_prefetch_poll poll(void * opaque, uint64_t event) noexcept {
        auto & self = *static_cast<rerank_stage_fake *>(opaque);
        for (const auto & item : self.events) if (item.id == event)
            return item.done ? llama_kv_prefetch_poll::completed : llama_kv_prefetch_poll::pending;
        return llama_kv_prefetch_poll::failed;
    }
    static void cancel(void * opaque, uint64_t event) noexcept {
        auto & self = *static_cast<rerank_stage_fake *>(opaque);
        for (auto & item : self.events) if (item.id == event) item.done = true;
    }
    static void release(void * opaque, uint64_t) noexcept {
        ++static_cast<rerank_stage_fake *>(opaque)->released;
    }
};

struct rerank_stage_source_fake {
    std::vector<uint8_t> bytes;
    uint64_t version = 7;
    static bool recheck(void * opaque, uint64_t version) noexcept {
        return static_cast<rerank_stage_source_fake *>(opaque)->version == version;
    }
    static bool read(void * opaque, uint64_t offset, void * dst, size_t bytes) noexcept {
        auto & self = *static_cast<rerank_stage_source_fake *>(opaque);
        if (offset > self.bytes.size() || bytes > self.bytes.size() - size_t(offset)) return false;
        std::memcpy(dst, self.bytes.data() + offset, bytes);
        return true;
    }
};

static void test_rerank_stage_ring() {
    uint8_t host0[4] = {}, host1[4] = {}, device0[4] = {}, device1[4] = {};
    rerank_stage_fake backend_state;
    llama_kv_rerank_stage_backend backend { &backend_state,
        &rerank_stage_fake::enqueue, &rerank_stage_fake::poll,
        &rerank_stage_fake::cancel, &rerank_stage_fake::release };
    llama_kv_rerank_stage_ring ring;
    assert(ring.configure({ host0, host1 }, { device0, device1 }, 4, backend));
    assert(ring.allocated_bytes() == 16);
    auto source_data = std::make_shared<rerank_stage_source_fake>();
    source_data->bytes = { 1, 2, 3, 4, 5, 6, 7, 8 };
    std::shared_ptr<const void> holder(source_data, source_data.get());
    llama_kv_rerank_stage_source source { holder, source_data.get(),
        &rerank_stage_source_fake::recheck, &rerank_stage_source_fake::read };
    uint32_t slot0 = UINT32_MAX, slot1 = UINT32_MAX, slot2 = UINT32_MAX;
    assert(ring.submit(source, 0, 4, 11, 7, slot0) == llama_kv_rerank_stage_status::ok);
    assert(ring.submit(source, 4, 4, 11, 7, slot1) == llama_kv_rerank_stage_status::ok);
    assert(slot0 != slot1 && ring.busy_slots() == 2 && holder.use_count() == 5);
    assert(ring.submit(source, 0, 1, 11, 7, slot2) == llama_kv_rerank_stage_status::backpressure);
    assert(std::memcmp(device0, source_data->bytes.data() + slot0 * 4, 4) == 0);
    assert(std::memcmp(device1, source_data->bytes.data() + slot1 * 4, 4) == 0);
    assert(ring.poll() == 0 && ring.busy_slots() == 2);
    for (auto & event : backend_state.events) event.done = true;
    assert(ring.poll() == 2 && ring.busy_slots() == 0 && backend_state.released == 2);
    source_data->version = 8;
    assert(ring.submit(source, 0, 4, 12, 7, slot2) == llama_kv_rerank_stage_status::stale_source);
}

static void test_rerank_bundle_order() {
    // A sparse useful head wins by peak mass even when another page has more
    // support and a much stronger average mass.
    const llama_kv_prefetch_ranked_bundle_score rare_head { 0.91f, 0.08f, 1, 12 };
    const llama_kv_prefetch_ranked_bundle_score generic_mean { 0.62f, 0.41f, 8, 4 };
    assert(llama_kv_prefetch_ranked_bundle_better(rare_head, generic_mean));

    const llama_kv_prefetch_ranked_bundle_score supported { 0.62f, 0.41f, 3, 8 };
    const llama_kv_prefetch_ranked_bundle_score weak_support { 0.62f, 0.41f, 2, 3 };
    assert(llama_kv_prefetch_ranked_bundle_better(supported, weak_support));

    const llama_kv_prefetch_ranked_bundle_score lower_id { 0.62f, 0.41f, 3, 3 };
    assert(llama_kv_prefetch_ranked_bundle_better(lower_id, supported));
}

static llama_kv_prefetch_intent intent(uint64_t page, uint32_t priority = 1) {
    return { page, 7, 8, 10, priority, false };
}

static llama_kv_page_id mailbox_page(uint32_t logical, uint32_t layer) {
    llama_kv_page_id id;
    id.session_generation = 1;
    id.sequence_id = 0;
    id.sequence_generation = 1;
    id.logical_page = logical;
    id.page_generation = logical + 1;
    id.representation_epoch = 1;
    id.model_identity = 1;
    id.topology_identity = 1;
    id.codec_digest = 1;
    id.codebook_digest = 1;
    id.rotation_digest = 1;
    id.meansub_digest = 1;
    id.position_begin = llama_pos(logical * 256);
    id.position_end = id.position_begin + 256;
    id.attention_layer = layer;
    return id;
}

struct mailbox_fake {
    bool complete = false;
    uint64_t cancelled = 0;
    uint64_t released = 0;

    static llama_kv_prefetch_mailbox_poll poll(
            void * opaque, uint64_t) noexcept {
        return static_cast<mailbox_fake *>(opaque)->complete
            ? llama_kv_prefetch_mailbox_poll::completed
            : llama_kv_prefetch_mailbox_poll::pending;
    }

    static void cancel(void * opaque, uint64_t) noexcept {
        ++static_cast<mailbox_fake *>(opaque)->cancelled;
    }

    static void release(void * opaque, uint64_t) noexcept {
        ++static_cast<mailbox_fake *>(opaque)->released;
    }
};

static void test_candidate_mailbox() {
    mailbox_fake fake;
    llama_kv_prefetch_mailbox mailbox({ 2, 4 });
    mailbox.set_backend({ &fake, mailbox_fake::poll, mailbox_fake::cancel,
                          mailbox_fake::release });

    uint32_t slot = UINT32_MAX;
    llama_kv_prefetch_candidate * records = nullptr;
    assert(mailbox.acquire(slot, records) == llama_kv_prefetch_mailbox_status::ok);
    assert(records != nullptr);
    records[0] = { mailbox_page(4, 0), 0, 9, 12, 5.0f, 270336, 3, 0, false };
    records[1] = { mailbox_page(4, 1), 1, 9, 12, 4.0f, 270336, 3, 0, false };
    assert(mailbox.publish_pending(slot, 2, 9, 41) ==
           llama_kv_prefetch_mailbox_status::ok);
    assert(mailbox.poll(9, 12) == llama_kv_prefetch_mailbox_status::ok);
    assert(mailbox.ready_slots() == 0);
    fake.complete = true;
    assert(mailbox.poll(9, 12) == llama_kv_prefetch_mailbox_status::ok);
    std::vector<llama_kv_prefetch_candidate> ready;
    assert(mailbox.take_ready(ready) == 2);
    assert(ready[0].identity.logical_page == 4 &&
           ready[1].identity.logical_page == 4 &&
           ready[0].attention_layer != ready[1].attention_layer);
    assert(mailbox.pending_slots() == 0 && mailbox.ready_slots() == 0 &&
           fake.released == 1);

    assert(mailbox.acquire(slot, records) == llama_kv_prefetch_mailbox_status::ok);
    records[0] = { mailbox_page(5, 0), 0, 10, 12, 3.0f, 270336, 4, 0, false };
    assert(mailbox.publish_pending(slot, 1, 10, 42) ==
           llama_kv_prefetch_mailbox_status::ok);
    assert(mailbox.poll(9, 12) == llama_kv_prefetch_mailbox_status::stale_generation);
    assert(fake.cancelled == 1 && mailbox.pending_slots() == 0);
}

static void test_scaled_candidate_mailbox() {
    constexpr uint32_t resident_pages = 200;
    constexpr uint32_t cold_pages = LLAMA_KV_QUERY_COLD_SELECTOR_PAGES;
    constexpr uint32_t layers = 16;
    const uint32_t width = (resident_pages + cold_pages) * layers;
    llama_kv_prefetch_mailbox mailbox({ 2, width });
    assert(mailbox.configured() && mailbox.candidates_per_slot() >= width);

    uint32_t slot = UINT32_MAX;
    llama_kv_prefetch_candidate * records = nullptr;
    assert(mailbox.acquire(slot, records) == llama_kv_prefetch_mailbox_status::ok);
    uint32_t count = 0;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        for (uint32_t rank = 0; rank < resident_pages + cold_pages; ++rank) {
            auto & candidate = records[count++];
            const uint32_t logical_page = rank < resident_pages
                ? rank + (rank >= 31 ? 1 : 0) : 1000 + rank - resident_pages;
            candidate.identity = mailbox_page(logical_page, layer);
            candidate.attention_layer = layer;
            candidate.generation = 17;
            candidate.table_epoch = 23;
            candidate.score = 1.0f / (1.0f + float(rank));
            candidate.requested_bytes = 4096;
            candidate.content_version = 100 + logical_page;
            candidate.summary_version = candidate.content_version;
            candidate.speculation_generation = 3;
            candidate.selector_rank = rank;
            candidate.query_position = 8193;
            candidate.cold = rank >= resident_pages;
            candidate.rollback_generation = candidate.identity.page_generation;
        }
    }
    assert(count == width);
    assert(mailbox.publish_ready(slot, count, 17) ==
           llama_kv_prefetch_mailbox_status::ok);
    std::vector<llama_kv_prefetch_candidate> ready;
    assert(mailbox.take_ready(ready) == width);
    assert(ready.back().attention_layer == layers - 1 && ready.back().cold &&
           ready.back().identity.logical_page == 1000 + cold_pages - 1 &&
           ready.back().content_version == 1100 + cold_pages - 1 &&
           ready.back().selector_rank == resident_pages + cold_pages - 1);
    assert(ready[(layers - 1) * (resident_pages + cold_pages) + 31].identity.logical_page ==
           32);

    assert(mailbox.configure({ 2, (32 + cold_pages) * 4 }));
    assert(mailbox.candidates_per_slot() == (32 + cold_pages) * 4);

    // Run the production completion decoder against compact IDs and candidate
    // records sharing the same acquired mailbox allocation, as the async
    // callback does. This exercises the in-place expansion path above 128 IDs.
    llama_kv_prefetch_mailbox expansion_mailbox({ 2, width });
    assert(expansion_mailbox.acquire(slot, records) == llama_kv_prefetch_mailbox_status::ok);
    std::vector<int32_t> raw(width);
    std::vector<std::vector<llama_kv_prefetch_page_descriptor>> pages(layers);
    std::vector<llama_kv_prefetch_selector_segment> segments;
    segments.reserve(layers);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        pages[layer].resize(resident_pages + cold_pages);
        for (uint32_t rank = 0; rank < resident_pages + cold_pages; ++rank) {
            const uint32_t logical = rank < resident_pages
                ? rank + (rank >= 31 ? 1 : 0) : 1000 + rank - resident_pages;
            auto & page = pages[layer][rank];
            page.identity = mailbox_page(logical, layer);
            page.content_version = 5000 + logical;
            page.summary_version = page.content_version;
            raw[layer * (resident_pages + cold_pages) + rank] = int32_t(rank);
        }
        llama_kv_prefetch_selector_segment segment{
            layer * (resident_pages + cold_pages), resident_pages + cold_pages,
            0, resident_pages, resident_pages, cold_pages,
            0, layer, 1, 1, 17, 23, 8193, 31, 4096, &pages[layer],
        };
        segment.byte_offset = uint64_t(segment.raw_offset) * sizeof(int32_t);
        segment.byte_count = uint64_t(segment.count) * sizeof(int32_t);
        segment.byte_stride = sizeof(int32_t);
        segments.push_back(segment);
    }
    std::memcpy(records, raw.data(), raw.size() * sizeof(raw[0]));
    std::vector<int32_t> copied;
    uint32_t expanded = 0;
    assert(llama_kv_prefetch_expand_selector_ids(
        reinterpret_cast<const int32_t *>(records), uint32_t(raw.size()), segments,
        expansion_mailbox.candidates_per_slot(), copied, records, expanded));
    assert(copied == raw && expanded == width && expanded > 128);
    const auto & remapped = records[(layers - 1) * (resident_pages + cold_pages) + 31];
    assert(remapped.identity.logical_page == 32 && remapped.attention_layer == layers - 1 &&
           remapped.generation == 17 && remapped.table_epoch == 23 &&
           remapped.selector_rank == 31 && remapped.content_version == 5032);
    const auto & last = records[expanded - 1];
    assert(last.identity.logical_page == 1000 + cold_pages - 1 &&
           last.attention_layer == layers - 1 &&
           last.cold && last.selector_rank == cold_pages - 1 &&
           last.content_version == 6000 + cold_pages - 1);
           last.content_version == 6004);
}

static void test_layer_duplicate_and_refresh_budget(prefetch_fake & fake) {
    llama_kv_prefetch_backend backend;
    backend.context = &fake;
    backend.submit = prefetch_fake::submit;
    backend.poll = prefetch_fake::poll;
    backend.cancel = prefetch_fake::cancel;
    backend.publish_complete = prefetch_fake::publish;
    llama_kv_prefetch_config config;
    config.max_queued_pages = 8;
    config.max_queued_bytes = 128;
    config.max_events = 2;
    config.max_pinned_slots = 4;
    config.staging_slots = 2;
    config.max_cold_pages_per_refresh = 1;
    config.bytes_per_refresh = 10;
    llama_kv_prefetch_status status;
    auto scheduler = llama_kv_prefetch_scheduler::create(config, backend, status);
    assert(scheduler && status == llama_kv_prefetch_status::ok);
    assert(scheduler->begin_refresh(1) == llama_kv_prefetch_status::ok);

    auto layer0 = intent(30);
    layer0.attention_layer = 0;
    auto layer1 = intent(30);
    layer1.attention_layer = 1;
    assert(scheduler->enqueue(layer0) == llama_kv_prefetch_status::ok);
    assert(scheduler->enqueue(layer1) == llama_kv_prefetch_status::backpressure);

    auto required = intent(31);
    required.required = true;
    required.attention_layer = 1;
    assert(scheduler->enqueue(required) == llama_kv_prefetch_status::ok);
    scheduler->shutdown();
}

static void test_mixed_selector_segments() {
    llama_kv_prefetch_mailbox mailbox({1, 8});
    uint32_t slot = UINT32_MAX;
    llama_kv_prefetch_candidate * records = nullptr;
    assert(mailbox.acquire(slot, records) == llama_kv_prefetch_mailbox_status::ok);
    std::vector<llama_kv_prefetch_page_descriptor> packed_pages(8), legacy_pages(8);
    for (uint32_t logical = 0; logical < 8; ++logical) {
        packed_pages[logical].identity = mailbox_page(logical, 0);
        packed_pages[logical].content_version = 100 + logical;
        packed_pages[logical].summary_version = 100 + logical;
        legacy_pages[logical].identity = mailbox_page(logical, 1);
        legacy_pages[logical].content_version = 200 + logical;
        legacy_pages[logical].summary_version = 200 + logical;
    }
    const ggml_kv_page_rank_record packed {1, 1u, 0.75f, 0.5f};
    const int32_t legacy[2] = {2, 3};
    std::memcpy(records, &packed, sizeof(packed));
    std::memcpy(reinterpret_cast<uint8_t *>(records) + sizeof(packed), legacy, sizeof(legacy));
    llama_kv_prefetch_selector_segment packed_segment{};
    packed_segment.raw_offset = 0;
    packed_segment.count = 1;
    packed_segment.resident_count = 1;
    packed_segment.sequence_id = 0;
    packed_segment.attention_layer = 0;
    packed_segment.session_generation = 1;
    packed_segment.sequence_generation = 1;
    packed_segment.query_generation = 7;
    packed_segment.table_epoch = 9;
    packed_segment.pages = &packed_pages;
    packed_segment.record_format = llama_kv_prefetch_selector_segment::format::packed_rank_records;
    packed_segment.byte_offset = 0;
    packed_segment.byte_count = sizeof(packed);
    packed_segment.byte_stride = sizeof(packed);
    auto legacy_segment = packed_segment;
    legacy_segment.raw_offset = 1;
    legacy_segment.count = 2;
    legacy_segment.resident_count = 1;
    legacy_segment.cold_count = 1;
    legacy_segment.attention_layer = 1;
    legacy_segment.pages = &legacy_pages;
    legacy_segment.record_format = llama_kv_prefetch_selector_segment::format::legacy_i32_ids;
    legacy_segment.byte_offset = sizeof(packed);
    legacy_segment.byte_count = sizeof(legacy);
    legacy_segment.byte_stride = sizeof(int32_t);
    const std::vector<llama_kv_prefetch_selector_segment> segments = {
        packed_segment, legacy_segment,
    };
    std::vector<int32_t> copied_ids;
    uint32_t written = 0;
    assert(llama_kv_prefetch_expand_selector_segments(records, 3,
            sizeof(packed) + sizeof(legacy), segments, mailbox.candidates_per_slot(),
            copied_ids, records, written));
    assert((copied_ids == std::vector<int32_t>{1, 2, 3}));
    assert(written == 3 && records[0].provenance ==
            llama_kv_prefetch_candidate::score_kind::probe_softmax &&
            records[0].peak_probability == packed.peak_probability);
    assert(records[1].identity.logical_page == 2 && records[1].attention_layer == 1 &&
            records[1].provenance != llama_kv_prefetch_candidate::score_kind::probe_softmax);
    assert(records[2].identity.logical_page == 3 && records[2].cold && records[2].selector_rank == 0);
}

int main() {
    llama_kv_prefetch_predictor predictor(2);
    assert(predictor.observe(41, 3, 100,
            { intent(10, 1), intent(11, 4), intent(12, 2) }));
    const auto predicted = predictor.predict(7, 3, 101, 2);
    assert(predicted.size() == 2 && predicted[0].page_id == 11 &&
           predicted[1].page_id == 12);
    assert(predicted[0].prediction && predicted[0].generation == 7 &&
           predicted[0].source_query_generation == 41 &&
           predicted[0].source_query_layer == 3 &&
           predicted[0].needed_by_token == 101);

    prefetch_fake fake;
    llama_kv_prefetch_backend backend;
    backend.context = &fake;
    backend.submit = prefetch_fake::submit;
    backend.poll = prefetch_fake::poll;
    backend.cancel = prefetch_fake::cancel;
    backend.publish_complete = prefetch_fake::publish;
    backend.timestamp_us = prefetch_fake::timestamp;

    llama_kv_prefetch_config config;
    config.max_queued_pages = 4;
    config.max_queued_bytes = 80;
    config.max_events = 2;
    config.max_pinned_slots = 5;
    config.staging_slots = 2;
    config.max_timeline_events = 64;
    llama_kv_prefetch_status status;
    auto scheduler = llama_kv_prefetch_scheduler::create(config, backend, status);
    assert(scheduler && status == llama_kv_prefetch_status::ok);

    assert(scheduler->prefetch(predicted) == llama_kv_prefetch_status::ok);
    assert(scheduler->active_events() == 2);

    // The second ticket (page 12) completes first.  A required page waits for the
    // event and is consumed only after publication; no queue-order shortcut
    // can expose an incomplete page.
    fake.complete_next = 2;
    auto required = predicted[1];
    required.required = true;
    const auto second = scheduler->ensure_ready({ required }, {}, 1);
    assert(second.readiness == llama_kv_prefetch_readiness::waited_ready);
    assert(second.ready.size() == 1 && second.ready[0] == 12);

    fake.complete_next = 1;
    required = predicted[0];
    required.required = true;
    const auto first = scheduler->ensure_ready({ required }, {}, 1);
    assert(first.readiness == llama_kv_prefetch_readiness::waited_ready);
    assert(first.ready.size() == 1 && first.ready[0] == 11);
    assert(scheduler->counters().prediction_hits == 2);
    assert(scheduler->counters().prediction_useful_bytes == 16);

    // Two active tickets plus one queued ticket are the complete bounded
    // occupancy.  A fourth request is refused without growing the queue.
    assert(scheduler->enqueue({ 20, 7, 8, 10, 1, false, 0, UINT32_MAX,
                                UINT64_MAX, 0, 102, true, false }) ==
           llama_kv_prefetch_status::ok);
    assert(scheduler->enqueue({ 21, 7, 8, 10, 1, false, 0, UINT32_MAX,
                                UINT64_MAX, 0, 103, true, false }) ==
           llama_kv_prefetch_status::ok);
    assert(scheduler->enqueue({ 22, 7, 8, 10, 1, false, 0, UINT32_MAX,
                                UINT64_MAX, 0, 104, true, false }) ==
           llama_kv_prefetch_status::event_full);
    assert(scheduler->pinned_slots() == 5);
    assert(scheduler->enqueue({ 23, 7, 8, 10, 1, false, 0, UINT32_MAX,
                                UINT64_MAX, 0, 105, true, false }) ==
           llama_kv_prefetch_status::backpressure);
    assert(scheduler->cancel(22, 7) == llama_kv_prefetch_status::cancelled);
    assert(scheduler->counters().prediction_wasted_bytes >= 8);

    // A page requested by the current query is not useful until publication.
    // A failed publication therefore remains wasted even if it was marked
    // needed while its asynchronous copy was in flight.
    fake.fail_publish = true;
    fake.complete_next = 3;
    required = predicted[0];
    required.page_id = 20;
    required.required = true;
    const auto failed = scheduler->ensure_ready({ required }, {}, 1);
    assert(failed.readiness == llama_kv_prefetch_readiness::fallback_larger_union);
    assert(scheduler->counters().prediction_useful_bytes == 16);
    assert(scheduler->counters().prediction_wasted_bytes >= 16);
    fake.fail_publish = false;

    bool saw_enqueue = false;
    bool saw_begin = false;
    bool saw_end = false;
    bool saw_needed = false;
    bool saw_wait = false;
    bool saw_consumed = false;
    for (const auto & event : scheduler->timeline()) {
        saw_enqueue |= event.kind == llama_kv_prefetch_timeline_kind::enqueue;
        saw_begin |= event.kind == llama_kv_prefetch_timeline_kind::copy_begin;
        saw_end |= event.kind == llama_kv_prefetch_timeline_kind::copy_end;
        saw_needed |= event.kind == llama_kv_prefetch_timeline_kind::needed;
        saw_wait |= event.kind == llama_kv_prefetch_timeline_kind::wait;
        saw_consumed |= event.kind == llama_kv_prefetch_timeline_kind::consumed;
    }
    assert(saw_enqueue && saw_begin && saw_end && saw_needed && saw_wait && saw_consumed);

    scheduler->shutdown();
    assert(scheduler->stopped() && scheduler->queued_pages() == 0 &&
           scheduler->active_events() == 0);
    test_candidate_mailbox();
    test_scaled_candidate_mailbox();
    test_mixed_selector_segments();
    test_layer_duplicate_and_refresh_budget(fake);
    test_rerank_stage_ring();
    test_rerank_bundle_order();
    std::cout << "kv prefetch checks passed\n";
}
