#include "llama-kv-prefetch.h"

#include <cassert>
#include <cstdint>
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
    test_layer_duplicate_and_refresh_budget(fake);
    std::cout << "kv prefetch checks passed\n";
}
