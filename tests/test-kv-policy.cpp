#include "llama-kv-policy.h"
#include "llama-kv-live-policy.h"
#include "llama-kv-prefetch.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

static llama_kv_page_id live_page_id(uint32_t logical) {
    llama_kv_page_id id;
    id.session_generation = 1;
    id.sequence_id = 0;
    id.sequence_generation = 1;
    id.logical_page = logical;
    id.page_generation = logical + 1;
    id.representation_epoch = 1;
    id.position_begin = llama_pos(logical * 256);
    id.position_end = id.position_begin + 256;
    return id;
}

static llama_kv_page_record live_resident(uint32_t logical, uint32_t slot) {
    llama_kv_page_record record;
    record.id = live_page_id(logical);
    record.physical_slot = slot;
    record.state = llama_kv_page_state::gpu_host_clean;
    record.host_valid = true;
    return record;
}

struct live_transfer_fake {
    bool fail_issue = false;

    static bool map_slot(void *, uint32_t) noexcept { return true; }
    static bool drop_slot(void *, uint32_t) noexcept { return true; }
    static bool issue(void * opaque, llama_kv_residency_transfer_direction,
                      const llama_kv_residency_completion &, uint64_t,
                      void *, size_t, uint64_t, bool) noexcept {
        return !static_cast<live_transfer_fake *>(opaque)->fail_issue;
    }
    static bool complete(void *, uint64_t) noexcept { return true; }
    static void cancel(void *, uint64_t) noexcept {}
    static bool host_read(void *, uint32_t, uint64_t, uint8_t * destination,
                          size_t size) noexcept {
        std::fill(destination, destination + size, uint8_t(7));
        return true;
    }
};

static llama_kv_live_policy_boundary live_boundary(
        const llama_kv_residency_snapshot & snapshot,
        const llama_kv_residency_transfer_plan & promotion) {
    llama_kv_live_policy_boundary boundary;
    boundary.snapshot = snapshot;
    boundary.hot_capacity = 2;
    boundary.logical_page_count = 3;
    boundary.has_write_page = true;
    boundary.write_page = live_page_id(0);
    boundary.retrieval.status = llama_kv_routing_retrieval_status::ok;
    boundary.retrieval.table_epoch = snapshot.epoch();
    boundary.retrieval.selected.push_back({
        live_page_id(1), llama_kv_routing_retrieval_reason::summary,
        9.0f, true, false, 1,
    });
    boundary.transaction.staging_capacity = 32;
    boundary.transaction.transfers.push_back(promotion);

    llama_kv_live_policy_page current;
    current.record = live_resident(0, 0);
    current.current = true;
    current.age = 1;
    current.recency = 10;
    boundary.pages.push_back(current);

    llama_kv_live_policy_page cold;
    cold.record.id = live_page_id(1);
    cold.record.state = llama_kv_page_state::host_clean;
    cold.record.host_valid = true;
    cold.age = 2;
    cold.recency = 9;
    boundary.pages.push_back(cold);

    llama_kv_live_policy_page other;
    other.record.id = live_page_id(2);
    other.record.state = llama_kv_page_state::host_clean;
    other.record.host_valid = true;
    other.age = 3;
    other.recency = 1;
    boundary.pages.push_back(other);
    return boundary;
}

static llama_kv_residency_transfer_plan live_promotion() {
    llama_kv_residency_transfer_page page;
    page.page = live_page_id(1);
    page.table_epoch = 1;
    page.physical_slot = 1;
    page.runs.push_back({ UINT32_MAX, 0, 0, 0, 0, 8, 1, 0, 0 });
    llama_kv_residency_transfer_plan output;
    assert(llama_kv_residency_build_transfer_plan(
        llama_kv_residency_transfer_direction::h2d_promotion,
        { page }, 4, {}, output));
    return output;
}

static llama_kv_residency_pool_backend live_pool_backend(
        live_transfer_fake & fake) {
    llama_kv_residency_pool_backend backend;
    backend.context = &fake;
    backend.map_slot = live_transfer_fake::map_slot;
    backend.drop_slot = live_transfer_fake::drop_slot;
    backend.issue_copy = live_transfer_fake::issue;
    backend.complete_copy = live_transfer_fake::complete;
    backend.cancel_copy = live_transfer_fake::cancel;
    return backend;
}

static void test_live_policy_publication() {
    llama_kv_residency_table table(2);
    auto initial = table.begin();
    assert(table.replace(initial, live_resident(0, 0)) == llama_kv_residency_status::ok);
    assert(table.publish(initial) == llama_kv_residency_status::ok);

    live_transfer_fake fake;
    auto backend = live_pool_backend(fake);
    llama_kv_residency_pool_status pool_status;
    auto pool = llama_kv_residency_pool::create(
        { 2, 64, 4, 4, 1024 }, backend, pool_status);
    assert(pool && pool_status == llama_kv_residency_pool_status::ok);
    vbr_h2d_status ring_status;
    auto ring = vbr_h2d_chunk_ring::create({ {} }, 128, 32, ring_status);
    assert(ring && ring_status == vbr_h2d_status::ok);
    llama_kv_residency_transfer_transport transport;
    transport.upload_ring = ring.get();
    transport.context = &fake;
    transport.host_read = live_transfer_fake::host_read;

    auto boundary = live_boundary(table.snapshot(), live_promotion());
    auto result = llama_kv_live_policy_apply(
        table, *pool, boundary, backend, transport);
    assert(result.status == llama_kv_live_policy_status::committed);
    assert(result.published && result.published_epoch == 2);
    assert(result.trace.hot_capacity == 2 && result.trace.target_pages == 2);
    assert(result.target_pages.size() == 2);
    assert(result.target_pages[0].id == live_page_id(0));
    assert(result.target_pages[1].id == live_page_id(1));
    assert(result.decisions.size() == 2);
    assert(result.decisions[1].added &&
           result.decisions[1].selection_reason == llama_kv_policy_reason::summary &&
           result.decisions[1].retrieval_score_available &&
           result.decisions[1].retention_score_q != 0);

    auto stale = boundary;
    stale.snapshot = boundary.snapshot;
    auto stale_result = llama_kv_live_policy_apply(
        table, *pool, stale, backend, transport);
    assert(stale_result.status == llama_kv_live_policy_status::stale_snapshot);
    assert(table.snapshot().epoch() == 2 && table.snapshot().pages().size() == 2);

    llama_kv_residency_table failed_table(2);
    auto failed_initial = failed_table.begin();
    assert(failed_table.replace(failed_initial, live_resident(0, 0)) == llama_kv_residency_status::ok);
    assert(failed_table.publish(failed_initial) == llama_kv_residency_status::ok);
    live_transfer_fake failed_fake;
    failed_fake.fail_issue = true;
    auto failed_backend = live_pool_backend(failed_fake);
    llama_kv_residency_pool_status failed_pool_status;
    auto failed_pool = llama_kv_residency_pool::create(
        { 2, 64, 4, 4, 1024 }, failed_backend, failed_pool_status);
    assert(failed_pool);
    vbr_h2d_status failed_ring_status;
    auto failed_ring = vbr_h2d_chunk_ring::create({ {} }, 128, 32, failed_ring_status);
    assert(failed_ring);
    llama_kv_residency_transfer_transport failed_transport;
    failed_transport.upload_ring = failed_ring.get();
    failed_transport.context = &failed_fake;
    failed_transport.host_read = live_transfer_fake::host_read;
    auto failed_result = llama_kv_live_policy_apply(
        failed_table, *failed_pool,
        live_boundary(failed_table.snapshot(), live_promotion()),
        failed_backend, failed_transport);
    assert(failed_result.status == llama_kv_live_policy_status::transaction_failed);
    assert(!failed_result.published && failed_table.snapshot().epoch() == 1 &&
           failed_table.snapshot().pages().size() == 1);
}

static llama_kv_policy_page page(uint64_t id, bool resident = true) {
    llama_kv_policy_page p; p.id = id; p.resident = resident; p.age = id; p.recency = id; return p;
}

static bool contains(const std::vector<uint64_t> & values, uint64_t id) {
    return std::find(values.begin(), values.end(), id) != values.end();
}

struct fake_prefetch_backend {
    struct ticket {
        uint64_t value = 0;
        bool complete = false;
    };

    std::vector<ticket> tickets;
    std::vector<uint64_t> submitted;
    std::vector<uint64_t> published;
    std::vector<uint64_t> cancelled;
    std::vector<uint64_t> resealed;
    std::vector<uint64_t> evicted;
    uint64_t complete_next = 0;
    uint64_t clock = 100;
    bool host_miss = false;
    bool submit_fail = false;
    uint64_t fail_poll_ticket = 0;
    bool overlap_seen = false;
    bool attention_active = false;

    static bool submit(void * opaque, const llama_kv_prefetch_intent & intent,
                       uint32_t, uint64_t ticket, bool asynchronous) noexcept {
        auto & fake = *static_cast<fake_prefetch_backend *>(opaque);
        if (!asynchronous || fake.submit_fail) return false;
        fake.tickets.push_back({ ticket, false });
        fake.submitted.push_back(intent.page_id);
        fake.overlap_seen = fake.overlap_seen || fake.attention_active;
        return true;
    }

    static llama_kv_prefetch_poll poll(void * opaque, uint64_t ticket) noexcept {
        auto & fake = *static_cast<fake_prefetch_backend *>(opaque);
        for (auto & item : fake.tickets) {
            if (item.value != ticket) continue;
            if (fake.fail_poll_ticket == ticket) return llama_kv_prefetch_poll::failed;
            if (fake.complete_next == ticket) {
                item.complete = true;
                fake.complete_next = 0;
            }
            return item.complete ? llama_kv_prefetch_poll::completed
                                 : llama_kv_prefetch_poll::pending;
        }
        return llama_kv_prefetch_poll::stale_generation;
    }

    static void cancel(void * opaque, uint64_t ticket) noexcept {
        static_cast<fake_prefetch_backend *>(opaque)->cancelled.push_back(ticket);
    }

    static bool publish(void * opaque, const llama_kv_prefetch_intent & intent) noexcept {
        static_cast<fake_prefetch_backend *>(opaque)->published.push_back(intent.page_id);
        return true;
    }

    static bool host_available(void * opaque, const llama_kv_prefetch_intent &) noexcept {
        return !static_cast<fake_prefetch_backend *>(opaque)->host_miss;
    }

    static bool reseal(void * opaque, const llama_kv_prefetch_intent & intent) noexcept {
        static_cast<fake_prefetch_backend *>(opaque)->resealed.push_back(intent.page_id);
        return true;
    }

    static bool evict(void * opaque, const llama_kv_prefetch_intent & intent) noexcept {
        static_cast<fake_prefetch_backend *>(opaque)->evicted.push_back(intent.page_id);
        return true;
    }

    static uint64_t timestamp(void * opaque) noexcept {
        return ++static_cast<fake_prefetch_backend *>(opaque)->clock;
    }
};

int main() {
    test_live_policy_publication();

    llama_kv_policy_trace trace;
    trace.epoch = 1; trace.capacity_pages = 2; trace.write_page = 4;
    auto p1 = page(1); p1.attention_ema_q = 20; p1.attention_observed = true;
    auto p2 = page(2); p2.attention_ema_q = 1; p2.attention_observed = true; p2.dirty_cost = 9;
    auto p3 = page(3); p3.application_pin = true;
    auto p4 = page(4, false); p4.anchor = true;
    auto p5 = page(5); // cold: must not be interpreted as observed attention == zero
    trace.pages = { p1, p2, p3, p4, p5 };
    trace.summary_top_k = { 5, 5 };
    trace.exploration = { 2 };
    const auto result = llama_kv_policy_replay(trace);
    assert(result.status == llama_kv_policy_status::ok);
    assert(result.retrieve.size() == 5);
    assert(result.victims.size() == 2);
    assert(result.victims[0] == 2); // observed low attention wins despite dirty cost
    assert(result.victims[1] == 5); // cold page follows explicit fallback, not zero attention

    trace.capacity_pages = 1;
    assert(llama_kv_policy_replay(trace).status == llama_kv_policy_status::pin_overflow);
    trace.capacity_pages = 3; trace.pages[0].application_pin = true;
    trace.pages[1].application_pin = true;
    trace.pages[2].inflight_pin = true;
    trace.pages[4].application_pin = true;
    assert(llama_kv_policy_replay(trace).status == llama_kv_policy_status::pin_overflow);

    llama_kv_policy_trace controller_trace;
    controller_trace.epoch = 7;
    controller_trace.write_page = 1;
    controller_trace.pages.resize(8);
    for (size_t i = 0; i < controller_trace.pages.size(); ++i) {
        controller_trace.pages[i] = page(i + 1, i < 6);
        controller_trace.pages[i].age = 8 - i;
        controller_trace.pages[i].recency = i;
        controller_trace.pages[i].attention_ema_q = i * 10;
    }
    controller_trace.pages[0].current = true;
    controller_trace.pages[1].structural = true;
    controller_trace.pages[2].recent = true;
    controller_trace.pages[3].attention_observed = true;
    controller_trace.pages[4].attention_observed = true;
    controller_trace.pages[5].speculative_pin = true;
    controller_trace.summary_top_k = { 7, 4, 7 };
    controller_trace.exploration = { 8, 4 };

    llama_kv_policy_controller_config controller_config;
    controller_config.capacity_pages = 6;
    controller_config.recent_pages = 1;
    controller_config.structural_pages = 1;
    controller_config.historical_pages = 2;
    controller_config.transient_pages = 2;
    const auto decision = llama_kv_policy_decide(controller_trace, controller_config);
    assert(decision.status == llama_kv_policy_status::ok);
    assert(decision.target.size() == controller_config.capacity_pages);
    assert(contains(decision.target, 1));
    assert(contains(decision.target, 6));
    assert(decision.unavailable_evidence == 6);
    const auto repeat = llama_kv_policy_decide(controller_trace, controller_config);
    assert(decision.target == repeat.target);
    const auto retained = llama_kv_policy_decide(controller_trace, controller_config, decision.target);
    assert(retained.keeps.size() == decision.target.size());
    controller_config.capacity_pages = 1;
    assert(llama_kv_policy_decide(controller_trace, controller_config).status == llama_kv_policy_status::pin_overflow);

    // Automatic policy quotas are capacity-relative. Exercise several admitted
    // capacities, including values below any former nominal quota.
    llama_kv_policy_trace dynamic_trace;
    dynamic_trace.epoch = 8;
    dynamic_trace.write_page = 1;
    dynamic_trace.pages.resize(16);
    for (size_t i = 0; i < dynamic_trace.pages.size(); ++i) {
        dynamic_trace.pages[i] = page(i + 1);
    }
    llama_kv_policy_controller_config dynamic_config;
    for (const uint32_t capacity : { 1u, 2u, 3u, 5u, 8u }) {
        dynamic_config.capacity_pages = capacity;
        const auto dynamic = llama_kv_policy_decide(dynamic_trace, dynamic_config);
        assert(dynamic.status == llama_kv_policy_status::ok);
        assert(dynamic.target.size() == capacity);
    }

    fake_prefetch_backend fake;
    llama_kv_prefetch_config prefetch_config;
    prefetch_config.max_queued_pages = 4;
    prefetch_config.max_queued_bytes = 40;
    prefetch_config.max_events = 2;
    prefetch_config.max_pinned_slots = 3;
    prefetch_config.staging_slots = 2;
    prefetch_config.wait_budget_steps = 2;
    llama_kv_prefetch_backend prefetch_backend;
    prefetch_backend.context = &fake;
    prefetch_backend.submit = fake_prefetch_backend::submit;
    prefetch_backend.poll = fake_prefetch_backend::poll;
    prefetch_backend.cancel = fake_prefetch_backend::cancel;
    prefetch_backend.publish_complete = fake_prefetch_backend::publish;
    prefetch_backend.host_available = fake_prefetch_backend::host_available;
    prefetch_backend.reseal_dirty = fake_prefetch_backend::reseal;
    prefetch_backend.evict_clean = fake_prefetch_backend::evict;
    prefetch_backend.timestamp_us = fake_prefetch_backend::timestamp;
    llama_kv_prefetch_status prefetch_status;
    auto scheduler = llama_kv_prefetch_scheduler::create(
        prefetch_config, prefetch_backend, prefetch_status);
    assert(scheduler && prefetch_status == llama_kv_prefetch_status::ok);
    const auto intent = [](uint64_t id, uint64_t generation, bool required = false) {
        return llama_kv_prefetch_intent { id, generation, 8, 10, 1, required };
    };
    fake.attention_active = true;
    assert(scheduler->enqueue(intent(10, 1, true)) == llama_kv_prefetch_status::ok);
    assert(scheduler->enqueue(intent(11, 1)) == llama_kv_prefetch_status::ok);
    assert(scheduler->enqueue(intent(12, 1)) == llama_kv_prefetch_status::event_full);
    assert(scheduler->queued_pages() == 1 && scheduler->pinned_slots() == 3);
    assert(fake.overlap_seen); // upload submission overlapped the fake attention window

    // The second ticket completes before the first; publication follows event
    // completion rather than queue order and never exposes a partial page.
    fake.complete_next = 2;
    assert(scheduler->advance() == llama_kv_prefetch_status::event_full);
    assert(fake.published.size() == 1 && fake.published[0] == 11);
    fake.complete_next = 1;
    assert(scheduler->advance() == llama_kv_prefetch_status::ok);
    assert(fake.published.size() == 2 && fake.published[1] == 10);

    assert(scheduler->counters().useful_bytes == 24);
    assert(scheduler->counters().aligned_bytes == 30);
    fake.complete_next = 3;
    const auto ready = scheduler->ensure_ready({ intent(12, 1, true) }, { 10, 11 }, 2);
    assert(ready.readiness == llama_kv_prefetch_readiness::waited_ready);
    assert(ready.ready.size() == 1 && ready.ready[0] == 12);
    assert(scheduler->counters().late_waits > 0);
    assert(scheduler->counters().stage_latency_us > 0);

    const auto old_set = scheduler->ensure_ready({ intent(13, 1, true) }, {}, 0);
    assert(old_set.readiness == llama_kv_prefetch_readiness::fallback_larger_union);
    assert(old_set.fallback.size() == 1 && old_set.fallback[0] == 13);

    assert(scheduler->enqueue(intent(20, 1)) == llama_kv_prefetch_status::ok);
    assert(scheduler->enqueue(intent(20, 2)) == llama_kv_prefetch_status::ok);
    assert(!fake.cancelled.empty() && scheduler->counters().stale_generation_rejects > 0);
    assert(scheduler->cancel(13, 1) == llama_kv_prefetch_status::cancelled);
    fake.host_miss = true;
    assert(scheduler->enqueue(intent(21, 1)) == llama_kv_prefetch_status::host_miss);
    fake.host_miss = false;
    fake.submit_fail = true;
    assert(scheduler->enqueue(intent(22, 1)) == llama_kv_prefetch_status::transfer_failed);
    fake.submit_fail = false;
    assert(!fake.tickets.empty());
    fake.fail_poll_ticket = fake.tickets.back().value;
    assert(scheduler->advance() == llama_kv_prefetch_status::transfer_failed);
    fake.fail_poll_ticket = 0;

    assert(scheduler->evict({ intent(30, 1), false }) == llama_kv_prefetch_status::ok);
    assert(scheduler->evict({ intent(31, 1), true }) == llama_kv_prefetch_status::ok);
    assert(fake.resealed.size() == 1 && fake.evicted.size() == 2);
    assert(scheduler->counters().reseals == 1 && scheduler->counters().evictions == 2);
    assert(scheduler->enqueue(intent(23, 1)) == llama_kv_prefetch_status::ok);
    const size_t cancelled_before_shutdown = fake.cancelled.size();
    scheduler->shutdown();
    assert(scheduler->stopped() && scheduler->queued_pages() == 0 && scheduler->active_events() == 0);
    if (fake.cancelled.size() <= cancelled_before_shutdown) return 1;
    scheduler->shutdown(); // teardown and cancellation are idempotent

    std::cout << "kv policy checks passed\n";
}
