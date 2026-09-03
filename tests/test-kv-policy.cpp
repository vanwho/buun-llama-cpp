#include "llama-kv-policy.h"
#include "llama-kv-prefetch.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

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
