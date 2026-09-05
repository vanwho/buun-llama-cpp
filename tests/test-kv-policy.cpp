#include "llama-kv-policy.h"
#include "llama-kv-live-policy.h"
#include "llama-kv-live-lifecycle.h"
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
    id.model_identity = 1;
    id.topology_identity = 1;
    id.codec_digest = 1;
    id.codebook_digest = 1;
    id.rotation_digest = 1;
    id.meansub_digest = 1;
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
    static bool recheck(void *, const llama_kv_residency_completion &) noexcept {
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
    cold.age = 0;
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
    if (!llama_kv_residency_build_transfer_plan(
            llama_kv_residency_transfer_direction::h2d_promotion,
            { page }, 4, {}, output)) return {};
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
    const auto initial_replace = table.replace(initial, live_resident(0, 0));
    assert(initial_replace == llama_kv_residency_status::ok);
    const auto initial_publish = table.publish(initial);
    assert(initial_publish == llama_kv_residency_status::ok);
    (void) initial_replace;
    (void) initial_publish;

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
    transport.recheck = live_transfer_fake::recheck;

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
    const auto failed_replace = failed_table.replace(failed_initial, live_resident(0, 0));
    assert(failed_replace == llama_kv_residency_status::ok);
    const auto failed_publish = failed_table.publish(failed_initial);
    assert(failed_publish == llama_kv_residency_status::ok);
    (void) failed_replace;
    (void) failed_publish;
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
    failed_transport.recheck = live_transfer_fake::recheck;
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

struct fake_live_lifecycle_hooks {
    uint32_t companion_publishes = 0;
    uint32_t companion_rollbacks = 0;
    uint32_t lifecycle_events = 0;
    bool reject_publish = false;
    bool reject_event = false;

    static bool publish(void * opaque,
            const llama_kv_live_lifecycle_generation &,
            const llama_kv_live_lifecycle_frontier &) noexcept {
        auto & fake = *static_cast<fake_live_lifecycle_hooks *>(opaque);
        ++fake.companion_publishes;
        return !fake.reject_publish;
    }

    static void rollback(void * opaque,
            const llama_kv_live_lifecycle_generation &,
            const llama_kv_live_lifecycle_frontier &) noexcept {
        ++static_cast<fake_live_lifecycle_hooks *>(opaque)->companion_rollbacks;
    }

    static bool event(void * opaque, llama_kv_live_lifecycle_event,
            const llama_kv_live_lifecycle_generation &) noexcept {
        auto & fake = *static_cast<fake_live_lifecycle_hooks *>(opaque);
        ++fake.lifecycle_events;
        return !fake.reject_event;
    }
};

static bool test_live_lifecycle() {
#define LIVE_CHECK(expression) do { if (!(expression)) { std::cerr << "live lifecycle check failed: " << #expression << "\n"; return false; } } while (false)
    fake_prefetch_backend prefetch_fake;
    llama_kv_live_lifecycle_config config;
    config.prefetch.max_queued_pages = 8;
    config.prefetch.max_queued_bytes = 128;
    config.prefetch.max_events = 2;
    config.prefetch.max_pinned_slots = 4;
    config.prefetch.staging_slots = 2;
    config.prefetch.prefetch_depth = 3;
    config.prefetch.wait_budget_steps = 1;
    fake_live_lifecycle_hooks hooks_fake;
    llama_kv_live_lifecycle_hooks hooks;
    hooks.context = &hooks_fake;
    hooks.publish_companions = fake_live_lifecycle_hooks::publish;
    hooks.rollback_companions = fake_live_lifecycle_hooks::rollback;
    hooks.event = fake_live_lifecycle_hooks::event;
    llama_kv_prefetch_backend backend;
    backend.context = &prefetch_fake;
    backend.submit = fake_prefetch_backend::submit;
    backend.poll = fake_prefetch_backend::poll;
    backend.cancel = fake_prefetch_backend::cancel;
    backend.publish_complete = fake_prefetch_backend::publish;
    backend.timestamp_us = fake_prefetch_backend::timestamp;

    llama_kv_live_lifecycle_status status;
    auto lifecycle = llama_kv_live_lifecycle::create(config, backend, hooks, status);
    LIVE_CHECK(lifecycle && status == llama_kv_live_lifecycle_status::ready);
    LIVE_CHECK(lifecycle->start({ 1, 0, 1, 1 }) == llama_kv_live_lifecycle_status::ready);
    LIVE_CHECK(lifecycle->set_frontier({ 1, 1, 1, 2, 1, 1 }) ==
           llama_kv_live_lifecycle_status::ready);
    const auto committed_frontier = lifecycle->frontier();
    LIVE_CHECK(lifecycle->set_frontier({ 0, 0, 0, 1, 0, 1 }) ==
           llama_kv_live_lifecycle_status::stale_generation);
    LIVE_CHECK(lifecycle->frontier().target_tokens == committed_frontier.target_tokens &&
           lifecycle->frontier().speculative_accepted ==
               committed_frontier.speculative_accepted);

    prefetch_fake.attention_active = true;
    const auto future = [](uint64_t id, uint32_t priority) {
        return llama_kv_prefetch_intent { id, 0, 8, 10, priority, false };
    };
    LIVE_CHECK(lifecycle->prefetch({ future(100, 1), future(101, 3), future(100, 9) }) ==
           llama_kv_live_lifecycle_status::ready);
    LIVE_CHECK(prefetch_fake.submitted.size() == 2 && prefetch_fake.submitted[0] == 100 &&
           prefetch_fake.submitted[1] == 101);
    LIVE_CHECK(prefetch_fake.overlap_seen);
    prefetch_fake.complete_next = 1;
    LIVE_CHECK(lifecycle->advance() == llama_kv_live_lifecycle_status::ready);
    const auto ready = lifecycle->ensure_ready({ future(100, 1) }, { 101 }, 0);
    LIVE_CHECK(ready.status == llama_kv_live_lifecycle_status::ready);
    LIVE_CHECK(ready.prefetch.readiness == llama_kv_prefetch_readiness::ready);
    LIVE_CHECK(lifecycle->ensure_ready({ { 102, 99, 8, 10, 1, true } }, {}, 0).status ==
           llama_kv_live_lifecycle_status::stale_generation);

    llama_kv_residency_table table(2);
    auto initial = table.begin();
    LIVE_CHECK(table.replace(initial, live_resident(0, 0)) == llama_kv_residency_status::ok);
    LIVE_CHECK(table.publish(initial) == llama_kv_residency_status::ok);
    live_transfer_fake transfer_fake;
    auto pool_backend = live_pool_backend(transfer_fake);
    llama_kv_residency_pool_status pool_status;
    auto pool = llama_kv_residency_pool::create({ 2, 64, 4, 4, 1024 },
                                                 pool_backend, pool_status);
    LIVE_CHECK(pool && pool_status == llama_kv_residency_pool_status::ok);
    vbr_h2d_status ring_status;
    auto ring = vbr_h2d_chunk_ring::create({ {} }, 128, 32, ring_status);
    LIVE_CHECK(ring && ring_status == vbr_h2d_status::ok);
    llama_kv_residency_transfer_transport transport;
    transport.upload_ring = ring.get();
    transport.context = &transfer_fake;
    transport.host_read = live_transfer_fake::host_read;
    transport.recheck = live_transfer_fake::recheck;
    const auto result = lifecycle->apply_policy(
            table, *pool, live_boundary(table.snapshot(), live_promotion()),
            pool_backend, transport);
    LIVE_CHECK(result.status == llama_kv_live_lifecycle_status::committed);
    LIVE_CHECK(result.companion_published && hooks_fake.companion_publishes == 1);
    LIVE_CHECK(table.snapshot().epoch() == 2);

    fake_live_lifecycle_hooks rejected_hooks_fake;
    rejected_hooks_fake.reject_publish = true;
    auto rejected_hooks = hooks;
    rejected_hooks.context = &rejected_hooks_fake;
    auto rejected = llama_kv_live_lifecycle::create(config, backend, rejected_hooks, status);
    LIVE_CHECK(rejected && rejected->start({ 1, 0, 1, 1 }) == llama_kv_live_lifecycle_status::ready);
    LIVE_CHECK(rejected->set_frontier({ 1, 1, 1, 0, 0, 0 }) == llama_kv_live_lifecycle_status::ready);
    llama_kv_residency_table rejected_table(2);
    auto rejected_initial = rejected_table.begin();
    LIVE_CHECK(rejected_table.replace(rejected_initial, live_resident(0, 0)) == llama_kv_residency_status::ok);
    LIVE_CHECK(rejected_table.publish(rejected_initial) == llama_kv_residency_status::ok);
    live_transfer_fake rejected_transfer_fake;
    auto rejected_pool_backend = live_pool_backend(rejected_transfer_fake);
    auto rejected_pool = llama_kv_residency_pool::create(
            { 2, 64, 4, 4, 1024 }, rejected_pool_backend, pool_status);
    LIVE_CHECK(rejected_pool);
    auto rejected_ring = vbr_h2d_chunk_ring::create({ {} }, 128, 32, ring_status);
    LIVE_CHECK(rejected_ring);
    llama_kv_residency_transfer_transport rejected_transport;
    rejected_transport.upload_ring = rejected_ring.get();
    rejected_transport.context = &rejected_transfer_fake;
    rejected_transport.host_read = live_transfer_fake::host_read;
    rejected_transport.recheck = live_transfer_fake::recheck;
    const auto rejected_result = rejected->apply_policy(
            rejected_table, *rejected_pool,
            live_boundary(rejected_table.snapshot(), live_promotion()),
            rejected_pool_backend, rejected_transport);
    LIVE_CHECK(rejected_result.status == llama_kv_live_lifecycle_status::companion_rejected);
    LIVE_CHECK(!rejected_result.policy.published && rejected_result.companion_rolled_back);
    LIVE_CHECK(rejected_table.snapshot().epoch() == 1 && rejected_hooks_fake.companion_rollbacks == 1);

    auto incomplete = live_boundary(table.snapshot(), live_promotion());
    incomplete.pages.front().record.id.model_identity = 0;
    const auto incomplete_result = lifecycle->apply_policy(
            table, *pool, incomplete, pool_backend, transport);
    LIVE_CHECK(incomplete_result.status == llama_kv_live_lifecycle_status::stale_generation);
    LIVE_CHECK(table.snapshot().epoch() == 2 && hooks_fake.companion_publishes == 1);

    auto wrong_version = live_boundary(table.snapshot(), live_promotion());
    wrong_version.version++;
    const auto wrong_version_result = lifecycle->apply_policy(
            table, *pool, wrong_version, pool_backend, transport);
    LIVE_CHECK(wrong_version_result.status == llama_kv_live_lifecycle_status::invalid_argument);
    LIVE_CHECK(table.snapshot().epoch() == 2);

    LIVE_CHECK(lifecycle->prefetch({ future(103, 1) }) == llama_kv_live_lifecycle_status::ready);
    const auto cancelled_before = prefetch_fake.cancelled.size();
    const auto published_before_reuse = prefetch_fake.published.size();
    LIVE_CHECK(lifecycle->prompt_adopt({ 1, 0, 1, 2 }) == llama_kv_live_lifecycle_status::ready);
    LIVE_CHECK(prefetch_fake.cancelled.size() > cancelled_before);
    prefetch_fake.complete_next = prefetch_fake.tickets.back().value;
    LIVE_CHECK(lifecycle->advance() == llama_kv_live_lifecycle_status::ready);
    LIVE_CHECK(prefetch_fake.published.size() == published_before_reuse);
    LIVE_CHECK(lifecycle->slot_reuse({ 1, 0, 1, 3 }) == llama_kv_live_lifecycle_status::ready);
    const auto fallback = lifecycle->ensure_ready({ future(104, 1) }, { 101 }, 0);
    LIVE_CHECK(fallback.status == llama_kv_live_lifecycle_status::reuse_old_hot_set);
    LIVE_CHECK(lifecycle->checkpoint_save() == llama_kv_live_lifecycle_status::ready);
    LIVE_CHECK(lifecycle->checkpoint_restore({ 1, 0, 1, 4 }) == llama_kv_live_lifecycle_status::ready);
    LIVE_CHECK(lifecycle->clear() == llama_kv_live_lifecycle_status::ready);
    LIVE_CHECK(hooks_fake.lifecycle_events >= 4);
    lifecycle->shutdown();
    LIVE_CHECK(lifecycle->stopped());
    const auto & counters = lifecycle->prefetch_counters();
    std::cout << "live lifecycle trace submitted=" << counters.submitted
              << " completed=" << counters.completed
              << " late_waits=" << counters.late_waits
              << " cancellations=" << counters.cancellations
              << " useful_bytes=" << counters.useful_bytes
              << " aligned_bytes=" << counters.aligned_bytes
              << " overlap=" << (prefetch_fake.overlap_seen ? 1 : 0)
              << " companion_publishes=" << hooks_fake.companion_publishes
              << " companion_rollbacks=" << hooks_fake.companion_rollbacks << "\n";
    LIVE_CHECK(llama_kv_live_lifecycle::create(
            llama_kv_live_lifecycle_config { 2, {} }, backend, hooks, status) == nullptr);
    LIVE_CHECK(status == llama_kv_live_lifecycle_status::unsupported_slots);

    fake_live_lifecycle_hooks failed_event_fake;
    failed_event_fake.reject_event = true;
    auto failed_event_hooks = hooks;
    failed_event_hooks.context = &failed_event_fake;
    auto failed_event = llama_kv_live_lifecycle::create(
            config, backend, failed_event_hooks, status);
    LIVE_CHECK(failed_event && failed_event->start({ 1, 0, 1, 1 }) ==
           llama_kv_live_lifecycle_status::ready);
    LIVE_CHECK(failed_event->set_frontier({ 1, 1, 1, 1, 1, 0 }) ==
           llama_kv_live_lifecycle_status::ready);
    LIVE_CHECK(failed_event->prefetch({ future(105, 1) }) ==
           llama_kv_live_lifecycle_status::ready);
    LIVE_CHECK(failed_event->clear() == llama_kv_live_lifecycle_status::companion_rejected);
    LIVE_CHECK(!failed_event->active() && failed_event->frontier().target_tokens == 0);
    LIVE_CHECK(failed_event->advance() == llama_kv_live_lifecycle_status::stale_generation);
#undef LIVE_CHECK
    return true;
}

int main() {
    test_live_policy_publication();
    if (!test_live_lifecycle()) return 1;

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
    assert(result.retrieve.size() == 4);
    assert(result.victims.size() == 2);
    assert(result.victims[0] == 2); // observed low attention wins despite dirty cost
    assert(result.victims[1] == 1); // remaining resident follows deterministic evidence ordering

    trace.capacity_pages = 1;
    assert(llama_kv_policy_replay(trace).status == llama_kv_policy_status::ok);
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
    dynamic_trace.pages.resize(512);
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

    const auto check_replay_candidate = [&](uint32_t recent_ratio,
                                            uint32_t structural_ratio,
                                            uint32_t historical_ratio,
                                            uint32_t transient_ratio,
                                            uint32_t ema_weight,
                                            uint32_t peak_weight,
                                            uint32_t frequency_weight,
                                            uint32_t recency_weight,
                                            uint64_t hysteresis_q) {
        auto candidate = llama_kv_policy_release_defaults();
        candidate.recent_ratio = recent_ratio;
        candidate.structural_ratio = structural_ratio;
        candidate.historical_ratio = historical_ratio;
        candidate.transient_ratio = transient_ratio;
        candidate.attention_ema_weight = ema_weight;
        candidate.recent_peak_weight = peak_weight;
        candidate.frequency_weight = frequency_weight;
        candidate.recency_weight = recency_weight;
        candidate.hysteresis_q = hysteresis_q;
        for (const uint32_t capacity : { 1u, 2u, 3u, 5u, 8u, 16u, 304u }) {
            candidate.capacity_pages = capacity;
            const auto result = llama_kv_policy_decide(dynamic_trace, candidate);
            if (result.status != llama_kv_policy_status::ok ||
                    result.target.size() != capacity) return false;
        }
        return true;
    };
    if (!check_replay_candidate(300000, 200000, 350000, 150000,
                                250000, 250000, 250000, 250000, 0) ||
        !check_replay_candidate(400000, 100000, 350000, 150000,
                                500000, 200000, 150000, 150000, 100000) ||
        !check_replay_candidate(350000, 150000, 350000, 150000,
                                600000, 200000, 100000, 100000, 200000)) return 1;

    const auto release = llama_kv_policy_release_defaults(5);
    assert(release.capacity_pages == 5);
    assert(release.recent_pages == 0 && release.structural_pages == 0 &&
           release.historical_pages == 0 && release.transient_pages == 0);
    assert(release.recent_ratio + release.structural_ratio +
           release.historical_ratio + release.transient_ratio == LLAMA_KV_POLICY_RATIO_SCALE);
    assert(release.recent_min_pages == 1 && release.transient_min_pages == 1);
    assert(release.attention_ema_weight == 500000 &&
           release.recent_peak_weight == 200000 && release.frequency_weight == 150000 &&
           release.recency_weight == 150000);
    assert(release.hysteresis_q == 100000);
    const auto release_decision = llama_kv_policy_decide(dynamic_trace, release);
    assert(release_decision.status == llama_kv_policy_status::ok &&
           release_decision.target.size() == release.capacity_pages);
    auto invalid_weights = release;
    invalid_weights.attention_ema_weight = 0;
    invalid_weights.recent_peak_weight = 0;
    invalid_weights.frequency_weight = 0;
    invalid_weights.recency_weight = 0;
    if (llama_kv_policy_decide(dynamic_trace, invalid_weights).status !=
            llama_kv_policy_status::invalid_trace) return 1;

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
