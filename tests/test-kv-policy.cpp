#include "llama-kv-policy.h"
#include "llama-kv-live-policy.h"
#include "llama-kv-live-lifecycle.h"
#include "llama-kv-prefetch.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

#undef NDEBUG
#include <cassert>

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

static void test_residency_snapshot_reconciles_stale_slots() {
    llama_kv_residency_table table(1);
    auto initial = table.begin();
    const auto current = live_resident(26, 0);
    assert(table.replace(initial, current) == llama_kv_residency_status::ok);
    assert(table.publish(initial) == llama_kv_residency_status::ok);

    auto stale_catalogue = live_resident(24, 0);
    std::vector<llama_kv_page_record> records { current, stale_catalogue };
    llama_kv_live_policy_reconcile_residency_records(table.snapshot(), records);
    assert(records[0].id == current.id && records[0].physical_slot == 0);
    assert(records[1].id == stale_catalogue.id &&
           records[1].physical_slot == UINT32_MAX &&
           records[1].state == llama_kv_page_state::host_clean &&
           !records[1].dirty && records[1].pin_count == 0);
    assert(records[1].host_valid); // preserve the authenticated cold source bit
    std::cout << "query_commit_reconciles_stale_resident_slot=pass\n";
}

struct live_transfer_fake {
    bool fail_issue = false;
    std::vector<std::vector<uint8_t>> copied;

    static bool map_slot(void *, uint32_t) noexcept { return true; }
    static bool drop_slot(void *, uint32_t) noexcept { return true; }
    static bool issue(void * opaque, llama_kv_residency_transfer_direction,
                      const llama_kv_residency_completion &, uint64_t,
                      void * host, size_t size, uint64_t, bool) noexcept {
        auto & self = *static_cast<live_transfer_fake *>(opaque);
        if (self.fail_issue) return false;
        try {
            self.copied.emplace_back(static_cast<uint8_t *>(host),
                    static_cast<uint8_t *>(host) + size);
            return true;
        } catch (...) {
            return false;
        }
    }
    static bool complete(void *, uint64_t) noexcept { return true; }
    static void cancel(void *, uint64_t) noexcept {}
    static bool host_read(void *, uint32_t page_index, uint64_t, uint8_t * destination,
                          size_t size) noexcept {
        std::fill(destination, destination + size, uint8_t(7 + page_index));
        return true;
    }
    static bool recheck(void *, const llama_kv_residency_completion &) noexcept {
        return true;
    }
    static bool drop_clean(void *, const llama_kv_page_record &) noexcept { return true; }
    static bool restore_clean(void *, const llama_kv_page_record &) noexcept { return true; }
};

struct live_cancel_phase {
    llama_kv_residency_transaction_phase cancel_at =
        llama_kv_residency_transaction_phase::_count;
    uint32_t callbacks = 0;

    static bool phase(void * opaque,
                      llama_kv_residency_transaction_phase value) noexcept {
        auto & self = *static_cast<live_cancel_phase *>(opaque);
        if (value != self.cancel_at) return true;
        ++self.callbacks;
        return false;
    }
};

static llama_kv_residency_transaction_hooks live_transaction_hooks() {
    llama_kv_residency_transaction_hooks hooks;
    hooks.drop_clean = live_transfer_fake::drop_clean;
    hooks.restore_clean = live_transfer_fake::restore_clean;
    return hooks;
}

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
    // The previous resident set is an anchor-biased safe table.  The cold
    // retrieval hit must supersede that fallback even though no page has yet
    // published an attention sample.
    boundary.previous_target = { live_page_id(0), live_page_id(2) };
    boundary.transaction.staging_capacity = 32;
    boundary.transaction.max_h2d_pages = 2;
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
    other.record = live_resident(2, 1);
    other.age = 3;
    other.recency = 1;
    boundary.pages.push_back(other);
    return boundary;
}

static llama_kv_residency_transfer_plan live_promotion_for(
        uint32_t logical, uint32_t slot);

static llama_kv_residency_transfer_plan live_promotion() {
    return live_promotion_for(1, 1);
}

static llama_kv_residency_transfer_plan live_promotions() {
    std::vector<llama_kv_residency_transfer_page> pages;
    for (const auto & target : std::array<std::pair<uint32_t, uint32_t>, 3>{
            std::pair<uint32_t, uint32_t>{ 1, 1 },
            std::pair<uint32_t, uint32_t>{ 3, 2 },
            std::pair<uint32_t, uint32_t>{ 5, 3 } }) {
        llama_kv_residency_transfer_page page;
        page.page = live_page_id(target.first);
        page.table_epoch = 1;
        page.physical_slot = target.second;
        page.runs.push_back({ UINT32_MAX, 0, 0, 0, 0, 8, 1, 0, 0 });
        pages.push_back(std::move(page));
    }
    llama_kv_residency_transfer_plan output;
    if (!llama_kv_residency_build_transfer_plan(
            llama_kv_residency_transfer_direction::h2d_promotion,
            pages, 4, {}, output)) return {};
    return output;
}

static llama_kv_residency_transfer_plan live_promotion_for(
        uint32_t logical, uint32_t slot) {
    llama_kv_residency_transfer_page page;
    page.page = live_page_id(logical);
    page.table_epoch = 1;
    page.physical_slot = slot;
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
    assert(table.replace(initial, live_resident(2, 1)) == llama_kv_residency_status::ok);
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
    llama_kv_live_policy_trace trace;
    llama_kv_policy_trace policy_trace;
    std::vector<llama_kv_policy_page> policy_pages;
    assert(llama_kv_live_policy_build_trace(
            boundary, trace, policy_trace, policy_pages));
    assert(policy_pages.size() == boundary.pages.size());
    // Routing summary selection is retrieval evidence only.  A selected cold
    // page is still unobserved until completed attention publishes mass.
    assert(!policy_pages[1].attention_observed);
    assert(policy_pages[1].attention_layer == 0);
    auto result = llama_kv_live_policy_apply(
        table, *pool, boundary, backend, transport, live_transaction_hooks());
    if (result.status != llama_kv_live_policy_status::committed) {
        std::fprintf(stderr, "live policy publication status=%d target=%zu policy=%d tx=%d phase=%d\n",
            int(result.status), result.target_pages.size(), int(result.policy.status),
            int(result.transaction.status), int(result.transaction.failed_phase));
        for (const auto & page : result.target_pages) {
            std::fprintf(stderr, "  target logical=%u slot=%u\n", page.id.logical_page,
                page.physical_slot);
        }
        std::fprintf(stderr, "  transfer plans=%zu staging=%llu pool=%u\n",
            boundary.transaction.transfers.size(),
            (unsigned long long) boundary.transaction.staging_capacity, pool->slot_capacity());
        for (const auto & plan : boundary.transaction.transfers) {
            std::fprintf(stderr, "  plan pages=%zu runs=%zu events=%u bytes=%llu\n",
                plan.pages.size(), plan.runs.size(), plan.event_count,
                (unsigned long long) plan.useful_bytes);
            for (const auto & page : plan.pages) std::fprintf(stderr,
                "    transfer logical=%u slot=%u\n", page.page.logical_page,
                page.physical_slot);
        }
        for (const auto & plan : boundary.transaction.transfers) for (const auto & page : plan.pages) {
            std::fprintf(stderr, "  transfer logical=%u slot=%u\n", page.page.logical_page,
                page.physical_slot);
        }
        for (const auto & plan : boundary.transaction.transfers) for (const auto & page : plan.pages) {
            std::fprintf(stderr, "  transfer logical=%u slot=%u\n", page.page.logical_page,
                page.physical_slot);
        }
    }
    assert(result.status == llama_kv_live_policy_status::committed);
    assert(result.published && result.published_epoch == 2);
    assert(result.trace.hot_capacity == 2 && result.trace.target_pages == 2);
    assert(result.target_pages.size() == 2);
    assert(result.target_pages[0].id == live_page_id(0));
    assert(result.target_pages[1].id == live_page_id(1));
    assert(result.decisions.size() == 3);
    assert(std::any_of(result.decisions.begin(), result.decisions.end(),
        [](const auto & decision) { return decision.victim && decision.id == live_page_id(2); }));
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
    assert(failed_table.replace(failed_initial, live_resident(2, 1)) == llama_kv_residency_status::ok);
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
        failed_backend, failed_transport, live_transaction_hooks());
    assert(failed_result.status == llama_kv_live_policy_status::transaction_failed);
    assert(!failed_result.published && failed_table.snapshot().epoch() == 1 &&
           failed_table.snapshot().pages().size() == 2);
}

static void test_live_policy_cancellation_boundaries() {
    llama_kv_residency_table table(2);
    auto initial = table.begin();
    assert(table.replace(initial, live_resident(0, 0)) == llama_kv_residency_status::ok);
    assert(table.replace(initial, live_resident(2, 1)) == llama_kv_residency_status::ok);
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
    transport.recheck = live_transfer_fake::recheck;

    auto boundary = live_boundary(table.snapshot(), live_promotion());
    const auto original = table.snapshot();
    llama_kv_live_policy_result completed_then_unpublished;
    auto cancel = [&](llama_kv_residency_transaction_phase phase) {
        live_cancel_phase cancellation;
        cancellation.cancel_at = phase;
        auto hooks = live_transaction_hooks();
        hooks.context = &cancellation;
        hooks.phase = live_cancel_phase::phase;
        const size_t copies_before = fake.copied.size();
        const auto result = llama_kv_live_policy_apply(
            table, *pool, boundary, backend, transport, hooks);
        assert(cancellation.callbacks == 1);
        const auto expected_status = phase == llama_kv_residency_transaction_phase::recheck
            ? llama_kv_live_policy_status::stale_snapshot
            : llama_kv_live_policy_status::transaction_failed;
        if (result.status != expected_status) {
            std::fprintf(stderr, "cancellation phase=%d returned status=%d expected=%d published=%d tx_status=%d failed_phase=%d rollback=%d\n",
                int(phase), int(result.status), int(expected_status), int(result.published),
                int(result.transaction.status), int(result.transaction.failed_phase),
                int(result.transaction.rollback_complete));
        }
        assert(result.status == expected_status);
        assert(!result.published && !result.transaction.published &&
            result.transaction.rollback_complete);
        if (phase == llama_kv_residency_transaction_phase::recheck) {
            // The upload was queued and its completion event retired, but a
            // later identity check prevented table publication. This is the
            // queued-without-publication boundary observed on H200: completion
            // must not be mistaken for residency becoming visible.
            assert(result.transaction.h2d_counters.queued > 0);
            assert(result.transaction.h2d_counters.event_completions > 0);
            assert(result.transaction.h2d_counters.copied_useful_bytes > 0);
            assert(!result.published && !result.transaction.published);
            completed_then_unpublished = result;
        }
        const auto current = table.snapshot();
        assert(current.epoch() == original.epoch() &&
            current.pages().size() == original.pages().size());
        for (size_t i = 0; i < original.pages().size(); ++i) {
            assert(current.pages()[i].id == original.pages()[i].id &&
                current.pages()[i].physical_slot == original.pages()[i].physical_slot);
        }
        assert(pool->mapped_slots() == 0);
        return fake.copied.size() - copies_before;
    };

    const size_t before_h2d_copies = cancel(
        llama_kv_residency_transaction_phase::load);
    assert(before_h2d_copies == 0);
    const size_t completed_h2d_copies = cancel(
        llama_kv_residency_transaction_phase::recheck);
    assert(completed_h2d_copies == 1);
    assert(completed_then_unpublished.transaction.h2d_counters.queued > 0 &&
        completed_then_unpublished.transaction.h2d_counters.event_completions > 0 &&
        !completed_then_unpublished.published);

    auto hooks = live_transaction_hooks();
    const auto retry = llama_kv_live_policy_apply(
        table, *pool, boundary, backend, transport, hooks);
    assert(retry.status == llama_kv_live_policy_status::committed && retry.published);
    assert(retry.target_pages.size() == 2);
    assert(retry.target_pages[0].id == live_page_id(0));
    assert(retry.target_pages[1].id == live_page_id(1));
    assert(pool->mapped_slots() == 1);
    assert(fake.copied.size() == 2);
    std::cout << "query_cancel_boundaries=pass before_h2d=atomic "
                 "after_h2d_before_publish=atomic retry_same_slot=pass\n";
}

static void test_live_policy_multi_promotion() {
    llama_kv_residency_table table(4);
    auto initial = table.begin();
    assert(table.replace(initial, live_resident(0, 0)) == llama_kv_residency_status::ok);
    assert(table.replace(initial, live_resident(2, 1)) == llama_kv_residency_status::ok);
    assert(table.publish(initial) == llama_kv_residency_status::ok);

    auto boundary = live_boundary(table.snapshot(), live_promotion());
    boundary.hot_capacity = 4;
    boundary.logical_page_count = 6;
    boundary.previous_target.clear();
    llama_kv_live_policy_page second_cold;
    second_cold.record.id = live_page_id(3);
    second_cold.record.state = llama_kv_page_state::host_clean;
    second_cold.record.host_valid = true;
    second_cold.record.content_version = 9;
    second_cold.age = 0;
    second_cold.recency = 8;
    boundary.pages.push_back(second_cold);
    llama_kv_live_policy_page third_cold;
    third_cold.record.id = live_page_id(5);
    third_cold.record.state = llama_kv_page_state::host_clean;
    third_cold.record.host_valid = true;
    third_cold.record.content_version = 10;
    third_cold.age = 0;
    third_cold.recency = 10;
    boundary.pages.push_back(third_cold);
    boundary.pages[1].record.content_version = 7;
    boundary.retrieval.selected.push_back({
        live_page_id(3), llama_kv_routing_retrieval_reason::summary,
        8.0f, true, false, 2,
    });
    boundary.retrieval.selected.push_back({
        live_page_id(5), llama_kv_routing_retrieval_reason::summary,
        7.0f, true, false, 3,
    });
    boundary.retrieval.query_generation = 3;
    boundary.retrieval.model_identity = live_page_id(0).model_identity;
    boundary.retrieval.session_generation = live_page_id(0).session_generation;
    boundary.retrieval.sequence_generation = live_page_id(0).sequence_generation;
    boundary.retrieval.sequence_id = 0;
    boundary.retrieval.representation_epoch = live_page_id(0).representation_epoch;
    boundary.retrieval.position = 300;
    boundary.query_commit.enabled = true;
    boundary.query_commit.turn_id = 2;
    boundary.query_commit.retrieval_epoch = 1;
    boundary.query_commit.query_generation = 3;
    boundary.query_commit.table_epoch = boundary.snapshot.epoch();
    boundary.query_commit.query_position = 300;
    boundary.query_commit.query_start = 299;
    boundary.query_commit.query_end = 301;
    boundary.query_commit.rollback_generation = live_page_id(5).page_generation;
    boundary.query_commit.model_identity = live_page_id(0).model_identity;
    boundary.query_commit.session_generation = live_page_id(0).session_generation;
    boundary.query_commit.sequence_generation = live_page_id(0).sequence_generation;
    boundary.query_commit.representation_epoch = live_page_id(0).representation_epoch;
    boundary.query_commit.sequence_id = 0;
    boundary.query_commit.retrieval_budget = 3;
    boundary.query_commit.generation_budget = 1;
    boundary.query_commit.selected = {
        { live_page_id(1), 7, { 0 } },
        { live_page_id(3), 9, { 0 } },
        { live_page_id(5), 10, { 0 } },
    };
    boundary.transaction.transfers.clear();
    boundary.transaction.transfers.push_back(live_promotions());
    boundary.transaction.max_h2d_pages = 3;

    live_transfer_fake fake;
    auto backend = live_pool_backend(fake);
    llama_kv_residency_pool_status pool_status;
    auto pool = llama_kv_residency_pool::create(
            { 4, 64, 4, 8, 1024 }, backend, pool_status);
    assert(pool && pool_status == llama_kv_residency_pool_status::ok);
    vbr_h2d_status ring_status;
    auto ring = vbr_h2d_chunk_ring::create({ {} }, 128, 32, ring_status);
    assert(ring && ring_status == vbr_h2d_status::ok);
    llama_kv_residency_transfer_transport transport;
    transport.upload_ring = ring.get();
    transport.context = &fake;
    transport.host_read = live_transfer_fake::host_read;
    transport.recheck = live_transfer_fake::recheck;

    auto bounded = boundary;
    bounded.transaction.max_h2d_pages = 2;
    const auto refused = llama_kv_live_policy_apply(
            table, *pool, bounded, backend, transport, live_transaction_hooks());
    if (refused.status != llama_kv_live_policy_status::transaction_failed) {
        std::fprintf(stderr, "three-page bounded refusal status=%d tx=%d phase=%d published=%d\n",
            int(refused.status), int(refused.transaction.status),
            int(refused.transaction.failed_phase), int(refused.published));
    }
    assert(refused.status == llama_kv_live_policy_status::transaction_failed);
    assert(!refused.published && table.snapshot().epoch() == 1);

    const auto result = llama_kv_live_policy_apply(
            table, *pool, boundary, backend, transport, live_transaction_hooks());
    if (result.status != llama_kv_live_policy_status::committed) {
        std::fprintf(stderr, "three-page query commit status=%d tx=%d phase=%d target=%zu loaded=%u\n",
            int(result.status), int(result.transaction.status),
            int(result.transaction.failed_phase), result.target_pages.size(),
            result.transaction.loaded_pages);
        for (const auto & page : result.target_pages) {
            std::fprintf(stderr, "  target logical=%u slot=%u\n", page.id.logical_page,
                page.physical_slot);
        }
    }
    assert(result.status == llama_kv_live_policy_status::committed);
    assert(result.target_pages.size() == 4);
    assert(result.target_pages[1].id == live_page_id(1));
    assert(result.target_pages[2].id == live_page_id(3));
    assert(result.target_pages[3].id == live_page_id(5));
    assert(result.transaction.loaded_pages == 3);
    assert(result.transaction.h2d_counters.copied_useful_bytes == 24);
    assert(fake.copied.size() == 3);
    assert(fake.copied[0] == std::vector<uint8_t>(8, 7));
    assert(fake.copied[1] == std::vector<uint8_t>(8, 8));
    assert(fake.copied[2] == std::vector<uint8_t>(8, 9));
}

static void test_query_commit_authoritative_admission() {
    // The query's current page consumes hot capacity alongside R historical
    // pages and the separately reserved G generation pages.
    assert(llama_kv_query_history_budget(16, 4470, 4551, 256) == 15);
    assert(llama_kv_query_history_budget(16, 4090, 4609, 256) == 12);
    assert(llama_kv_query_history_budget(1, 4470, 4551, 256) == 0);
    assert(llama_kv_query_history_budget(16, -1, 8, 256) == 0);

    llama_kv_query_commit bundle_commit;
    bundle_commit.enabled = true;
    bundle_commit.retrieval_budget = 1;
    assert(llama_kv_query_commit_add_candidate(bundle_commit, live_page_id(1), 7, 0));
    assert(llama_kv_query_commit_add_candidate(bundle_commit, live_page_id(1), 7, 1));
    assert(bundle_commit.selected.size() == 1 &&
           bundle_commit.selected[0].attention_layers == std::vector<uint32_t>({ 0, 1 }));
    auto layer_copy = live_page_id(1);
    layer_copy.attention_layer = 1;
    assert(!llama_kv_query_commit_add_candidate(bundle_commit, layer_copy, 7, 1));

    std::vector<llama_kv_routing_retrieval_entry> ranked_candidates = {
        { live_page_id(2), llama_kv_routing_retrieval_reason::summary,
          1.0f, true, false, 0 },
        { live_page_id(1), llama_kv_routing_retrieval_reason::summary,
          9.0f, true, false, 0 },
    };
    assert(llama_kv_live_policy_rank_query_candidates(ranked_candidates, 1));
    assert(ranked_candidates.size() == 1 && ranked_candidates[0].id == live_page_id(1));

    llama_kv_residency_table table(2);
    auto initial = table.begin();
    auto mutable_page = live_resident(0, 0);
    mutable_page.content_version = 4;
    mutable_page.pin_count = 1;
    assert(table.replace(initial, mutable_page) == llama_kv_residency_status::ok);
    auto prior_resident = live_resident(2, 1);
    prior_resident.content_version = 9;
    assert(table.replace(initial, prior_resident) == llama_kv_residency_status::ok);
    assert(table.publish(initial) == llama_kv_residency_status::ok);

    auto boundary = live_boundary(table.snapshot(), live_promotion_for(1, 1));
    boundary.hot_capacity = 2;
    boundary.logical_page_count = 3;
    boundary.policy.hysteresis_q = UINT64_MAX;
    boundary.pages[0].record.content_version = 4;
    llama_kv_routing_page_attributes query_page_attributes;
    query_page_attributes.id = boundary.pages[0].record.id;
    query_page_attributes.current = true;
    query_page_attributes.structural = true;
    llama_kv_live_policy_apply_routing_attributes(
            boundary.pages[0], query_page_attributes);
    assert(boundary.pages[0].current && boundary.pages[0].structural);
    boundary.pages[1].record.content_version = 7;
    boundary.pages[2].record.content_version = 9;
    // The catalog can report different sealing/lifecycle metadata for the
    // current pinned page. Query admission must retain the published record,
    // not turn that observation into a forbidden pinned-page replacement.
    boundary.pages[0].record.host_valid = false;
    boundary.pages[0].record.state = llama_kv_page_state::gpu_dirty;
    boundary.pages[0].record.dirty = true;
    boundary.query_commit.enabled = true;
    boundary.query_commit.turn_id = 2;
    boundary.query_commit.retrieval_epoch = 1;
    boundary.retrieval.query_generation = 3;
    boundary.retrieval.model_identity = live_page_id(0).model_identity;
    boundary.retrieval.session_generation = live_page_id(0).session_generation;
    boundary.retrieval.sequence_generation = live_page_id(0).sequence_generation;
    boundary.retrieval.sequence_id = 0;
    boundary.retrieval.representation_epoch = live_page_id(0).representation_epoch;
    boundary.retrieval.position = 300;
    boundary.query_commit.query_generation = boundary.retrieval.query_generation;
    boundary.query_commit.table_epoch = boundary.snapshot.epoch();
    boundary.query_commit.query_position = 300;
    boundary.query_commit.query_start = 299;
    boundary.query_commit.query_end = 301;
    boundary.query_commit.rollback_generation = live_page_id(2).page_generation;
    boundary.query_commit.model_identity = live_page_id(0).model_identity;
    boundary.query_commit.session_generation = live_page_id(0).session_generation;
    boundary.query_commit.sequence_generation = live_page_id(0).sequence_generation;
    boundary.query_commit.representation_epoch = live_page_id(0).representation_epoch;
    boundary.query_commit.sequence_id = 0;
    boundary.query_commit.retrieval_budget = 1;
    boundary.query_commit.generation_budget = 1;
    boundary.query_commit.selected = {{ live_page_id(1), 7, { 0, 1 } }};

    std::vector<llama_kv_page_record> prepared;
    assert(llama_kv_live_policy_prepare_query_target(boundary, prepared));
    assert(prepared.size() == 2);
    assert(prepared[0].id == live_page_id(0));
    assert(prepared[0].host_valid == mutable_page.host_valid &&
           prepared[0].state == mutable_page.state &&
           prepared[0].dirty == mutable_page.dirty &&
           prepared[0].pin_count == mutable_page.pin_count);
    assert(prepared[1].id == live_page_id(1));
    assert(prepared[1].physical_slot == 1);

    auto query_overlap = boundary;
    query_overlap.query_commit.query_start = 512;
    query_overlap.query_commit.query_end = 600;
    query_overlap.query_commit.query_position = 599;
    query_overlap.retrieval.position = 599;
    query_overlap.query_commit.selected.clear();
    std::vector<llama_kv_page_record> overlap_target;
    assert(llama_kv_live_policy_prepare_query_target(query_overlap, overlap_target));
    assert(overlap_target.size() == 2 && overlap_target[0].id == live_page_id(0) &&
           overlap_target[1].id == live_page_id(2));

    live_transfer_fake fake;
    auto backend = live_pool_backend(fake);
    llama_kv_residency_pool_status pool_status;
    auto pool = llama_kv_residency_pool::create({ 2, 64, 4, 4, 1024 }, backend, pool_status);
    assert(pool && pool_status == llama_kv_residency_pool_status::ok);
    vbr_h2d_status ring_status;
    auto ring = vbr_h2d_chunk_ring::create({ {} }, 128, 32, ring_status);
    assert(ring && ring_status == vbr_h2d_status::ok);
    llama_kv_residency_transfer_transport transport;
    transport.upload_ring = ring.get();
    transport.context = &fake;
    transport.host_read = live_transfer_fake::host_read;
    transport.recheck = live_transfer_fake::recheck;

    auto stale = boundary;
    stale.query_commit.selected[0].content_version++;
    const auto stale_result = llama_kv_live_policy_apply(table, *pool, stale, backend, transport);
    assert(stale_result.status == llama_kv_live_policy_status::query_capacity_refused);
    assert(!stale_result.published && table.snapshot().epoch() == 1);
    auto overflow = boundary;
    overflow.query_commit.generation_budget = 0;
    const auto overflow_result = llama_kv_live_policy_apply(table, *pool, overflow, backend, transport);
    assert(overflow_result.status == llama_kv_live_policy_status::query_capacity_refused);
    assert(!overflow_result.published && table.snapshot().epoch() == 1);

    const auto committed = llama_kv_live_policy_apply(
            table, *pool, boundary, backend, transport, live_transaction_hooks());
    assert(committed.status == llama_kv_live_policy_status::committed && committed.published);
    assert(committed.target_pages.size() == prepared.size());
    assert(committed.target_pages[0].id == prepared[0].id &&
           committed.target_pages[0].physical_slot == prepared[0].physical_slot);
    assert(committed.target_pages[1].id == prepared[1].id &&
           committed.target_pages[1].physical_slot == prepared[1].physical_slot);
    assert(std::none_of(committed.target_pages.begin(), committed.target_pages.end(),
            [](const auto & page) { return page.id == live_page_id(2); }));

    const auto before_stale = table.snapshot().epoch();
    assert(before_stale == 2);
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
    LIVE_CHECK(lifecycle->observe_query(3, 10,
            { future(100, 1), future(101, 3) }) ==
           llama_kv_live_lifecycle_status::ready);
    const auto predicted = lifecycle->predict_next(3, 11, 2);
    LIVE_CHECK(predicted.size() == 2 && predicted[0].prediction &&
           predicted[0].page_id == 101 && predicted[1].page_id == 100 &&
           predicted[0].source_query_token == 10 &&
           predicted[0].needed_by_token == 11);
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

static llama_kv_prefetch_candidate exact_history_candidate(
        uint32_t logical, uint32_t layer, float peak, float mean, bool cold) {
    llama_kv_prefetch_candidate candidate;
    candidate.identity = live_page_id(logical);
    candidate.identity.page_generation = logical + 1;
    candidate.attention_layer = layer;
    candidate.content_version = candidate.summary_version = 1;
    candidate.provenance = llama_kv_prefetch_candidate::score_kind::exact_mass;
    candidate.peak_probability = peak;
    candidate.mean_probability = mean;
    candidate.cold = cold;
    return candidate;
}

static void test_common_history_competition() {
    std::vector<llama_kv_prefetch_common_history_bundle> ranked;
    std::vector<llama_kv_prefetch_candidate> candidates {
        exact_history_candidate(1, 0, 0.4f, 0.4f, false),
        exact_history_candidate(2, 0, 0.1f, 0.1f, true),
    };
    assert(llama_kv_prefetch_rank_common_history(candidates, 1, ranked));
    auto selected = llama_kv_prefetch_select_common_history(ranked, 1, 1);
    assert(selected.size() == 1 && selected[0].representative.identity.logical_page == 1);

    candidates[1] = exact_history_candidate(2, 0, 0.8f, 0.8f, true);
    assert(llama_kv_prefetch_rank_common_history(candidates, 1, ranked));
    selected = llama_kv_prefetch_select_common_history(ranked, 1, 1);
    assert(selected.size() == 1 && selected[0].representative.identity.logical_page == 2);

    // Equal score/support uses ascending logical identity. Missing layer mass
    // contributes zero: page 3's .6 across two nominated layers beats page 4's
    // .9 from one layer when the participating layer count is two.
    candidates = {
        exact_history_candidate(3, 0, 0.7f, 0.6f, false),
        exact_history_candidate(3, 1, 0.7f, 0.6f, false),
        exact_history_candidate(4, 0, 0.7f, 0.9f, false),
    };
    assert(llama_kv_prefetch_rank_common_history(candidates, 2, ranked));
    assert(ranked[0].representative.identity.logical_page == 3);
    assert(std::fabs(ranked[0].mean_probability_sum / 2.0f - 0.6f) < 1e-6f);
    assert(ranked[0].supporting_layers == 2);

    candidates = {
        exact_history_candidate(8, 0, 0.9f, 0.9f, true),
        exact_history_candidate(7, 0, 0.9f, 0.9f, true),
    };
    assert(llama_kv_prefetch_rank_common_history(candidates, 1, ranked));
    assert(ranked[0].representative.identity.logical_page == 7);

    // A blocked cold page is skipped and the next feasible resident refills H.
    candidates = {
        exact_history_candidate(10, 0, 0.9f, 0.9f, true),
        exact_history_candidate(11, 0, 0.8f, 0.8f, true),
        exact_history_candidate(12, 0, 0.7f, 0.7f, false),
    };
    assert(llama_kv_prefetch_rank_common_history(candidates, 1, ranked));
    bool budget_limited = false;
    selected = llama_kv_prefetch_select_common_history(ranked, 2, 1, &budget_limited);
    assert(budget_limited && selected.size() == 2 &&
            selected[0].representative.identity.logical_page == 10 &&
            selected[1].representative.identity.logical_page == 12);

    // A pinned incumbent is accounted outside historical capacity and remains
    // protected while the one available history slot is competed for.
    const uint32_t pinned_incumbent = 20;
    candidates = { exact_history_candidate(21, 0, 0.1f, 0.1f, false),
        exact_history_candidate(22, 0, 0.8f, 0.8f, true) };
    assert(llama_kv_prefetch_rank_common_history(candidates, 1, ranked));
    selected = llama_kv_prefetch_select_common_history(ranked, 1, 1);
    assert(pinned_incumbent == 20 && selected.size() == 1 &&
            selected[0].representative.identity.logical_page == 22);

    auto legacy = exact_history_candidate(30, 0, 1.0f, 1.0f, true);
    legacy.provenance = llama_kv_prefetch_candidate::score_kind::legacy_rank;
    assert(llama_kv_prefetch_rank_common_history({ legacy }, 1, ranked) && ranked.empty());
}

int main(int argc, char ** argv) {
    test_common_history_competition();
    if (argc == 2 && std::strcmp(argv[1], "--query-commit-authoritative-admission") == 0) {
        test_residency_snapshot_reconciles_stale_slots();
        test_live_policy_publication();
        test_live_policy_cancellation_boundaries();
        test_query_commit_authoritative_admission();
        test_live_policy_multi_promotion();
        std::cout << "query_commit_authoritative_admission=pass\n";
        return 0;
    }
    test_residency_snapshot_reconciles_stale_slots();
    test_live_policy_publication();
    test_live_policy_cancellation_boundaries();
    test_live_policy_multi_promotion();
    test_query_commit_authoritative_admission();
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
    // Leave room for the required queued page and the subsequent probe
    // intents while earlier completed pages remain ready for the test's
    // ordering assertions.
    prefetch_config.max_pinned_slots = 8;
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
    const auto second_advance = scheduler->advance();
    if (second_advance != llama_kv_prefetch_status::ok &&
            second_advance != llama_kv_prefetch_status::event_full &&
            second_advance != llama_kv_prefetch_status::backpressure) {
        std::fprintf(stderr, "second prefetch advance status=%s\n",
            llama_kv_prefetch_status_name(second_advance));
    }
    assert(second_advance == llama_kv_prefetch_status::ok ||
           second_advance == llama_kv_prefetch_status::event_full ||
           second_advance == llama_kv_prefetch_status::backpressure);
    assert(fake.published.size() == 2 && fake.published[1] == 10);

    assert(scheduler->counters().useful_bytes == 24);
    assert(scheduler->counters().aligned_bytes == 30);
    const auto required_submit = std::find(fake.submitted.begin(), fake.submitted.end(), 12);
    assert(required_submit != fake.submitted.end());
    fake.complete_next = fake.tickets[size_t(required_submit - fake.submitted.begin())].value;
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
