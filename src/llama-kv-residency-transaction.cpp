#include "llama-kv-residency-transaction.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace {

bool same_logical_page(const llama_kv_page_id & a,
                       const llama_kv_page_id & b) noexcept {
    return a.session_generation == b.session_generation &&
           a.sequence_id == b.sequence_id &&
           a.sequence_generation == b.sequence_generation &&
           a.logical_page == b.logical_page;
}

bool plan_contains_page(
        const llama_kv_residency_transfer_plan & plan,
        const llama_kv_page_id & page) noexcept {
    for (const auto & item : plan.pages) {
        if (item.page == page) {
            return true;
        }
    }
    return false;
}

const llama_kv_residency_transfer_plan * find_page_plan(
        const std::vector<llama_kv_residency_transfer_plan> & plans,
        const llama_kv_page_id & page,
        llama_kv_residency_transfer_direction direction) noexcept {
    for (const auto & plan : plans) {
        if (plan.direction == direction && plan_contains_page(plan, page)) {
            return &plan;
        }
    }
    return nullptr;
}

uint64_t page_plan_bytes(
        const llama_kv_residency_transfer_plan & plan,
        const llama_kv_page_id & page) noexcept {
    uint64_t total = 0;
    for (const auto & item : plan.pages) {
        if (item.page != page) {
            continue;
        }
        for (const auto & run : item.runs) {
            if (run.useful_bytes() > std::numeric_limits<uint64_t>::max() - total) {
                return 0;
            }
            total += run.useful_bytes();
        }
    }
    return total;
}

bool call_phase(
        const llama_kv_residency_transaction_hooks & hooks,
        llama_kv_residency_transaction_phase phase) noexcept {
    return !hooks.phase || hooks.phase(hooks.context, phase);
}

bool contains_page(
        const std::vector<llama_kv_page_record> & pages,
        const llama_kv_page_id & id) noexcept {
    return std::any_of(pages.begin(), pages.end(), [&](const auto & page) {
        return page.id == id;
    });
}

} // namespace

const char * llama_kv_residency_transaction_phase_name(
        llama_kv_residency_transaction_phase phase) noexcept {
    switch (phase) {
        case llama_kv_residency_transaction_phase::snapshot: return "snapshot";
        case llama_kv_residency_transaction_phase::plan: return "plan";
        case llama_kv_residency_transaction_phase::reserve: return "reserve";
        case llama_kv_residency_transaction_phase::pin: return "pin";
        case llama_kv_residency_transaction_phase::reseal: return "reseal";
        case llama_kv_residency_transaction_phase::drop: return "drop";
        case llama_kv_residency_transaction_phase::load: return "load";
        case llama_kv_residency_transaction_phase::fence: return "fence";
        case llama_kv_residency_transaction_phase::recheck: return "recheck";
        case llama_kv_residency_transaction_phase::publish: return "publish";
        case llama_kv_residency_transaction_phase::retire: return "retire";
        case llama_kv_residency_transaction_phase::release: return "release";
        case llama_kv_residency_transaction_phase::_count: break;
    }
    return "invalid";
}

const char * llama_kv_residency_transaction_status_name(
        llama_kv_residency_transaction_status status) noexcept {
    switch (status) {
        case llama_kv_residency_transaction_status::committed: return "committed";
        case llama_kv_residency_transaction_status::invalid_argument: return "invalid_argument";
        case llama_kv_residency_transaction_status::shutdown: return "shutdown";
        case llama_kv_residency_transaction_status::stale_generation: return "stale_generation";
        case llama_kv_residency_transaction_status::stale_epoch: return "stale_epoch";
        case llama_kv_residency_transaction_status::insufficient_slots: return "insufficient_slots";
        case llama_kv_residency_transaction_status::all_pinned: return "all_pinned";
        case llama_kv_residency_transaction_status::dirty_victim: return "dirty_victim";
        case llama_kv_residency_transaction_status::missing_host_source: return "missing_host_source";
        case llama_kv_residency_transaction_status::short_page: return "short_page";
        case llama_kv_residency_transaction_status::phase_failed: return "phase_failed";
        case llama_kv_residency_transaction_status::transfer_failed: return "transfer_failed";
        case llama_kv_residency_transaction_status::publish_failed: return "publish_failed";
        case llama_kv_residency_transaction_status::rollback_failed: return "rollback_failed";
        case llama_kv_residency_transaction_status::internal_error: return "internal_error";
        case llama_kv_residency_transaction_status::_count: break;
    }
    return "invalid";
}

llama_kv_residency_transaction_result
llama_kv_residency_execute_transaction(
        llama_kv_residency_table & table,
        llama_kv_residency_pool & pool,
        const llama_kv_residency_transaction_request & request,
        const llama_kv_residency_pool_backend & backend,
        const llama_kv_residency_transfer_transport & transport,
        const llama_kv_residency_transaction_hooks & hooks) noexcept {
    llama_kv_residency_transaction_result result;
    std::vector<llama_kv_page_record> old_pages;
    std::vector<llama_kv_page_record> desired;
    std::vector<llama_kv_page_record> victims;
    std::vector<llama_kv_page_record> dropped;
    std::vector<llama_kv_page_record> loaded;
    std::vector<llama_kv_page_id> pinned;
    std::vector<llama_kv_residency_transfer_claim> claims;
    uint64_t committed_catalog_bytes = 0;
    bool table_published = false;

    const auto fail_phase = [&](llama_kv_residency_transaction_phase phase)
            noexcept {
        result.failed_phase = phase;
        result.status = llama_kv_residency_transaction_status::phase_failed;
    };

    const auto release_pins = [&]() noexcept {
        for (const auto & page : pinned) {
            if (hooks.unpin) {
                hooks.unpin(hooks.context, page);
            }
        }
        pinned.clear();
    };

    const auto release_catalog = [&]() noexcept {
        if (committed_catalog_bytes && request.catalog.release) {
            request.catalog.release(
                request.catalog.context, committed_catalog_bytes);
            committed_catalog_bytes = 0;
        }
    };

    const auto rollback_resources = [&]() noexcept {
        bool complete = true;
        for (auto it = loaded.rbegin(); it != loaded.rend(); ++it) {
            bool dropped_page = false;
            const auto status = pool.drop_logical_page(it->id, dropped_page);
            if (status != llama_kv_residency_pool_status::ok || !dropped_page) {
                complete = false;
            }
        }
        for (auto it = dropped.rbegin(); it != dropped.rend(); ++it) {
            if (!hooks.restore_clean ||
                !hooks.restore_clean(hooks.context, *it)) {
                complete = false;
            }
        }
        for (auto & claim : claims) {
            if (claim.active()) {
                pool.rollback(claim);
            }
        }
        release_catalog();
        release_pins();
        result.rollback_complete = complete;
        if (!complete) {
            result.status = llama_kv_residency_transaction_status::rollback_failed;
        }
    };

    try {
        if (request.shutting_down) {
            result.status = llama_kv_residency_transaction_status::shutdown;
            result.failed_phase = llama_kv_residency_transaction_phase::snapshot;
            result.rollback_complete = true;
            return result;
        }
        if (!call_phase(hooks, llama_kv_residency_transaction_phase::snapshot)) {
            fail_phase(llama_kv_residency_transaction_phase::snapshot);
            result.rollback_complete = true;
            return result;
        }
        const auto snapshot = table.snapshot();
        result.base_epoch = snapshot.epoch();
        old_pages = snapshot.pages();
        desired = request.desired_pages;

        if (!call_phase(hooks, llama_kv_residency_transaction_phase::plan)) {
            fail_phase(llama_kv_residency_transaction_phase::plan);
            result.rollback_complete = true;
            return result;
        }
        if (desired.size() > pool.slot_capacity() ||
            request.transfers.size() > pool.slot_capacity() ||
            request.staging_capacity == 0) {
            result.status = llama_kv_residency_transaction_status::insufficient_slots;
            result.failed_phase = llama_kv_residency_transaction_phase::plan;
            result.rollback_complete = true;
            return result;
        }
        for (size_t i = 0; i < desired.size(); ++i) {
            const auto & page = desired[i];
            if (!llama_kv_page_id_valid(
                        page.id, page.state == llama_kv_page_state::filling_gpu) ||
                page.physical_slot >= pool.slot_capacity() ||
                page.state == llama_kv_page_state::absent ||
                page.state == llama_kv_page_state::invalid ||
                (std::count_if(desired.begin(), desired.end(), [&](const auto & other) {
                    return other.id == page.id;
                }) != 1)) {
                result.status = llama_kv_residency_transaction_status::invalid_argument;
                result.failed_phase = llama_kv_residency_transaction_phase::plan;
                result.rollback_complete = true;
                return result;
            }
            for (size_t j = 0; j < i; ++j) {
                if (same_logical_page(desired[j].id, page.id) ||
                    desired[j].physical_slot == page.physical_slot) {
                    result.status = same_logical_page(desired[j].id, page.id)
                        ? llama_kv_residency_transaction_status::stale_generation
                        : llama_kv_residency_transaction_status::insufficient_slots;
                    result.failed_phase = llama_kv_residency_transaction_phase::plan;
                    result.rollback_complete = true;
                    return result;
                }
            }
        }
        for (const auto & plan : request.transfers) {
            if (plan.pages.empty() || plan.runs.empty() ||
                plan.event_count != plan.runs.size() ||
                plan.direction >= llama_kv_residency_transfer_direction::_count ||
                plan.useful_bytes == 0) {
                result.status = llama_kv_residency_transaction_status::short_page;
                result.failed_phase = llama_kv_residency_transaction_phase::plan;
                result.rollback_complete = true;
                return result;
            }
            for (const auto & page : plan.pages) {
                const bool known_old = contains_page(old_pages, page.page);
                const bool known_desired = contains_page(desired, page.page);
                if ((plan.direction ==
                         llama_kv_residency_transfer_direction::h2d_promotion &&
                     !known_desired) ||
                    (plan.direction !=
                         llama_kv_residency_transfer_direction::h2d_promotion &&
                     !known_old)) {
                    result.status = llama_kv_residency_transaction_status::stale_generation;
                    result.failed_phase = llama_kv_residency_transaction_phase::plan;
                    result.rollback_complete = true;
                    return result;
                }
            }
        }

        auto table_tx = table.begin();
        std::vector<bool> replace_desired(desired.size(), false);
        for (size_t i = 0; i < desired.size(); ++i) {
            for (const auto & old : old_pages) {
                if (desired[i].id == old.id) {
                    desired[i].pin_count = std::max(
                        desired[i].pin_count, old.pin_count);
                    break;
                }
            }
        }
        for (const auto & old : old_pages) {
            size_t exact = desired.size();
            for (size_t i = 0; i < desired.size(); ++i) {
                if (desired[i].id == old.id) {
                    exact = i;
                    break;
                }
                if (same_logical_page(desired[i].id, old.id)) {
                    result.status = llama_kv_residency_transaction_status::stale_generation;
                    result.failed_phase = llama_kv_residency_transaction_phase::plan;
                    result.rollback_complete = true;
                    return result;
                }
            }
            const bool remove = exact == desired.size();
            const bool changed = !remove &&
                (desired[exact].physical_slot != old.physical_slot ||
                 desired[exact].state != old.state ||
                 desired[exact].host_valid != old.host_valid ||
                 desired[exact].dirty != old.dirty ||
                 desired[exact].pin_count != old.pin_count);
            if (!remove && !changed) {
                continue;
            }
            if (old.pin_count != 0) {
                result.status = llama_kv_residency_transaction_status::all_pinned;
                result.failed_phase = llama_kv_residency_transaction_phase::plan;
                result.rollback_complete = true;
                return result;
            }
            const bool has_reseal = find_page_plan(
                request.transfers, old.id,
                llama_kv_residency_transfer_direction::d2h_reseal) != nullptr;
            const bool has_seal = find_page_plan(
                request.transfers, old.id,
                llama_kv_residency_transfer_direction::d2h_seal) != nullptr;
            if (old.dirty && (remove || changed) && !has_reseal) {
                result.status = llama_kv_residency_transaction_status::dirty_victim;
                result.failed_phase = llama_kv_residency_transaction_phase::reseal;
                result.rollback_complete = true;
                return result;
            }
            if (!old.host_valid && (remove || changed) && !has_seal) {
                result.status = llama_kv_residency_transaction_status::missing_host_source;
                result.failed_phase = llama_kv_residency_transaction_phase::reseal;
                result.rollback_complete = true;
                return result;
            }
            const bool needs_drop = remove ||
                desired[exact].physical_slot != old.physical_slot;
            if (needs_drop &&
                ((!old.dirty && old.host_valid) || has_reseal || has_seal)) {
                victims.push_back(old);
            }
            const auto status = table.erase(table_tx, old.id);
            if (status != llama_kv_residency_status::ok) {
                result.status = llama_kv_residency_transaction_status::publish_failed;
                result.failed_phase = llama_kv_residency_transaction_phase::plan;
                result.rollback_complete = true;
                return result;
            }
            if (!remove) {
                replace_desired[exact] = true;
            }
        }
        for (size_t i = 0; i < desired.size(); ++i) {
            const auto old = std::find_if(
                old_pages.begin(), old_pages.end(), [&](const auto & page) {
                    return page.id == desired[i].id;
                });
            const bool existed = old != old_pages.end();
            const bool requires_load = !existed ||
                desired[i].physical_slot != old->physical_slot;
            if (replace_desired[i] || !existed) {
                if (requires_load && !find_page_plan(
                            request.transfers, desired[i].id,
                            llama_kv_residency_transfer_direction::h2d_promotion)) {
                    result.status = llama_kv_residency_transaction_status::missing_host_source;
                    result.failed_phase = llama_kv_residency_transaction_phase::plan;
                    result.rollback_complete = true;
                    return result;
                }
                const auto status = table.replace(table_tx, desired[i]);
                if (status != llama_kv_residency_status::ok) {
                    result.status = llama_kv_residency_transaction_status::insufficient_slots;
                    result.failed_phase = llama_kv_residency_transaction_phase::plan;
                    result.rollback_complete = true;
                    return result;
                }
            }
        }

        if (!call_phase(hooks, llama_kv_residency_transaction_phase::reserve)) {
            fail_phase(llama_kv_residency_transaction_phase::reserve);
            result.rollback_complete = true;
            return result;
        }
        claims.reserve(request.transfers.size());
        for (const auto & plan : request.transfers) {
            llama_kv_residency_transfer_claim claim;
            const auto status = pool.reserve(
                plan, request.staging_capacity, request.catalog, claim);
            if (status != llama_kv_residency_pool_status::ok) {
                result.status = status == llama_kv_residency_pool_status::slot_unavailable
                    ? llama_kv_residency_transaction_status::insufficient_slots
                    : llama_kv_residency_transaction_status::transfer_failed;
                result.failed_phase = llama_kv_residency_transaction_phase::reserve;
                for (auto & prior : claims) {
                    if (prior.active()) pool.rollback(prior);
                }
                result.rollback_complete = true;
                return result;
            }
            claims.push_back(std::move(claim));
        }

        const auto execute_transfer = [&](size_t index) -> bool {
            const auto transfer_result = llama_kv_residency_execute_transfer(
                pool, request.transfers[index], claims[index], backend, transport);
            result.transfer_counters.queued += transfer_result.counters.queued;
            result.transfer_counters.submitted += transfer_result.counters.submitted;
            result.transfer_counters.copied_useful_bytes +=
                transfer_result.counters.copied_useful_bytes;
            result.transfer_counters.copied_aligned_bytes +=
                transfer_result.counters.copied_aligned_bytes;
            result.transfer_counters.waits += transfer_result.counters.waits;
            result.transfer_counters.cancellations +=
                transfer_result.counters.cancellations;
            result.transfer_counters.stale_completions +=
                transfer_result.counters.stale_completions;
            result.transfer_counters.event_completions +=
                transfer_result.counters.event_completions;
            if (transfer_result.status != llama_kv_residency_pool_status::ok) {
                result.status = transfer_result.status ==
                    llama_kv_residency_pool_status::stale_completion
                    ? llama_kv_residency_transaction_status::stale_generation
                    : llama_kv_residency_transaction_status::transfer_failed;
                return false;
            }
            if (request.transfers[index].direction ==
                    llama_kv_residency_transfer_direction::h2d_promotion) {
                for (const auto & page : request.transfers[index].pages) {
                    loaded.push_back({ page.page, page.physical_slot,
                                       llama_kv_page_state::gpu_host_clean,
                                       true, false, 0 });
                    ++result.loaded_pages;
                }
            } else {
                if (request.transfers[index].useful_bytes >
                    std::numeric_limits<uint64_t>::max() -
                        committed_catalog_bytes) {
                    result.status = llama_kv_residency_transaction_status::short_page;
                    return false;
                }
                committed_catalog_bytes += request.transfers[index].useful_bytes;
            }
            return true;
        };

        if (!call_phase(hooks, llama_kv_residency_transaction_phase::pin)) {
            fail_phase(llama_kv_residency_transaction_phase::pin);
            rollback_resources();
            return result;
        }
        for (const auto & page : old_pages) {
            if (!hooks.pin || hooks.pin(hooks.context, page.id)) {
                pinned.push_back(page.id);
            } else {
                fail_phase(llama_kv_residency_transaction_phase::pin);
                rollback_resources();
                return result;
            }
        }
        result.pinned_pages = uint32_t(pinned.size());

        if (!call_phase(hooks, llama_kv_residency_transaction_phase::reseal)) {
            fail_phase(llama_kv_residency_transaction_phase::reseal);
            rollback_resources();
            return result;
        }
        for (size_t i = 0; i < request.transfers.size(); ++i) {
            if (request.transfers[i].direction !=
                    llama_kv_residency_transfer_direction::h2d_promotion &&
                !execute_transfer(i)) {
                result.failed_phase = llama_kv_residency_transaction_phase::reseal;
                rollback_resources();
                return result;
            }
        }
        for (const auto & plan : request.transfers) {
            if (plan.direction != llama_kv_residency_transfer_direction::h2d_promotion) {
                continue;
            }
            for (const auto & page : plan.pages) {
                if (hooks.has_clean_host &&
                    !hooks.has_clean_host(
                        hooks.context, page.page, page_plan_bytes(plan, page.page))) {
                    result.status = llama_kv_residency_transaction_status::missing_host_source;
                    result.failed_phase = llama_kv_residency_transaction_phase::reseal;
                    rollback_resources();
                    return result;
                }
            }
        }

        if (!call_phase(hooks, llama_kv_residency_transaction_phase::drop)) {
            fail_phase(llama_kv_residency_transaction_phase::drop);
            rollback_resources();
            return result;
        }
        if (!victims.empty() && (!hooks.drop_clean || !hooks.restore_clean)) {
            result.status = llama_kv_residency_transaction_status::insufficient_slots;
            result.failed_phase = llama_kv_residency_transaction_phase::drop;
            rollback_resources();
            return result;
        }
        for (const auto & page : victims) {
            if (!hooks.drop_clean(hooks.context, page)) {
                result.status = llama_kv_residency_transaction_status::transfer_failed;
                result.failed_phase = llama_kv_residency_transaction_phase::drop;
                rollback_resources();
                return result;
            }
            dropped.push_back(page);
            ++result.dropped_pages;
        }

        if (!call_phase(hooks, llama_kv_residency_transaction_phase::load)) {
            fail_phase(llama_kv_residency_transaction_phase::load);
            rollback_resources();
            return result;
        }
        for (size_t i = 0; i < request.transfers.size(); ++i) {
            if (request.transfers[i].direction ==
                    llama_kv_residency_transfer_direction::h2d_promotion &&
                !execute_transfer(i)) {
                result.failed_phase = llama_kv_residency_transaction_phase::load;
                rollback_resources();
                return result;
            }
        }

        if (!call_phase(hooks, llama_kv_residency_transaction_phase::fence)) {
            fail_phase(llama_kv_residency_transaction_phase::fence);
            rollback_resources();
            return result;
        }
        if (!call_phase(hooks, llama_kv_residency_transaction_phase::recheck) ||
            (hooks.recheck && !hooks.recheck(
                hooks.context, result.base_epoch, table_tx.pages()))) {
            result.status = llama_kv_residency_transaction_status::stale_epoch;
            result.failed_phase = llama_kv_residency_transaction_phase::recheck;
            rollback_resources();
            return result;
        }
        if (!call_phase(hooks, llama_kv_residency_transaction_phase::retire)) {
            fail_phase(llama_kv_residency_transaction_phase::retire);
            rollback_resources();
            return result;
        }
        if (!call_phase(hooks, llama_kv_residency_transaction_phase::publish)) {
            fail_phase(llama_kv_residency_transaction_phase::publish);
            rollback_resources();
            return result;
        }
        const auto publish_status = table.publish(table_tx);
        if (publish_status != llama_kv_residency_status::ok) {
            result.status = publish_status == llama_kv_residency_status::stale_epoch
                ? llama_kv_residency_transaction_status::stale_epoch
                : llama_kv_residency_transaction_status::publish_failed;
            result.failed_phase = llama_kv_residency_transaction_phase::publish;
            rollback_resources();
            return result;
        }
        table_published = true;
        result.published = true;
        result.published_epoch = result.base_epoch + 1;

        if (hooks.retire) {
            for (const auto & page : dropped) {
                hooks.retire(hooks.context, page);
            }
        }
        if (!call_phase(hooks, llama_kv_residency_transaction_phase::release)) {
            result.status = llama_kv_residency_transaction_status::committed;
        }
        release_pins();
        result.status = llama_kv_residency_transaction_status::committed;
        result.rollback_complete = true;
        return result;
    } catch (...) {
        if (!table_published) {
            rollback_resources();
        }
        result.status = table_published
            ? llama_kv_residency_transaction_status::committed
            : llama_kv_residency_transaction_status::internal_error;
        return result;
    }
}
