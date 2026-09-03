#include "../src/llama-kv-residency.h"

#include <cassert>

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

int main() {
    test_page_geometry();
    test_table_identity_and_snapshot();
    test_rejection_and_stale_publication();
    return 0;
}
