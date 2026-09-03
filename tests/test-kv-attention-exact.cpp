#include "llama-kv-attention-exact.h"
#include "llama-kv-attention-execution.h"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

static llama_kv_page_id page_id(uint32_t logical, uint32_t tokens) {
    llama_kv_page_id id;
    id.sequence_id = 0;
    id.logical_page = logical;
    id.position_begin = llama_pos(logical * VBR_GENERATION_PAGE_CELLS);
    id.position_end = id.position_begin + tokens;
    return id;
}

static std::vector<llama_kv_attention_exact_page> all_pages() {
    // Deliberately permuted. The plan must make hot pages the first partition
    // and restore logical order for the cold wave list.
    return {
        { page_id(2, 256), 7, 256, true,  false },
        { page_id(0, 256), 5, 256, true, false },
        { page_id(3, 17), UINT32_MAX, 17, false, true },
        { page_id(1, 256), UINT32_MAX, 256, false, true },
    };
}

static llama_kv_residency_snapshot resident_snapshot() {
    llama_kv_residency_table table(8);
    auto tx = table.begin();
    llama_kv_page_record page0;
    page0.id = page_id(0, 256);
    page0.physical_slot = 5;
    page0.state = llama_kv_page_state::gpu_host_clean;
    assert(table.replace(tx, page0) == llama_kv_residency_status::ok);

    llama_kv_page_record page2;
    page2.id = page_id(2, 256);
    page2.physical_slot = 7;
    page2.state = llama_kv_page_state::gpu_host_clean;
    assert(table.replace(tx, page2) == llama_kv_residency_status::ok);
    assert(table.publish(tx) == llama_kv_residency_status::ok);
    return table.snapshot();
}

static void test_merge_against_direct() {
    const float logits[] = { -100.0f, 2.0f, -3.0f, 100.0f };
    const float values[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    const uint8_t mask[] = { 1, 1, 0, 1 };
    llama_kv_attention_online_state direct;
    assert(llama_kv_attention_online_state_from_rows(
        logits, values, 4, 2, mask, direct) == llama_kv_attention_exact_status::ok);

    llama_kv_attention_online_state first;
    llama_kv_attention_online_state second;
    assert(llama_kv_attention_online_state_from_rows(
        logits, values, 2, 2, mask, first) == llama_kv_attention_exact_status::ok);
    assert(llama_kv_attention_online_state_from_rows(
        logits + 2, values + 4, 2, 2, mask + 2, second) == llama_kv_attention_exact_status::ok);

    llama_kv_attention_online_state merged;
    assert(llama_kv_attention_online_state_merge(first, second, merged) ==
        llama_kv_attention_exact_status::ok);
    float direct_output[2];
    float merged_output[2];
    assert(llama_kv_attention_online_state_normalize(direct, direct_output, 2) ==
        llama_kv_attention_exact_status::ok);
    assert(llama_kv_attention_online_state_normalize(merged, merged_output, 2) ==
        llama_kv_attention_exact_status::ok);
    assert(std::fabs(direct_output[0] - merged_output[0]) < 1.0e-5f);
    assert(std::fabs(direct_output[1] - merged_output[1]) < 1.0e-5f);

    llama_kv_attention_online_state empty;
    const uint8_t no_rows[] = { 0, 0 };
    assert(llama_kv_attention_online_state_from_rows(
        logits, values, 2, 2, no_rows, empty) == llama_kv_attention_exact_status::empty_partition);
    assert(llama_kv_attention_online_state_merge(empty, direct, merged) ==
        llama_kv_attention_exact_status::ok);
}

struct fake_backend_state {
    uint32_t uploads = 0;
    uint32_t waits = 0;
    uint32_t asynchronous_uploads = 0;
};

static bool upload_wave(
        void * context, const llama_kv_attention_exact_wave & wave,
        uint32_t staging_slot, bool asynchronous) noexcept {
    auto & state = *static_cast<fake_backend_state *>(context);
    assert(wave.contains_cold_pages && staging_slot != UINT32_MAX);
    ++state.uploads;
    if (asynchronous) ++state.asynchronous_uploads;
    return true;
}

static bool wait_wave(
        void * context, const llama_kv_attention_exact_wave & wave) noexcept {
    auto & state = *static_cast<fake_backend_state *>(context);
    assert(wave.contains_cold_pages);
    ++state.waits;
    return true;
}

static bool compute_wave(
        void *, const llama_kv_attention_exact_wave & wave,
        llama_kv_attention_online_state & output) noexcept {
    std::vector<float> logits;
    std::vector<float> values;
    for (const auto & page : wave.pages) {
        for (uint32_t row = 0; row < page.valid_tokens; ++row) {
            const uint32_t logical_position = page.logical_page * VBR_GENERATION_PAGE_CELLS + row;
            logits.push_back(-0.01f * float(logical_position));
            values.push_back(float(page.logical_page + 1));
            values.push_back(1.0f + 0.001f * float(logical_position));
        }
    }
    return llama_kv_attention_online_state_from_rows(
        logits.data(), values.data(), uint32_t(logits.size()), 2, nullptr, output) ==
        llama_kv_attention_exact_status::ok;
}

static void test_plan_and_executor() {
    llama_kv_attention_exact_config config;
    config.logical_page_count = 4;
    config.pages_per_wave = 1;
    config.staging_slots = 1;
    config.page_bytes = 100;
    llama_kv_attention_exact_status status;
    auto plan = llama_kv_attention_exact_wave_plan::build(
        all_pages(), resident_snapshot(), config, status);
    assert(status == llama_kv_attention_exact_status::ok && plan.valid());
    assert(plan.waves().size() == 3);
    assert(plan.waves()[0].pages.size() == 2 && !plan.waves()[0].contains_cold_pages);
    assert(plan.waves()[1].pages[0].logical_page == 1);
    assert(plan.waves()[2].pages[0].logical_page == 3);
    assert(plan.ledger().resident_pages == 2 && plan.ledger().cold_pages == 2);
    assert(plan.ledger().h2d_useful_bytes == 200);
    assert(plan.ledger().peak_staging_pages == 1);

    fake_backend_state fake;
    llama_kv_attention_exact_backend backend;
    backend.context = &fake;
    backend.upload_cold_wave = upload_wave;
    backend.wait_wave = wait_wave;
    backend.compute_wave = compute_wave;
    llama_kv_attention_online_state result;
    llama_kv_attention_exact_executor executor;
    assert(executor.execute(plan, backend, result) == llama_kv_attention_exact_status::ok);
    assert(fake.uploads == 2 && fake.waits == 2);
    assert(fake.asynchronous_uploads == 0);
    assert(plan.ledger().pages_visited == 4 && plan.ledger().missing_pages == 0);
    float output[2];
    assert(llama_kv_attention_online_state_normalize(result, output, 2) ==
        llama_kv_attention_exact_status::ok);
    assert(std::isfinite(output[0]) && std::isfinite(output[1]));

    llama_kv_attention_exact_config double_config = config;
    double_config.pages_per_wave = 2;
    double_config.staging_slots = 2;
    double_config.schedule = llama_kv_attention_exact_schedule::double_buffered;
    auto double_plan = llama_kv_attention_exact_wave_plan::build(
        all_pages(), resident_snapshot(), double_config, status);
    assert(status == llama_kv_attention_exact_status::ok && double_plan.valid());
    assert(double_plan.waves().size() == 2);
    assert(double_plan.waves()[1].staging_slot == 1);
    fake = {};
    assert(executor.execute(double_plan, backend, result) == llama_kv_attention_exact_status::ok);
    assert(fake.uploads == 1 && fake.waits == 1 && fake.asynchronous_uploads == 1);
    float double_output[2];
    assert(llama_kv_attention_online_state_normalize(result, double_output, 2) ==
        llama_kv_attention_exact_status::ok);
    assert(std::fabs(output[0] - double_output[0]) < 1.0e-5f);
    assert(std::fabs(output[1] - double_output[1]) < 1.0e-5f);
}

static void test_coverage_failures() {
    llama_kv_attention_exact_config config;
    config.logical_page_count = 4;
    config.pages_per_wave = 2;
    config.staging_slots = 1;
    llama_kv_attention_exact_status status;
    auto plan = llama_kv_attention_exact_wave_plan::build(
        all_pages(), resident_snapshot(), config, status);
    assert(status == llama_kv_attention_exact_status::ok);
    assert(plan.record_visit(0) == llama_kv_attention_exact_status::ok);
    assert(plan.record_visit(0) == llama_kv_attention_exact_status::duplicate_page);
    assert(plan.finish() == llama_kv_attention_exact_status::missing_page);

    auto broken = all_pages();
    broken[3].host_valid = false;
    auto refused = llama_kv_attention_exact_wave_plan::build(
        broken, resident_snapshot(), config, status);
    assert(!refused.valid() && status == llama_kv_attention_exact_status::invalid_page);
}

static void test_large_page_coverage() {
    constexpr uint32_t page_count = 305;
    std::vector<llama_kv_attention_exact_page> pages;
    pages.reserve(page_count);
    for (uint32_t logical = 0; logical < page_count; ++logical) {
        const uint32_t tokens = logical + 1 == page_count ? 17 : VBR_GENERATION_PAGE_CELLS;
        const bool resident = logical == 0;
        pages.push_back({ page_id(logical, tokens), resident ? 5 : UINT32_MAX,
            tokens, resident, !resident });
    }

    llama_kv_attention_exact_config config;
    config.logical_page_count = page_count;
    config.pages_per_wave = 17;
    config.staging_slots = 2;
    config.page_bytes = 4096;
    llama_kv_attention_exact_status status;
    auto plan = llama_kv_attention_exact_wave_plan::build(
        pages, resident_snapshot(), config, status);
    assert(status == llama_kv_attention_exact_status::ok && plan.valid());
    assert(plan.ledger().resident_pages == 1 && plan.ledger().cold_pages == 304);
    assert(plan.ledger().pages_visited == 0 && plan.ledger().peak_staging_pages == 17);
    for (const auto & wave : plan.waves()) {
        for (const auto & page : wave.pages) {
            assert(plan.record_visit(page.logical_page) == llama_kv_attention_exact_status::ok);
        }
    }
    assert(plan.finish() == llama_kv_attention_exact_status::ok);
    assert(plan.ledger().pages_visited == page_count);
    assert(plan.ledger().valid_tokens == 304ULL * VBR_GENERATION_PAGE_CELLS + 17);
}

static void test_exact_execution_route() {
    llama_kv_attention_execution execution(llama_kv_attention_execution_mode::exact);
    llama_kv_attention_scratch_request scratch;
    const auto decision = execution.prepare({}, llama_kv_attention_execution_phase::decode,
        0, 0, false, scratch);
    assert(decision.status == llama_kv_attention_execution_status::ok);
    assert(decision.route == llama_kv_attention_execution_route::exact_reference);
    assert(std::string(llama_kv_attention_execution_mode_name(
        llama_kv_attention_execution_mode::exact)) == "exact");
}

int main() {
    test_merge_against_direct();
    test_plan_and_executor();
    test_coverage_failures();
    test_large_page_coverage();
    test_exact_execution_route();
    return 0;
}
