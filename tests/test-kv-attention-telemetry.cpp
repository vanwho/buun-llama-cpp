#include "llama-kv-attention-telemetry.h"

#include <array>
#include <cassert>
#include <cmath>

static llama_kv_page_record make_page(uint32_t logical, uint32_t slot, llama_pos end,
                                      uint32_t generation = 1) {
    llama_kv_page_record page;
    page.id.sequence_id = 0;
    page.id.logical_page = logical;
    page.id.page_generation = generation;
    page.id.position_begin = llama_pos(logical * VBR_GENERATION_PAGE_CELLS);
    page.id.position_end = end;
    page.physical_slot = slot;
    page.state = uint32_t(end - page.id.position_begin) < VBR_GENERATION_PAGE_CELLS
        ? llama_kv_page_state::filling_gpu : llama_kv_page_state::gpu_host_clean;
    page.host_valid = true;
    return page;
}

static llama_kv_residency_snapshot make_snapshot(uint32_t generation = 1) {
    llama_kv_residency_table table(8);
    auto tx = table.begin();
    assert(table.replace(tx, make_page(0, 4, 256, generation)) == llama_kv_residency_status::ok);
    assert(table.replace(tx, make_page(1, 1, 512)) == llama_kv_residency_status::ok);
    assert(table.replace(tx, make_page(2, 7, 700)) == llama_kv_residency_status::ok);
    assert(table.publish(tx) == llama_kv_residency_status::ok);
    return table.snapshot();
}

static llama_kv_attention_telemetry_sample make_sample(
        const llama_kv_residency_snapshot & snapshot,
        const float * mass, size_t page_count) {
    llama_kv_attention_telemetry_sample sample;
    sample.table_epoch = snapshot.epoch();
    sample.head_count = 2;
    sample.head_stride_bytes = 4 * sizeof(float);
    sample.layer_stride_bytes = 2 * sample.head_stride_bytes;
    sample.token_stride_bytes = sample.layer_stride_bytes;
    sample.page_mass = mass;
    sample.pages = snapshot.pages().data();
    sample.page_count = page_count;
    return sample;
}

int main() {
    const auto snapshot = make_snapshot();
    llama_kv_attention_telemetry_config config;
    config.mode = llama_kv_attention_telemetry_mode::observe;
    config.logical_page_count = 4;
    config.ema_alpha = 0.5f;
    config.peak_decay = 0.5f;
    llama_kv_attention_telemetry_config large_config = config;
    large_config.logical_page_count = 2048;
    llama_kv_attention_telemetry large(large_config);
    assert(large.accounting().logical_page_count == 2048);
    assert(large.accounting().controller_score_bytes == 4096 * sizeof(float));

    llama_kv_attention_telemetry telemetry(config);
    assert(telemetry.initialize(snapshot) == llama_kv_attention_telemetry_status::ok);
    assert(telemetry.accounting().controller_score_bytes == 8 * sizeof(float));
    assert(telemetry.accounting().metadata_bytes == 4 * sizeof(llama_kv_page_id));

    // One layer, two heads, four logical bins.  This is the CPU oracle for
    // the bounded GPU page-mass reduction: page 0 averages to .375 and page 1
    // averages to .125 across heads.
    const std::array<float, 8> mass = { 0.50f, 0.25f, 0.0f, 0.0f,
                                        0.25f, 0.00f, 0.0f, 0.0f };
    auto sample = make_sample(snapshot, mass.data(), snapshot.pages().size());
    assert(telemetry.publish_completed(snapshot, sample) == llama_kv_attention_telemetry_status::ok);
    llama_kv_attention_telemetry_page page;
    assert(telemetry.page_state(0, page) && page.observed);
    assert(std::fabs(page.normalized_ema - 0.375f) < 1.0e-6f);
    assert(std::fabs(page.recent_peak - 0.375f) < 1.0e-6f);
    assert(telemetry.page_state(2, page) && page.observed);
    assert(page.frequency == 1);
    assert(telemetry.counters().sampled_tokens == 1);
    assert(telemetry.counters().sampled_pages == 3);
    assert(telemetry.counters().samples == 1);
    assert(telemetry.counters().sampled_layers == 1);
    assert(telemetry.counters().sampled_heads == 2);

    // Layer and head aggregation is over participating page bins only. The
    // second layer gives page 0 a second .30 contribution and page 1 a .10
    // contribution, so the two-layer averages are .30 and .15.
    llama_kv_attention_telemetry aggregated(config);
    assert(aggregated.initialize(snapshot) == llama_kv_attention_telemetry_status::ok);
    const std::array<float, 16> layered_mass = {
        0.20f, 0.10f, 0.0f, 0.0f,  0.40f, 0.30f, 0.0f, 0.0f,
        0.40f, 0.20f, 0.0f, 0.0f,  0.20f, 0.00f, 0.0f, 0.0f,
    };
    auto layered = make_sample(snapshot, layered_mass.data(), snapshot.pages().size());
    layered.layer_count = 2;
    layered.head_stride_bytes = 4 * sizeof(float);
    layered.layer_stride_bytes = 2 * layered.head_stride_bytes;
    layered.token_stride_bytes = 2 * layered.layer_stride_bytes;
    assert(aggregated.publish_completed(snapshot, layered) == llama_kv_attention_telemetry_status::ok);
    assert(aggregated.page_state(0, page) && std::fabs(page.normalized_ema - 0.30f) < 1.0e-6f);
    assert(aggregated.page_state(1, page) && std::fabs(page.normalized_ema - 0.15f) < 1.0e-6f);

    // A second completed sample exercises EMA and recent-peak decay.
    const std::array<float, 8> second_mass = { 0.0f, 0.0f, 0.0f, 0.0f,
                                               0.0f, 0.50f, 0.0f, 0.0f };
    sample.page_mass = second_mass.data();
    assert(telemetry.publish_completed(snapshot, sample) == llama_kv_attention_telemetry_status::ok);
    assert(telemetry.page_state(0, page));
    assert(std::fabs(page.normalized_ema - 0.1875f) < 1.0e-6f);
    assert(std::fabs(page.recent_peak - 0.1875f) < 1.0e-6f);

    std::array<float, 4> ema = {}, peak = {};
    assert(telemetry.copy_scores(ema.data(), peak.data(), 4));
    assert(std::fabs(ema[0] - 0.1875f) < 1.0e-6f);
    assert(peak[2] == 0.0f); // an unobserved page is not an observed zero.

    // Selective mode only publishes selected pages; omitted resident pages
    // remain explicitly unobserved rather than being assigned zero mass.
    config.mode = llama_kv_attention_telemetry_mode::selective;
    llama_kv_attention_telemetry selective(config);
    assert(selective.initialize(snapshot) == llama_kv_attention_telemetry_status::ok);
    sample.pages = snapshot.pages().data();
    sample.page_count = 1;
    assert(selective.publish_completed(snapshot, sample) == llama_kv_attention_telemetry_status::ok);
    assert(selective.page_state(0, page) && page.observed);
    assert(selective.page_state(1, page) && !page.observed);

    // Sampling interval is explicit and skipped tokens do no state update.
    config.sample_interval_tokens = 2;
    llama_kv_attention_telemetry sampled(config);
    assert(sampled.initialize(snapshot) == llama_kv_attention_telemetry_status::ok);
    sample = make_sample(snapshot, mass.data(), snapshot.pages().size());
    sample.token_index = 1;
    assert(sampled.publish_completed(snapshot, sample) == llama_kv_attention_telemetry_status::sampling_skipped);
    assert(sampled.counters().sampled_tokens == 0);
    assert(sampled.counters().skipped == 1);
    sample.token_index = 2;
    assert(sampled.publish_completed(snapshot, sample) == llama_kv_attention_telemetry_status::ok);

    // Off has no telemetry work and rejects even malformed/null samples.
    config.mode = llama_kv_attention_telemetry_mode::off;
    llama_kv_attention_telemetry off(config);
    assert(off.initialize(snapshot) == llama_kv_attention_telemetry_status::disabled);
    assert(off.publish_completed({}, {}) == llama_kv_attention_telemetry_status::disabled);
    assert(off.counters().sampled_tokens == 0);

    // Epochs are table-local; identity validation catches a reused epoch with
    // a changed page generation before any EMA is modified.
    llama_kv_attention_telemetry_config no_capacity;
    llama_kv_attention_telemetry stale(no_capacity);
    stale.set_mode(llama_kv_attention_telemetry_mode::observe);
    // Constructing with off intentionally has no page capacity to configure;
    // this also proves that enabling a disabled instance cannot invent storage.
    assert(stale.initialize(snapshot) == llama_kv_attention_telemetry_status::invalid_argument);
    config.mode = llama_kv_attention_telemetry_mode::observe;
    llama_kv_attention_telemetry active(config);
    assert(active.initialize(snapshot) == llama_kv_attention_telemetry_status::ok);
    const auto changed = make_snapshot(2);
    sample = make_sample(changed, mass.data(), changed.pages().size());
    assert(active.publish_completed(changed, sample) == llama_kv_attention_telemetry_status::stale_epoch);
    assert(active.counters().stale_dropped == 1);

    // Rebinding a rebuilt table preserves only identical full page identities;
    // a changed generation starts cold and can be observed by the next sample.
    llama_kv_attention_telemetry rebound(config);
    assert(rebound.initialize(snapshot) == llama_kv_attention_telemetry_status::ok);
    assert(rebound.publish_completed(snapshot, make_sample(snapshot, mass.data(), snapshot.pages().size())) ==
            llama_kv_attention_telemetry_status::ok);
    assert(rebound.reconcile(changed) == llama_kv_attention_telemetry_status::ok);
    assert(rebound.page_state(0, page) && !page.observed);
    assert(rebound.page_state(1, page) && page.observed);
    return 0;
}
