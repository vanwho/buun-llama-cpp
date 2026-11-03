#include "llama-kv-attention-view.h"
#include "llama-kv-attention-op.h"
#include "llama-graph.h"

#include <algorithm>
#include <cassert>
#include <vector>

static llama_kv_page_record page(uint32_t logical, uint32_t slot, llama_pos end) {
    llama_kv_page_record result;
    result.id.sequence_id = 0;
    result.id.logical_page = logical;
    result.id.position_begin = llama_pos(logical * VBR_GENERATION_PAGE_CELLS);
    result.id.position_end = end;
    result.physical_slot = slot;
    result.state = llama_kv_page_state::gpu_host_clean;
    result.host_valid = true;
    return result;
}

static llama_kv_residency_snapshot make_snapshot() {
    llama_kv_residency_table table(8);
    auto tx = table.begin();
    const auto replace0 = table.replace(tx, page(0, 5, 256));
    const auto replace1 = table.replace(tx, page(1, 1, 512));
    const auto replace2 = table.replace(tx, page(2, 7, 768));
    auto tail = page(3, 3, 900);
    tail.state = llama_kv_page_state::filling_gpu;
    const auto replace3 = table.replace(tx, tail);
    assert(replace0 == llama_kv_residency_status::ok);
    assert(replace1 == llama_kv_residency_status::ok);
    assert(replace2 == llama_kv_residency_status::ok);
    assert(replace3 == llama_kv_residency_status::ok);
    const auto publish = table.publish(tx);
    assert(publish == llama_kv_residency_status::ok);
    (void) replace0;
    (void) replace1;
    (void) replace2;
    (void) replace3;
    (void) publish;
    return table.snapshot();
}

static llama_kv_residency_snapshot make_contiguous_snapshot() {
    llama_kv_residency_table table(8);
    auto tx = table.begin();
    assert(table.replace(tx, page(0, 0, 256)) == llama_kv_residency_status::ok);
    assert(table.replace(tx, page(1, 1, 512)) == llama_kv_residency_status::ok);
    assert(table.replace(tx, page(2, 2, 768)) == llama_kv_residency_status::ok);
    assert(table.replace(tx, page(3, 3, 900)) == llama_kv_residency_status::ok);
    assert(table.publish(tx) == llama_kv_residency_status::ok);
    return table.snapshot();
}

static llama_kv_page_record certified_page(uint32_t logical, uint32_t slot,
        llama_pos end, uint32_t valid_length = VBR_GENERATION_PAGE_CELLS,
        uint64_t version = 1) {
    auto result = page(logical, slot, end);
    result.state = llama_kv_page_state::gpu_host_clean;
    result.valid_length = valid_length;
    result.content_version = version;
    result.id.page_generation = 1;
    result.id.sequence_generation = 1;
    return result;
}

static llama_kv_residency_snapshot snapshot_of(
        const std::vector<llama_kv_page_record> & pages, uint32_t capacity = 8,
        uint32_t publishes = 1) {
    llama_kv_residency_table table(capacity);
    auto tx = table.begin();
    for (const auto & record : pages) {
        assert(table.replace(tx, record) == llama_kv_residency_status::ok);
    }
    assert(table.publish(tx) == llama_kv_residency_status::ok);
    for (uint32_t i = 1; i < publishes; ++i) {
        auto next = table.begin();
        assert(table.publish(next) == llama_kv_residency_status::ok);
    }
    return table.snapshot();
}

static std::vector<llama_kv_page_record> complete_history(uint64_t first_version = 1) {
    return {
        certified_page(0, 0, 256, 256, first_version),
        certified_page(1, 1, 512, 256, 1),
        certified_page(2, 2, 600, 88, 1),
    };
}

static void test_sequence_scoped_lookup() {
    llama_kv_residency_table table(8);
    auto tx = table.begin();
    const auto seq0 = page(0, 1, 256);
    auto seq1 = page(0, 6, 256);
    seq1.id.sequence_id = 1;
    const auto first = table.replace(tx, seq0);
    const auto second = table.replace(tx, seq1);
    const auto published = table.publish(tx);
    assert(first == llama_kv_residency_status::ok);
    assert(second == llama_kv_residency_status::ok);
    assert(published == llama_kv_residency_status::ok);
    (void) first;
    (void) second;
    (void) published;

    const auto snapshot = table.snapshot();
    const auto filtered = snapshot.for_sequence(1);
    assert(filtered.epoch() == snapshot.epoch());
    assert(filtered.slot_capacity() == snapshot.slot_capacity());
    assert(filtered.pages().size() == 1);
    assert(filtered.pages()[0].id.sequence_id == 1);
    assert(filtered.pages()[0].physical_slot == 6);

    llama_kv_attention_view_status status;
    const auto view = llama_kv_attention_view::build(snapshot, { 0 }, 1, status);
    assert(status == llama_kv_attention_view_status::ok);
    assert(view.valid() && view.pages()[0].source_physical_slot == 6);
}

static void test_compact_permuted_gapped_view() {
    llama_kv_attention_view_status status;
    const auto view = llama_kv_attention_view::build(make_snapshot(), { 2, 0 }, status);
    assert(status == llama_kv_attention_view_status::ok);
    assert(view.valid() && view.graph_epoch() == 1);
    assert(view.get_n_kv() == 512);
    assert(view.pages().size() == 2);
    // The selected set is still {2, 0}, but compact rows are canonicalized
    // into native order before masks and FA descriptors are built.
    assert(view.pages()[0].logical_page == 0);
    assert(view.pages()[0].source_physical_slot == 5);
    assert(view.pages()[0].compact_row_begin == 0);
    assert(view.pages()[1].compact_row_begin == 256);
    assert(view.native_positions()[0] == 0);
    assert(view.native_positions()[255] == 255);
    assert(view.native_positions()[256] == 512);
    assert(view.native_positions().back() == 767);
    assert(view.native_mask().size() == view.get_n_kv());
    assert(view.native_mask().back() == 1);

    auto fence = view.acquire_graph_fence();
    assert(fence.active());
    fence.release();
    assert(!fence.active());
}

static void test_tail_and_rejections() {
    llama_kv_attention_view_status status;
    const auto tail = llama_kv_attention_view::build(make_snapshot(), { 3 }, status);
    assert(status == llama_kv_attention_view_status::ok && tail.get_n_kv() == 132);

    const auto duplicate = llama_kv_attention_view::build(make_snapshot(), { 0, 0 }, status);
    assert(!duplicate.valid() && status == llama_kv_attention_view_status::duplicate_page);
    const auto absent = llama_kv_attention_view::build(make_snapshot(), { 4 }, status);
    assert(!absent.valid() && status == llama_kv_attention_view_status::not_resident);
}

static void test_contiguous_prefix_after_reordering() {
    llama_kv_attention_view_status view_status;
    const auto view = llama_kv_attention_view::build(
            make_contiguous_snapshot(), { 3, 1, 0, 2 }, view_status);
    assert(view_status == llama_kv_attention_view_status::ok);
    assert(view.pages().size() == 4);
    assert(view.pages()[0].logical_page == 0 && view.pages()[0].source_physical_slot == 0);
    assert(view.pages()[3].logical_page == 3 && view.pages()[3].source_physical_slot == 3);

    llama_kv_attention_operator_params params;
    params.mode = llama_kv_attention_operator_mode::selective;
    params.type_k = GGML_TYPE_TURBO4_0;
    params.type_v = GGML_TYPE_TURBO4_0;
    params.domain_k = llama_kv_attention_representation_domain::turbo_rotated;
    params.domain_v = llama_kv_attention_representation_domain::turbo_rotated;
    params.page_tokens = VBR_GENERATION_PAGE_CELLS;
    params.head_dim_k = 256;
    params.head_dim_v = 256;
    params.n_head_q = 16;
    params.n_head_kv = 4;
    params.n_query_tokens = 1;
    params.n_batch = 1;
    params.query_positions = { 899 };
    llama_kv_attention_operator_status status;
    const auto metadata = llama_kv_attention_operator_metadata::build(view, params, status);
    assert(status == llama_kv_attention_operator_status::ok);
    const auto dense = llama_kv_attention_dense_view_check(metadata, 8);
    assert(dense.eligible && dense.source_row_begin == 0 && dense.row_count == 900);
    assert(dense.padded_row_count == 1024);
    (void) dense;
}

static void test_operator_contract() {
    llama_kv_attention_view_status view_status;
    const auto view = llama_kv_attention_view::build(make_snapshot(), { 2, 0 }, view_status);
    assert(view_status == llama_kv_attention_view_status::ok);

    llama_kv_attention_operator_params params;
    params.mode = llama_kv_attention_operator_mode::selective;
    params.type_k = GGML_TYPE_TURBO4_0;
    params.type_v = GGML_TYPE_TURBO4_0;
    params.head_dim_k = 256;
    params.head_dim_v = 256;
    params.n_head_q = 16;
    params.n_head_kv = 4;
    params.n_query_tokens = 2;
    params.n_batch = 2;
    params.query_positions = { 700, 701, 500, 501 };

    llama_kv_attention_operator_status status;
    const auto metadata = llama_kv_attention_operator_metadata::build(view, params, status);
    assert(status == llama_kv_attention_operator_status::ok);
    assert(metadata.valid() && metadata.enabled());
    assert(metadata.table_epoch() == view.graph_epoch());
    assert(metadata.graph_reuse_key() == 1);
    assert(metadata.get_n_kv() == view.get_n_kv());
    assert(metadata.n_head_q() / metadata.n_head_kv() == 4);
    assert(metadata.query_positions().size() == 4);
    assert(metadata.domain_k() == llama_kv_attention_representation_domain::original);
    assert(metadata.domain_v() == llama_kv_attention_representation_domain::original);
    auto rotated_source = params;
    rotated_source.domain_k = llama_kv_attention_representation_domain::turbo_rotated;
    rotated_source.domain_v = llama_kv_attention_representation_domain::turbo_rotated;
    const auto rotated_metadata = llama_kv_attention_operator_metadata::build(
            view, rotated_source, status);
    assert(rotated_metadata.valid());
    assert(rotated_metadata.graph_content_key() != metadata.graph_content_key());
    assert(rotated_metadata.graph_layout_key() != metadata.graph_layout_key());
    assert(rotated_metadata.domain_k() == llama_kv_attention_representation_domain::turbo_rotated);
    const auto rotated_gapped = llama_kv_attention_dense_view_check(rotated_metadata, 8);
    assert(!rotated_gapped.eligible);
    (void) rotated_gapped;

    llama_kv_attention_view_status tail_view_status;
    const auto tail_view = llama_kv_attention_view::build(make_snapshot(), { 3 }, tail_view_status);
    assert(tail_view_status == llama_kv_attention_view_status::ok);
    auto tail_params = rotated_source;
    tail_params.n_query_tokens = 1;
    tail_params.n_batch = 1;
    tail_params.query_positions = { 899 };
    llama_kv_attention_operator_status tail_status;
    const auto tail_metadata = llama_kv_attention_operator_metadata::build(
            tail_view, tail_params, tail_status);
    assert(tail_status == llama_kv_attention_operator_status::ok);
    const auto tail_dense = llama_kv_attention_dense_view_check(tail_metadata, 8);
    assert(tail_dense.eligible && tail_dense.source_row_begin == 3 * VBR_GENERATION_PAGE_CELLS &&
           tail_dense.row_count == 132);
    assert(tail_dense.padded_row_count == 256);
    assert(tail_dense.source_row_begin + tail_dense.padded_row_count <= 8 * VBR_GENERATION_PAGE_CELLS);
    (void) tail_dense;
    assert(llama_kv_attention_operator_check_backend(
                GGML_BACKEND_DEVICE_TYPE_CPU, metadata) ==
           llama_kv_attention_backend_status::supported_reference);
    assert(llama_kv_attention_operator_check_backend(
                GGML_BACKEND_DEVICE_TYPE_GPU, metadata) ==
           llama_kv_attention_backend_status::unsupported_backend);
    assert(llama_kv_attention_operator_check_backend(
                static_cast<ggml_backend_t>(nullptr), metadata) ==
           llama_kv_attention_backend_status::unsupported_backend);

    // Dense GPU attention now accepts F16 alongside the quantized types;
    // use a representation that is actually unsupported for the rejection.
    auto mixed_type = params;
    mixed_type.type_v = GGML_TYPE_F16;
    assert(llama_kv_attention_operator_metadata::build(view, mixed_type, status).valid());
    auto wrong_type = params;
    wrong_type.type_v = GGML_TYPE_I32;
    assert(!llama_kv_attention_operator_metadata::build(view, wrong_type, status).valid());
    assert(status == llama_kv_attention_operator_status::invalid_type);

    auto wrong_gqa = params;
    wrong_gqa.n_head_q = 15;
    assert(!llama_kv_attention_operator_metadata::build(view, wrong_gqa, status).valid());
    assert(status == llama_kv_attention_operator_status::invalid_shape);

    auto non_causal = params;
    non_causal.causal = false;
    assert(!llama_kv_attention_operator_metadata::build(view, non_causal, status).valid());
    assert(status == llama_kv_attention_operator_status::non_causal);

    params.mode = llama_kv_attention_operator_mode::off;
    assert(!llama_kv_attention_operator_metadata::build(view, params, status).valid());
    assert(status == llama_kv_attention_operator_status::disabled);

    llm_graph_params graph_a = {};
    graph_a.kv_attention_table_epoch = view.graph_epoch();
    graph_a.kv_attention_content_key = metadata.graph_content_key();
    graph_a.kv_attention_layout_key = metadata.graph_layout_key();
    auto graph_b = graph_a;
    assert(graph_a.allow_reuse(graph_a));
    graph_b.kv_attention_table_epoch++;
    graph_b.kv_attention_content_key++;
    assert(graph_a.allow_reuse(graph_b));
    graph_b.kv_attention_layout_key++;
    assert(!graph_a.allow_reuse(graph_b));
}

static void test_full_prefix_coverage_proof() {
    auto pages = complete_history();
    const auto contiguous = snapshot_of(pages);
    assert(llama_kv_attention_full_prefix(contiguous, 0, 600, 3));
    assert(llama_kv_attention_full_prefix(contiguous, 0, 336, 3));
    assert(!llama_kv_attention_full_prefix(contiguous, 1, 600, 3));
    assert(!llama_kv_attention_full_prefix(contiguous, 0, 600, 0));
    assert(!llama_kv_attention_full_prefix(contiguous, 0, 600, 2));

    // A resident page count does not prove that the requested prefix has no holes.
    const auto hole = snapshot_of({ pages[0], pages[2] });
    assert(!llama_kv_attention_full_prefix(hole, 0, 600, 3));

    // Two residency generations claiming the same native range overlap even
    // though either record on its own would provide complete coverage.
    auto overlapping_generation = pages[0];
    overlapping_generation.id.sequence_generation++;
    overlapping_generation.physical_slot = 3;
    const auto overlap = snapshot_of({ pages[0], overlapping_generation });
    assert(!llama_kv_attention_full_prefix(overlap, 0, 256, 2));

    // Adjacent pages from different model/representation owners cannot be
    // combined into one causal prefix.
    auto mixed_representation_pages = complete_history();
    mixed_representation_pages[1].id.representation_epoch++;
    const auto mixed_representation = snapshot_of(mixed_representation_pages);
    assert(!llama_kv_attention_full_prefix(mixed_representation, 0, 600, 3));

    // Host-only inventory cannot be used as a live full-attention prefix.
    pages[1].state = llama_kv_page_state::host_clean;
    const auto cold = snapshot_of(pages);
    assert(!llama_kv_attention_full_prefix(cold, 0, 600, 3));

    // valid_length can shorten an otherwise full position range, and coverage
    // uses that committed extent rather than the larger identity range.
    pages = complete_history();
    pages[1].valid_length = 80;
    const auto short_extent = snapshot_of(pages);
    assert(llama_kv_attention_full_prefix(short_extent, 0, 336, 3));
    assert(!llama_kv_attention_full_prefix(short_extent, 0, 337, 3));

    // A maximum valid_length alone cannot prove that every interior row is
    // present; the residency record must also certify a contiguous prefix.
    pages = complete_history();
    pages[0].contiguous_prefix = false;
    const auto interior_hole = snapshot_of(pages);
    assert(!llama_kv_attention_full_prefix(interior_hole, 0, 600, 3));

    llama_kv_attention_replay_certificate cert;
    cert.reset(0, 300, 350);
    cert.observe(interior_hole, { 0, 1, 2 }, 300, 349);
    assert(!cert.matches(interior_hole, { 0, 1, 2 }));
}

static void test_replay_certificate_identity_and_submission() {
    const auto baseline = snapshot_of(complete_history());
    llama_kv_attention_replay_certificate cert;
    cert.reset(0, 300, 350);
    cert.observe(baseline, { 0, 1, 2 }, 300, 349);
    assert(cert.matches(baseline, { 0, 1, 2 }));
    assert(cert.matches(baseline, { 2, 0, 1 }));
    // Page 2 starts after this request's causal horizon and may be omitted.
    assert(cert.matches(baseline, { 0, 1 }));
    // Page 0 contributes observed history and must remain selected.
    assert(!cert.matches(baseline, { 1, 2 }));

    llama_kv_attention_replay_certificate unobserved;
    unobserved.reset(0, 300, 350);
    assert(!unobserved.matches(baseline, { 0, 1, 2 }));

    llama_kv_attention_replay_certificate gap;
    gap.reset(0, 300, 350);
    gap.observe(baseline, { 0, 1, 2 }, 301, 349);
    assert(!gap.matches(baseline, { 0, 1, 2 }));

    llama_kv_attention_replay_certificate incomplete;
    incomplete.reset(0, 300, 350);
    incomplete.observe(baseline, { 0, 1, 2 }, 300, 348);
    assert(!incomplete.matches(baseline, { 0, 1, 2 }));

    llama_kv_attention_replay_certificate wrong_sequence;
    wrong_sequence.reset(0, 300, 350);
    auto wrong_seq_pages = complete_history();
    for (auto & p : wrong_seq_pages) p.id.sequence_id = 1;
    const auto wrong_seq = snapshot_of(wrong_seq_pages);
    wrong_sequence.observe(wrong_seq, { 0, 1, 2 }, 300, 349);
    assert(!wrong_sequence.matches(wrong_seq, { 0, 1, 2 }));

    llama_kv_attention_replay_certificate invalidated;
    invalidated.reset(0, 300, 350);
    invalidated.observe(baseline, { 0, 1, 2 }, 300, 349);
    invalidated.invalidate();
    assert(!invalidated.matches(baseline, { 0, 1, 2 }));
}

static void test_replay_certificate_snapshot_changes() {
    const auto baseline = snapshot_of(complete_history());
    llama_kv_attention_replay_certificate cert;
    cert.reset(0, 300, 350);
    cert.observe(baseline, { 0, 1, 2 }, 300, 349);

    // Slot assignment, snapshot epoch, and caller page ordering do not alter
    // the logical causal history submitted by the graph.
    auto permuted_pages = complete_history();
    permuted_pages[0].physical_slot = 6;
    permuted_pages[1].physical_slot = 4;
    permuted_pages[2].physical_slot = 7;
    std::reverse(permuted_pages.begin(), permuted_pages.end());
    const auto relocated = snapshot_of(permuted_pages, 8, 2);
    assert(relocated.epoch() != baseline.epoch());
    assert(cert.matches(relocated, { 2, 1, 0 }));

    // A rewrite to a full historical page must invalidate even when logical
    // coverage and every other page identity stay the same.
    auto rewritten_pages = complete_history(2);
    const auto rewritten = snapshot_of(rewritten_pages);
    assert(!cert.matches(rewritten, { 0, 1, 2 }));

    // A changed causal generation also invalidates the certificate.
    auto new_generation_pages = complete_history();
    new_generation_pages[0].id.page_generation++;
    const auto new_generation = snapshot_of(new_generation_pages);
    assert(!cert.matches(new_generation, { 0, 1, 2 }));
}

static void test_replay_certificate_append_seam() {
    // The first query is inside page 1. Its already committed prefix ends at
    // the user-span seam, so an append to that page may extend the page/version
    // while preserving the observed causal prefix.
    auto before_pages = std::vector<llama_kv_page_record> {
        certified_page(0, 0, 256, 256, 4),
        certified_page(1, 1, 350, 94, 8),
    };
    const auto before = snapshot_of(before_pages);
    llama_kv_attention_replay_certificate cert;
    cert.reset(0, 300, 350);
    cert.observe(before, { 0, 1 }, 300, 349);
    assert(cert.matches(before, { 0, 1 }));

    auto appended_pages = before_pages;
    appended_pages[1].id.position_end = 400;
    appended_pages[1].valid_length = 144;
    appended_pages[1].content_version++;
    const auto appended = snapshot_of(appended_pages);
    assert(cert.matches(appended, { 0, 1 }));

    // Page 0 is wholly historical, so changing its version is never covered
    // by the append-only seam exception.
    appended_pages[0].content_version++;
    const auto rewritten_full_page = snapshot_of(appended_pages);
    assert(!cert.matches(rewritten_full_page, { 0, 1 }));

    // A page whose old extent ends exactly at query_begin is wholly historical;
    // extending the query span cannot excuse a version change to that page.
    const auto exact_seam_before = snapshot_of(complete_history());
    llama_kv_attention_replay_certificate exact_seam_cert;
    exact_seam_cert.reset(0, 512, 540);
    exact_seam_cert.observe(exact_seam_before, { 0, 1, 2 }, 512, 539);
    assert(exact_seam_cert.matches(exact_seam_before, { 0, 1, 2 }));
    auto exact_seam_pages = complete_history();
    exact_seam_pages[1].content_version++;
    const auto changed_at_seam = snapshot_of(exact_seam_pages);
    assert(!exact_seam_cert.matches(changed_at_seam, { 0, 1, 2 }));
}

static void test_replay_certificate_rolling_query_window() {
    auto pages = complete_history();
    pages[2].id.position_end = 768;
    pages[2].valid_length = 256;
    const auto first_window = snapshot_of({ pages[0] }, 2);
    const auto second_window = snapshot_of({ pages[0], pages[1] }, 2);
    // The query window rolls forward while staying within the two resident
    // attention pages available to each graph submission.
    auto third_pages = std::vector<llama_kv_page_record> { pages[1], pages[2] };
    third_pages[0].physical_slot = 0;
    third_pages[1].physical_slot = 1;
    const auto third_window = snapshot_of(third_pages, 2);

    llama_kv_attention_replay_certificate cert;
    cert.reset(0, 0, 768, VBR_GENERATION_PAGE_CELLS, 2);
    cert.observe(first_window, { 0 }, 0, 255);
    cert.observe(second_window, { 0, 1 }, 256, 511);
    cert.observe(third_window, { 1, 2 }, 512, 767);
    assert(cert.matches(third_window, { 1, 2 }));
    assert(!cert.matches(third_window, { 2 }));
}

static void test_replay_certificate_rolling_query_after_non_aligned_seam() {
    auto pages = complete_history();
    pages[2].id.position_end = 768;
    pages[2].valid_length = 256;
    const auto first_window = snapshot_of({ pages[0] }, 2);
    const auto second_window = snapshot_of({ pages[0], pages[1] }, 2);
    auto third_pages = std::vector<llama_kv_page_record> { pages[1], pages[2] };
    third_pages[0].physical_slot = 0;
    third_pages[1].physical_slot = 1;
    const auto third_window = snapshot_of(third_pages, 2);

    llama_kv_attention_replay_certificate cert;
    cert.reset(0, 100, 768, VBR_GENERATION_PAGE_CELLS, 2);
    cert.observe(first_window, { 0 }, 100, 255);
    cert.observe(second_window, { 0, 1 }, 256, 511);
    cert.observe(third_window, { 1, 2 }, 512, 767);
    assert(cert.matches(third_window, { 1, 2 }));
}

static void test_replay_certificate_keeps_full_old_history() {
    auto pages = complete_history();
    pages[2].id.position_end = 768;
    pages[2].valid_length = 256;
    const auto first_window = snapshot_of({ pages[0], pages[1] }, 2);
    // Logical page 2 takes the slot released by page 1 in this two-slot pool.
    pages[2].physical_slot = 1;
    const auto second_window = snapshot_of({ pages[0], pages[2] }, 2);

    llama_kv_attention_replay_certificate cert;
    cert.reset(0, 256, 768, VBR_GENERATION_PAGE_CELLS, 2);
    cert.observe(first_window, { 0, 1 }, 256, 511);
    cert.observe(second_window, { 0, 2 }, 512, 767);
    assert(cert.matches(second_window, { 0, 2 }));
    assert(!cert.matches(second_window, { 2 }));
}

int main() {
    test_compact_permuted_gapped_view();
    test_tail_and_rejections();
    test_contiguous_prefix_after_reordering();
    test_sequence_scoped_lookup();
    test_operator_contract();
    test_full_prefix_coverage_proof();
    test_replay_certificate_identity_and_submission();
    test_replay_certificate_snapshot_changes();
    test_replay_certificate_append_seam();
    test_replay_certificate_rolling_query_window();
    test_replay_certificate_rolling_query_after_non_aligned_seam();
    test_replay_certificate_keeps_full_old_history();
    return 0;
}
