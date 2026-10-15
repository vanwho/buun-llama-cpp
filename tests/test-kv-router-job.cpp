#include "llama-kv-router-job.h"
#include "ggml-backend.h"
#include "ggml-quants.h"

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <cstring>
#include <memory>
#include <limits>
#include <string>
#include <vector>

struct fake_event {
    uint64_t id = 0;
    bool done = false;
    bool failed = false;
};

struct fake_stage_backend {
    uint64_t next = 1;
    bool fail_enqueue = false;
    bool fail_terminal = false;
    std::vector<fake_event> events;
    uint32_t releases = 0;

    static bool enqueue(void * opaque, uint32_t, const void * host, void * device,
            size_t bytes, const llama_kv_rerank_chunk_task &, uint64_t * out) noexcept {
        auto & self = *static_cast<fake_stage_backend *>(opaque);
        if (self.fail_enqueue) {
            *out = self.next++;
            self.events.push_back({ *out, false, false });
            return false;
        }
        std::memcpy(device, host, bytes);
        const uint64_t id = self.next++;
        self.events.push_back({ id, false, false });
        *out = id;
        return true;
    }
    static llama_kv_prefetch_poll poll(void * opaque, uint64_t id) noexcept {
        auto & self = *static_cast<fake_stage_backend *>(opaque);
        for (const auto & event : self.events) if (event.id == id) {
            if (!event.done) return llama_kv_prefetch_poll::pending;
            return event.failed ? llama_kv_prefetch_poll::failed : llama_kv_prefetch_poll::completed;
        }
        return llama_kv_prefetch_poll::failed;
    }
    static void cancel(void * opaque, uint64_t id) noexcept {
        auto & self = *static_cast<fake_stage_backend *>(opaque);
        for (auto & event : self.events) if (event.id == id) event.done = true;
    }
    static void release(void * opaque, uint64_t) noexcept {
        ++static_cast<fake_stage_backend *>(opaque)->releases;
    }
    void complete(uint64_t id, bool failed = false) {
        for (auto & event : events) if (event.id == id) {
            event.done = true;
            event.failed = failed;
        }
    }
};

struct source_data {
    std::vector<uint8_t> bytes { 1, 2, 3, 4, 5, 6, 7, 8 };
    uint64_t version = 9;
    static bool recheck(void * opaque, uint64_t version) noexcept {
        return static_cast<source_data *>(opaque)->version == version;
    }
    static bool read(void * opaque, uint64_t offset, void * destination, size_t size) noexcept {
        const auto & bytes = static_cast<source_data *>(opaque)->bytes;
        if (offset > bytes.size() || size > bytes.size() - size_t(offset)) return false;
        std::memcpy(destination, bytes.data() + offset, size);
        return true;
    }
};

struct cache_adapter_fixture {
    std::vector<llama_kv_router_layer_plan> plans;
    llama_kv_router_job * job = nullptr;
    std::vector<llama_kv_prefetch_common_history_bundle> selected;
    static bool prepare(void * opaque, int32_t sequence, uint64_t turn,
            std::vector<llama_kv_router_layer_plan> & plans,
            llama_kv_router_plan_error & error) noexcept {
        auto & self = *static_cast<cache_adapter_fixture *>(opaque);
        plans = self.plans;
        error = llama_kv_router_plan_error::none;
        return !plans.empty() && plans.front().identity.sequence_id == sequence &&
            plans.front().identity.turn_id == turn;
    }
    static llama_kv_router_execution_status execute(void * opaque,
            const llama_kv_router_query_identity & identity,
            const std::vector<llama_kv_router_layer_plan> & plans,
            std::vector<llama_kv_router_exact_page_record> & records) noexcept {
        auto * self = static_cast<cache_adapter_fixture *>(opaque);
        return self != nullptr && self->job != nullptr
            ? self->job->execute_query(identity, plans, records)
            : llama_kv_router_execution_status::failed;
    }
    static void policy(void * opaque, llama_kv_router_execution_status status,
            const std::vector<llama_kv_router_exact_page_record> &) noexcept {
        auto & self = *static_cast<cache_adapter_fixture *>(opaque);
        if (status != llama_kv_router_execution_status::ready || self.job == nullptr) return;
        std::vector<llama_kv_prefetch_common_history_bundle> ranked;
        if (llama_kv_prefetch_rank_common_history(self.job->exact_candidates(), 1, ranked))
            self.selected = llama_kv_prefetch_select_common_history(ranked, ranked.size(), 4);
    }
};

struct key_reader_state {
    uint64_t content = 7;
    uint64_t summary = 9;
    static bool current(void * opaque, const llama_kv_page_id &,
            uint64_t content, uint64_t summary) noexcept {
        const auto & self = *static_cast<key_reader_state *>(opaque);
        return self.content == content && self.summary == summary;
    }
};

static void run_cpu_key_reader_contract() {
    llama_kv_page_id identity;
    identity.sequence_id = 0;
    identity.position_begin = 0;
    identity.position_end = 256;
    identity.logical_page = 0;
    identity.attention_layer = 11;
    auto chain = std::make_shared<artifact_segment_chain>(6);
    const uint8_t first[] = { 1, 2 };
    const uint8_t second[] = { 3, 4, 5, 6 };
    assert(chain->append(first, sizeof(first)));
    assert(chain->append(second, sizeof(second)));
    assert(chain->size() == 6 && chain->segment_count() == 2);
    vbr_selected_page_unit_descriptor unit;
    unit.logical_unit_id = 17;
    unit.layer = 1;
    unit.side = vbr_artifact_side::key;
    unit.valid_rows = 3;
    unit.row_bytes = 2;
    unit.bytes = chain;
    key_reader_state state;
    auto reader = llama_kv_router_key_reader::create(identity, 7, 9, unit,
            2, &state, &key_reader_state::current);
    assert(reader != nullptr);
    auto source = reader->source();
    reader.reset(); // the source holder must retain the reader and chain
    assert(source.page_holder != nullptr && source.recheck(source.context, 7));
    uint8_t bytes[4]{};
    assert(source.read(source.context, 0, bytes, sizeof(bytes)));
    const uint8_t expected[] = { 1, 2, 3, 4 };
    assert(std::memcmp(bytes, expected, sizeof(bytes)) == 0);
    assert(!source.read(source.context, 4, bytes, 2)); // partial page bounds
    assert(!source.read(source.context, 1, bytes, 2)); // complete rows only
    state.content = 8;
    assert(!source.recheck(source.context, 7)); // stale content version
    state.content = 7;
    state.summary = 10;
    assert(!source.recheck(source.context, 7)); // stale summary version

    // Compact attention ordinals are distinct from model layer IDs. Two KV
    // heads still share one stream tensor; an empty cold shortlist is valid.
    llama_kv_router_layer_plan plan;
    plan.compact_layer = 1;
    plan.model_layer = 11;
    plan.probe_capture_generation = 3;
    plan.query_heads = 4;
    plan.kv_heads = 2;
    plan.page_tokens = 256;
    plan.key_dimension = 2;
    plan.attention_scale = 0.5f;
    plan.pages.resize(1);
    plan.pages[0].candidate.identity = identity;
    plan.pages[0].candidate.identity.attention_layer = plan.compact_layer;
    plan.pages[0].candidate.content_version = 7;
    plan.pages[0].candidate.summary_version = 9;
    plan.pages[0].stream = 0;
    plan.pages[0].physical_slot = 3;
    plan.pages[0].valid_rows = 256;
    plan.pages[0].page_generation = 3;
    plan.pages[0].content_generation = 7;
    plan.pages[0].summary_generation = 9;
    assert(plan.compact_layer != plan.model_layer && plan.kv_heads == 2 &&
            plan.pages[0].stream == 0 && !plan.pages[0].candidate.cold &&
            llama_kv_router_layer_plan_metadata_valid(plan));
    llama_kv_router_layer_plan empty_cold_plan = plan;
    empty_cold_plan.pages.clear();
    assert(llama_kv_router_layer_plan_metadata_valid(empty_cold_plan));
}

static void run_cpu_lifecycle() {
    run_cpu_key_reader_contract();
    struct counting_executor {
        uint32_t starts = 0;
        static llama_kv_router_execution_status execute(void * opaque,
                const llama_kv_router_query_identity &,
                const std::vector<llama_kv_router_layer_plan> &,
                std::vector<llama_kv_router_exact_page_record> &) noexcept {
            ++static_cast<counting_executor *>(opaque)->starts;
            return llama_kv_router_execution_status::ready;
        }
    } adapter;
    llama_kv_router_query_executor call_gate;
    llama_kv_router_query_identity gate_identity;
    gate_identity.sequence_id = 0;
    gate_identity.session_generation = 1;
    gate_identity.turn_id = 41;
    gate_identity.query_generation = 6;
    gate_identity.table_epoch = 7;
    gate_identity.content_generation = 8;
    std::vector<llama_kv_router_layer_plan> gate_plans(1);
    std::vector<llama_kv_router_exact_page_record> gate_records;
    assert(call_gate.execute_once(gate_identity, true, gate_plans, gate_records,
            &adapter, &counting_executor::execute) == llama_kv_router_execution_status::ready);
    assert(call_gate.execute_once(gate_identity, true, gate_plans, gate_records,
            &adapter, &counting_executor::execute) == llama_kv_router_execution_status::ready);
    assert(adapter.starts == 1);
    call_gate.cancel();
    gate_identity.turn_id++;
    assert(call_gate.execute_once(gate_identity, false, gate_plans, gate_records,
            &adapter, &counting_executor::execute) == llama_kv_router_execution_status::stale);
    assert(adapter.starts == 1); // replay/generation/MTP phases never start execution

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(backend != nullptr);
    const auto pinned_buft = ggml_backend_cpu_buffer_type();
    fake_stage_backend fake;
    llama_kv_rerank_stage_backend callbacks { &fake, nullptr,
        &fake_stage_backend::poll, &fake_stage_backend::cancel,
        &fake_stage_backend::release, &fake_stage_backend::enqueue };
    auto job = llama_kv_router_job::create(backend, pinned_buft, 4, callbacks);
    assert(job != nullptr && job->slot_bytes() == 4);
    auto empty_owner = llama_kv_router_job::create(backend, pinned_buft, 4, callbacks);
    assert(empty_owner != nullptr);
    llama_kv_router_query_identity no_history_identity;
    no_history_identity.sequence_id = 0;
    no_history_identity.session_generation = 1;
    no_history_identity.turn_id = 40;
    no_history_identity.query_generation = 5;
    no_history_identity.table_epoch = 6;
    no_history_identity.content_generation = 7;
    assert(empty_owner->complete_empty_query(no_history_identity, 64));
    assert(empty_owner->owner_state().stage == llama_kv_router_owner_stage::ready &&
            empty_owner->exact_candidates().empty());
    assert(empty_owner->complete_empty_query(no_history_identity, 64));
    assert(empty_owner->owner_state().stage == llama_kv_router_owner_stage::ready &&
            empty_owner->exact_candidates().empty());
    empty_owner.reset();
    const size_t empty_row_bytes = ggml_row_size(GGML_TYPE_TURBO4_0, 128);
    auto empty_job = llama_kv_router_job::create(backend, pinned_buft,
            empty_row_bytes, callbacks, GGML_TYPE_TURBO4_0, 128);
    assert(empty_job != nullptr);
    ggml_init_params empty_params = {};
    empty_params.mem_size = 1 << 20;
    empty_params.no_alloc = true;
    ggml_context * empty_ctx = ggml_init(empty_params);
    assert(empty_ctx != nullptr);
    auto * empty_probes = ggml_new_tensor_3d(empty_ctx, GGML_TYPE_F32, 128, 1, 4);
    auto * empty_keys = ggml_new_tensor_2d(empty_ctx, GGML_TYPE_TURBO4_0, 128, 1);
    auto * empty_validity = ggml_new_tensor_1d(empty_ctx, GGML_TYPE_I64, 9);
    assert(empty_probes && empty_keys && empty_validity);
    ggml_backend_buffer_t empty_buffer = ggml_backend_alloc_ctx_tensors(empty_ctx, backend);
    assert(empty_buffer != nullptr);
    llama_kv_router_query_identity empty_identity;
    empty_identity.sequence_id = 0;
    empty_identity.session_generation = 1;
    empty_identity.turn_id = 42;
    empty_identity.query_generation = 7;
    empty_identity.table_epoch = 8;
    empty_identity.content_generation = 9;
    assert(empty_job->retain_coarse_shortlist(empty_identity, 0, {}));
    llama_kv_router_layer_plan empty_plan;
    empty_plan.identity = empty_identity;
    empty_plan.compact_layer = empty_plan.model_layer = 0;
    empty_plan.probe_capture_generation = 4;
    empty_plan.probes = empty_probes;
    empty_plan.validity = empty_validity;
    empty_plan.resident_keys = empty_keys;
    empty_plan.page_tokens = 1;
    empty_plan.query_heads = empty_plan.kv_heads = 1;
    empty_plan.key_dimension = 128;
    empty_plan.attention_scale = 0.1f;
    std::vector<llama_kv_router_exact_page_record> empty_records;
    assert(empty_job->execute_query(empty_identity, { empty_plan }, empty_records) ==
            llama_kv_router_execution_status::ready && empty_records.empty());
    assert(empty_job->execute_query(empty_identity, { empty_plan }, empty_records) ==
            llama_kv_router_execution_status::ready && empty_records.empty());
    empty_job.reset();
    ggml_backend_buffer_free(empty_buffer);
    ggml_free(empty_ctx);
    const auto & ledger = job->allocations();
    assert(ledger.pinned_requested == 8 && ledger.pinned_realized >= 8);
    assert(ledger.device_requested >= 8 && ledger.device_realized >= 8);

    auto data = std::make_shared<source_data>();
    std::shared_ptr<const void> holder(data, data.get());
    llama_kv_rerank_stage_source source { holder, data.get(),
        &source_data::recheck, &source_data::read };
    llama_kv_rerank_chunk_task task;
    task.candidate_index = 4;
    task.compact_layer_index = 2;
    task.row_offset = 10;
    task.rows = 2;
    task.row_bytes = 2;
    task.query_generation = 3;
    task.content_version = 9;
    uint32_t first = UINT32_MAX, second = UINT32_MAX, unused = UINT32_MAX;
    uint64_t ticket_first = 0, ticket_second = 0;
    assert(job->submit(source, 0, 2, task, first, &ticket_first) == llama_kv_rerank_stage_status::ok);
    task.row_offset = 12;
    assert(job->submit(source, 4, 2, task, second, &ticket_second) == llama_kv_rerank_stage_status::ok);
    assert(first != second && ticket_first != ticket_second && holder.use_count() == 5);
    task.row_offset = 14;
    assert(job->submit(source, 0, 2, task, unused) == llama_kv_rerank_stage_status::backpressure);
    assert(std::memcmp(job->encoded_key_slot(first)->data, data->bytes.data(), 4) == 0);

    fake.complete(ticket_first);
    llama_kv_rerank_stage_ticket completed[2];
    assert(job->poll(completed, 2) == 1);
    assert(completed[0].terminal == llama_kv_rerank_stage_terminal::succeeded);
    assert(completed[0].task.candidate_index == 4 && completed[0].task.row_offset == 10);
    assert(job->submit(source, 0, 2, task, unused) == llama_kv_rerank_stage_status::ok);
    // Fail a terminal key-reader event and make sure production ownership latches it.
    const uint64_t failed_ticket = fake.events.back().id;
    fake.complete(failed_ticket, true);
    assert(job->poll(completed, 2) == 1 && job->failed());
    assert(fake.releases == 2);
    job->cancel();
    assert(job->poll(completed, 2) == 1);
    assert(completed[0].terminal == llama_kv_rerank_stage_terminal::cancelled);
    data->version = 10;
    task.content_version = 10;
    job.reset();

    auto owner_job = llama_kv_router_job::create(backend, pinned_buft, 4, callbacks);
    assert(owner_job != nullptr);
    llama_kv_router_query_identity identity;
    identity.sequence_id = 0;
    identity.session_generation = 1;
    identity.turn_id = 7;
    identity.query_generation = 3;
    identity.table_epoch = 11;
    identity.content_generation = 2;
    llama_kv_prefetch_candidate coarse;
    coarse.identity.sequence_id = 0;
    coarse.identity.session_generation = 1;
    coarse.identity.sequence_generation = 5;
    coarse.identity.page_generation = 9;
    coarse.identity.logical_page = 4;
    coarse.content_version = coarse.summary_version = 9;
    coarse.attention_layer = 0;
    coarse.generation = 3;
    coarse.table_epoch = 11;
    coarse.speculation_generation = 5;
    coarse.cold = true;
    coarse.provenance = llama_kv_prefetch_candidate::score_kind::probe_softmax;
    coarse.peak_probability = coarse.mean_probability = 1.0f;
    assert(owner_job->retain_coarse_shortlist(identity, 0, { coarse }));
    llama_kv_router_exact_page_record exact_record;
    exact_record.candidate = coarse;
    exact_record.descriptor_content_version = exact_record.descriptor_summary_version = 9;
    assert(!owner_job->complete_query(identity, false, { exact_record }));
    assert(!owner_job->complete_query(identity, true, { exact_record }));
    identity.turn_id = 8;
    assert(owner_job->retain_coarse_shortlist(identity, 0, { coarse }));
    assert(!owner_job->complete_query(identity, true, { exact_record, exact_record }));
    assert(owner_job->retain_coarse_shortlist(identity, 0, { coarse }));
    assert(owner_job->complete_query(identity, true, { exact_record }));
    assert(owner_job->owner_state().stage == llama_kv_router_owner_stage::ready);
    assert(owner_job->exact_candidates().size() == 1);
    assert(owner_job->exact_candidates()[0].provenance ==
            llama_kv_prefetch_candidate::score_kind::exact_mass);
    assert(owner_job->complete_query(identity, false, {})); // repeated completion is idempotent
    auto stale_identity = identity;
    ++stale_identity.session_generation; // same turn must not adopt another session's terminal result
    assert(!owner_job->complete_query(stale_identity, true, { exact_record }));
    assert(owner_job->exact_candidates().empty());
    owner_job.reset();

    fake_stage_backend partial;
    partial.fail_enqueue = true;
    callbacks.context = &partial;
    auto second_job = llama_kv_router_job::create(backend, pinned_buft, 4, callbacks);
    assert(second_job != nullptr);
    task.content_version = 9;
    assert(second_job->submit(source, 0, 2, task, unused) ==
            llama_kv_rerank_stage_status::stale_source);
    data->version = 10;
    task.content_version = 10;
    assert(second_job->submit(source, 0, 2, task, unused) ==
            llama_kv_rerank_stage_status::enqueue_failed);
    assert(second_job->poll(completed, 2) == 0 && !second_job->failed());
    assert(partial.releases == 1);
    second_job->cancel();
    second_job->cancel();
    second_job.reset();
    ggml_backend_free(backend);
}

static int run_cuda_executor_fixture(ggml_backend_t backend, ggml_backend_dev_t device,
        bool through_cache_adapter = false, uint32_t stream = 0,
        bool nonfinite_query = false, bool needle_case = false) {
    // Two encoded rows fit in the owned slot: three cold submissions exercise
    // both slot reuse and the one-row partial tail.
    constexpr int64_t dim = 128, heads = 4, probes_count = 4;
    const int64_t rows = needle_case ? 256 : 7;
    constexpr uint64_t capture_generation = 701;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_TURBO4_0, dim * 2);
    ggml_init_params params = {};
    params.mem_size = 1 << 20;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);
    ggml_tensor * probes = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, heads, probes_count);
    ggml_tensor * resident = ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, dim * 2, rows, 2);
    ggml_tensor * validity = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 9);
    assert(probes && resident && validity);
    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(input_buffer != nullptr);

    std::vector<float> query(size_t(dim * heads * probes_count), nonfinite_query
            ? std::numeric_limits<float>::quiet_NaN() : 1.0f);
    if (needle_case) for (int64_t probe = 0; probe < probes_count; ++probe) {
        const float direction = probe % 2 == 0 ? 1.0f : -1.0f;
        for (int64_t head = 0; head < heads; ++head) for (int64_t d = 0; d < dim; ++d) {
            query[size_t(d + dim * (head + heads * probe))] = direction;
        }
    }
    std::vector<uint8_t> resident_bytes(size_t(rows) * row_bytes * 2);
    std::vector<float> row(size_t(dim * 2), needle_case ? -0.25f : 1.0f);
    for (uint32_t s = 0; s < 2; ++s) for (int64_t r = 0; r < rows; ++r) {
        quantize_row_turbo4_0_ref(row.data(), reinterpret_cast<block_turbo4_0 *>(
                resident_bytes.data() + (size_t(s) * rows + size_t(r)) * row_bytes), dim * 2);
    }
    int64_t valid[9] = { int64_t(capture_generation), 5, 4, 3, 2, 1, 1, 1, 1 };
    if (needle_case) {
        valid[1] = 511;
        valid[2] = 510;
        valid[3] = 508;
        valid[4] = 504;
    }
    ggml_backend_tensor_set(probes, query.data(), 0, query.size() * sizeof(float));
    ggml_backend_tensor_set(resident, resident_bytes.data(), 0, resident_bytes.size());
    ggml_backend_tensor_set(validity, valid, 0, sizeof(valid));

    std::vector<uint8_t> cold_bytes(size_t(rows) * row_bytes);
    row.assign(size_t(dim * 2), -0.5f);
    for (int64_t r = 0; r < rows; ++r) {
        if (needle_case && r + 1 == rows) row.assign(size_t(dim * 2), 0.5f);
        quantize_row_turbo4_0_ref(row.data(), reinterpret_cast<block_turbo4_0 *>(
                cold_bytes.data() + size_t(r) * row_bytes), dim * 2);
    }
    auto cold_chain = std::make_shared<artifact_segment_chain>(cold_bytes.size());
    assert(cold_chain->append(cold_bytes.data(), cold_bytes.size()));
    vbr_selected_page_unit_descriptor cold_unit;
    cold_unit.logical_unit_id = 44;
    cold_unit.layer = stream == 0 ? 0 : 11;
    cold_unit.side = vbr_artifact_side::key;
    cold_unit.valid_rows = rows;
    cold_unit.row_bytes = row_bytes;
    cold_unit.bytes = cold_chain;

    llama_kv_router_query_identity identity;
    identity.sequence_id = 0;
    identity.session_generation = 1;
    identity.turn_id = 19;
    identity.query_generation = 31;
    identity.rollback_generation = 2;
    identity.table_epoch = 7;
    identity.content_generation = 4;
    auto make_candidate = [&](uint32_t logical, uint32_t page_generation,
            uint64_t version, bool cold, float coarse_score) {
        llama_kv_prefetch_candidate candidate;
        candidate.identity.sequence_id = 0;
        candidate.identity.session_generation = 1;
        candidate.identity.sequence_generation = 5;
        candidate.identity.page_generation = page_generation;
        candidate.identity.logical_page = logical;
        candidate.identity.position_begin = int64_t(logical * 256);
        candidate.identity.position_end = candidate.identity.position_begin + 256;
        candidate.identity.attention_layer = stream;
        candidate.attention_layer = stream;
        candidate.generation = identity.query_generation;
        candidate.table_epoch = identity.table_epoch;
        candidate.rollback_generation = identity.rollback_generation;
        candidate.speculation_generation = 5;
        candidate.content_version = candidate.summary_version = version;
        candidate.cold = cold;
        candidate.provenance = llama_kv_prefetch_candidate::score_kind::probe_softmax;
        candidate.score = coarse_score;
        candidate.peak_probability = candidate.mean_probability = coarse_score;
        return candidate;
    };
    const auto resident_candidate = make_candidate(0, 9, 9, false, 0.99f);
    const auto cold_candidate = make_candidate(1, 10, 10, true, 0.01f);
    key_reader_state cold_reader_state;
    cold_reader_state.content = cold_reader_state.summary = 10;
    auto cold_reader = llama_kv_router_key_reader::create(cold_candidate.identity,
            cold_candidate.content_version, cold_candidate.summary_version,
            cold_unit, rows, &cold_reader_state, &key_reader_state::current);
    assert(cold_reader != nullptr);
    auto job = llama_kv_router_job::create(backend,
            ggml_backend_dev_host_buffer_type(device), row_bytes * (needle_case ? rows : 2), {},
            GGML_TYPE_TURBO4_0, dim * 2);
    assert(job != nullptr);
    assert(job->retain_coarse_shortlist(identity, stream, { resident_candidate, cold_candidate }));

    llama_kv_router_layer_plan plan;
    plan.identity = identity;
    plan.compact_layer = stream;
    plan.model_layer = stream == 0 ? 0 : 11;
    plan.probe_capture_generation = capture_generation;
    plan.probes = probes;
    plan.validity = validity;
    plan.resident_keys = resident;
    plan.page_tokens = 256;
    plan.query_heads = heads;
    plan.kv_heads = 2;
    plan.key_dimension = dim;
    plan.attention_scale = 0.1f;
    plan.pages.resize(2);
    auto & resident_page = plan.pages[0];
    resident_page.candidate = resident_candidate;
    resident_page.valid_rows = rows;
    resident_page.stream = stream;
    resident_page.physical_slot = 0;
    resident_page.page_generation = 9;
    resident_page.content_generation = 9;
    resident_page.summary_generation = 9;
    auto & cold_page = plan.pages[1];
    cold_page.candidate = cold_candidate;
    cold_page.valid_rows = rows;
    cold_page.stream = stream;
    cold_page.page_generation = 10;
    cold_page.content_generation = 10;
    cold_page.summary_generation = 10;
    cold_page.key_source = cold_reader->source();
    assert(llama_kv_router_layer_plan_metadata_valid(plan));
    std::vector<llama_kv_router_exact_page_record> records;
    llama_kv_router_execution_status execution;
    cache_adapter_fixture cache_fixture;
    if (through_cache_adapter) {
        cache_fixture.plans = { plan };
        cache_fixture.job = job.get();
        llama_kv_router_query_executor gate;
        const auto result = llama_kv_router_cache_adapter::execute(gate,
                identity.sequence_id, identity.turn_id, &cache_fixture,
                &cache_adapter_fixture::prepare, &cache_fixture,
                &cache_adapter_fixture::execute, &cache_fixture,
                &cache_adapter_fixture::policy);
        execution = result.status;
        records = result.records;
        assert(result.layer == stream && result.plan_error == llama_kv_router_plan_error::none);
        if (!nonfinite_query && cache_fixture.selected.size() != 2) std::fprintf(stderr,
                "cache adapter selection mismatch stream=%u records=%zu exact=%zu selected=%zu\n",
                stream, records.size(), job->exact_candidates().size(), cache_fixture.selected.size());
        if (nonfinite_query) {
            assert(cache_fixture.selected.empty());
            assert(job->owner_state().stage == llama_kv_router_owner_stage::failed);
            assert(job->exact_candidates().empty());
        } else {
            assert(cache_fixture.selected.size() == 2);
            assert(cache_fixture.selected[0].representative.identity.logical_page == 1);
            assert(cache_fixture.selected[0].cold);
        }
        assert(plan.identity.query_generation != plan.probe_capture_generation);
        const auto before_repeat = job->execution();
        const auto repeated_adapter = llama_kv_router_cache_adapter::execute(gate,
                identity.sequence_id, identity.turn_id, &cache_fixture,
                &cache_adapter_fixture::prepare, &cache_fixture,
                &cache_adapter_fixture::execute, &cache_fixture,
                &cache_adapter_fixture::policy);
        assert(repeated_adapter.status == (nonfinite_query
                    ? llama_kv_router_execution_status::failed
                    : llama_kv_router_execution_status::ready) &&
                repeated_adapter.records.size() == records.size());
        const auto after_repeat = job->execution();
        assert(before_repeat.cold_reader_graphs == after_repeat.cold_reader_graphs &&
                before_repeat.mass_graphs == after_repeat.mass_graphs);
        std::fprintf(stderr,
                "cache_adapter final_user=sequence:%d/session:%llu/turn:%llu/query:%llu/rollback:%llu/table:%llu/content:%llu "
            "capture_generation=%llu compact_layer=%u model_layer=%u stream=%u source_unit=%u "
                "source_bytes=%zu reader_submissions=%u rerank_graphs=%u mass_graphs=%u "
                "terminal_records=%zu exact_batch=%zu proposed_cold_page=%u published=0 repeated_start=0\n",
                identity.sequence_id, (unsigned long long)identity.session_generation,
                (unsigned long long)identity.turn_id, (unsigned long long)identity.query_generation,
                (unsigned long long)identity.rollback_generation,
                (unsigned long long)identity.table_epoch,
                (unsigned long long)identity.content_generation,
                (unsigned long long)plan.probe_capture_generation, plan.compact_layer,
                plan.model_layer, stream, cold_unit.logical_unit_id, cold_bytes.size(),
                after_repeat.reader_events, after_repeat.cold_reader_graphs,
                after_repeat.mass_graphs, records.size(), cache_fixture.selected.size(),
                cache_fixture.selected.empty() ? UINT32_MAX :
                    cache_fixture.selected.front().representative.identity.logical_page);
    } else {
        execution = job->execute_query(identity, { plan }, records);
    }
    if (execution != llama_kv_router_execution_status::ready) {
        std::fprintf(stderr, "execute_query status=%u failed=%d stage=%u\n",
                unsigned(execution), int(job->failed()), unsigned(job->owner_state().stage));
    }
    assert(execution == (nonfinite_query ? llama_kv_router_execution_status::failed
            : llama_kv_router_execution_status::ready));
    assert(nonfinite_query ? records.empty() && job->exact_candidates().empty()
            : records.size() == 2 && job->exact_candidates().size() == 2);
    if (nonfinite_query) {
        assert(job->owner_state().stage == llama_kv_router_owner_stage::failed);
        std::vector<llama_kv_router_exact_page_record> repeated;
        assert(job->execute_query(identity, { plan }, repeated) ==
                llama_kv_router_execution_status::failed && repeated.empty());
        ggml_backend_buffer_free(input_buffer);
        ggml_free(ctx);
        return 0;
    }
    for (size_t i = 0; i < records.size(); ++i) {
        assert(records[i].descriptor_content_version == records[i].candidate.content_version);
        assert(records[i].descriptor_summary_version == records[i].candidate.summary_version);
        assert(job->exact_candidates()[i].provenance ==
                llama_kv_prefetch_candidate::score_kind::exact_mass);
        assert(job->exact_candidates()[i].content_version == records[i].candidate.content_version);
    }
    const auto & execution_ledger = job->execution();
    const uint32_t expected_cold_chunks = needle_case ? 1 : 4;
    assert(execution_ledger.resident_rerank_graphs == 1 &&
            execution_ledger.cold_reader_graphs == expected_cold_chunks &&
            execution_ledger.reader_events == expected_cold_chunks && execution_ledger.mass_graphs == 1 &&
            execution_ledger.output_records == 2);
    const auto & allocation_ledger = job->allocations();
    std::fprintf(stderr,
            "owned_executor graphs resident=%u cold=%u mass=%u reader_events=%u records=%u "
            "state_bytes=256 mass_temp_bytes=128 descriptors_bytes=160 identity_bytes=48 "
            "rerank_output_bytes=128 records_bytes=32 slot_bytes=%zu device_requested=%llu "
            "device_realized=%llu pinned_requested=%llu pinned_realized=%llu\n",
            execution_ledger.resident_rerank_graphs, execution_ledger.cold_reader_graphs,
            execution_ledger.mass_graphs, execution_ledger.reader_events,
            execution_ledger.output_records, job->slot_bytes(),
            (unsigned long long)allocation_ledger.device_requested,
            (unsigned long long)allocation_ledger.device_realized,
            (unsigned long long)allocation_ledger.pinned_requested,
            (unsigned long long)allocation_ledger.pinned_realized);
    assert(records[1].candidate.peak_probability > records[0].candidate.peak_probability);
    assert(std::fabs(records[0].candidate.mean_probability +
            records[1].candidate.mean_probability - 1.0f) < 0.02f);
    if (needle_case) {
        // Independent decoded-Turbo4 oracle: one opposing-query channel sees
        // the final key row as a needle among 255 distractors. The exact LSE
        // for each page is reduced across its actual encoded rows, then page
        // masses are normalized across resident and cold competitors.
        std::vector<float> decoded(size_t(dim * 2));
        double mass_sum[2] = { 0.0, 0.0 };
        double mass_peak[2] = { 0.0, 0.0 };
        int channels = 0;
        for (int64_t probe = 0; probe < probes_count; ++probe) for (int64_t head = 0; head < heads; ++head) {
            double page_lse[2] = { -INFINITY, -INFINITY };
            for (int page = 0; page < 2; ++page) {
                const auto & encoded = page == 0 ? resident_bytes : cold_bytes;
                double maximum = -INFINITY, sum = 0.0;
                for (int64_t r = 0; r < rows; ++r) {
                    const uint8_t * row_bytes_ptr = encoded.data() +
                        ((page == 0 ? size_t(stream) * size_t(rows) : 0) + size_t(r)) * row_bytes;
                    dequantize_row_turbo4_0(reinterpret_cast<const block_turbo4_0 *>(row_bytes_ptr),
                            decoded.data(), dim * 2);
                    const int64_t kv_head = head / (heads / 2);
                    double dot = 0.0;
                    for (int64_t d = 0; d < dim; ++d) {
                        dot += double(query[size_t(d + dim * (head + heads * probe))]) *
                            decoded[size_t(kv_head * dim + d)];
                    }
                    const double logit = dot * plan.attention_scale;
                    if (logit > maximum) { sum = sum * std::exp(maximum - logit) + 1.0; maximum = logit; }
                    else sum += std::exp(logit - maximum);
                }
                page_lse[page] = maximum + std::log(sum);
            }
            const double maximum = std::max(page_lse[0], page_lse[1]);
            const double p0 = std::exp(page_lse[0] - maximum) /
                (std::exp(page_lse[0] - maximum) + std::exp(page_lse[1] - maximum));
            const double p1 = 1.0 - p0;
            mass_sum[0] += p0; mass_sum[1] += p1;
            mass_peak[0] = std::max(mass_peak[0], p0); mass_peak[1] = std::max(mass_peak[1], p1);
            ++channels;
        }
        for (const auto & record : records) {
            const int page = int(record.candidate.identity.logical_page);
            assert(page == 0 || page == 1);
            assert(std::fabs(record.candidate.mean_probability - mass_sum[page] / channels) < 0.01);
            assert(std::fabs(record.candidate.peak_probability - mass_peak[page]) < 0.01);
        }
        assert(records[1].candidate.identity.logical_page == 1);
        std::fprintf(stderr, "decoded Turbo4 needle oracle rows=256 distractors=255 opposing_probes=4 exact_cold_mass=%.6f\n",
                mass_sum[1] / channels);
    }
    std::vector<llama_kv_router_exact_page_record> repeated;
    assert(job->execute_query(identity, { plan }, repeated) ==
            llama_kv_router_execution_status::ready && repeated.size() == records.size());
    job.reset();

    // A separate actual GPU invocation covers a resident-only layer plan.
    auto resident_only_identity = identity;
    resident_only_identity.turn_id += 1;
    auto resident_only_job = llama_kv_router_job::create(backend,
            ggml_backend_dev_host_buffer_type(device), row_bytes * 2, {},
            GGML_TYPE_TURBO4_0, dim * 2);
    assert(resident_only_job != nullptr &&
            resident_only_job->retain_coarse_shortlist(resident_only_identity, stream,
                { resident_candidate }));
    llama_kv_router_layer_plan resident_only_plan = plan;
    resident_only_plan.identity = resident_only_identity;
    resident_only_plan.pages.resize(1);
    resident_only_plan.pages[0] = resident_page;
    resident_only_plan.pages[0].candidate = resident_candidate;
    std::vector<llama_kv_router_exact_page_record> resident_only_records;
    assert(resident_only_job->execute_query(resident_only_identity,
            { resident_only_plan }, resident_only_records) ==
            llama_kv_router_execution_status::ready);
    assert(resident_only_records.size() == 1 &&
            resident_only_job->execution().resident_rerank_graphs == 1 &&
            resident_only_job->execution().cold_reader_graphs == 0 &&
            resident_only_job->execution().reader_events == 0 &&
            resident_only_job->execution().mass_graphs == 1);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(ctx);
    return 0;
}

static int run_cuda_owner_contract(const char *) {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name("CUDA0");
    if (device == nullptr) return 77;
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) return 77;
    const int result = run_cuda_executor_fixture(backend, device);
    ggml_backend_free(backend);
    return result;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    if (std::string(argv[1]) == "--cpu-only") {
        run_cpu_lifecycle();
        return 0;
    }
    if (std::string(argv[1]) == "--cuda") return run_cuda_owner_contract(argv[0]);
    if (std::string(argv[1]) == "--cache-cuda") {
        ggml_backend_load_all();
        ggml_backend_dev_t device = ggml_backend_dev_by_name("CUDA0");
        if (device == nullptr) return 77;
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (backend == nullptr) return 77;
        int result = run_cuda_executor_fixture(backend, device, true, 0);
        if (result == 0) result = run_cuda_executor_fixture(backend, device, true, 1);
        if (result == 0) result = run_cuda_executor_fixture(backend, device, true, 1, true);
        if (result == 0) result = run_cuda_executor_fixture(backend, device, true, 0, false, true);
        ggml_backend_free(backend);
        return result;
    }
    return 2;
}
