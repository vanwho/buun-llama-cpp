#include "llama-kv-router-job.h"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <memory>
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

static void run_cpu_lifecycle() {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(backend != nullptr);
    const auto pinned_buft = ggml_backend_cpu_buffer_type();
    fake_stage_backend fake;
    llama_kv_rerank_stage_backend callbacks { &fake, nullptr,
        &fake_stage_backend::poll, &fake_stage_backend::cancel,
        &fake_stage_backend::release, &fake_stage_backend::enqueue };
    auto job = llama_kv_router_job::create(backend, pinned_buft, 4, callbacks);
    assert(job != nullptr && job->slot_bytes() == 4);
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
    ++stale_identity.query_generation;
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

static int run_cuda_owner_contract(const char * executable) {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name("CUDA0");
    if (device == nullptr) return 77;
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) return 77;
    auto job = llama_kv_router_job::create(backend,
            ggml_backend_dev_host_buffer_type(device), 4, {});
    assert(job != nullptr);
    auto data = std::make_shared<source_data>();
    std::shared_ptr<const void> holder(data, data.get());
    llama_kv_rerank_stage_source source { holder, data.get(),
        &source_data::recheck, &source_data::read };
    llama_kv_rerank_chunk_task task;
    task.candidate_index = 0;
    task.compact_layer_index = 0;
    task.rows = 2;
    task.row_bytes = 2;
    task.query_generation = 31;
    task.content_version = 9;
    uint32_t slot = UINT32_MAX;
    uint64_t ticket = 0;
    assert(job->submit(source, 0, 2, task, slot, &ticket) ==
            llama_kv_rerank_stage_status::ok);
    llama_kv_rerank_stage_ticket completed[2];
    uint32_t count = 0;
    for (uint32_t i = 0; i < 1000000 && count == 0; ++i) count = job->poll(completed, 2);
    assert(count == 1 && completed[0].ticket == ticket &&
            completed[0].terminal == llama_kv_rerank_stage_terminal::succeeded);
    uint8_t copied[4]{};
    ggml_backend_tensor_get(job->encoded_key_slot(slot), copied, 0, sizeof(copied));
    assert(std::memcmp(copied, data->bytes.data(), sizeof(copied)) == 0);

    llama_kv_router_query_identity identity;
    identity.sequence_id = 0;
    identity.session_generation = 1;
    identity.turn_id = 19;
    identity.query_generation = 31;
    identity.rollback_generation = 2;
    identity.table_epoch = 7;
    identity.content_generation = 4;
    llama_kv_prefetch_candidate coarse;
    coarse.identity.sequence_id = 0;
    coarse.identity.session_generation = 1;
    coarse.identity.sequence_generation = 5;
    coarse.identity.page_generation = 9;
    coarse.identity.logical_page = 4;
    coarse.attention_layer = 0;
    coarse.generation = 31;
    coarse.table_epoch = 7;
    coarse.rollback_generation = 2;
    coarse.speculation_generation = 5;
    coarse.content_version = coarse.summary_version = 9;
    coarse.cold = true;
    coarse.provenance = llama_kv_prefetch_candidate::score_kind::probe_softmax;
    coarse.peak_probability = coarse.mean_probability = 1.0f;
    assert(job->retain_coarse_shortlist(identity, 0, { coarse }));
    llama_kv_router_exact_page_record record;
    record.candidate = coarse;
    record.descriptor_content_version = record.descriptor_summary_version = 9;
    assert(job->complete_query(identity, true, { record }));
    assert(job->exact_candidates().size() == 1 &&
            job->exact_candidates()[0].provenance ==
                llama_kv_prefetch_candidate::score_kind::exact_mass);
    job.reset();
    ggml_backend_free(backend);

    const auto directory = std::filesystem::absolute(executable).parent_path();
    const auto rank_path = directory / "test-kv-page-select";
    const auto mass_path = directory / "test-cuda-kv-page-summary";
    const std::string rank_command = "LLAMA_TEST_KV_PAGE_SELECT_CUDA=1 \"" +
        rank_path.string() + "\"";
    const std::string mass_command = "\"" + mass_path.string() + "\"";
    if (std::system(rank_command.c_str()) != 0 || std::system(mass_command.c_str()) != 0) return 1;
    return 0;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    if (std::string(argv[1]) == "--cpu-only") {
        run_cpu_lifecycle();
        return 0;
    }
    if (std::string(argv[1]) == "--cuda") return run_cuda_owner_contract(argv[0]);
    return 2;
}
