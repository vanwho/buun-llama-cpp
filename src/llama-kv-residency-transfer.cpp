#include "llama-kv-residency-transfer.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <new>
#include <utility>
#include <chrono>

namespace {

bool add_u64(uint64_t a, uint64_t b, uint64_t & out) noexcept {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

bool mul_u64(uint64_t a, uint64_t b, uint64_t & out) noexcept {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

bool round_up(uint64_t value, uint64_t alignment, uint64_t & output) noexcept {
    if (alignment == 0) {
        return false;
    }
    const uint64_t remainder = value % alignment;
    if (remainder == 0) {
        output = value;
        return true;
    }
    return add_u64(value, alignment - remainder, output);
}

bool same_run_prefix(
        const llama_kv_residency_transfer_run & a,
        const llama_kv_residency_transfer_run & b) noexcept {
    uint64_t a_rows = 0;
    return a.page_index == b.page_index && a.lane == b.lane &&
        a.layer == b.layer && a.side == b.side &&
        a.row_bytes == b.row_bytes &&
        add_u64(a.first_physical_row, a.row_count, a_rows) &&
        a_rows == b.first_physical_row &&
        add_u64(a.host_offset, a.useful_bytes(), a_rows) &&
        a_rows == b.host_offset &&
        add_u64(a.device_offset, a.useful_bytes(), a_rows) &&
        a_rows == b.device_offset;
}

struct run_context {
    const llama_kv_residency_pool_backend * backend = nullptr;
    const llama_kv_residency_transfer_transport * transport = nullptr;
    llama_kv_residency_transfer_direction direction =
        llama_kv_residency_transfer_direction::d2h_seal;
    llama_kv_residency_completion completion;
    const llama_kv_residency_transfer_run * run = nullptr;
    llama_kv_residency_transfer_result * result = nullptr;
    bool completed = false;
    bool cancelled = false;
};

bool transfer_continue(void * opaque) noexcept {
    auto & context = *static_cast<run_context *>(opaque);
    return !context.transport->continue_transfer ||
        context.transport->continue_transfer(context.transport->context);
}

bool d2h_issue(
        const void * opaque, uint64_t offset, uint8_t * destination,
        size_t size, uint64_t ticket, bool asynchronous) noexcept {
    auto & context = *static_cast<run_context *>(const_cast<void *>(opaque));
    if (!context.backend->issue_copy(
            context.backend->context, context.direction, context.completion,
            context.run->device_offset + offset, destination, size, ticket,
            asynchronous)) {
        return false;
    }
    if (!asynchronous) {
        context.result->counters.waits++;
        context.completed = context.backend->complete_copy(
            context.backend->context, ticket);
        return context.completed;
    }
    return true;
}

bool d2h_complete(const void * opaque, uint64_t ticket) noexcept {
    auto & context = *static_cast<run_context *>(const_cast<void *>(opaque));
    context.result->counters.waits++;
    context.completed = context.backend->complete_copy(
        context.backend->context, ticket);
    return context.completed;
}

void d2h_cancel(const void * opaque, uint64_t ticket) noexcept {
    auto & context = *static_cast<run_context *>(const_cast<void *>(opaque));
    context.cancelled = true;
    context.result->counters.cancellations++;
    if (context.backend->cancel_copy) {
        context.backend->cancel_copy(context.backend->context, ticket);
    } else {
        context.result->counters.waits++;
        (void) context.backend->complete_copy(
            context.backend->context, ticket);
    }
}

bool h2d_read(
        const void * opaque, uint64_t offset, uint8_t * destination,
        size_t size) noexcept {
    const auto & context = *static_cast<const run_context *>(opaque);
    return context.transport->host_read(
        context.transport->context, context.run->page_index,
        context.run->host_offset + offset, destination, size);
}

bool h2d_issue(
        void * opaque, uint64_t ticket, uint64_t offset,
        const uint8_t * data, size_t size, bool asynchronous) noexcept {
    auto & context = *static_cast<run_context *>(opaque);
    return context.backend->issue_copy(
        context.backend->context, context.direction, context.completion,
        context.run->device_offset + offset,
        const_cast<uint8_t *>(data), size, ticket, asynchronous);
}

bool h2d_complete(void * opaque, uint64_t ticket) noexcept {
    auto & context = *static_cast<run_context *>(opaque);
    context.result->counters.waits++;
    context.completed = context.backend->complete_copy(
        context.backend->context, ticket);
    return context.completed;
}

void h2d_cancel(void * opaque, uint64_t ticket) noexcept {
    auto & context = *static_cast<run_context *>(opaque);
    context.cancelled = true;
    context.result->counters.cancellations++;
    if (context.backend->cancel_copy) {
        context.backend->cancel_copy(context.backend->context, ticket);
    } else {
        context.result->counters.waits++;
        (void) context.backend->complete_copy(
            context.backend->context, ticket);
    }
}

llama_kv_residency_pool_status map_stream_status(
        vbr_capture_stream_status status) noexcept {
    switch (status) {
        case vbr_capture_stream_status::ok:
            return llama_kv_residency_pool_status::ok;
        case vbr_capture_stream_status::cancelled:
            return llama_kv_residency_pool_status::cancelled;
        case vbr_capture_stream_status::ring_unavailable:
            return llama_kv_residency_pool_status::staging_unavailable;
        default:
            return llama_kv_residency_pool_status::transfer_failed;
    }
}

llama_kv_residency_pool_status map_h2d_status(
        vbr_h2d_status status) noexcept {
    switch (status) {
        case vbr_h2d_status::ok:
            return llama_kv_residency_pool_status::ok;
        case vbr_h2d_status::cancelled:
            return llama_kv_residency_pool_status::cancelled;
        case vbr_h2d_status::ring_unavailable:
            return llama_kv_residency_pool_status::staging_unavailable;
        default:
            return llama_kv_residency_pool_status::transfer_failed;
    }
}

} // namespace

const char * llama_kv_residency_ggml_status_name(
        llama_kv_residency_ggml_status status) noexcept {
    switch (status) {
        case llama_kv_residency_ggml_status::ok: return "ok";
        case llama_kv_residency_ggml_status::invalid_argument: return "invalid_argument";
        case llama_kv_residency_ggml_status::unavailable: return "unavailable";
        case llama_kv_residency_ggml_status::allocation_failed: return "allocation_failed";
        case llama_kv_residency_ggml_status::event_failed: return "event_failed";
        case llama_kv_residency_ggml_status::_count: break;
    }
    return "invalid";
}

struct llama_kv_residency_ggml_adapter::pending_event {
    uint64_t ticket = 0;
    ggml_backend_event_t event = nullptr;
};

llama_kv_residency_ggml_adapter::llama_kv_residency_ggml_adapter(
        const llama_kv_residency_ggml_adapter_config & config) noexcept
    : backend_(config.backend),
      device_(config.backend ? ggml_backend_get_device(config.backend) : nullptr),
      buffer_(config.buffer), slot_capacity_(config.slot_capacity),
      bytes_per_slot_(config.bytes_per_slot),
      force_synchronous_(config.force_synchronous) {}

llama_kv_residency_ggml_adapter::~llama_kv_residency_ggml_adapter() {
    drain();
    release_slots();
}

std::unique_ptr<llama_kv_residency_ggml_adapter>
llama_kv_residency_ggml_adapter::create(
        const llama_kv_residency_ggml_adapter_config & config,
        llama_kv_residency_ggml_status & status) noexcept {
    status = llama_kv_residency_ggml_status::invalid_argument;
    if (!config.backend || !config.buffer || config.slot_capacity == 0 ||
        config.bytes_per_slot == 0 ||
        config.bytes_per_slot > SIZE_MAX/config.slot_capacity) {
        return nullptr;
    }
    try {
        auto output = std::unique_ptr<llama_kv_residency_ggml_adapter>(
                new (std::nothrow) llama_kv_residency_ggml_adapter(config));
        if (!output || ggml_backend_buffer_get_size(config.buffer) <
                size_t(config.bytes_per_slot * config.slot_capacity) ||
            !output->initialize_slots()) {
            status = llama_kv_residency_ggml_status::allocation_failed;
            return nullptr;
        }
        status = llama_kv_residency_ggml_status::ok;
        return output;
    } catch (...) {
        status = llama_kv_residency_ggml_status::allocation_failed;
        return nullptr;
    }
}

bool llama_kv_residency_ggml_adapter::initialize_slots() noexcept {
    try {
        const size_t bytes = size_t(bytes_per_slot_ * slot_capacity_);
        const size_t overhead = ggml_tensor_overhead();
        if (overhead == 0 || slot_capacity_ + 1 > SIZE_MAX/overhead) return false;
        tensor_context_ = ggml_init({ size_t(slot_capacity_ + 1)*overhead,
                                      nullptr, true });
        if (!tensor_context_) return false;
        void * base = ggml_backend_buffer_get_base(buffer_);
        if (!base) return false;
        storage_tensor_ = ggml_new_tensor_1d(
                tensor_context_, GGML_TYPE_I8, int64_t(bytes));
        if (!storage_tensor_ || ggml_backend_tensor_alloc(
                buffer_, storage_tensor_, base) != GGML_STATUS_SUCCESS) {
            release_slots();
            return false;
        }
        slot_tensors_.reserve(slot_capacity_);
        mapped_.assign(slot_capacity_, 0);
        for (uint32_t slot = 0; slot < slot_capacity_; ++slot) {
            auto * tensor = ggml_view_1d(
                    tensor_context_, storage_tensor_, int64_t(bytes_per_slot_),
                    size_t(slot)*bytes_per_slot_);
            if (!tensor || ggml_backend_view_init(tensor) !=
                    GGML_STATUS_SUCCESS) {
                release_slots();
                return false;
            }
            slot_tensors_.push_back(tensor);
        }
        return slot_tensors_.size() == slot_capacity_;
    } catch (...) {
        release_slots();
        return false;
    }
}

void llama_kv_residency_ggml_adapter::release_slots() noexcept {
    pending_.clear();
    mapped_.clear();
    slot_tensors_.clear();
    storage_tensor_ = nullptr;
    if (tensor_context_) {
        ggml_free(tensor_context_);
        tensor_context_ = nullptr;
    }
}

llama_kv_residency_pool_backend
llama_kv_residency_ggml_adapter::pool_backend() noexcept {
    return {
        this,
        &llama_kv_residency_ggml_adapter::reserve_slots,
        &llama_kv_residency_ggml_adapter::release_slots,
        &llama_kv_residency_ggml_adapter::map_slot,
        &llama_kv_residency_ggml_adapter::drop_slot,
        &llama_kv_residency_ggml_adapter::issue_copy,
        &llama_kv_residency_ggml_adapter::complete_copy,
        &llama_kv_residency_ggml_adapter::cancel_copy,
        backend_, &llama_kv_residency_ggml_adapter::slot_tensor,
    };
}

uint32_t llama_kv_residency_ggml_adapter::mapped_slots() const noexcept {
    return uint32_t(std::count(mapped_.begin(), mapped_.end(), uint8_t(1)));
}

uint32_t llama_kv_residency_ggml_adapter::pending_events() const noexcept {
    return uint32_t(pending_.size());
}

void llama_kv_residency_ggml_adapter::drain() noexcept {
    for (auto & pending : pending_) {
        if (pending.event) {
            ggml_backend_event_synchronize(pending.event);
            ggml_backend_event_free(pending.event);
        }
    }
    pending_.clear();
}

bool llama_kv_residency_ggml_adapter::reserve_slots(
        void * context, uint32_t slots, uint64_t bytes) noexcept {
    auto * self = static_cast<llama_kv_residency_ggml_adapter *>(context);
    return self && slots == self->slot_capacity_ && bytes == self->bytes_per_slot_;
}

void llama_kv_residency_ggml_adapter::release_slots(
        void *, uint32_t, uint64_t) noexcept {}

bool llama_kv_residency_ggml_adapter::map_slot(
        void * context, uint32_t slot) noexcept {
    auto * self = static_cast<llama_kv_residency_ggml_adapter *>(context);
    if (!self || slot >= self->mapped_.size() || self->mapped_[slot]) return false;
    self->mapped_[slot] = 1;
    return true;
}

bool llama_kv_residency_ggml_adapter::drop_slot(
        void * context, uint32_t slot) noexcept {
    auto * self = static_cast<llama_kv_residency_ggml_adapter *>(context);
    if (!self || slot >= self->mapped_.size() || !self->mapped_[slot]) return false;
    self->mapped_[slot] = 0;
    return true;
}

ggml_tensor * llama_kv_residency_ggml_adapter::slot_tensor(
        void * context, uint32_t slot) noexcept {
    auto * self = static_cast<llama_kv_residency_ggml_adapter *>(context);
    return self && slot < self->slot_tensors_.size()
        ? self->slot_tensors_[slot] : nullptr;
}

bool llama_kv_residency_ggml_adapter::issue(
        llama_kv_residency_transfer_direction direction,
        const llama_kv_residency_completion & completion,
        uint64_t device_offset, void * host, size_t size,
        uint64_t ticket, bool asynchronous) noexcept {
    if (completion.physical_slot >= slot_tensors_.size() || !host || size == 0 ||
        device_offset > ggml_nbytes(slot_tensors_[completion.physical_slot]) ||
        size > ggml_nbytes(slot_tensors_[completion.physical_slot]) - device_offset) {
        return false;
    }
    auto * tensor = slot_tensors_[completion.physical_slot];
    const bool async = asynchronous && !force_synchronous_ && device_ != nullptr;
    if (direction == llama_kv_residency_transfer_direction::h2d_promotion) {
        ggml_backend_tensor_set_async(
                backend_, tensor, host, size_t(device_offset), size);
    } else {
        ggml_backend_tensor_get_async(
                backend_, tensor, host, size_t(device_offset), size);
    }
    if (!async) {
        ggml_backend_synchronize(backend_);
        return true;
    }
    ggml_backend_event_t event = ggml_backend_event_new(device_);
    if (!event) return false;
    ggml_backend_event_record(event, backend_);
    try {
        pending_.push_back({ ticket, event });
        return true;
    } catch (...) {
        ggml_backend_event_synchronize(event);
        ggml_backend_event_free(event);
        return false;
    }
}

bool llama_kv_residency_ggml_adapter::complete(uint64_t ticket) noexcept {
    const auto found = std::find_if(pending_.begin(), pending_.end(),
            [ticket](const pending_event & value) { return value.ticket == ticket; });
    if (found == pending_.end()) return true;
    bool ok = true;
    if (found->event) {
        ggml_backend_event_synchronize(found->event);
        ggml_backend_event_free(found->event);
    }
    pending_.erase(found);
    return ok;
}

void llama_kv_residency_ggml_adapter::cancel(uint64_t ticket) noexcept {
    (void) complete(ticket);
}

bool llama_kv_residency_ggml_adapter::issue_copy(
        void * context, llama_kv_residency_transfer_direction direction,
        const llama_kv_residency_completion & completion,
        uint64_t device_offset, void * host, size_t size,
        uint64_t ticket, bool asynchronous) noexcept {
    auto * self = static_cast<llama_kv_residency_ggml_adapter *>(context);
    return self && self->issue(direction, completion, device_offset, host,
                               size, ticket, asynchronous);
}

bool llama_kv_residency_ggml_adapter::complete_copy(
        void * context, uint64_t ticket) noexcept {
    auto * self = static_cast<llama_kv_residency_ggml_adapter *>(context);
    return self && self->complete(ticket);
}

void llama_kv_residency_ggml_adapter::cancel_copy(
        void * context, uint64_t ticket) noexcept {
    auto * self = static_cast<llama_kv_residency_ggml_adapter *>(context);
    if (self) self->cancel(ticket);
}

uint64_t llama_kv_residency_transfer_run::useful_bytes() const noexcept {
    uint64_t result = 0;
    return mul_u64(row_count, row_bytes, result) ? result : 0;
}

bool llama_kv_residency_build_transfer_plan(
        llama_kv_residency_transfer_direction direction,
        const std::vector<llama_kv_residency_transfer_page> & pages,
        uint64_t alignment,
        const llama_kv_residency_transfer_limits & limits,
        llama_kv_residency_transfer_plan & output) noexcept {
    output = {};
    output.direction = direction;
    try {
        if (direction >= llama_kv_residency_transfer_direction::_count ||
            alignment == 0 || pages.empty() ||
            pages.size() > limits.max_pages || limits.max_runs == 0) {
            return false;
        }
        output.pages.reserve(pages.size());
        uint64_t runs = 0;
        for (size_t page_index = 0; page_index < pages.size(); ++page_index) {
            const auto & input = pages[page_index];
            if (!llama_kv_page_id_valid(input.page, false) &&
                !llama_kv_page_id_valid(input.page, true)) {
                output = {};
                return false;
            }
            for (const auto & prior : output.pages) {
                if (prior.page == input.page ||
                    prior.physical_slot == input.physical_slot) {
                    output = {};
                    return false;
                }
            }
            if (input.physical_slot == UINT32_MAX || input.runs.empty() ||
                input.runs.size() > limits.max_runs - runs) {
                output = {};
                return false;
            }
            for (const auto & run : input.runs) {
                if (run.lane == UINT32_MAX || run.layer >= VBR_SELECTED_PAGE_TARGET_LAYERS ||
                    run.side > 1 || run.row_count == 0 || run.row_bytes == 0 ||
                    run.first_physical_row > UINT32_MAX - run.row_count ||
                    run.useful_bytes() == 0) {
                    output = {};
                    return false;
                }
            }
            output.pages.push_back(input);
            ++runs;
            if (input.runs.size() > 1) {
                runs += input.runs.size() - 1;
            }
            if (runs > limits.max_runs) {
                output = {};
                return false;
            }
        }

        output.runs.reserve(size_t(runs));
        for (uint32_t page_index = 0; page_index < output.pages.size(); ++page_index) {
            auto & page = output.pages[page_index];
            for (const auto & input_run : page.runs) {
                auto run = input_run;
                run.page_index = page_index;
                if (!output.runs.empty() && same_run_prefix(output.runs.back(), run)) {
                    auto & prior = output.runs.back();
                    if (prior.row_count > UINT32_MAX - run.row_count) {
                        output = {};
                        return false;
                    }
                    prior.row_count += run.row_count;
                    continue;
                }
                output.runs.push_back(run);
            }
        }
        if (output.runs.empty() || output.runs.size() > UINT32_MAX) {
            output = {};
            return false;
        }
        output.event_count = uint32_t(output.runs.size());
        for (const auto & run : output.runs) {
            const uint64_t useful = run.useful_bytes();
            uint64_t aligned = 0;
            if (!round_up(useful, alignment, aligned) ||
                !add_u64(output.useful_bytes, useful, output.useful_bytes) ||
                !add_u64(output.aligned_bytes, aligned, output.aligned_bytes) ||
                output.useful_bytes > limits.max_useful_bytes) {
                output = {};
                return false;
            }
        }
        return true;
    } catch (...) {
        output = {};
        return false;
    }
}

const char * llama_kv_residency_pool_status_name(
        llama_kv_residency_pool_status status) noexcept {
    switch (status) {
        case llama_kv_residency_pool_status::ok: return "ok";
        case llama_kv_residency_pool_status::invalid_argument: return "invalid_argument";
        case llama_kv_residency_pool_status::not_configured: return "not_configured";
        case llama_kv_residency_pool_status::backend_unavailable: return "backend_unavailable";
        case llama_kv_residency_pool_status::slot_unavailable: return "slot_unavailable";
        case llama_kv_residency_pool_status::event_unavailable: return "event_unavailable";
        case llama_kv_residency_pool_status::staging_unavailable: return "staging_unavailable";
        case llama_kv_residency_pool_status::catalog_unavailable: return "catalog_unavailable";
        case llama_kv_residency_pool_status::transfer_failed: return "transfer_failed";
        case llama_kv_residency_pool_status::cancelled: return "cancelled";
        case llama_kv_residency_pool_status::stale_completion: return "stale_completion";
        case llama_kv_residency_pool_status::dirty_page: return "dirty_page";
        case llama_kv_residency_pool_status::not_found: return "not_found";
        case llama_kv_residency_pool_status::transaction_closed: return "transaction_closed";
        case llama_kv_residency_pool_status::internal_error: return "internal_error";
        case llama_kv_residency_pool_status::_count: break;
    }
    return "invalid";
}

struct llama_kv_residency_pool::slot {
    bool mapped = false;
    bool reserved = false;
    bool host_valid = false;
    bool dirty = false;
    llama_kv_page_id page;
};

struct llama_kv_residency_pool::impl {
    llama_kv_residency_pool_config config;
    llama_kv_residency_pool_backend backend;
    std::vector<slot> slots;
    mutable std::mutex mutex;
    uint32_t reserved_slots = 0;
    uint32_t mapped_slots = 0;
    uint32_t pending_events = 0;
};

llama_kv_residency_transfer_claim::~llama_kv_residency_transfer_claim() {
    reset();
}

llama_kv_residency_transfer_claim::llama_kv_residency_transfer_claim(
        llama_kv_residency_transfer_claim && other) noexcept
    : pool_(other.pool_), slots_(std::move(other.slots_)),
      events_(other.events_), catalog_bytes_(other.catalog_bytes_),
      catalog(other.catalog), mapped_(other.mapped_), active_(other.active_) {
    other.pool_ = nullptr;
    other.events_ = 0;
    other.catalog_bytes_ = 0;
    other.catalog = {};
    other.mapped_ = false;
    other.active_ = false;
}

llama_kv_residency_transfer_claim &
llama_kv_residency_transfer_claim::operator=(
        llama_kv_residency_transfer_claim && other) noexcept {
    if (this != &other) {
        reset();
        pool_ = other.pool_;
        slots_ = std::move(other.slots_);
        events_ = other.events_;
        catalog_bytes_ = other.catalog_bytes_;
        catalog = other.catalog;
        mapped_ = other.mapped_;
        active_ = other.active_;
        other.pool_ = nullptr;
        other.events_ = 0;
        other.catalog_bytes_ = 0;
        other.catalog = {};
        other.mapped_ = false;
        other.active_ = false;
    }
    return *this;
}

bool llama_kv_residency_transfer_claim::active() const noexcept {
    return active_;
}

uint32_t llama_kv_residency_transfer_claim::reserved_slots() const noexcept {
    return active_ && !mapped_ ? uint32_t(slots_.size()) : 0;
}

uint32_t llama_kv_residency_transfer_claim::reserved_events() const noexcept {
    return active_ ? events_ : 0;
}

void llama_kv_residency_transfer_claim::reset() noexcept {
    if (active_ && pool_) {
        pool_->release_claim(*this);
    }
    if (!active_) {
        pool_ = nullptr;
        slots_.clear();
        events_ = 0;
        catalog_bytes_ = 0;
        catalog = {};
        mapped_ = false;
    }
}

llama_kv_residency_pool::llama_kv_residency_pool(
        std::unique_ptr<impl> state) noexcept
    : impl_(std::move(state)) {}

std::unique_ptr<llama_kv_residency_pool>
llama_kv_residency_pool::create(
        const llama_kv_residency_pool_config & config,
        const llama_kv_residency_pool_backend & backend,
        llama_kv_residency_pool_status & status) noexcept {
    status = llama_kv_residency_pool_status::invalid_argument;
    try {
        if (config.slot_capacity == 0 || config.bytes_per_slot == 0 ||
            config.allocation_alignment == 0 || config.event_capacity == 0 ||
            config.max_transfer_bytes == 0 || !backend.issue_copy ||
            !backend.complete_copy) {
            return nullptr;
        }
        if (backend.reserve_slots &&
            !backend.reserve_slots(
                backend.context, config.slot_capacity, config.bytes_per_slot)) {
            status = llama_kv_residency_pool_status::backend_unavailable;
            return nullptr;
        }
        std::unique_ptr<impl> state(new impl);
        state->config = config;
        state->backend = backend;
        state->slots.resize(config.slot_capacity);
        status = llama_kv_residency_pool_status::ok;
        return std::unique_ptr<llama_kv_residency_pool>(
            new llama_kv_residency_pool(std::move(state)));
    } catch (...) {
        if (backend.release_slots) {
            backend.release_slots(
                backend.context, config.slot_capacity, config.bytes_per_slot);
        }
        status = llama_kv_residency_pool_status::internal_error;
        return nullptr;
    }
}

llama_kv_residency_pool::~llama_kv_residency_pool() {
    if (!impl_) {
        return;
    }
    if (impl_->backend.drop_slot) {
        for (uint32_t slot = 0; slot < impl_->slots.size(); ++slot) {
            if (impl_->slots[slot].mapped) {
                (void) impl_->backend.drop_slot(impl_->backend.context, slot);
            }
        }
    }
    if (impl_->backend.release_slots) {
        impl_->backend.release_slots(
            impl_->backend.context, impl_->config.slot_capacity,
            impl_->config.bytes_per_slot);
    }
}

uint32_t llama_kv_residency_pool::slot_capacity() const noexcept {
    return impl_ ? impl_->config.slot_capacity : 0;
}

uint64_t llama_kv_residency_pool::bytes_per_slot() const noexcept {
    return impl_ ? impl_->config.bytes_per_slot : 0;
}

uint32_t llama_kv_residency_pool::mapped_slots() const noexcept {
    if (!impl_) return 0;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->mapped_slots;
}

uint64_t llama_kv_residency_pool::resident_bytes() const noexcept {
    if (!impl_) return 0;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    uint64_t result = 0;
    return mul_u64(impl_->mapped_slots, impl_->config.bytes_per_slot, result)
        ? result : 0;
}

uint32_t llama_kv_residency_pool::pending_events() const noexcept {
    if (!impl_) return 0;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->pending_events;
}

llama_kv_residency_pool_status llama_kv_residency_pool::reserve(
        const llama_kv_residency_transfer_plan & plan,
        uint64_t staging_capacity,
        const llama_kv_residency_catalog_reservation & catalog,
        llama_kv_residency_transfer_claim & output) noexcept {
    output.reset();
    if (!impl_ || plan.pages.empty() || plan.runs.empty() ||
        plan.useful_bytes > impl_->config.max_transfer_bytes ||
        staging_capacity == 0 || plan.event_count == 0 ||
        plan.event_count > impl_->config.event_capacity) {
        return llama_kv_residency_pool_status::invalid_argument;
    }
    if (plan.direction >= llama_kv_residency_transfer_direction::_count) {
        return llama_kv_residency_pool_status::invalid_argument;
    }
    if (plan.direction == llama_kv_residency_transfer_direction::d2h_seal ||
        plan.direction == llama_kv_residency_transfer_direction::d2h_reseal) {
        if (!catalog.reserve || !catalog.release) {
            return llama_kv_residency_pool_status::catalog_unavailable;
        }
    }
    bool catalog_reserved = false;
    try {
        std::vector<uint32_t> slots;
        if (plan.direction == llama_kv_residency_transfer_direction::h2d_promotion) {
            slots.reserve(plan.pages.size());
            for (const auto & page : plan.pages) {
                if (page.physical_slot >= impl_->config.slot_capacity ||
                    std::find(slots.begin(), slots.end(), page.physical_slot) != slots.end()) {
                    return llama_kv_residency_pool_status::slot_unavailable;
                }
                slots.push_back(page.physical_slot);
            }
        }
        const bool downloads =
            plan.direction == llama_kv_residency_transfer_direction::d2h_seal ||
            plan.direction == llama_kv_residency_transfer_direction::d2h_reseal;
        const uint64_t catalog_bytes = downloads ? plan.useful_bytes : 0;
        if (downloads && catalog.reserve &&
            !catalog.reserve(catalog.context, catalog_bytes)) {
            return llama_kv_residency_pool_status::catalog_unavailable;
        }
        catalog_reserved = downloads;
        bool admitted = false;
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            if (plan.event_count <= impl_->config.event_capacity -
                    impl_->pending_events &&
                slots.size() <= impl_->config.slot_capacity -
                    impl_->reserved_slots) {
                admitted = true;
                for (uint32_t slot : slots) {
                    if (impl_->slots[slot].mapped || impl_->slots[slot].reserved) {
                        admitted = false;
                        break;
                    }
                }
                if (admitted) {
                    for (uint32_t slot : slots) {
                        impl_->slots[slot].reserved = true;
                    }
                    impl_->reserved_slots += uint32_t(slots.size());
                    impl_->pending_events += plan.event_count;
                }
            }
        }
        if (!admitted) {
            if (downloads && catalog.release) {
                catalog.release(catalog.context, catalog_bytes);
            }
            return slots.empty()
                ? llama_kv_residency_pool_status::event_unavailable
                : llama_kv_residency_pool_status::slot_unavailable;
        }
        output.pool_ = this;
        output.slots_ = std::move(slots);
        output.events_ = plan.event_count;
        output.catalog_bytes_ = catalog_bytes;
        output.catalog = catalog;
        output.active_ = true;
        return llama_kv_residency_pool_status::ok;
    } catch (...) {
        if (catalog_reserved && catalog.release) {
            catalog.release(catalog.context, plan.useful_bytes);
        }
        return llama_kv_residency_pool_status::internal_error;
    }
}

llama_kv_residency_pool_status llama_kv_residency_pool::map_reserved(
        llama_kv_residency_transfer_claim & claim,
        const llama_kv_residency_transfer_plan & plan,
        bool host_valid) noexcept {
    if (!impl_ || !claim.active_ || claim.pool_ != this ||
        claim.mapped_ ||
        plan.direction != llama_kv_residency_transfer_direction::h2d_promotion ||
        claim.slots_.size() != plan.pages.size() || !impl_->backend.map_slot ||
        !impl_->backend.drop_slot) {
        return llama_kv_residency_pool_status::not_configured;
    }
    std::vector<uint32_t> mapped;
    try {
        mapped.reserve(claim.slots_.size());
        for (uint32_t slot : claim.slots_) {
            if (!impl_->backend.map_slot(impl_->backend.context, slot)) {
                for (uint32_t prior : mapped) {
                    (void) impl_->backend.drop_slot(impl_->backend.context, prior);
                }
                return llama_kv_residency_pool_status::backend_unavailable;
            }
            mapped.push_back(slot);
        }
        bool valid = true;
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            for (size_t i = 0; i < claim.slots_.size(); ++i) {
                const uint32_t slot = claim.slots_[i];
                if (!impl_->slots[slot].reserved || impl_->slots[slot].mapped) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                for (size_t i = 0; i < claim.slots_.size(); ++i) {
                    const uint32_t slot = claim.slots_[i];
                    impl_->slots[slot].reserved = false;
                    impl_->slots[slot].mapped = true;
                    impl_->slots[slot].host_valid = host_valid;
                    impl_->slots[slot].dirty = false;
                    impl_->slots[slot].page = plan.pages[i].page;
                }
                impl_->reserved_slots -= uint32_t(claim.slots_.size());
                impl_->mapped_slots += uint32_t(claim.slots_.size());
                claim.mapped_ = true;
            }
        }
        if (!valid) {
            for (uint32_t prior : mapped) {
                (void) impl_->backend.drop_slot(impl_->backend.context, prior);
            }
            return llama_kv_residency_pool_status::slot_unavailable;
        }
        return llama_kv_residency_pool_status::ok;
    } catch (...) {
        for (uint32_t prior : mapped) {
            (void) impl_->backend.drop_slot(impl_->backend.context, prior);
        }
        return llama_kv_residency_pool_status::internal_error;
    }
}

void llama_kv_residency_pool::release_claim(
        llama_kv_residency_transfer_claim & claim) noexcept {
    if (!impl_ || claim.pool_ != this || !claim.active_) return;
    bool all_dropped = true;
    if (claim.mapped_ && impl_->backend.drop_slot) {
        for (uint32_t slot : claim.slots_) {
            if (!impl_->backend.drop_slot(impl_->backend.context, slot)) {
                all_dropped = false;
            }
        }
    } else if (claim.mapped_) {
        all_dropped = false;
    }
    if (!all_dropped) return;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        for (uint32_t slot : claim.slots_) {
            if (slot >= impl_->slots.size()) continue;
            if (impl_->slots[slot].mapped) {
                impl_->slots[slot] = {};
                if (impl_->mapped_slots > 0) --impl_->mapped_slots;
            } else if (impl_->slots[slot].reserved) {
                impl_->slots[slot].reserved = false;
                if (impl_->reserved_slots > 0) --impl_->reserved_slots;
            }
        }
        if (impl_->pending_events >= claim.events_) {
            impl_->pending_events -= claim.events_;
        } else {
            impl_->pending_events = 0;
        }
    }
    if (claim.catalog.release) {
        claim.catalog.release(claim.catalog.context, claim.catalog_bytes_);
    }
    claim.pool_ = nullptr;
    claim.slots_.clear();
    claim.events_ = 0;
    claim.catalog_bytes_ = 0;
    claim.catalog = {};
    claim.mapped_ = false;
    claim.active_ = false;
}

void llama_kv_residency_pool::rollback(
        llama_kv_residency_transfer_claim & claim) noexcept {
    release_claim(claim);
}

void llama_kv_residency_pool::commit(
        llama_kv_residency_transfer_claim & claim) noexcept {
    if (!impl_ || claim.pool_ != this || !claim.active_) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->pending_events >= claim.events_) {
        impl_->pending_events -= claim.events_;
    } else {
        impl_->pending_events = 0;
    }
    claim.pool_ = nullptr;
    claim.slots_.clear();
    claim.events_ = 0;
    claim.catalog_bytes_ = 0;
    claim.catalog = {};
    claim.mapped_ = false;
    claim.active_ = false;
}

llama_kv_residency_pool_status llama_kv_residency_pool::drop_logical_page(
        const llama_kv_page_id & page, bool & dropped) noexcept {
    dropped = false;
    if (!impl_ || !impl_->backend.drop_slot) {
        return llama_kv_residency_pool_status::not_configured;
    }
    uint32_t slot = UINT32_MAX;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        for (uint32_t i = 0; i < impl_->slots.size(); ++i) {
            if (impl_->slots[i].mapped && impl_->slots[i].page == page) {
                if (!impl_->slots[i].host_valid || impl_->slots[i].dirty) {
                    return llama_kv_residency_pool_status::dirty_page;
                }
                slot = i;
                break;
            }
        }
    }
    if (slot == UINT32_MAX) return llama_kv_residency_pool_status::not_found;
    if (!impl_->backend.drop_slot(impl_->backend.context, slot)) {
        return llama_kv_residency_pool_status::backend_unavailable;
    }
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->slots[slot].mapped || impl_->slots[slot].page != page) {
            return llama_kv_residency_pool_status::stale_completion;
        }
        impl_->slots[slot] = {};
        --impl_->mapped_slots;
    }
    dropped = true;
    return llama_kv_residency_pool_status::ok;
}

llama_kv_residency_pool_status llama_kv_residency_pool::mark_dirty(
        const llama_kv_page_id & page, bool dirty) noexcept {
    if (!impl_) return llama_kv_residency_pool_status::not_configured;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (auto & slot : impl_->slots) {
        if (slot.mapped && slot.page == page) {
            slot.dirty = dirty;
            return llama_kv_residency_pool_status::ok;
        }
    }
    return llama_kv_residency_pool_status::not_found;
}

bool llama_kv_residency_pool::find_logical_page(
        const llama_kv_page_id & page, uint32_t & slot) const noexcept {
    slot = UINT32_MAX;
    if (!impl_) return false;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (uint32_t i = 0; i < impl_->slots.size(); ++i) {
        if (impl_->slots[i].mapped && impl_->slots[i].page == page) {
            slot = i;
            return true;
        }
    }
    return false;
}

llama_kv_residency_transfer_result llama_kv_residency_execute_transfer(
        llama_kv_residency_pool & pool,
        const llama_kv_residency_transfer_plan & plan,
        llama_kv_residency_transfer_claim & claim,
        const llama_kv_residency_pool_backend & backend,
        const llama_kv_residency_transfer_transport & transport) noexcept {
    llama_kv_residency_transfer_result result;
    result.status = llama_kv_residency_pool_status::invalid_argument;
    result.counters.queued = plan.runs.size();
    result.counters.copied_aligned_bytes = plan.aligned_bytes;
    try {
    const bool tensor_route = backend.backend != nullptr &&
        backend.slot_tensor != nullptr;
    if (!claim.active() || claim.reserved_events() != plan.event_count ||
        plan.direction >= llama_kv_residency_transfer_direction::_count ||
        plan.pages.empty() || plan.runs.empty() ||
        !backend.issue_copy || !backend.complete_copy || !transport.recheck ||
        (plan.direction == llama_kv_residency_transfer_direction::h2d_promotion
            ? (!transport.upload_ring || !transport.host_read ||
               (tensor_route && !backend.backend))
            : (!transport.download_ring || !transport.host_write ||
               (tensor_route && !backend.backend)))) {
        pool.rollback(claim);
        return result;
    }
    const uint64_t staging_capacity = plan.direction ==
        llama_kv_residency_transfer_direction::h2d_promotion
        ? transport.upload_ring->capacity_bytes()
        : transport.download_ring->capacity_bytes();
    if (staging_capacity == 0) {
        result.status = llama_kv_residency_pool_status::staging_unavailable;
        pool.rollback(claim);
        return result;
    }
    if (plan.direction == llama_kv_residency_transfer_direction::h2d_promotion) {
        result.status = pool.map_reserved(claim, plan, true) ==
            llama_kv_residency_pool_status::ok
            ? llama_kv_residency_pool_status::ok
            : llama_kv_residency_pool_status::backend_unavailable;
        if (result.status != llama_kv_residency_pool_status::ok) {
            ++result.counters.map_failures;
            pool.rollback(claim);
            return result;
        }
    } else {
        result.status = llama_kv_residency_pool_status::ok;
        for (const auto & page : plan.pages) {
            uint32_t slot = UINT32_MAX;
            if (!pool.find_logical_page(page.page, slot) ||
                slot != page.physical_slot) {
                result.status = llama_kv_residency_pool_status::slot_unavailable;
                break;
            }
        }
        if (result.status != llama_kv_residency_pool_status::ok) {
            pool.rollback(claim);
            return result;
        }
    }

    for (uint32_t run_index = 0; run_index < plan.runs.size(); ++run_index) {
        const auto & run = plan.runs[run_index];
        if (run.page_index >= plan.pages.size()) {
            result.status = llama_kv_residency_pool_status::invalid_argument;
            break;
        }
        const auto & page = plan.pages[run.page_index];
        run_context context;
        context.backend = &backend;
        context.transport = &transport;
        context.direction = plan.direction;
        context.completion.page = page.page;
        context.completion.table_epoch = page.table_epoch;
        context.completion.physical_slot = page.physical_slot;
        context.completion.run_index = run_index;
        context.run = &run;
        context.result = &result;

        std::unique_ptr<artifact_segment_chain> downloaded;
        const auto transfer_start = std::chrono::steady_clock::now();
        if (plan.direction == llama_kv_residency_transfer_direction::h2d_promotion) {
            vbr_artifact_byte_source source;
            source.size = run.useful_bytes();
            source.context = &context;
            source.read = h2d_read;
            vbr_h2d_transfer transfer;
            transfer.lane = run.lane;
            transfer.source = source;
            transfer.size = source.size;
            transfer.fake.context = &context;
            transfer.fake.issue = h2d_issue;
            transfer.fake.complete = h2d_complete;
            transfer.fake.cancel = h2d_cancel;
            transfer.fake.supports_events = !transport.force_synchronous;
            if (tensor_route) {
                transfer.backend = backend.backend;
                transfer.device = ggml_backend_get_device(backend.backend);
                transfer.destination = backend.slot_tensor(
                        backend.context, page.physical_slot);
                transfer.destination_offset = run.device_offset;
                transfer.fake = {};
                if (!transfer.destination) {
                    result.status = llama_kv_residency_pool_status::backend_unavailable;
                    break;
                }
            }
            transfer.continue_context = transport.continue_transfer ? &context : nullptr;
            transfer.continue_transfer = transport.continue_transfer
                ? transfer_continue : nullptr;
            vbr_h2d_stats stats;
            const vbr_h2d_status status = transport.upload_ring->stream(
                transfer, stats);
            result.counters.submitted += stats.chunks;
            result.counters.backpressure_waits += stats.backpressure_waits;
            result.counters.event_completions += stats.event_completions;
            result.counters.copied_useful_bytes += stats.bytes;
            if (status != vbr_h2d_status::ok) {
                result.status = map_h2d_status(status);
                break;
            }
        } else {
            downloaded.reset(new artifact_segment_chain(run.useful_bytes()));
            vbr_capture_stream_source source;
            source.lane = run.lane;
            source.size = run.useful_bytes();
            if (tensor_route) {
                source.backend = backend.backend;
                source.device = ggml_backend_get_device(backend.backend);
                source.tensor = backend.slot_tensor(
                        backend.context, page.physical_slot);
                source.tensor_offset = run.device_offset;
                if (!source.tensor) {
                    result.status = llama_kv_residency_pool_status::backend_unavailable;
                    break;
                }
            } else {
                source.context = &context;
                source.async_read = d2h_issue;
                source.complete = d2h_complete;
                source.cancel = d2h_cancel;
            }
            source.continue_context = transport.continue_transfer ? &context : nullptr;
            source.continue_transfer = transport.continue_transfer
                ? transfer_continue : nullptr;
            const vbr_capture_stream_range range { 0, source.size };
            vbr_capture_stream_stats stats;
            const auto status = transport.download_ring->stream_ranges(
                source, { range }, *downloaded, stats);
            result.counters.submitted += stats.submitted_chunks;
            result.counters.backpressure_waits += stats.backpressure_waits;
            result.counters.event_completions += stats.event_completions;
            result.counters.copied_useful_bytes += stats.bytes;
            if (status != vbr_capture_stream_status::ok) {
                result.status = map_stream_status(status);
                break;
            }
        }

        if (!transport.recheck(
                transport.context, context.completion)) {
            ++result.counters.stale_completions;
            result.status = llama_kv_residency_pool_status::stale_completion;
            break;
        }

        if (downloaded) {
            // The bounded ring already owns each transfer chunk. Copy the
            // completed chain into the catalog in the same bounded quantum;
            // no full-page temporary is formed.
            const size_t chunk_bytes = transport.download_ring->chunk_bytes();
            std::vector<uint8_t> buffer(chunk_bytes);
            uint64_t offset = 0;
            while (offset < run.useful_bytes()) {
                const size_t size = size_t(std::min<uint64_t>(
                    buffer.size(), run.useful_bytes() - offset));
                if (!downloaded->read(offset, buffer.data(), size) ||
                    !transport.host_write(
                        transport.context, run.page_index,
                        run.host_offset + offset, buffer.data(), size)) {
                    result.status = llama_kv_residency_pool_status::transfer_failed;
                    break;
                }
                offset += size;
            }
            if (result.status != llama_kv_residency_pool_status::ok) {
                break;
            }
        }
        result.counters.transfer_time_us += uint64_t(std::chrono::duration_cast<
                std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - transfer_start).count());
    }
    if (result.status == llama_kv_residency_pool_status::ok &&
        result.counters.copied_useful_bytes == plan.useful_bytes) {
        pool.commit(claim);
    } else {
        pool.rollback(claim);
    }
    return result;
    } catch (...) {
        pool.rollback(claim);
        result.status = llama_kv_residency_pool_status::internal_error;
        return result;
    }
}
