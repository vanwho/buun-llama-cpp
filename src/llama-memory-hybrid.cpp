#include "llama-memory-hybrid.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

#include <limits>
#include <utility>

namespace {

void llama_memory_hybrid_binding_add(
        vbr_operation_binding & binding,
        vbr_operation_kind kind,
        vbr_operation_class operation_class,
        const llama_kv_cache * cache,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1) {
    if (cache == nullptr || !cache->vbr_operation_armed()) {
        return;
    }

    if (!vbr_binding_add_instance_target(binding, kind, operation_class,
            cache->vbr_instance_id(), VBR_STREAM_ANY, seq_id, p0, p1)) {
        binding.n_targets = 0;
    }
}

} // namespace

llama_memory_hybrid::mutation_scope::mutation_scope(
        llama_memory_hybrid * mem,
        vbr_operation_kind kind,
        vbr_operation_class operation_class,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1,
        llama_kv_cache * companion) :
    mem(mem), companion(companion), exceptions_at_entry(std::uncaught_exceptions()) {
    vbr_normalize_edit_range(p0, p1);

    vbr_operation_binding binding;
    binding.kind = kind;
    binding.child_phase = vbr_operation_phase::mutate;
    llama_memory_hybrid_binding_add(binding, kind, operation_class,
            mem->get_mem_attn(), seq_id, p0, p1);
    llama_memory_hybrid_binding_add(binding, kind, operation_class,
            companion, seq_id, p0, p1);

    if (binding.n_targets > 0) {
        operation.emplace(binding);
        adopt(mem->get_mem_attn());
        adopt(companion);
    }
}

llama_memory_hybrid::mutation_scope::mutation_scope(
        llama_memory_hybrid * mem,
        vbr_operation_binding binding,
        llama_kv_cache * companion) :
    mem(mem), companion(companion), exceptions_at_entry(std::uncaught_exceptions()) {
    if (binding.n_targets > 0) {
        operation.emplace(std::move(binding));
        adopt(mem->get_mem_attn());
        adopt(companion);
    }
}

void llama_memory_hybrid::mutation_scope::adopt(llama_kv_cache * cache) {
    if (cache == nullptr || !cache->vbr_operation_armed()) {
        return;
    }

    if (operation && *operation) {
        cache->vbr_adopt_operation(operation->id());
        adopted = true;
    } else {
        cache->vbr_adopt_refused();
        refused = true;
    }
}

void llama_memory_hybrid::mutation_scope::release() {
    if (mem != nullptr) {
        mem->get_mem_attn()->vbr_release_adopted();
    }
    if (companion != nullptr && companion->vbr_operation_armed()) {
        companion->vbr_release_adopted();
    }
    adopted = false;
    refused = false;
}

void llama_memory_hybrid::mutation_scope::adopt_composite(int32_t declared) {
    if (adopted && operation && *operation) {
        composite = std::make_shared<llama_kv_cache::vbr_composite_outcome>();
        composite->operation_id = operation->id();
        composite->declared = declared;
        mem->get_mem_attn()->vbr_adopt_composite(composite);
        if (companion != nullptr && companion->vbr_operation_armed()) {
            companion->vbr_adopt_composite(composite);
        }
    }
}

void llama_memory_hybrid::mutation_scope::finish(bool ok) {
    if (finished) {
        return;
    }
    finished = true;

    const bool has_adoption = adopted || refused;
    if (has_adoption) {
        release();
    }

    if (composite) {
        operation->release();
        composite->seal(ok);
    } else if (operation && *operation) {
        operation->close(ok ? vbr_operation_outcome::committed : vbr_operation_outcome::failed);
    }
}

llama_memory_hybrid::mutation_scope::~mutation_scope() {
    if (finished) {
        return;
    }

    const bool unwinding = std::uncaught_exceptions() > exceptions_at_entry;
    if (adopted || refused) {
        release();
        if (composite) {
            operation->release();
            composite->seal(false);
        }
    }

    if (operation && *operation && !composite) {
        operation->close(unwinding ? vbr_operation_outcome::failed
                                    : vbr_operation_outcome::committed);
    }
}

//
// llama_memory_hybrid
//

llama_memory_hybrid::llama_memory_hybrid(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const llama_memory_vbr_params & vbr,
    const struct llama_kv_pager_snapshot * pager_plan) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache(
        model,
        model.hparams,
        type_k,
        type_v,
        v_trans,
        offload,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr,
        vbr,
        "",
        pager_plan
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )) {}

llama_memory_context_ptr llama_memory_hybrid::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            // DFlash target models need per-seq ubatches so the per-ubatch slot
            // switch in llama_context::decode() can route hidden-state capture
            // and tape writes to the correct slot.
            if (embd_all || force_split_seq) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (mem_attn->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = mem_recr->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto heads_attn = mem_attn->plan_slots(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to plan attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // Recurrent prepare is a dry preflight. Keep it before the first publishing child so a
        // bounded recurrent failure leaves the attention page tree untouched.
        if (!mem_recr->prepare(ubatches)) {
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        mutation_scope mutation(this, vbr_operation_kind::sequence_edit,
                vbr_operation_class::state_api, -1, 0, std::numeric_limits<llama_pos>::max());
        heads_attn = mem_attn->prepare_with_slots(ubatches, std::move(heads_attn));
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            mutation.finish(false);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }
        mutation.finish(true);

        return std::make_unique<llama_memory_hybrid_context>(
                this, std::move(heads_attn), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid::init_full() {
    return std::make_unique<llama_memory_hybrid_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_context>(this, lctx, optimize);
}

bool llama_memory_hybrid::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

void llama_memory_hybrid::clear(bool data) {
    mutation_scope mutation(this, vbr_operation_kind::sequence_edit,
            vbr_operation_class::state_api, -1, 0, std::numeric_limits<llama_pos>::max());
    mem_attn->clear(data);
    mem_recr->clear(data);
}

bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    mutation_scope mutation(this, vbr_operation_kind::sequence_edit,
            vbr_operation_class::state_api, seq_id, p0, p1);
    // Try removing from the recurrent cache first since a bounded rollback may
    // be rejected without mutation. Keep the attention and recurrent children
    // on the same timeline when that happens.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        mutation.finish(false);
        return false;
    }
    const bool result = mem_attn->seq_rm(seq_id, p0, p1);
    mutation.finish(result);
    return result;
}

bool llama_memory_hybrid::seq_rm_attn(
        llama_seq_id seq_id,
        llama_pos    p0,
        llama_pos    p1) {
    mutation_scope mutation(this, vbr_operation_kind::sequence_edit,
            vbr_operation_class::state_api, seq_id, p0, p1);
    const bool result = mem_attn->seq_rm(seq_id, p0, p1);
    mutation.finish(result);
    return result;
}

bool llama_memory_hybrid::seq_rm_transient(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    mutation_scope mutation(this, vbr_operation_kind::sequence_edit,
            vbr_operation_class::state_api, seq_id, p0, p1);
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        mutation.finish(false);
        return false;
    }
    const bool result = mem_attn->seq_rm_transient(seq_id, p0, p1);
    mutation.finish(result);
    return result;
}

bool llama_memory_hybrid::seq_rm_attn_transient(
        llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    mutation_scope mutation(this, vbr_operation_kind::sequence_edit,
            vbr_operation_class::state_api, seq_id, p0, p1);
    const bool result = mem_attn->seq_rm_attn_transient(seq_id, p0, p1);
    mutation.finish(result);
    return result;
}

void llama_memory_hybrid::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    (void) try_seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

bool llama_memory_hybrid::try_seq_cp(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos    p0,
        llama_pos    p1) {
    mutation_scope mutation(this, vbr_operation_kind::sequence_edit,
            vbr_operation_class::state_api, seq_id_dst, p0, p1);

    // Recurrent copy is the rejecting preflight. It must run before page publication, so a
    // capacity or rollback refusal leaves both destinations at their previous generation.
    if (!mem_recr->try_seq_cp(seq_id_src, seq_id_dst, p0, p1)) {
        mutation.finish(false);
        return false;
    }
    const bool result = mem_attn->try_seq_cp(seq_id_src, seq_id_dst, p0, p1);
    if (!result) {
        (void) mem_recr->seq_rm_transient(seq_id_dst, -1, -1);
    }
    mutation.finish(result);
    return result;
}

bool llama_memory_hybrid::try_seq_cp_transient(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos    p0,
        llama_pos    p1) {
    mutation_scope mutation(this, vbr_operation_kind::sequence_edit,
            vbr_operation_class::state_api, seq_id_dst, p0, p1);
    if (!mem_recr->try_seq_cp(seq_id_src, seq_id_dst, p0, p1)) {
        mutation.finish(false);
        return false;
    }
    const bool result = mem_attn->try_seq_cp_transient(seq_id_src, seq_id_dst, p0, p1);
    if (!result) {
        (void) mem_recr->seq_rm_transient(seq_id_dst, -1, -1);
    }
    mutation.finish(result);
    return result;
}

void llama_memory_hybrid::seq_keep(llama_seq_id seq_id) {
    mutation_scope mutation(this, vbr_operation_kind::sequence_edit,
            vbr_operation_class::state_api, -1, 0, std::numeric_limits<llama_pos>::max());
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
}

void llama_memory_hybrid::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mutation_scope mutation(this, vbr_operation_kind::sequence_edit,
            vbr_operation_class::state_api, seq_id, p0, p1);
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mutation_scope mutation(this, vbr_operation_kind::sequence_edit,
            vbr_operation_class::state_api, seq_id, p0, p1);
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid::memory_breakdown_vbr_managed() const {
    // Recurrent state is fixed and n_seq_max-sized. Only the attention child follows the KV
    // representation policy.
    return mem_attn->memory_breakdown_vbr_managed();
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid::memory_breakdown_fixed() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown_fixed();
    for (const auto & buft_size : mem_recr->memory_breakdown_fixed()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
    }
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    mutation_scope mutation(this, vbr_operation_kind::state_import,
            vbr_operation_class::state_api, seq_id, 0, std::numeric_limits<llama_pos>::max());
    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            mem_attn->state_read(io, seq_id, flags);
        }
        mem_recr->state_read(io, seq_id, flags);
    } catch (...) {
        // State import is a composite publication. If either child rejects the stream, drop the
        // imported sequence from both children before the failed operation is closed.
        if (seq_id < 0) {
            mem_attn->clear(true);
            mem_recr->clear(true);
        } else {
            mem_attn->seq_rm(seq_id, -1, -1);
            mem_recr->seq_rm(seq_id, -1, -1);
        }
        mutation.finish(false);
        throw;
    }
}

llama_kv_cache * llama_memory_hybrid::get_mem_attn() const {
    return mem_attn.get();
}

llama_memory_recurrent * llama_memory_hybrid::get_mem_recr() const {
    return mem_recr.get();
}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_hybrid * mem) :
    ctx_recr(mem->get_mem_recr()->init_full()),
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), ctx_recr->get_max_graph_seqs())),
    mem(mem),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize) :
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    mem(mem),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), std::move(sinfos_attn), this->ubatches)),
    mem(mem),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool llama_memory_hybrid_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    return apply_atomic(nullptr, nullptr);
}

void llama_memory_hybrid_context::finish(bool graph_succeeded) {
    if (ctx_attn) {
        ctx_attn->finish(graph_succeeded);
    }
}

bool llama_memory_hybrid_context::apply_atomic(
        llama_memory_context_i * companion_context,
        llama_kv_cache * companion_cache) {
    if (mem == nullptr) {
        return false;
    }

    vbr_operation_binding binding;
    int32_t declared = 0;

    if (ubatches.empty()) {
        binding.kind = vbr_operation_kind::sequence_edit;
        binding.child_phase = vbr_operation_phase::mutate;
        llama_memory_hybrid_binding_add(binding, binding.kind,
                vbr_operation_class::state_api, mem->get_mem_attn(), -1, 0,
                std::numeric_limits<llama_pos>::max());
        llama_memory_hybrid_binding_add(binding, binding.kind,
                vbr_operation_class::state_api, companion_cache, -1, 0,
                std::numeric_limits<llama_pos>::max());
    } else {
        binding.kind = vbr_operation_kind::decode;
        binding.child_phase = vbr_operation_phase::mutate;

        const auto add_decode_target = [&](llama_kv_cache * cache) {
            if (cache == nullptr || !cache->vbr_operation_armed()) {
                return true;
            }
            ++declared;
            return llama_kv_cache::vbr_decode_targets_from_ubatch(
                    binding, cache->vbr_instance_id(), cache->get_n_swa() != 0,
                    VBR_STREAM_ANY, ubatches[i_next]);
        };

        if (!add_decode_target(mem->get_mem_attn()) ||
            !add_decode_target(companion_cache)) {
            return false;
        }
    }

    llama_memory_hybrid::mutation_scope mutation(mem, std::move(binding), companion_cache);
    if (!ubatches.empty() && declared > 0) {
        mutation.adopt_composite(declared);
    }

    bool result = ctx_recr->apply();
    if (result && companion_context != nullptr) {
        result = companion_context->apply();
    }
    if (result) {
        result = ctx_attn->apply();
    }

    mutation.finish(result);
    return result;
}

llama_memory_status llama_memory_hybrid_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

uint32_t llama_memory_hybrid_context::get_max_graph_seqs() const {
    if (!ctx_attn || !ctx_recr) {
        return 0;
    }
    return std::min(ctx_attn->get_max_graph_seqs(), ctx_recr->get_max_graph_seqs());
}

uint64_t llama_memory_hybrid_context::get_vbr_epoch() const {
    return get_attn()->get_vbr_epoch();
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_attn() const {
    return static_cast<const llama_kv_cache_context *>(ctx_attn.get());
}

ggml_tensor * llama_memory_hybrid_context::get_turbo_rot_forward() const {
    return ctx_attn ? ctx_attn->get_turbo_rot_forward() : nullptr;
}

ggml_tensor * llama_memory_hybrid_context::get_turbo_rot_inverse() const {
    return ctx_attn ? ctx_attn->get_turbo_rot_inverse() : nullptr;
}

const llama_memory_recurrent_context * llama_memory_hybrid_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}
