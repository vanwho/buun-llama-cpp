#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"

#include <memory>
#include <optional>
#include <vector>

//
// llama_memory_hybrid
//

// utilizes instances of llama_memory_recurrent and llama_kv_cache to
//   support models where each layer may be either attention-based or recurrent

class llama_memory_hybrid : public llama_memory_i {
public:
    // Owns the single VBR operation for a hybrid mutation.  The recurrent child is fixed GPU
    // state, but it must remain inside the same lifetime as the attention and optional companion
    // cache mutations so a failed child cannot publish a newer attention generation alone.
    class mutation_scope {
    public:
        mutation_scope(
                llama_memory_hybrid * mem,
                vbr_operation_kind kind,
                vbr_operation_class operation_class,
                llama_seq_id seq_id,
                llama_pos p0,
                llama_pos p1,
                llama_kv_cache * companion = nullptr);
        mutation_scope(
                llama_memory_hybrid * mem,
                vbr_operation_binding binding,
                llama_kv_cache * companion = nullptr);
        ~mutation_scope();

        mutation_scope(const mutation_scope &) = delete;
        mutation_scope & operator=(const mutation_scope &) = delete;

        void adopt_composite(int32_t declared);
        void finish(bool ok);

    private:
        void adopt(llama_kv_cache * cache);
        void release();

        llama_memory_hybrid * mem = nullptr;
        llama_kv_cache * companion = nullptr;
        std::optional<vbr_scoped_operation> operation;
        std::shared_ptr<llama_kv_cache::vbr_composite_outcome> composite;
        bool adopted = false;
        bool refused = false;
        bool finished = false;
        int exceptions_at_entry = 0;
    };

    llama_memory_hybrid(
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
    const layer_filter_cb & filter_attn = nullptr,
    const layer_filter_cb & filter_recr = nullptr,
    const llama_memory_vbr_params & vbr = {},
    const struct llama_kv_pager_snapshot * pager_plan = nullptr);

    ~llama_memory_hybrid() = default;

    void set_kv_pager(llama_kv_pager * pager) override {
        mem_attn->set_kv_pager(pager);
    }

    void seal_kv_pager_pages() override {
        mem_attn->seal_kv_pager_pages();
    }

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;
    bool can_seq_rm_partial() const override {
        return mem_attn->can_seq_rm_partial() && mem_recr->can_seq_rm_partial();
    }

    double kv_bpv() const override { return mem_attn->kv_bpv(); }

    llama_memory_vbr_representation_identity
    vbr_representation_identity() const override {
        return mem_attn->vbr_representation_identity();
    }

    // recurrent state has no VBR controller and clears with rm-all; the attn cache answers
    llama_memory_vbr_state_data_v2 memory_vbr_state_v2(
            llama_seq_id seq_id, uint32_t n_tokens_extra) const override {
        return mem_attn->memory_vbr_state_v2(seq_id, n_tokens_extra);
    }

    bool vbr_accumulate_exclusive_cells(
            uint32_t * counts, size_t size) const override {
        return mem_attn->vbr_accumulate_exclusive_cells(counts, size);
    }
    bool vbr_capture_readiness_cells(
            uint64_t logical_growth,
            uint64_t & committed,
            uint64_t & projected,
            uint64_t & capacity) const override {
        return mem_attn->vbr_capture_readiness_cells(
            logical_growth, committed, projected, capacity);
    }

    bool vbr_operation_armed() const override {
        return mem_attn->vbr_operation_armed();
    }
    bool vbr_retier_freeze_begin(const char * owner, vbr_operation_id operation_id) override {
        return mem_attn->vbr_retier_freeze_begin(owner, operation_id);
    }
    void vbr_retier_freeze_end(const char * owner, vbr_operation_id operation_id) override {
        mem_attn->vbr_retier_freeze_end(owner, operation_id);
    }
    void vbr_commit_submitted() override {
        mem_attn->vbr_commit_submitted();
    }
    void vbr_decode_ops_finish(bool ok) override {
        mem_attn->vbr_decode_ops_finish(ok);
    }
    void vbr_adopt_operation(vbr_operation_id operation_id) override {
        mem_attn->vbr_adopt_operation(operation_id);
    }
    void vbr_release_adopted() override {
        mem_attn->vbr_release_adopted();
    }
    llama_memory_vbr_preflight_data vbr_retier_preflight(
            uint32_t n_tokens_extra,
            std::vector<llama_memory_vbr_physical_growth> * physical = nullptr) const override {
        return mem_attn->vbr_retier_preflight(n_tokens_extra, physical);
    }

    double memory_vbr_floor_bits_per_token(ggml_type entry_k, ggml_type entry_v, double floor_bpv) override {
        return mem_attn->memory_vbr_floor_bits_per_token(entry_k, entry_v, floor_bpv);
    }

    double memory_vbr_entry_bits_per_token(ggml_type entry_k, ggml_type entry_v) override {
        return mem_attn->memory_vbr_entry_bits_per_token(entry_k, entry_v);
    }

    double memory_vbr_scratch_bytes_per_token(ggml_type entry_k, ggml_type entry_v, double floor_bpv) override {
        return mem_attn->memory_vbr_scratch_bytes_per_token(entry_k, entry_v, floor_bpv);
    }

    // recurrent state has no deferred maintenance; the attn cache breathes
    void breathe() override { mem_attn->breathe(); }

    void vbr_cotenancy_accum(uint64_t & d, uint32_t & g, uint64_t & o, uint64_t & p) const override {
        mem_attn->vbr_cotenancy_accum(d, g, o, p);
    }

    bool vbr_ledger_tree_active() const override {
        return mem_attn->vbr_ledger_tree_active();
    }
    void vbr_hard_seal_guard_set(vbr_hard_seal_guard guard) override {
        mem_attn->vbr_hard_seal_guard_set(std::move(guard));
    }
    bool vbr_hard_seal_blocked_take(bool decode_failed) override {
        return mem_attn->vbr_hard_seal_blocked_take(decode_failed);
    }
    void vbr_hard_seal_evidence_take(
            std::vector<vbr_hard_seal_subject> & out) override {
        mem_attn->vbr_hard_seal_evidence_take(out);
    }

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    bool seq_rm_attn(llama_seq_id seq_id,                            llama_pos p0, llama_pos p1) override;
    bool seq_rm_transient(llama_seq_id seq_id,                       llama_pos p0, llama_pos p1) override;
    bool seq_rm_attn_transient(llama_seq_id seq_id,                  llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    bool try_seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    bool try_seq_cp_transient(
            llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown_vbr_managed() const override;
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown_fixed() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid specific API
    //

    llama_kv_cache * get_mem_attn() const;
    llama_memory_recurrent * get_mem_recr() const;

    void set_force_split_seq(bool v) override { force_split_seq = v; }

private:
    const llama_hparams & hparams;

    const std::unique_ptr<llama_kv_cache> mem_attn;
    const std::unique_ptr<llama_memory_recurrent> mem_recr;

    bool force_split_seq = false;
};

class llama_memory_hybrid_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // init failure
    explicit llama_memory_hybrid_context(llama_memory_status status);

    // init full
    explicit llama_memory_hybrid_context(llama_memory_hybrid * mem);

    // init update
    explicit llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize);

    // init success
    llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_context() = default;

    bool next()  override;
    bool apply() override;
    void finish(bool graph_succeeded) override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    uint32_t get_max_graph_seqs() const override;

    // tier epoch of the attention child (the recurrent child has no VBR)
    uint64_t get_vbr_epoch() const override;

    // TurboQuant: delegate to the KV cache context
    ggml_tensor * get_turbo_rot_forward() const override;
    ggml_tensor * get_turbo_rot_inverse() const override;

    //
    // llama_memory_hybrid_context
    //

    const llama_kv_cache_context * get_attn() const;
    const llama_memory_recurrent_context * get_recr() const;

protected:
    // Apply the recurrent and attention contexts, optionally inserting a fixed-layout companion
    // before attention publication.  Indexed hybrid memory uses this to keep QSA metadata in the
    // same decode transaction.
    bool apply_atomic(llama_memory_context_i * companion_context, llama_kv_cache * companion_cache);

private:
    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_recr;
    const llama_memory_context_ptr ctx_attn;

    llama_memory_hybrid * const mem = nullptr;

    const llama_memory_status status;
};
