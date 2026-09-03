#pragma once

#include "llama-kv-attention-op.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// This is an internal execution seam.  It deliberately does not add a public
// C API: selection policy owns the page list, while this object owns the
// prompt/decode route and the graph lifetime of the selected view.
enum class llama_kv_attention_execution_mode : uint8_t {
    off = 0,
    observe,
    selective,
    exact,
};

enum class llama_kv_attention_execution_phase : uint8_t {
    prefill = 0,
    decode,
};

enum class llama_kv_attention_execution_route : uint8_t {
    dense = 0,
    observe,
    selected_reference,
    selected_direct,
    exact_reference,
    refusal,
};

const char * llama_kv_attention_execution_mode_name(
        llama_kv_attention_execution_mode mode) noexcept;
const char * llama_kv_attention_execution_phase_name(
        llama_kv_attention_execution_phase phase) noexcept;
const char * llama_kv_attention_execution_route_name(
        llama_kv_attention_execution_route route) noexcept;

enum class llama_kv_attention_execution_status : uint8_t {
    ok = 0,
    disabled,
    invalid_metadata,
    invalid_prefill_transition,
    overflow,
};

const char * llama_kv_attention_execution_status_name(
        llama_kv_attention_execution_status status) noexcept;

// Scratch is charged from the physical work actually needed by the selected
// submission.  logical_context_rows is intentionally absent: a sparse graph
// must not reserve a full-context gather merely because the logical context is
// large.
struct llama_kv_attention_scratch_request {
    uint64_t resident_rows = 0;
    uint64_t transfer_rows = 0;
    uint64_t router_rows = 0;
    size_t bytes_per_row = 0;

    uint64_t required_rows() const noexcept;
    size_t required_bytes() const noexcept;
};

struct llama_kv_attention_execution_decision {
    llama_kv_attention_execution_status status = llama_kv_attention_execution_status::disabled;
    llama_kv_attention_execution_route route = llama_kv_attention_execution_route::dense;
    llama_kv_attention_execution_phase phase = llama_kv_attention_execution_phase::prefill;
    bool graph_rebuild = false;
    uint64_t table_epoch = 0;
    uint64_t representation_epoch = 0;
    uint64_t shape_epoch = 0;
    uint64_t scratch_rows = 0;
    size_t scratch_bytes = 0;
    std::string reason;

    bool accepted() const noexcept {
        return status == llama_kv_attention_execution_status::ok ||
               status == llama_kv_attention_execution_status::disabled;
    }
};

// Admission is kept separate from residency publication.  A page can be
// written by several prompt chunks, but a new page cannot be admitted until
// the preceding page is full.  The final short page is explicitly finalized
// before decode begins and remains part of the selected view.
class llama_kv_attention_prefill_admission {
public:
    llama_kv_attention_execution_status append(
            uint32_t logical_page, uint32_t row_count) noexcept;
    llama_kv_attention_execution_status finish_tail() noexcept;
    llama_kv_attention_execution_status begin_decode() noexcept;

    llama_kv_attention_execution_phase phase() const noexcept { return phase_; }
    bool decode_ready() const noexcept { return decode_ready_; }
    uint32_t page_count() const noexcept { return uint32_t(pages_.size()); }
    uint32_t resident_rows() const noexcept { return resident_rows_; }
    const std::vector<uint32_t> & page_rows() const noexcept { return pages_; }

private:
    std::vector<uint32_t> pages_;
    uint32_t resident_rows_ = 0;
    llama_kv_attention_execution_phase phase_ = llama_kv_attention_execution_phase::prefill;
    bool tail_finished_ = false;
    bool decode_ready_ = false;
};

class llama_kv_attention_execution {
public:
    explicit llama_kv_attention_execution(
            llama_kv_attention_execution_mode mode = llama_kv_attention_execution_mode::off) noexcept;

    void set_mode(llama_kv_attention_execution_mode mode) noexcept;
    llama_kv_attention_execution_mode mode() const noexcept { return mode_; }

    // direct_capable is supplied by the backend loader after it has checked
    // the actual device.  The reference route remains available for any valid
    // selected metadata and for all prompt shapes.
    llama_kv_attention_execution_decision prepare(
            const llama_kv_attention_operator_metadata & metadata,
            llama_kv_attention_execution_phase phase,
            uint64_t representation_epoch,
            uint64_t shape_epoch,
            bool direct_capable,
            const llama_kv_attention_scratch_request & scratch);

    // One lease is retained for every submitted graph, including graph reuse.
    // Releasing in submission order lets a changed table coexist with an old
    // in-flight graph without allowing either view to be reclaimed early.
    void complete_one_graph() noexcept;
    void clear() noexcept;

    bool has_graph() const noexcept { return have_graph_; }
    size_t in_flight_graphs() const noexcept { return graph_fences_.size(); }
    llama_kv_attention_execution_route route() const noexcept { return route_; }
    uint64_t table_epoch() const noexcept { return table_epoch_; }
    uint64_t representation_epoch() const noexcept { return representation_epoch_; }
    uint64_t shape_epoch() const noexcept { return shape_epoch_; }
    const llama_kv_attention_operator_metadata & metadata() const noexcept { return metadata_; }

private:
    bool same_graph(
            const llama_kv_attention_operator_metadata & metadata,
            llama_kv_attention_execution_phase phase,
            uint64_t representation_epoch,
            uint64_t shape_epoch,
            llama_kv_attention_execution_route route) const noexcept;

    llama_kv_attention_execution_mode mode_;
    llama_kv_attention_execution_route route_ = llama_kv_attention_execution_route::dense;
    llama_kv_attention_operator_metadata metadata_;
    llama_kv_attention_execution_phase phase_ = llama_kv_attention_execution_phase::prefill;
    uint64_t table_epoch_ = 0;
    uint64_t representation_epoch_ = 0;
    uint64_t shape_epoch_ = 0;
    bool have_graph_ = false;
    std::vector<llama_kv_attention_view::graph_fence> graph_fences_;
};
