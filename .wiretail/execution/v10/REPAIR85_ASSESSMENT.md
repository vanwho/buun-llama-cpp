# Phase 85 corrective source audit — 2026-09-21

Revision: `hotpath-v10-20260914`. Audited source:
`374349bce47f0b5723da9017ef575e02c1b409bb`. This is the active explanation for tasks 85-09
onward; earlier summaries are historical observations, not design authority.

## What is established, and what is not

Prefill is not inherently a CPU operation when canonical KV lives in RAM.
Target weights, attention, recurrent computation and draft inference should
execute on CUDA. Cold Turbo4 bytes are storage, not a request to execute
TurboQuant CPU attention. A CPU-only fixture refusing TurboQuant is not a
failure of this architecture. Small decode workloads need not draw maximum
GPU watts; 50 W alone does not identify a kernel or prove CPU fallback.

`L` is allocated logical context; `C` is actually committed/occupied context;
`H` is target resident capacity; `A` is selected rows per attention layer;
`B` is batch capacity; `U` is physical microbatch. A durable frontier is the
committed prefix usable by a subsequent cached request, not an allocated
context or proof of high throughput. `selected packed` means selected
compressed rows are exposed through a compact GPU tensor consumed by mature
FA. Its route counter counts dispatches, not tokens or successful promotion.

### Three concrete numerical defects in the direct MMA path

1. **Row/head strides are interchanged at the graph/backend boundary.**
   `llama_kv_cache::get_k/get_v` return `[D, KV-head, row, stream]` views.
   Their `nb[1]` is one encoded head, `nb[2]` is all heads in a token row.
   In `llm_graph_context::build_attn`, both direct and exact-wave branches set
   `k_raw.nb[1] = row_size(D)` and `k_raw.nb[2] = k.nb[2]` (likewise V).
   The CUDA graph wrapper in `fattn.cu` interprets these as **row stride** and
   **head stride**, respectively. This is the reverse of the backing layout.
   Fix the producer/consumer contract, not the comments. Formula and tests
   are in 85-09. Single-KV-head tests cannot distinguish these strides.
2. **MMA Q/output addresses omit the KV-group offset.**
   `ggml_cuda_fattn_mma_turbo4_paged_kernel` sets `policy.kv_head = blockIdx.y`.
   `flash_attn_ext_f16_process_tile` passes `zt_gqa*ncols2+c` to policy
   `q_value` and `store_output`: that is a head *within* a KV group. The
   policy addresses it as a global query head. Different KV groups read the
   first group's queries and write overlapping output addresses; other
   output heads are unwritten. The global index must include
   `kv_head * (n_head_q / n_head_kv)`. Partial tiles already have query/head
   guards in `fattn-mma-f16.cuh`; do not invent a missing Q-tail guard there.
3. **The direct MMA consumer omits Turbo4's Q transform.**
   `q_direct` is untransformed `q_cur`; the graph wrapper forwards it
   unchanged. `flash_attn_ext_turbo4_decode_row` decodes transformed-domain K.
   The ordinary fused Turbo path explicitly runs `k_turbo_fwht_forward`;
   the cooperative paged path uses `turbo4_paged_fwht`. The paged MMA path
   does neither. Use the existing forward transform, including applicable
   channel scaling, once per Q, and leave V inverse transformation once at
   graph output. Nonzero, head-distinct Q is essential to catch this.

These are source-established defects, not profiler guesses. They invalidate
the current automatic direct route. They do not, by themselves, identify
every historical CUDA crash or the separate feature-off dense corruption.
The standalone test uses one KV head in its principal correctness case;
its Q is also zero, hiding a missing linear transform. The multi-head
maintenance fixture attaches split scratch/page-mass fields which bypass MMA.
Large MMA timing cases measure latency without numerical
parity. Passing that executable was therefore insufficient.

### Why the fallback is slow, and why MTP is not certified

`llama_kv_attention_execution::planned_route` now excludes automatic native
MTP prefill and all `mtp_verify` from direct dispatch. They use an eligible
contiguous dense view or otherwise `selected_reference`. The latter gathers
selected rows with `ggml_get_rows`, materializes floating-point K/V, inverse
transforms K, casts and runs ordinary attention. It is a diagnostic oracle,
not the desired compressed paged fast path. Repeated growing materialization
and host maintenance can prevent sustained GPU work. The old packed run used
an older binary; its precise time breakdown must not be attributed to today's
fallback without a matched trace.

There are additional real maintenance costs: `pager_routing_summary_build`
iterates pages/layers/heads, copies catalog views, decodes whole all-head rows
on CPU once per head, and can synchronously read device rows. A GPU summary
operator already exists (`kv-page-summary.cu`, `build_kv_page_select` in
`llama-kv-cache.cpp`); it currently coexists with this host path and copies/
recomputes a logical-sized catalogue. Reuse and wire it correctly, do not add
another summary engine. `seal_kv_pager_pages` also short-circuits the required
call in `dirty = dirty || seal_ready_pages(...) != 0` when already dirty.
This is a definite missed-work bug, not an optimization opportunity to skip
sealing. Async snapshot preparation currently reads mutable cache metadata
from a worker; source pins, generations and backend/stream ownership need
explicit synchronization (85-11).

No valid timing decomposition yet assigns percentages to gathering, kernels,
sealing, hashing, checkpoints, graph setup or waits. A zero/unimplemented
`kernel_us` field cannot prove that kernels took no time. 85-14 obtains a
short measured decomposition rather than another long occupancy campaign.

85-08's workaround produced only **1 accepted draft out of 5**, eight output
tokens, roughly **187 prompt / 14 decode tok/s** on its short ending. A longer
attempt stalled. This is not normal canonical MTP performance or proof that
rollback and direct consumption are fixed. Conversely 84-03 selected rows
reported roughly 83–91% acceptance, showing that native Turbo4 MTP is not
intrinsically incompatible with selected attention. Those results do not
certify the current direct path. Feature-off/all-GPU controls also emitted
slash-like output with 0% acceptance, including MTP-off controls. Investigate
their target logits independently before blaming draft prediction.
`target_argmax_ids=-1` can mean no argmax side-output was produced; it is not
proof of NaN logits. Actual finite counts and first divergent tensors decide.

## Corrections to earlier summaries

- **99,483,648 D2H bytes = 23 x 4,325,376-byte pages.** 2,208 transfer chunks
  = 23 x 96, and 1,472 summary records = 23 x 64. These counts alone are not
  evidence of copying complete history repeatedly. GPU-generated newly
  committed pages require a first D2H capture to establish canonical RAM.
  Optimize blocking/duplicate versions, not the necessary bytes away.
- Phase-85 natural-proof data already included a cold page, H2D publication
  and a target-use indication. The selected request did not finish. Neither
  "no promotion ever happened" nor "paging works end to end" follows.
- CPU main-KV control at matched short geometry is not the 40 tok/s
  all-CPU diagnostic. Do not mix prompt speed, decode speed, entire-request
  elapsed time or cached tokens in denominators.
- Positive draft acceptance, successful startup, changing table epochs,
  finite logits and passing CPU contracts are each necessary/useful but do
  not replace a completed CUDA request with correct placement and output.

## Historical speed observations, not accepted final comparisons

84-03 short fixed-coordinate diagnostics reported approximately:

| route | prompt tok/s | decode tok/s | qualification |
| --- | ---: | ---: | --- |
| all-GPU, MTP off | 1,590 | 49.7 | output validity/control problem |
| all-GPU, native MTP | 1,488 | 36.7 | 0/123 accepted; invalid normal baseline |
| CPU main KV + GPU MTP | 634 | 10.2 | 0/123 accepted; invalid normal baseline |
| selected, MTP off | 135 | 24.8 | old route/build, diagnostic workload |
| selected, native MTP | 139 | 36.3 | high acceptance in these bounded rows only |

Sources: `handoffs/84-03.md`, `evidence/V10_84-03.json`,
`evidence/V10_SUMMARY_85.json`, and
`/srv/ai/paged-kv/results/v10/85-08/attempt-4-final/`.
Do not load their giant ancestry. Inspect only a named raw row when needed.

## Decision

Repair numerical/addressing correctness first, then independently restore a
finite dense/native control, make host publication and speculative ownership
safe, remove reference fallback, and optimize existing GPU kernels/catalogue.
Use short GPU regressions and matched timing at small C>H. Keep all caches
Turbo4 and full-L draft on GPU. Do not restart VBR compression work, implement
CPU attention, add proof-only attention passes, or pursue huge occupancy before
the small path works. Full-context scale tests belong at the end.
