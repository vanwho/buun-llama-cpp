# Phase 21 capability gate

Status: `partially_capable`.

The immutable 20-07 runtime bundle starts at logical context `262144` with
selective paging, 256-token pages, CUDA Turbo4 target K/V, and native GPU
Turbo4 MTP. The four-page configuration is retained as the required negative
control. Its one-generation sentinel passed (`207/214` MTP acceptance), the
dense all-fit warm control passed `1/1`, and the selected-all correctness
sentinel passed `6/6` at diagnostic context `22016`.

The live gate is not capability-complete. The validated 262144 receipt shows
prefill `reference=1`, MTP verification `reference=108`, direct routes `0`,
faults `0`, and H2D/D2H actual and useful bytes `0`. This establishes bounded
resident backing and valid telemetry, but does not establish physical cold-page
movement or direct attention execution. The stopped selective corpus receipt
has 8 passes followed by 11 `answer_mismatch` failures, beginning with the
repeated `competing-a` case; no equivalent long rerun was made.

The bounded timing diagnostic used the same 4171-token prompt at context
`22016` with auto hot capacity. The off-mode 262144 auto-budget control was
blocked at startup by the recorded CUDA OOM. Observe measured `1836.1346`
prompt tokens/s
(`2271.62 ms`); selective measured `14.7180` prompt tokens/s (`283394.497
ms`). These are diagnostic observations, not a final speed claim. A separate
auto-budget 262144 startup failed with the exact CUDA OOM receipt recorded in
the JSON file; the service was restored to the exact four-page release
identity afterward.

## Owner and next boundary

The primary implementation owner is
`llama_context::prepare_kv_attention_graph()` in `src/llama-context.cpp`, where
the selective view is built from resident pager pages and direct eligibility is
decided. The promotion boundary is
`llama_kv_cache::apply_pager_live_policy()` in `src/llama-kv-cache.cpp`.
The next task should repair serialized paged prefill and selected cold-page
promotion/direct attention, using this receipt as the minimized reproducer.

## Verification

- CUDA/attention/pager CTest selection: 6/6 passed.
- Benchmark contract unit tests: 46/46 passed.
- Full corpus speed curve and physical H2D movement remain deferred by the
  bounded-gate contract and zero live movement counters.

Raw receipts, hashes, exact commands, and deferred checks are in
`PHASE21_CAPABILITY.json`.
