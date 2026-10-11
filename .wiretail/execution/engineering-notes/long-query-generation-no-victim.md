# Generation after a query larger than the hot pool

## Observed failure

On 2026-10-11, the managed `qwen38-aco` candidate used L=262144,
H=65536 (256 pages of 256 tokens), B=1024/U=256, GPU Turbo4 target
and GPU Turbo4 native draft-MTP with n-max=2. Task 5248 returned:

`KV pager batch write reservation failed: no_victim sequence=0 positions=72448..72448 rows=1 frozen_history_pages=0 phase=4`.

The service journal reported a fresh 72061-token prompt (cached prefix and
LCP both zero). Its rendered final-user span was ambiguous/outside the
tokenized prompt, so the server used the entire prompt as the compatibility
query span. This implies Q=[0,72061), exclusive end, for that request.
The refusal occurred about 387 positions after its prompt boundary, at
logical page 283. Phase 4 is generation. This is a physical-slot admission
failure; the error does not report a CUDA allocation failure.

## Source cause and correction

`tools/server/server-context.cpp` opens a request turn from the final-user
span where available, or the complete prompt for compatibility requests.
`llama_kv_pager::oldest_unselected_history_victim()` in
`src/llama-kv-pager.cpp` previously rejected every page overlapping that
query span, even after prefill/replay completed and generation began.
When the full hot pool contained Q, this excluded host-clean prompt pages
from the fallback used by both scalar and batch write reservation.

This disagreed with `llama_context::prepare_kv_attention_graph()` in
`src/llama-context.cpp`: committed generation freezes selected retrieved R,
while Q/output remain ordinary KV from the current bounded resident
snapshot. Completed older Q pages must remain eligible for safe recycling.

The repair retains the existing generation FIFO preference and permits
older completed Q pages as history victims when the FIFO cannot supply a
slot. The final Q page stays protected as prior history until generation
writes into it; the active write pin and existing generation FIFO then
govern that mixed Q/output page. Explicit frozen R, active write pins,
and host-copy pins remain protected. A victim must still have authenticated current host
bytes and routing summary, matching content versions, a clean state, and
no transfer in flight. Eviction keeps the canonical host page; it does not
delete logical context or substitute unrelated page contents.

The batch failure message now includes hot-page capacity and query bounds,
so future error excerpts can identify this geometry without a catalogue
dump. Those fields are constructed on failure only.

## Validation and continuation

Use a scaled deterministic long-Q reproduction with the same page size:
Q must exceed H, fill the resident pool, enter generation, and cross at
least two output-page boundaries. Include empty retrieved history, frozen
retrieved history, a partial Q tail, and a protected active write. Verify
that only authenticated unprotected pages are reclaimed and spilled Q
remains host-backed. The previous compiled library provides the failing
control; compare it with the repaired library before accepting the fix.

Build incrementally with parallel 16. Applying the new binary to the
managed service requires a restart after its active request ends. Keep
the configured L/H, B/U, quants and native MTP unchanged. Record focused
validation separately from scored matrix runs, and freeze the new source
and loaded library identities before resuming task 105-15.

### Focused validation performed

The new long-query regression linked against the original compiled
library failed at the first generation batch reservation. Against the
repaired library, `test-kv-pager`, `test-kv-page-select`, and
`test-kv-prefetch` passed. The regression crosses two generation page
boundaries and verifies that a reclaimed closed query page keeps its
authenticated host copy and content version, while the active batch page
stays pinned. Existing pager cases retain frozen-history and FIFO coverage.
The CPU-side policy tests were deliberately run with CUDA hidden; their
CUDA initialization warning is not a live GPU availability failure.

Incremental CUDA server and test builds used `--parallel 16`. Compact
validation logs are under
`/srv/ai/paged-kv/results/forward/long-query-no-victim-20261011/`.
Managed-service validation remains separate and must use the repaired
loaded library, not just a new executable file beside an old process.
