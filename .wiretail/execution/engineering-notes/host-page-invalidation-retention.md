# Host page invalidation retained obsolete KV buffers

## Evidence and cause

The 105-20 kernel excerpt records a global OOM killing llama-server with
55,996,908 KiB anonymous RSS during the 256,044-token prompt. The recorded
candidate source and the current source both retained invalidated host page
payloads in `llama_vbr_selected_page_host_catalog::impl::pages`.

`invalidate()` released ledger operations and marked a page obsolete, but
kept its descriptor's shared ownership of `artifact_segment_chain` byte
buffers. `snapshot()` excluded obsolete pages. Repeated isolated slot erases
and refills could therefore accumulate physical host payload while releasing
its accounting charge and hiding it from live-byte telemetry. This defect is
established by source inspection; the exact fraction of the OOM RSS caused by
it was not measured in that historical run.

## Repair and lifecycle

After ledger retirement succeeds, invalidation drops the page descriptor,
including its payload and position-vector ownership. Existing immutable
reader copies keep their own shared references until they finish. A failed
ledger retirement leaves the live page intact. Unaffected cold pages remain
available for selection and promotion.

An intrusive free list reuses retired catalog entries in constant time during
publication. Table storage follows peak simultaneous page count rather than
the number of erased requests. No new per-token synchronization, copying,
allocator flush, or attention-route change is introduced.

Server slot erasure, sequence removal/rewind, supported context shifts, and
cache clear already use the pager mutation/invalidation path and therefore
receive this cleanup. Standard context-limit behavior is retained: with
context shifting disabled, an oversized request is rejected or generation
stops at the context limit. Reaching the limit alone does not discard the
valid conversation; its pages are freed when that context is removed or
replaced. The patch does not enable shifts unsupported by the model.

## Focused validation

`test-vbr-artifact-capture` checks actual byte-chain lifetime with weak
references, a reader that outlives invalidation, 64 single-page refill/erase
cycles, repeated two-page slot reuse, and preservation of an unrelated cold
page. `test-kv-pager` and `test-kv-prefetch` exercise pager mutation and async
capture/promotion ownership. Build with `--parallel 16`.

Validation completed: the three focused targets built successfully in the
CPU build, and all three CTest cases passed (1.43 seconds total). Existing
compiler warnings were present. `git diff --check` passed.

The running managed server requires rebuilding and reloading to acquire the
fix. No claim is made that the historical 256K matrix has been rerun.
