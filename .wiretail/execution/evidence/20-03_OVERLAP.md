# 20-03 bounded promotion overlap evidence

Result: local implementation and CPU/CUDA verification passed. The named live
Qwen cold-promotion, overlap, quality-parity, and warm zero-copy checkpoints
are deferred because the RTX 4080 was occupied and the isolated Qwen35
model/corpus fixture was not available.

The owner-side scheduler now has bounded previous-query evidence, priority
lookahead, asynchronous ticket identity, staging slots, event limits, pinned
slot limits, queue-byte limits, cancellation, stale-generation cleanup, and a
bounded lifecycle timeline. `ensure_ready()` remains authoritative for the
current query: it waits for complete publication or returns the documented
larger-union/old-hot-set fallback. Failed or stale speculative tickets are
cancelled and cannot count useful prediction bytes. The lifecycle owner exposes
query observation and lookahead while preserving generation checks.

The deterministic fake completed ticket 2 before ticket 1, exercised event and
pinned-slot saturation, cancellation, and failed publication. It measured 16
useful predicted bytes and at least 16 wasted predicted bytes; the timeline
contained enqueue, copy begin/end, needed-by, wait, and consumed events.
Existing VBR run/chunk planning remains the transfer-level coalescing boundary;
the scheduler does not merge distinct page identities.

Local verification:

- CPU and CUDA focused CTest matrices: 6/6 passed each.
- CPU changed-target build, including `llama-server`: passed.
- CUDA changed-target build: passed.
- State validation: `Valid state: 122 tasks`.
- `git diff --check`: passed.

## Deferred verification

Observed hardware was an NVIDIA GeForce RTX 4080, driver 595.84, with 16,376
MiB total and 1,128 MiB free. An existing
`qwen38-fast-turbo4-mtp` server was using the device, so no live workload was
started or disturbed. The isolated Qwen35 model/corpus fixture was not
available. No live bytes, event overlap, attention checksum, wall-time pair,
or warm repeat is claimed here.

After isolation and fixture provisioning, run the runtime-supported cold
promotion trace with a small observed hot budget; capture page/table/slot
identity, host and device checksums, completed CUDA event, useful/aligned
bytes, attention checksum, failed replacement rollback, and a warm repetition
with zero redundant upload. Repeat the same history with the explicit
synchronous baseline and compare event timelines and wall-time tails.
