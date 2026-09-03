# Selective-attention and KV-pager semantics

Status: internal experimental contract for the Qwen3.8-27B/qwen35 target. This document defines behavior; it does not add a CLI or public API.

## Scope and vocabulary

The pager initially applies to one causal target sequence and one logical slot, on the supported Qwen3.8 geometry. A target attention page is one cross-layer unit of 256 logical token positions containing K and V for every target full-attention layer and both sides. The final partial page is valid with an explicit position run and remains pinned while written.

`llama_kv_representation`/VBR answers what encoding a row has. `llama_kv_residency` answers whether its logical page has a GPU mapping. They are separate objects, but one composite transaction authority advances representation and residency generations together. VBR pages are not pager pages, and physical slot IDs are never logical identity.

## Modes

- **off** preserves existing dense/current KV behavior and performs no pager work. It is the feature-disabled zero-overhead path.
- **observe** preserves dense/current execution and performs no movement or selected attention. It records routing and retention evidence for replay.
- **selective** executes attention over a bounded resident set chosen by retrieval. It is an intentionally disclosed approximation, not dense or bit-equivalent attention.
- **exact** makes every logical page participate. Its initial reference is bounded GPU page waves merging online-softmax `(max, sum, weighted_value)` states. It is a late correctness/quality oracle and may be slow.

The first release is fail-closed unless target geometry, causal attention, Turbo4 K/V, host backing, budget, and MTP reservation are compatible. Unsupported backends, non-causal attention, missing host budget, incompatible K/V types, multiple VBR authorities, or insufficient MTP space must select no pager behavior and report a typed diagnostic. No silent fallback is allowed after partial publication; the old valid snapshot remains authoritative.

## Retrieval and retention

Retrieval chooses candidates from an all-page routing index, including cold pages, and unions structural anchors, application-mandated pages, the recent window, top-K query-summary results, and bounded exploration. Retention is separate: after attention, resident-page mass, recent peak, recency, frequency, fault cost, dirty cost, and hysteresis decide what remains hot. Missing observation for a cold page is not zero importance. Until a required page is ready, policy may wait, use the previous hot set, or use a larger resident union; it must never consume a partial transfer.

Structural anchors include system/first-token and conversation-boundary positions. Pins include application pins, the filling page, graph consumers, and speculative checkpoints. Mandatory/pinned overflow is a typed refusal. The recent window is a policy input, not an entitlement beyond the budget.

Example: page 2 becomes cold after a long exchange as its scores decay. A later query-summary match recalls it from canonical host Turbo4 bytes; it is published only after identity and all segments validate, then participates in the next attention operation.

Example: a resident-only scorer cannot discover an evicted “needle” page. The all-page router ranks its summaries, while exploration samples cold pages, allowing the needle to be promoted despite having no resident attention weight.

## Identity, state, and publication

Every page/table reference carries session generation, sequence ID and generation, logical page index and position-run digest, representation epoch, page mutation generation, model/execution/topology identity, and Turbo4 codec/codebook/rotation/mean-subtraction digests. Stale asynchronous completions are rejected. A sealed page has canonical opaque host Turbo4 bytes; clean eviction drops its GPU mapping without D2H. Dirty pages reseal before eviction.

The published table is immutable for one graph and contains only validated GPU mappings. A transaction snapshots generations, reserves and pins, transfers, revalidates, atomically swaps the complete table, and releases old pins after consumer fences. Any failure rolls back to the prior table.

Target attention, recurrent/linear state, and MTP form one composite mutation for sequence copy/remove, shifts, rewind, prompt reuse, speculative rejection, and checkpoint restore. Recurrent state is exact and GPU-resident. MTP owns a separate full-length 262,144-token Turbo4 GPU allocation, follows the accepted frontier, and is never a target victim. Prompt artifacts may seed sealed host pages but cannot publish a live mutable page as an immutable artifact.

## Tail pages and supported scope

Pages seal only when their 256-token range is complete; a partial tail is GPU-authoritative, pinned, and represented by its valid begin/end run. Clears, failed validation, and generation changes invalidate the page. The initial implementation supports one target sequence/slot, one supported topology, and one pager/VBR authority. Multi-slot, multi-sequence, non-causal, mixed-representation, and unsupported-backend use is explicitly refused until independently proven.

## Telemetry and knobs

Internal telemetry is disabled unless requested. When enabled it reports mode, logical/resident pages and rows, pins, epochs, candidate counts, exploration, hits, faults, evictions, useful/amplified bytes, queue/copy/wait time, canceled transfers, stale rejects, retention scores, and fallback choice. Observe may collect the same bounded evidence; off allocates none and emits no pager work.

Page size 256, VRAM/host budgets, recent pin count, router top-K, exploration, prefetch depth, hot-set policy, debug output, and diagnostic hot-token cap remain fork-local experimental controls. The budget is authoritative; a token cap may only reduce admitted residency. Exact spelling and retention weights/normalizers remain provisional. No stable public C API is promised.
