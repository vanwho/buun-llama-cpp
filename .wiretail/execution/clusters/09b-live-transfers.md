# Cluster 09b — real Turbo4 host sealing and CUDA residency

Tasks: `09-04`, `09-05`.

Purpose: replace fake callbacks with actual CUDA/backend ownership while retaining the transaction and
catalog invariants proven earlier.

Carry forward:

- canonical host backing stores opaque Turbo4 bytes for all sealed target pages and every full-attention
  layer/K/V segment; pageable backing plus bounded pinned staging is the default until measurements say
  otherwise;
- D2H occurs at seal/reseal, not at ordinary clean eviction; clean eviction only drops a slot/mapping;
- H2D promotion uses a dedicated stream, coalesces compatible segment runs, publishes only after event
  completion and complete generation/identity revalidation, and keeps the prior table valid on failure;
- device slots and virtual mappings are charged at real allocator/VMM granularity; all pins, events,
  staging slots, and mappings are released on cancellation/teardown;
- test corruption, short copy, wrong codec/topology, stale session/sequence/page/representation/table
  generation, ring exhaustion, all-pinned pressure, allocation denial, and event reordering;
- no full-context F16 buffer or CPU Turbo4 attention fallback is permitted.

Read: plan sections 8.4–8.8 and 22; handoffs 02-01–02-05, 04-04, 06-02;
`src/llama-vbr-artifact-*`, `src/llama-kv-residency-*`, `src/llama-kv-prefetch.*`, CUDA VMM/pinned-ring
backends, and their tests.

Exit gate includes a model-backed fixed-selection smoke: ingest beyond admitted GPU capacity, seal host
pages, promote previously cold pages, verify checksums and attention storage bytes, observe zero D2H for
clean victims, reconcile VRAM/RAM ledgers, preserve MTP on GPU, and restore service health. Preserve raw
logs under `/srv/ai/paged-kv/results/`.
