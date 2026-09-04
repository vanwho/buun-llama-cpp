# Cluster 02b — backend residency, transfers, and fixed-window proof

Tasks: `02-03`, `02-04`, `02-05`.

Purpose: turn host backing into real bounded GPU residency before attention or dynamic policy.

Carry forward:

- create a separate backend KV residency pool; share low-level VMM/map/event helpers with VBR but never
  put token-page policy into `ggml_vbr_vmm_pool`;
- partial write page is pinned/GPU-authoritative; sealing performs one logical batched D2H;
- sealed host-valid eviction drops/unmaps the GPU slot with zero D2H;
- dirty mutation must invalidate/reseal before eviction;
- H2D promotion and table publication are separated by completion events and full generation recheck;
- no waits under global locks, no unbounded host vectors, and no hidden full-262K GPU target tensor;
- first live proof uses a fixed/manual 304-page set, not attention-driven policy.

Read: plan sections 8.1–8.5, 8.8, 9.1–9.3, 17; cluster 02a and Phase 01 handoffs; VMM batching, pinned
ring, backend event APIs, `ggml/src/ggml-cuda/vbr-vmm.cu`, `vbr-vmm-policy.h`, and existing VBR
transaction/occupied-replacement sources.

Exit artifacts: tested residency pool, async plans, atomic transaction/rollback, and fixed-window ledger
showing bounded physical bytes, canonical host bytes, checksums, and zero-D2H clean eviction.
