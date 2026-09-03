# Cluster 05a — Qwen hybrid atomicity and pinned MTP

Tasks: `05-01`, `05-02`.

Purpose: integrate target attention paging with Qwen3.8's exact recurrent companions and a separately
budgeted full-length Turbo4 native MTP cache.

Carry forward:

- only target full-attention KV is paged; recurrent/linear state stays exact, fixed, GPU-resident;
- one composite mutation covers attention metadata, recurrent state, index companions, and MTP frontier;
- MTP is 262,144-token T4/T4, expected encoded payload 264 MiB for measured geometry, reserved before
  target hot pages and never a pager/VBR victim;
- use 00-05 independent draft placement; do not reintroduce target `no_kv_offload` coupling;
- actual tensor type/backend/rows/bytes must be logged and startup fails on unsafe fit.

Read: plan sections 3.3–3.4, 5.4, 8.1, 8.9, 11 Phase 05, 17; 00-05, budget, policy, and scheduler
handoffs; hybrid/recurrent and current MTP fit/accounting sources.

Exit artifacts: atomic hybrid page operations and hardware-verifiable full Turbo4 MTP reservation with
separate ledger identity.
