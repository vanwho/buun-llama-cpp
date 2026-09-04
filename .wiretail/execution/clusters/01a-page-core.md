# Cluster 01a — pager semantics and pure page core

Tasks: `01-01`, `01-02`.

Purpose: turn locked product semantics into CPU-only page identity and immutable residency-table
primitives before any backend movement.

Carry forward:

- modes are `off`, `observe`, `selective`, and late `exact`; selective is disclosed approximation;
- retrieval and retention are separate;
- representation/VBR and residency/pager are separate objects under one composite authority;
- page size is 256 logical tokens across every target attention layer/side;
- physical slot IDs are ephemeral; session/sequence/page/representation/table generations prevent ABA;
- graph consumers see an immutable snapshot and pinned slots cannot be replaced.

Read: plan sections 2, 8.1–8.3, 9.1, 10, 15, 17; Phase 00 handoffs; current VBR generation and occupied
replacement primitives. Do not inspect CUDA implementation yet.

Exit artifacts: `design/SEMANTICS.md`, tested `llama_kv_*` identity/table primitives, and a handoff with
the publication/epoch invariants used by all later clusters.
