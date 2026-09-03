# Cluster 08a — dynamic contracts and live-gap closure map

Tasks: `08-01`, `08-02`, `08-03`.

Purpose: correct inherited fixed-capacity assumptions before production wiring. The output of this
cluster is executable code and tests for context-sized native MTP plus budget-derived target capacity,
not another design-only layer.

Carry forward:

- phases 00–07 built internal seams but did not expose a live pager;
- native MTP KV rows equal the resolved target per-sequence context `C`, never
  `max(C, n_ctx_train)`; `-c 0` follows normal target resolution;
- a conflicting explicit native-MTP `-cd` is rejected; external drafters keep existing behavior;
- MTP K/V is Turbo4/Turbo4, GPU-only, statically allocated, separately accounted, and not a pager/VBR
  victim;
- target hot capacity `H` is computed after actual/projected fixed resources, context-sized MTP,
  graph/kernel scratch, transfer/routing staging, allocator granularity, headroom, and actual page charge;
- remove the former fixed hot-page/token defaults, fixed maximum-page telemetry arrays, and fixed MTP
  context defaults. Numeric
  values remain valid only in explicit fixtures or historical evidence;
- capacity-relative policy shares are reconciled after mandatory-page deduplication and must work for
  small, odd-tail, all-pages-fit, and severely constrained capacities.

Read: plan sections 20–26; `design/SEMANTICS.md`; final evidence index; handoffs 01-03, 02-05, 04-03,
05-02, 07-02; `common/speculative.*`, `common/fit.*`, `src/llama-cache-budget.*`,
`src/llama-kv-{fixed-window,policy,attention-telemetry}.*`, associated tests.

Exit artifacts: `handoffs/08-01.md` with a symbol/caller/gap map; dynamic MTP implementation and tests;
dynamic admission/policy implementation and tests; a production-source scan proving no inherited hot-set
constant remains. Do not run long acceptance yet, but task 08-02 must perform a model-backed startup
matrix when the CUDA build is available.
