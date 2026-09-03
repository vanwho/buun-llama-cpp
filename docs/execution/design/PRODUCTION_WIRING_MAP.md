# Production wiring map — phase 08 reopening

Status: source audit for task 08-01, 2026-09-03. This is a call-graph map,
not a claim that the internal pager seams are a live product.

## Audit basis

The current branch is `a0d028f26ae36516d528c698d09553ce9a935a94` (107 commits
ahead of both fetched remote HEADs). `origin/HEAD`, `origin/master`,
`upstream/HEAD`, and `upstream/master` all resolve to
`cb703be37e3628dadb71912f3b3b25b82090555b`; the remotes have no divergence
from one another, while the working branch contains the local phase-00–07
work.

The source call graph was checked from `llama_context`, `llama_graph`, the
server context, CMake registration, and direct non-test references to pager
symbols. The decisive result is:

```text
llama_context::set_kv_attention_mode()
  -> llama_kv_attention_execution::set_mode()
llama_context::prepare_kv_attention()
  -> llama_kv_attention_execution::prepare()
llama_context::complete_kv_attention_graph()
  -> llama_kv_attention_execution::complete_one_graph()
llama_graph_params::kv_attention_metadata
  -> graph fence acquisition/release only
```

Those edges are at [`src/llama-context.cpp:1166-1182`](../../../src/llama-context.cpp:1166)
and [`src/llama-graph.cpp:1787-1788`](../../../src/llama-graph.cpp:1787).
No production caller reaches a pager constructor or feeds it live target K/V.
The registrations in [`src/CMakeLists.txt:31-37`](../../../src/CMakeLists.txt:31)
and [`tests/CMakeLists.txt:498-533`](../../../tests/CMakeLists.txt:498) prove
build/test inclusion only; they do not establish runtime ownership.

## Component map

| Component and production symbol | Constructor/owner | Production caller and data source | Completion event and teardown | Existing tests | Missing edge / next owner |
| --- | --- | --- | --- | --- | --- |
| Fixed-window geometry/page state: `llama_kv_fixed_window` | `create()`/`begin_fill()` in [`src/llama-kv-fixed-window.cpp:209-312`](../../../src/llama-kv-fixed-window.cpp:209); no context owner | No non-test caller. Inputs are explicit geometry/page fixtures, not target KV tensors. | `seal()`/`promote()`/`evict_clean()` are synchronous seam operations; no production owner. | `test-kv-residency.cpp` and phase-02 fixed-window evidence | Replace fixture owner with live pager storage; bind target writes, tails, mutations, and slots in 09-02/09-03. |
| Residency table/snapshot: `llama_kv_residency_table` | Caller-created table; [`src/llama-kv-residency.h:68-119`](../../../src/llama-kv-residency.h:68) | No production construction or snapshot feed from `llama_context`; only tests construct it. | `publish()` advances the table epoch; no context/server teardown releases a live table. | [`tests/test-kv-residency.cpp:46-82`](../../../tests/test-kv-residency.cpp:46) and transaction cases | Instantiate one target pager with sequence generations and teardown in 09-02/09-03. |
| Host catalog/canonical backing: VBR artifact catalog/capture/stage | VBR artifact owners and `llama_vbr_artifact_*` APIs; transfer headers include capture/stage types | VBR server paths use VBR catalog/budget, but no residency page seal calls the catalog and no live target Turbo4 segment source is connected. | VBR artifact lifecycle owns its stores; no pager seal/dirty-reseal completion is wired. | VBR artifact/capture tests and phase-02 catalog tests | Bind sealed target K/V segments, authentication, pageable host bytes, and tails in 09-04. |
| Transfer plan/pool: `llama_kv_residency_build_transfer_plan`, `llama_kv_residency_pool` | Caller supplies pool config/backend callbacks; [`src/llama-kv-residency-transfer.h:43-130`](../../../src/llama-kv-residency-transfer.h:43) | No CUDA or target-KV adapter calls the plan/executor. The backend is test/fake infrastructure. | Fake completion/cancel and transaction hooks exist; no production CUDA stream/event owner or context teardown. | [`tests/test-kv-residency.cpp:83-478`](../../../tests/test-kv-residency.cpp:83) | Add real CUDA mapping/copy/event callbacks in 09-05. |
| Transaction publication: `llama_kv_residency_execute_transaction` | Hook-driven free function; [`src/llama-kv-residency-transaction.h:56-115`](../../../src/llama-kv-residency-transaction.h:56) | No production caller supplies snapshot, transfer, pin, publish, or retire hooks. | Tests model rollback/stale completion; no graph fence or context owner drives retire. | [`tests/test-kv-residency.cpp:498-655`](../../../tests/test-kv-residency.cpp:498) | Controller owns transaction boundaries and publishes immutable tables in 09-05/11-04. |
| Selected attention view: `llama_kv_attention_view::build` | Caller-created immutable view; [`src/llama-kv-attention-view.h:33-79`](../../../src/llama-kv-attention-view.h:33) | No production caller builds it from a residency snapshot or policy target; graph only stores/releases metadata fences. | `graph_fence` is released with graph lifetime; no live attention submission owns a view lease. | [`tests/test-kv-attention-view.cpp:20-129`](../../../tests/test-kv-attention-view.cpp:20) | Build from the published table at prefill/decode boundaries in 10-01/10-04. |
| Operator metadata/backend gate: `llama_kv_attention_operator_metadata::build` | Caller-created metadata; [`src/llama-kv-attention-op.h:50-116`](../../../src/llama-kv-attention-op.h:50) | No Qwen graph node supplies page table, native positions, K/V types, or backend loader metadata; CUDA is unsupported by the current gate. | Shared metadata lifetime exists, but no production graph node owns a selected operator. | `test-kv-attention-view.cpp:66-125` | Wire selected reference then direct Turbo4 loaders in 10-01/10-03. |
| Attention execution seam: `llama_kv_attention_execution` | Member `llama_context::kv_attention_execution`; [`src/llama-context.h:296-304,581`](../../../src/llama-context.h:296) | Context wrappers are callable, but no caller prepares metadata from live KV or dispatches an attention op. | `synchronize()` drains graph leases at [`src/llama-context.cpp:1121-1122`](../../../src/llama-context.cpp:1121); no pager clear/teardown is connected. | `tests/test-kv-attention-execution.cpp` | Connect real prefill/decode graph construction and backend dispatch in 10-01–10-03. |
| Exact waves: `llama_kv_attention_exact_wave_plan` / executor | Caller-created plan/executor; [`src/llama-kv-attention-exact.h:124-181`](../../../src/llama-kv-attention-exact.h:124) | No production graph or catalog caller; backend callback is a bounded fixture seam. | Per-wave callback/ledger completion exists only inside executor; no live context owner. | `tests/test-kv-attention-exact.cpp` | Supply live catalog, transfer, CUDA kernel, and graph callbacks in 10-05. |
| Routing summaries: `llama_kv_routing_summary_store` | Static `build`/`update`; [`src/llama-kv-routing-summary.h:67-107`](../../../src/llama-kv-routing-summary.h:67) | No production producer creates summaries from sealed pages and no request/query consumer calls `score()`. | Store is caller-owned; no page-seal or request teardown event. | `tests/test-kv-routing-summary.cpp` | Produce on seal and consume all-page scores in 11-01/11-02. |
| Attention telemetry: `llama_kv_attention_telemetry` | Caller-created object; [`src/llama-kv-attention-telemetry.h:86-129`](../../../src/llama-kv-attention-telemetry.h:86) | No graph completion submits `publish_completed`; server metrics do not expose these counters. | API requires completed graph/event, but no production event invokes it and no server teardown owns it. | `tests/test-kv-attention-telemetry.cpp` | Add GPU reduction, completion callback, lifecycle ownership, and export in 11-03/12-02. |
| Policy/controller: `llama_kv_policy_decide` / `replay` | Pure free functions; config caller-owned; [`src/llama-kv-policy.h:75-107`](../../../src/llama-kv-policy.h:75) | No pager controller calls policy with live H, table, router, or telemetry. Existing VBR controller is a different representation controller. | No publication, transfer, or graph completion is attached. | `tests/test-kv-policy.cpp` | Replace absolute defaults with runtime-H admission and connect decisions to transactions in 08-03/11-04. |
| Budget/admission: `llama_cache_budget_admit`, `llama_cache_budget_coordinator` | VBR/cache authority and fit callers own current objects; [`src/llama-cache-budget.h:42-81`](../../../src/llama-cache-budget.h:42) | Server/VBR fit paths call generic budget code, but no target pager supplies realized MTP, graph, transfer, routing, and page-charge terms. | Results feed VBR/cache setup; no target pager reservation or teardown. | `tests/test-cache-budget.cpp` and cache authority/plan tests | Make MTP and pager one resolved admission candidate in 08-02/08-03 and 09-02. |
| Prefetch/backpressure | No `llama_kv_*` owner; existing VBR/mmap prefetch is unrelated | No production target-page queue, prefetch caller, or H2D overlap path. | No completion or teardown owner. | Existing VBR/mmap tests only | Implement against live transaction/policy lifecycle in 11-05. |
| Server metrics/CLI | `server_routes`/`server_context`; metrics route exists, pager option does not. [`tools/server/server-context.h:343-356`](../../../tools/server/server-context.h:343) and [`tools/server/README.md:45-84`](../../../tools/server/README.md:45) | `/metrics` serves ordinary metrics; `--kv-pager` is not parser-registered and benchmark adapter reports `not_configured`. | Server owns ordinary route teardown; no pager object/metric registry exists. | Server route/benchmark tests; no pager integration test | Add experimental option/diagnostics, real ledger/counters, and mode lifecycle in 12-01/12-03. |

## Fixed-capacity and trained-context audit

The following occurrences were located deliberately; test and historical
numbers are not treated as production defaults.

| Location | Finding | Classification / disposition |
| --- | --- | --- |
| [`src/llama-cache-budget.h:42-63`](../../../src/llama-cache-budget.h:42), `mtp_tokens = 262144` | Admission input defaults native MTP to the trained/reference context instead of receiving resolved target C. | Production bug; 08-02 must remove the floor and use the constructed target context. |
| [`src/llama-kv-policy.h:7-9,75-81`](../../../src/llama-kv-policy.h:7) | `DEFAULT_CAPACITY = 304` and 96/32/140/36 absolute quotas are encoded in controller configuration. | Production bug; 08-03 must accept runtime H, deduplicate mandatory pages, and derive ratios/minima. |
| [`src/llama-kv-attention-telemetry.h:9-12,127`](../../../src/llama-kv-attention-telemetry.h:9) | Fixed 1024-page array is allocated for the 256K model. | Production bug for dynamic contexts; 11-03 must use dynamic/context-sized storage or an explicitly accounted bound. |
| [`src/llama-kv-residency-transfer.h:50-53`](../../../src/llama-kv-residency-transfer.h:50) | `max_pages = 1024` is a default transfer-plan bound. | Production bug when used as hidden capacity; retain only as an explicit accounted safety limit in 09-05. |
| [`src/llama-kv-fixed-window.h`](../../../src/llama-kv-fixed-window.h), [`tests/test-kv-residency.cpp:29-30`](../../../tests/test-kv-residency.cpp:29) | 304 and 1024 occur in fixed-window geometry/page-count fixtures. | Explicit fixture/historical geometry, not itself a production default. |
| [`src/llama-context.cpp:178,442-449`](../../../src/llama-context.cpp:178) and model APIs | `n_ctx_train` resolves `-c 0` and logs model capability comparisons. | Model capability bound/target resolution is valid; native-MTP callers at server lines 5571-5575 and 7492-7498 incorrectly use `max(target, n_ctx_train)` and are production bugs owned by 08-02. |
| [`tools/server/server-context.cpp:6969-7005`](../../../tools/server/server-context.cpp:6969) | Fit path preserves model-frontier/full-target MTP behavior. | Production MTP-fit bug when it determines native MTP rows; correct with 08-02. |
| [`tools/server/README.md:54`](../../../tools/server/README.md:54), evidence and handoffs | About 304 pages / 77,824 tokens is an operator/design target. | Historical/evidence value; keep immutable, never use as runtime default. |
| `src/llama-vbr-artifact*.{h,cpp}` and unrelated mmap/VBR ring limits | Bounds protect artifact streams or a different representation controller. | Explicit safety bounds/separate subsystem; not target hot capacity unless a future pager caller supplies and accounts them. |

## Compact dependency graph and write ownership

```text
08-02 resolved C + native MTP GPU reservation
  -> 08-03 realized ledger -> H + dynamic policy inputs
  -> 09-01 config gate -> 09-02 pager owner/storage -> 09-03 target writes
  -> 09-04 host seal/catalog -> 09-05 CUDA transfer/publication
  -> 10-01 reference graph -> 10-02 direct decode -> 10-03 epochs
  -> 10-04 selected prefill -> 10-05 exact waves
  -> 11-01 summaries -> 11-02 retrieval -> 11-03 telemetry
  -> 11-04 policy publication -> 11-05 prefetch/lifecycle
  -> 12-01 CLI -> 12-02 metrics/adapter -> 12-03 mode/fail-closed proof
```

Likely write ownership is symbol-specific: `common/speculative.*` and
`tools/server/server-context.cpp` for 08-02; `src/llama-cache-budget.*` and
`src/llama-kv-policy.*` for 08-03; `common/arg.*`, server parameter plumbing,
and tests for 09-01/12-01; `llama_context` plus a new pager owner and
`llama-memory` integration for 09-02/09-03; VBR artifact catalog/capture and
residency transfer/transaction adapters for 09-04/09-05; graph/model attention
construction and CUDA backend launch sites for 10-01–10-05; routing, telemetry,
policy, and prefetch seams for 11-01–11-05; and `tools/server/server-context.*`,
server routes/metrics, benchmark adapter, and parser-generated docs for
12-01–12-03. Later tasks must preserve the split between representation and
target-residency ownership.

No direct source implementation was performed by this audit.
