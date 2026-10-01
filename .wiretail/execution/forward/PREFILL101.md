# Selected prefill: remove repeated CPU work and unnecessary submission barriers

Revision: `hotpath-v10-20260914`. Amendment: `prefill-owner-101-20261001`.
This is the implementation direction for tasks 101-13–18 plus 101-12h, after the
currently running 101-12g measurement and before 102-01. It supersedes the generic
"profile another bottleneck" successor instructions for this boundary. Do not
reload completed phase-100/101 packets, the entire journal, or raw Codex JSONL.

## Findings, confidence and what has not been established

The completed 101-12d matched campaign reports selected prefill medians
331.28/386.66/331.21 tok/s, CPU-main-KV/GPU-MTP 659.75/661.72/662.30, and
pager-off GPU 1566.34/1562.65/1564.67. Selected decode is 34.69/34.30/37.11;
MTP acceptance is 48.88/48.44/57.05%. These are measurements, not forecasts.
All have the same frozen prefix, L8192/H4096/P256/B1024/U256, fresh input,
reasoning off and GPU native Turbo4 MTP. The summary-vector move in 101-12c
did not materially improve prefill. Controls are matched in model and binary;
ordinary CPU attention's supported codec must remain explicitly labeled.

Source inspection identifies two definite redundant-work patterns:

1. `llama_kv_pager::seal_ready_pages` iterates configurations before pages.
   `llama_kv_cache::pager_routing_summary_build` caches **one page**, rebuilding
   all layers/heads when identity/content version changes. For P changed pages
   and N configurations, a configuration-major A/B/A/B visitation can perform
   P*N all-layer builds instead of P. Each build reads encoded host K, unpacks
   Turbo4 and reduces min/max/mean on the calling CPU thread. This is CPU
   metadata work, not a CPU attention fallback. Its multiplication factor in
   the current live run must be measured, not assumed. Existing single-page
   tests and the 101-12c move do not cover multi-page cache locality.
2. `llama_kv_routing_summary_store::update_pages/reconcile/invalidate_*` copy
   tables containing owning float vectors. `rebuild_accounting` hashes every
   float in every retained page after an update. Thus even a one-page delta
   copies and hashes unchanged catalogue payload; repeated prefix growth can
   make work quadratic in page count. This is avoidable payload work on the
   inference CPU thread, independent of whether CPU storage uses Turbo4.

Two further serialization risks require current operation-level evidence:

- `llama_context::decode` calls `synchronize()` on `prefill_page_wave_boundary`;
  Before101-12f, U=P=256 normally fenced every page. Its fresh-sequence fix
  now coalesces to physical admission/outer-batch boundaries; preserve it.
  Existing/irregular layouts still fall back to page fencing. Synchronize seals
  summaries and retires graph/source ownership. 101-07 kept this fence; its async host worker
  did not remove it. Event-driven D2H alone does not pipeline CPU submission.
  Removing the call without replacing those owners would corrupt data.
- Actual CUDA graphs require stable node properties and two compatible warmup
  calls in `ggml_backend_cuda_graph_compute`. Selected shape/pointer churn can
  prevent that warmup. Pager `graph_capture/replay_count` records the pager's
  graph bookkeeping, not actual CUDA driver captures/launches. Use existing
  `GGML_CUDA_GRAPH_DIAGNOSTICS`, actual timeline events and reuse reasons.

Selector sidebands in `set_kv_page_select_inputs` use tensor_set, but **do not
assume each call blocks**: merged CUDA already stages small writes through an
8 MiB per-thread pinned ring (`ggml_cuda_upload_async`). Fallback, allocation
failure and ring wrap can synchronize. Measure those cases and reuse the
existing owner; do not invent a second upload pipeline or naively replace
stack/vector-backed writes with unsafe async copies.

The old 101-01 trace showed only 1.098 s union GPU activity in a 19 s prefill
window. It supports looking for CPU gaps, but predates subsequent changes.
Current `wait_us` times **whole scheduler completion**, including GPU math;
`queue_us` times **whole graph_compute_async submission**, not H2D queue delay.
Request deltas include generation/verification. Do not add overlapping scopes
or interpret these counters as proof of cold-page stalls. Current supplemental
rows have zero useful H2D: cold promotion traffic cannot explain their loss.
CPU attention fallback, serial GDN and packed materialization are not proven
dominant causes of the selected-only regression. Dense native-MTP controls
are already fast; do not rewrite their common arithmetic without attribution.

## Chosen sequence and hard invariants

101-13 obtains one current CPU/CUDA attribution and a multi-page reproduction.
101-14 fixes cache visitation and honest physical-read accounting.
101-15 shares immutable catalogue payload and caches per-page hashes.
101-16 verifies/fixes actual graph reuse and ordered mutable sidebands.
101-17 extends the fresh101-12f repair to safe cached contiguous waves
and capacity-driven source/graph ownership; no duplicate fresh-path rewrite.
101-18 runs focused affected proofs and the final canonical paired benchmark.
101-12h reviews that new result and inserts a **measured-owner** successor if
needed. 102-01 depends on 101-12h, not the historical 101-12e speed miss.

All model operations stay on GPU. RAM is inclusive encoded Turbo4 storage;
small routing metadata on CPU is permitted if bounded and demonstrably cheap.
Keep native draft cache full logical L, GPU-resident, K/V Turbo4; no F16 draft
cache, CPU attention, selected_reference, disabled MTP, changed prompts, or
fixed production hot/context/head counts as a shortcut. Preserve final-user
query selection/replay, frozen historical membership through generation and
verification, exact identity/version authentication, and safe rollback.

CPU summary unpack is only metadata reduction. If after 101-14/15 it still
occupies >=10% of isolated prefill or a comparable GPU idle gap, 101-12h must
schedule GPU page-summary reduction using the existing
`GGML_OP_KV_PAGE_SUMMARY` owner in `ggml/src/ggml-cuda/kv-page-summary.cu`
and encoded keys, not CPU attention or a new
pager. Otherwise retain the now-bounded implementation. Do not prematurely
offload bookkeeping whose measured cost is negligible.

## Small tests, useful context, truthful completion

New clusters group summary/catalogue work separately from GPU submission and
final live measurement. Startup context is the current packet/cluster, this
file, TESTING, and one compact immediate predecessor handoff. Source pointers
are instructions to inspect symbol-sized spans on demand, not entire files to
inject into every prompt. Do not load 101-01 raw trace or whole historical
receipts; retain a small owner/time/count table and raw-path/hash references.

Use affected deterministic tests first. Most implementation tasks need no live
model: one fixed-prefix, <=80-output row is allowed after a material change.
101-13's attribution and 101-18's canonical suite are the only mandatory live
campaigns here. Keep B1024/U256, L8192/H4096 fixture geometry; production sizes
remain tunable/derived. Use the existing single managed Qwen process lifecycle,
matching binary/DSO/model/PID/argv, and passwordless `sudo -n` as documented in
TESTING. Never launch a second weights copy or touch 8092.

Tasks 101-13–17 complete from their named executable regression/attribution,
not a 500 tok/s pass. Measured no-change dispositions for a serialization
hypothesis require real evidence and an executable safety/regression check;
they must not be a generic source-read assertion. Final missed speeds/acceptance
are findings requiring ordered repairs, not reasons to repeat unchanged tests
or falsely mark the whole goal complete. Receipt/task transitions use the
existing V10 validator and generic state CLI; preserve all usage accounting.
