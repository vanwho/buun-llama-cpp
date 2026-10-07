# Occupied-context scaling costs (105)

Source-only assessment recorded 2026-10-08. This note distinguishes costs
visible in source from measured attribution. No production change, build, or
runtime profile was performed for this assessment.

## What the current measurements establish

The completed 105-03b attempt02 split-layout trajectory reached C=249,293
(707 below its requested 250,000 target) at L=262,144/H=51,200. All 22 scheduled
requests completed; A2 and both extra source-specific facts were correct, and
all 12 occupied-base canonical rows passed. The occupancy checker reports
`execution_status=complete` and numeric `goal_status=goal_miss` for the 707
token target gap; retain both facts. Physical cold-page rank/promotion remains
unknown. Candidate identity fingerprint was
`eb71506818a87d651a72794ff3380a0c73d9ca84eb3857211a27260aa9967d70` (server
SHA-256 begins `2d46af95`; model SHA-256
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`).

Across 21 fill rows median fresh/processed input was 787.27/789.39 tok/s and
median decode41.35 tok/s. Fresh points: ~1,224 tok/s at C=4,293; ~1,214 at
C=16,586; ~816 at C=126,960; ~739 at C=200,212; and 699.11 fresh / 700.99
executed prompt tok/s at C=249,075 (11,878 fresh, 11,910 executed prompt,
16,990 ms). The 64-fresh-token final A2 is not a bulk-prefill point. Canonical
MTP medians and outcomes are compactly recorded in `handoffs/105-03b.md`:
merge 43.32 tok/s / 88.28%, mmap 31.17 / 48.91% (pooled 48.78%), watcher
40.75 / 79.78%. The paired 105-03a small-context split arm remains 1236.61
fresh tok/s versus
703.20 legacy.

The older 105-03 curve reached C=257,445; near C=248,771 it measured322.6
fresh / 645.0 processed tok/s and 38–39 decode tok/s. The newer split-layout
curve is substantially faster near a similar C, but the candidate/layout
experiments are not a controlled same-binary comparison, so do not attribute
that delta solely to split layout or to any one source operation. No exact
same-candidate long CPU comparison or stage attribution was collected. Do not
turn these rates into a new pass threshold: 105-05 uses the matched small
uncached-input benchmark for bulk-prefill, with large-context rates kept as a
separate curve.

## Source-proven C-dependent work

1. Summary-accounting work grows with retained cold pages. In
   `src/llama-kv-pager.cpp::seal_ready_pages`, changed pages are grouped by
   summary configuration and published atomically. For each layer/head,
   `src/llama-kv-routing-summary.cpp::llama_kv_routing_summary_store::update_pages`
   merges a seal wave into the full retained descriptor vector (lines783-805)
   and calls `rebuild_accounting` (806-810). That function scans all retained
   descriptors for allocation byte accounting, source totals, and a content
   hash (1019-1098). Cost is O(number of retained pages) per configured
   layer/head per seal wave. This is a real asymptotic cost; its contribution
   to the recorded request times is unmeasured. The prior nested O(P^2)
   pointer-dedup loop was already replaced with an `unordered_set`; do not
   repeat that diagnosis.

2. Selector metadata is repeatedly materialized at occupied-context size.
   `llama_kv_pager::exact_page_records` scans the growing `logical_catalogue_`,
   unions cold and resident records, and sorts them
   (`src/llama-kv-pager.cpp:1098-1132`). A refreshed selector calls it from
   `can_reuse_kv_page_select` and again from
   `set_kv_page_select_inputs` (`src/llama-kv-cache.cpp:20004-07, 20058-66`);
   selector inputs are prepared per layer. The latter also constructs and sorts
   current logical-page indices from the inventory (`20231-35`). This is
   O(C-pages log C-pages) metadata preparation per layer/fence, independent of
   hot H. Device ranking must still consider eligible cold candidates; the
   source establishes repeated inventory/sort work, not that GPU scoring is
   the bottleneck.

3. Full-L MTP has genuine context-sized attention. MTP contexts bypass target
   pager planning (`src/llama-context.cpp::plan_kv_pager`, 1403-10), and model
   cache construction retains the `il >= n_layer()` nextn layer(s) at
   `cparams.n_ctx_seq` (`src/llama-model.cpp:3208-25, 3304-24`). The MTP-only
   cache therefore has no H-bounded target page view; dense MTP attention over
   its populated nextn KV grows with C. Target selected attention separately
   uses the H-bounded resident snapshot (`src/llama-context.cpp:3400-07`). The
   draft cost is structurally real, but no stage timing here quantifies it.

## Narrow owner-level follow-ups (not routing changes)

- For summary accounting, keep immutable exact content-hash and unique-payload
  charging semantics. A follow-up should cache/reuse accounting across
  append-only seal waves or maintain a validated incremental digest/index,
  with rebuild on invalidation, replacement, reconcile, and restore. Preserve
  all-layer/head atomic publication and old payload lifetime. The current
  source alone does not prove a cheap hash shortcut; do not weaken content
  identity or byte-budget accounting.
- For selector preparation, share one immutable, sorted inventory snapshot per
  query/sequence/table epoch across layer callbacks, rather than rebuilding it
  twice per layer. Cache key must bind sequence ID + sequence generation,
  query generation/position, and the residency/catalogue epoch or exact content
  generation. Retire on page identity/content change, catalogue mutation,
  restore, and query-safe-boundary change. Preserve per-layer bounds uploads and
  fail-closed completeness.
- MTP windowing would alter draft attention semantics unless it preserves the
  full target-visible Q/R history and position mapping. Treat it as an
  architectural experiment, not a cheap speed fix.

Scheduling: 105-03b attempt02 now supplies the post-layout curve, occupied-base
canonical rows, and two extra facts; its 707-token frontier shortfall remains a
finding. Proceed with the prescribed 105-04 summary and 105-05 review. The
measured near-250K split-layout rate (~699 fresh tok/s) does not justify
preempting those tasks with a new performance repair by itself. Keep the
source-proven costs as future candidates; require a same-candidate stage
measurement or a reproduced inefficiency before assigning an optimization
task. No new gate follows from this note.

Attempt02 compact artifact hashes: `occupied-frontier.json`
`d7811d038527482253424ce191a50397d5fc02db1fca3cc0bd6a3c19703c2e88`,
`incremental-state.json`
`d1f7c0b1a901b6c2c9a73452df4212503d860eb714dbc1f8462d1c6351b5d893`,
`candidate-identity.json`
`3d9166c3b1bd14f72275c2deaa4ee0bec73c61b1f4ad413a68ead97e8ece7290`, and
`extra-facts-findings.json`
`9fc6b9466159b8e1bdf11eb6dc2945f0305eabe25f8ac2ede364f2d06198a7a4` under
`/srv/ai/paged-kv/results/forward/105-03b/attempt-02/`.

## 105-04 summary/validator recommendation

`FORWARD_FINAL_SUMMARY.json` still identifies `summary_refresh_task` as
`103-04` and its 15 goal rows/results predate 105-03b. Refresh only the
task-required current fields: separate the old 105-03 occupancy curve from
the completed split-layout 105-03b curve; record L/C/H, A2 plus two extra
semantic facts, 12/12 occupied-base canonical outcomes with per-prompt MTP,
the 707-token target gap, unknown physical rank/promotion, and unknown route or
performance attribution. Keep prior capability evidence/provenance intact and
do not mark exact C=L or overall adoption as passed.

The generic `.wiretail/execution/v10/validate.py` check validates required
receipt keys, command/exit status, and referenced artifact paths/hashes; I
found no dedicated validator that reconciles the forward summary JSON with
its Markdown or checks current findings against the 15 goal rows. For the
105-04 proof `refreshed_forward_goal_summary`, run a small explicit consistency
check that validates 15 unique goal IDs, cross-checks new geometry and curve
endpoints/semantic statuses between JSON and Markdown, and verifies the named
current-evidence hashes. Record the actual command, zero exit, output, and
output hash in V10_105-04. This is a summary-integrity check, not a new product
or performance gate.

## Review of current uncommitted pager/test changes

`src/llama-kv-pager.cpp` adds failure-only transaction diagnostics and excludes
dirty pages from clean-victim candidates in both single and batch admission.
The dirty check is consistent with the existing host-valid/content-version/
state checks. Diagnostics do not run on the successful reservation path. I
found no concrete correctness issue in this diff.

`tests/test-kv-pager.cpp` uses public write, turn-transition, clear, seal, and
catalogue APIs to reproduce the sealed 255-row tail at page447 and the next
256-row crossing. It checks exact identity, canonical host survival through
snapshot lifetime and invalidation, physical ticket mapping, cold victim
retention, and canceled-row extent. The target passed CPU-only in the recorded
run, but this note is not a substitute for that test result or evidence that a
specific production failure has been reproduced.
