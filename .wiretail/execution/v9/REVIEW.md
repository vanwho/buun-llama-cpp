# V9 review algorithm — clone this contract for future remediation reviews

Input is **only the just-completed phase's compact benchmark summary** and
its referenced new raw receipts as needed to resolve a specific inconsistency.
For 38-01 that is `evidence/V9_SUMMARY.json/.md` from 37-04. Do not review
all old task gates, acceptance documents, diaries or phases 00–32.

Write `V9_REVIEW.json/.md` with boolean `goal_met`, list `next_task_ids`,
capability status, key rates/ratios and sample
counts, failure cause with current source symbol, uncertainty, next action,
and whether another phase was actually appended to state. Keep analysis
under ~200 lines. No target-speed claim without matched raw measurements.

## Deterministic decision order

1. Identity/types wrong or evidence inconsistent -> repair the exact identity,
   configuration or measurement boundary first; no new scale campaign.
2. Crashes/non-finite output/invalid indexing -> smallest reproducer and owner
   fix before further optimization. Do not revive custom direct MMA just to
   avoid repairing the selected production path.
3. Missing real Q-to-cold-to-target-use proof -> catalogue/producer/selection/
   copy/publication fix at the first missing edge. Forced page transfer is
   insufficient. Freeze longer contexts until that edge is proven.
4. Stable proof but slow -> attribute actual costs (GPU attention vs packing,
   CPU scheduling/sync, router, H2D/D2H wait, graph rebuilding, weight/recurrent
   kernels, or lost MTP acceptance). Add the smallest context-cohesive set of
   targeted implementation tasks needed for the dominant measured costs;
   keep related repairs in one cluster when their stable context remains
   useful, and split at a real subsystem or context-budget boundary. Every task names current symbols,
   invariant, intended change, bounded fixture, before/after quick test and
   stop condition. No “research every optimization” task.
5. Small fast but scale fails -> fix measured ledger/working-set/copy/selector
   scaling, preserving L and full-L GPU MTP; use the failing frontier, not the
   entire six-coordinate curve, for validation.
6. If stable natural paging + native GPU Turbo4 MTP and speed measurements
   across the active campaign are demonstrated, report achieved dimensions
   explicitly. “Much faster” requires actual CPU-KV paired results, not just
   beating older broken pager code. The historical 3x/5x aspirations are
   annotations, not hidden universal gates. A modest gain is reported as such;
   if it fails the practical goal, schedule the dominant repair.

The complete 256K goal is distinct from the current up-to128K live campaign.
Report `active_campaign_met` and `overall_256k_demonstrated` separately. If
128K succeeds and 256K remains plausible, propose the next bounded capacity /
incremental validation phase from measured ledger/ETA, with no YaRN or
mandatory long all-at-once prefill. Do not label 128K as 256K achievement.
If current speed problems make 256K exploration unhelpful, fix them first.

## If the goal is not met: append actual tasks before completing review

Choose the next unused remediation phase (initially 39), and the following
review phase (40). Add packets, bounded new cluster(s), and `WORK_STATE.json`
entries in execution order; **do not just mention a next action in prose**.
Use the task-appropriate `recommended_model` (`Luna Medium` for bounded
measurement/metadata or `Luna High` for implementation and diagnosis),
`retry1_reasoning: high`, new revision
ID, explicit `context_files`, `status: todo`, null commit/blocker, ordered
dependencies. Preserve old tasks and all usage accounting. Group tasks by
stable context and token/cache efficiency: keep a cohesive sequence together
while its context remains useful and within provider/runner guardrails; split
at subsystem, risk, provenance, toolchain, model, or context-budget boundaries.
There is no fixed tasks-per-cluster cap; a one-task cluster is appropriate for
an isolated live test or review, while a larger cohesive cluster is allowed
when its prompt context remains bounded and productive.

Copy this review's `completion_check` argv into the new review task, changing
its `--review` receipt path to that review's new JSON. This outer-runner check
rejects missing/malformed review or prose-only scheduling even if an agent
incorrectly sets the task status to done. Keep the validator under the current
metadata tree, extending its revision rules when introducing new contracts.

After identified implementation repairs, always append:

1. A bounded quick + natural cold proof and matched original-three-prompt
   performance evidence task using B1–B5. Reuse the current harness/receipts,
   do not create another incompatible acceptance framework. Scale only if
   appropriate to the defect and pilot budget.
2. A compact new phase summary task with identity, rates, memory, actual cold
   use, failures, uncertainty and recommended next action.
3. A review task in the following phase containing **this same review
   algorithm**, updated to read only that new phase summary. It must itself
   append remediation/benchmark/summary/review if still incomplete.

Dependency for the first added task is this review; subsequent tasks depend
on the owning repair/benchmark. Add entries while this review is still
in_progress, validate, then mark review done so Wiretail selects the first
new task automatically. Runner completion of a review is not product success.

If the result is a genuine external impossibility requiring new hardware or
authority, state that exact measured limitation and block/request direction;
do not manufacture endless identical phases. Available sudo, target model or
candidate replacement is not such a blocker. Do not lower target types,
move draft to CPU, silently shrink L, or weaken proof to close the list.

Before done: run `python3 .wiretail/execution/v9/validate_plan.py --review
.wiretail/execution/evidence/V9_REVIEW.json` (one shell command), validate all
new packets/cluster/state paths; verify first next
task is returned by `task_state.py next` after the review transition; record
new task IDs in review JSON. This catches the phase-32 prose-only scheduling
failure without reloading any phase-32 context.
