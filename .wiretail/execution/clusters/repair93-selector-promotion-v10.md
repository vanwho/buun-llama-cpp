# Cluster repair93-selector-promotion-v10

Revision: `hotpath-v10-20260914`. Amendment: `repair93-selector-promotion-20260925`.

This cluster starts after the phase-93 upstream integration and benchmark
contract lock. It repairs/proves natural cold-page promotion and then owns the
dependent paired cold-context speed screen and final phase review. The existing
`repair93-mtp-diagnostics-v10.md` performance contract remains normative for
the 93-12/93-13 speed and quality thresholds. Keep the tasks separate by code
boundary so selector semantics, decision evidence, and live results cannot be
confused.

## Shared context and invariants

- Read the current task packet and only the listed source/test files. Do not
  load WORK_LOG, WORK_STATE, prior attempt JSONL, or unrelated phase history.
- The candidate is Qwen3.8-27B UD-IQ4_XS. Target K/V and native-MTP draft K/V
  remain Turbo4; MTP remains GPU-resident. Preserve the current successful
  managed server after each task unless the task explicitly loads its candidate.
- The test workload is a natural three-turn same-slot conversation using the
  existing ten immutable Python/Bash fixtures. No forced page IDs, eviction,
  promotion, or route override may count as natural proof.
- Selector diagnostics must be opt-in and bounded to refresh/scheduler
  boundaries: selected cold IDs (at most the configured cold top-k), target
  page eligibility/version, mailbox result, policy decision, and transfer /
  publication / use stages. Never copy the attention matrix, scan/log the
  complete page table per token, or add a synchronous GPU fence to the normal
  decode path.
- An empty `natural_proof` means no completed promotion proof; it is not direct
  evidence that no selector nomination occurred. Report stage-specific facts.
- Every failed setup/configuration is repaired and rerun within its owning task.
  A failed model answer is reported separately from physical page movement.

## Task sequence

1. `93-11l`: select the last valid causal Q row of each selector-bearing
   microbatch, pair it with that row's logical position, and add a deterministic
   regression where row zero is unrelated but the final row targets a cold page.
2. `93-11m`: add bounded, opt-in selector→mailbox→policy→H2D diagnostics and
   deterministic tests/receipt validation. Prove the diagnostics distinguish
   ineligible, ranked-out, mailbox-dropped, policy-rejected, transfer-rejected,
   and promoted/used outcomes without per-token logging.
3. `93-11n`: rebuild/load the exact corrected candidate and run one natural
   candidate-bound file-recall/promotion campaign. Do not repeat an unchanged
   failed conversation; fix the first measured failing stage and create a
   dependent repair task if that requires source work.
4. Deferred `93-11e` remains historical only; do not run it.
5. `93-12` runs paired cold-context speed after 93-11n plus the existing
   upstream and benchmark-contract prerequisites; `93-13` reviews those
   results and schedules only evidence-driven follow-up work.

After those tasks, 93-12 may run only when 93-11n proves cold page → selector
nomination → completed H2D → mapping publication → target graph use, with
per-request GPU Turbo4 MTP placement verified independently. Answer scoring,
MTP acceptance, and physical promotion remain separate evidence fields.
