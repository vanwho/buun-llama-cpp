# Cluster repair93-selector-promotion-v10

Revision: `hotpath-v10-20260914`. Amendment: `repair93-selector-promotion-20260925`.

This cluster records the completed selector-row and bounded stage-diagnostic
work in 93-11l/93-11m. The live graph-registration repair, natural-promotion
campaign, paired speed screen, and phase review now have fresh, narrower
clusters so a failed long-running session cannot be reused across those
boundaries. This file is historical context for the completed tasks only; do
not use it as the active cluster context for 93-11o onward.

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

## Completed tasks

1. `93-11l`: select the last valid causal Q row of each selector-bearing
   microbatch, pair it with that row's logical position, and add a deterministic
   regression where row zero is unrelated but the final row targets a cold page.
2. `93-11m`: add bounded, opt-in selector→mailbox→policy→H2D diagnostics and
   deterministic tests/receipt validation. Prove the diagnostics distinguish
   ineligible, ranked-out, mailbox-dropped, policy-rejected, transfer-rejected,
   and promoted/used outcomes without per-token logging.
The successor clusters own the still-open work. Consult their task packet and
cluster only; do not load this file for those tasks.
