# R8 checkpoint retention investigation (read-only)

## Confirmed runtime behavior

For fork CPU/MTP-on R8 at L262144, C12288 records reuse nearly the whole occupied history (after the warmup, cached tokens are 12,284). C28672 likewise reports cache=28,668 and short backend suffixes. At C61440, the first 40-token warmup reports cache=61,440, but the next request reports cache=29,349 and backend prompt_n=32,187; later cases remain at 29,349. At C126976, the same cache count leaves about 97.7K tokens to reprocess.

The server log at `comparison-completion-20261104-r8/fork-cpu-mtp-on/attempt-001-L262144/cell/server-startup-20261105T000013Z.log` proves the large drop is a retained-checkpoint fallback, rather than a 32K loss of matching prompt prefix: task 20166 logs edit sample `(61540/61536/61444/61444/96/92/1)` at line 41281, then restores checkpoint position 29348 (`n_tokens=n_past=29349`) and logs `CHECKPOINT_ATTN_ONLY_TRIM p0=29349 target=1 draft=1` at lines 41282–83. The last logged creation before this is checkpoint 32/32 at position 29348, line 29010.

The retained records show the checkpoint problem appears after early cells: C12288 has 36 cases with near-frontier reuse; C28672 has short suffixes; C61440 and C126976 retain only 29,349 tokens. The fixed fill strategy has four mapped repository append requests by C28672. Therefore the earlier claim that the four repository appends alone consumed all 32 checkpoint slots was incorrect; ordinary canonical QA completions also create checkpoints.

## Source policy and limits of evidence

- Checkpoints may be taken at a prompt end even when `checkpoint_min_step` spacing has not elapsed: `tools/server/server-context.cpp:23904–09` includes `near_prompt_end` and `is_last_user_message` as bypasses.
- In lifecycle-authority mode, full-ring replacement asks `checkpoint_capacity_floor` for a victim (`server-context.cpp:24469–24506`). The floor excludes recovery-pinned, mandatory-anchor, hard-lease, and current-task entries (`tools/server/server-task.cpp:4714–45`; `tools/server/server-cache-authority.cpp:298–339`).
- A failed capacity-floor attempt is latched per ring generation and only a committed drop/publication advances the generation (`tools/server/server-cache-authority.h:90–117`; `tools/server/server-cache-authority.cpp:4635–47`). Thus a refused floor can leave a full ring unable to reconsider candidates until something mutates it.
- Retention scoring marks a checkpoint mandatory when its mapped turn ordinal is the first or last boundary in its recorded turn table (`common/common-retention-sidecar.cpp:616–28`). Old request-final checkpoints are therefore a plausible source of protected entries as conversation histories grow.

The saved `/slots` snapshots expose request/cache-status fields but not checkpoint-candidate inventory or protection reasons. The run did not emit destruction receipts, so there is no direct evidence identifying which of the 32 entries were `mandatory_anchor`, hard-leased, recovery-pinned, or current-task entries. The endpoint-anchor explanation is a source-supported hypothesis, not a proven per-entry diagnosis. Do not attribute the blockage solely to the four mapped appends or to `near_prompt_end`.

## Safe operational conclusion

No client/workload rewrite or inference-binary change is justified from this evidence. A fresh recovery-configuration series with the exact same flags and map, skipping already completed lower QA cells and running only the remaining upper cells, avoids inheriting the prior series’ checkpoint lineage. Keep the checkpoint policy and MTP state unchanged. Any future code fix needs cache-plan candidate inventory around capacity-floor refusal to identify the actual pin owner first.
