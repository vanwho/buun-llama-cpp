# Active R8: normal conversation benchmark completion

Root verified and activated this plan. Server commands are unchanged; only the client carries all prior Q/A forward.

# Comparison completion R8 — active, 2026-11-05

**Active and frozen after root review.** The normal append-history harness is implemented;35 tests pass and driver hashes are refreshed. Server configurations remain unchanged from R7.

- Maximum canonical requests: **492** across **120 remaining feasible missing comparison cells plus 3 independent outlier cells**, four requests per cell (one warmup, three measured).
- The 448-request lower-profile catalog is an alternative fallback catalog, not additive. Maximum queue budget stays 492.
- R7 master CPU MTP-off completed 27 missing comparison cells and is retained; do not rerun it. R7 master GPU MTP-off is retained as a partial diagnostic after two valid C6144 requests and a checkpoint-restore runtime fault.
- All valid R4/R5/R6/R7 records remain retained; existing quarantines remain quarantined. C130048 GPU endpoints remain explicitly capacity N/A.
- Preserve every R7 server argv/configuration: binaries, model, template, batch/ubatch, quantization, MTP/pager choices, threads, context limits, checkpoints, cache flags, and sampling. Only external artifact paths and client conversation construction change. No benchmark-owned slot save/restore/erase.
- Maximum nominal fill targets: **54** across all primary and alternative profiles; a fallback replaces the failed profile and is not additive. If accumulated Q/A already exceeds a nominal target, skip new repository ingestion for that target.

## Ordered work

| # | Family | Planned work | Primary cells | Alternatives / notes |
|---:|---|---|---:|---|
| 1 | master-gpu-mtp-off | 42 remaining feasible cells at L122880 / 32 CP | 42 | C130048 remains N/A |
| 2 | fork-cpu-mtp-on | Primary L262144 / 32 CP | 37 | Startup-capacity alternatives: L200704 (37 selected), L131072 (36 selected); not additive |
| 3 | master-cpu-mtp-on | Primary L200704 / 32 CP | 41 | Startup-capacity alternative: L131072 (39 selected); not additive |
| 4 | master-cpu-mtp-on outlier A | Independent clean C28672/off/Prompt 3 cohort | 1 | One mapped fill for this series |
| 5 | master-cpu-mtp-on outlier B | Independent clean C28672/off/Prompt 3 cohort | 1 | One mapped fill for this series |
| 6 | master-cpu-mtp-on outlier C | Independent clean C28672/off/Prompt 3 cohort | 1 | One mapped fill for this series |
| 7 | fork-gpu-mtp-off | L122880 mapped-prefill-only profile | 0 | No canonical decode |

## Append-only request history

Each series uses one mutable, append-only conversation history. Every canonical question follows all earlier canonical user questions and complete assistant answers, including the assistant reasoning content; this continues across prompts, reasoning modes, and nominal stages. Never return to a base prompt, discard prior Q/A, rewind, restore a slot, or reload already ingested repository bytes. Each independent outlier cohort starts a fresh history.

Nominal C remains a grouping label. Record actual occupied context immediately before and after every request. Append only new disjoint bytes from the frozen source map when more repository content is needed. If Q/A history already reaches or exceeds the next nominal C, skip the repository fill and keep the actual C. Before appending more content, reserve (40 warmup + 3×400 measured output + 4×128 request/history overhead) for every remaining canonical case in the series, then add a 1024-token guard. Ensure the aggregate reserve fits under L. Do not shorten response caps or use context shifting.

No automatic full-curve escalation or automatic retry of completed cells. Pause for assessment on actual runtime/HTTP faults and preserve valid partial evidence. Report outlier cohorts after the planned series finish; do not pick the fastest cohort.

## Retained evidence

R7 step 1 is complete and retained at `/srv/ai/paged-kv/results/comparison-completion-20261104-r7/master-cpu-mtp-off/attempt-001-L262144`. R7 step 2 is retained at `/srv/ai/paged-kv/results/comparison-completion-20261104-r7/master-gpu-mtp-off/attempt-001-L122880`: C6144 had two valid requests before the next checkpoint-restore request faulted with CUDA illegal memory access. This partial cell does not count as complete. All prior approved complete cells and quarantines are retained as listed in the JSON.
