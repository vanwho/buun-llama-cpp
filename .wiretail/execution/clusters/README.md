# Cluster context contract

`WORK_STATE.json` assigns each task to one session cluster. The clustered runner reuses a Codex thread
only while consecutive tasks have the same cluster, and its prompt tells the agent to read the matching
file in this directory.

Clusters are intentionally smaller than phases. Two or three tasks share a session only when they use
the same source ownership and intermediate mental model. Repository provenance, CUDA kernel work,
server lifecycle, benchmarks, exact reference work, and upstream slicing rotate to fresh sessions.

All current and generated task recommendations use `gpt-5.6-luna` with High reasoning. This uniform
baseline applies even to checklist/documentation work because every task can affect live CUDA,
service lifecycle, benchmark provenance, or resumability. Every task gets three bounded substantive
retries after its initial approach. Retry 1 is only the original task model/reasoning with an artifact-aware direct prefix.
Before retry 2, the task's own model family runs a High-reasoning assessment; retry 2 then uses the
original task model/reasoning with no retry prefix. Before retry 3, the next family in the escalation map
`luna -> terra -> sol` (Sol remains the ceiling) runs a High-reasoning assessment; retry 3 also uses the
original task model/reasoning with no retry prefix.

For every task, read in this order:

1. repository instructions;
2. `WORK_STATE.json` and this cluster file;
3. the current task packet;
4. handoffs named here and by the task dependencies;
5. only the source paths named by those files.

The canonical plan's decision ledger is authoritative when a task encounters an ambiguity. Read only
the named plan sections first; load the full plan only for a genuine unresolved conflict. Cluster files
do not override task acceptance or repository instructions.

| Cluster | Tasks | Shared context |
| --- | --- | --- |
| `00a-provenance` | 00-01–00-02 | Governance, fork/model/hardware provenance |
| `00b-baseline-draft` | 00-03–00-05 | Baselines, salvage map, independent draft placement |
| `01a-page-core` | 01-01–01-02 | Product semantics, page identity/table |
| `01b-budget-policy` | 01-03–01-04 | Admission arithmetic, pure retrieval/retention policy |
| `02a-host-backing` | 02-01–02-02 | Selected capture seam, canonical host catalog |
| `02b-residency-transfers` | 02-03–02-05 | Backend pool, transfers, transactions, fixed-window proof |
| `03a-attention-view` | 03-01–03-02 | Compact native-position view and graph operator plumbing |
| `03b-turbo4-fa` | 03-03–03-04 | Direct Turbo4 CUDA FA and graph integration |
| `04a-routing-telemetry` | 04-01–04-02 | All-page retrieval summaries and observed attention scores |
| `04b-policy-prefetch` | 04-03–04-04 | Hotset decisions and asynchronous intent scheduling |
| `05a-hybrid-mtp` | 05-01–05-02 | Qwen hybrid atomicity and separately pinned MTP |
| `05b-server-speculation` | 05-03–05-04 | Slot lifecycle, checkpoints, speculative rollback |
| `06a-correctness` | 06-01–06-02 | CPU/fake and CUDA fault/correctness matrix |
| `06b-benchmarks` | 06-03–06-04 | Harness integration and selective acceptance |
| `06c-exact-reference` | 06-05 | Online-softmax all-pages oracle |
| `07a-upstream-handoff` | 07-01–07-02 | Reviewable slices, evidence and operator docs |
| `08a-dynamic-contract` | 08-01–08-03 | Live-gap map, context-sized MTP, dynamic hot capacity |
| `08b-benchmark-contract` | 08-04–08-05 | Frozen quality/performance corpus and corrected controls |
| `09a-live-memory` | 09-01–09-03 | Runtime configuration, pager owner, target writes and mutations |
| `09b-live-transfers` | 09-04–09-05 | Canonical host sealing and real CUDA residency transfers |
| `10a-live-decode` | 10-01–10-03 | Production selected reference/direct decode and graph epochs |
| `10b-live-prefill-exact` | 10-04–10-05 | Bounded prefill and live exact page-wave oracle |
| `11a-live-routing` | 11-01–11-03 | Production summaries, cold retrieval, attention telemetry |
| `11b-live-policy` | 11-04–11-05 | Dynamic hot tables, prefetch overlap, atomic lifecycle |
| `12a-operator-surface` | 12-01–12-03 | Experimental CLI, metrics, harness, fail-closed lifecycle |
| `13a-kernel-transfer-opt` | 13-01–13-03 | Reproducible profiles, Turbo4 kernel, transfer overlap |
| `13b-system-opt` | 13-04–13-05 | Prefill/graph optimization and policy Pareto calibration |
| `14a-correctness-quality` | 14-01–14-03 | Full fault matrix, dynamic ladder, held-out quality |
| `14b-performance-soak` | 14-04–14-05 | Final speed gates and endurance/churn/concurrency soak |
| `15a-corrective-integration` | 15-01 | Synced-fork integration and immutable V2 evidence |
| `15b-corpus-contract` | 15-02 | Fact-bearing multi-page corpus and semantic validation |
| `15c-harness-lifecycle` | 15-03 | Live launcher and persistent benchmark server state |
| `15d-mtp-fit` | 15-04 | Dynamic context-sized MTP and VRAM admission |
| `15e-runtime-lifecycle` | 15-05 | Speculative rollback and page lifecycle atomicity |
| `15f-exact-telemetry` | 15-06 | Exact waves, transfers, and live telemetry |
| `15g-quality` | 15-07 | Corrective model-backed quality acceptance |
| `15h-performance` | 15-08 | Corrective paired speed acceptance |
| `15i-soak` | 15-09 | Endurance and handoff lifecycle |
| `16a-upstream-handoff-v2` | 16-01–16-02 | Rebase/slices and portable operator evidence |
| `17a-harness-contract` | 17-01–17-03 | Benchmark lifecycle, runtime identity, telemetry contract |
| `17b-mtp-capacity` | 17-04–17-05 | Dynamic Turbo4 MTP admission and rollback atomicity |
| `17c-corpus-exact` | 17-06–17-07 | Fact-bearing corpus and exact/selected-all reference |
| `17d-benchmark-quality` | 17-08–17-09 | Model-backed quality and paired performance evidence |
| `17e-benchmark-soak` | 17-10–17-11 | Initial soak campaign and compact benchmark summary |
| `17f-context-pressure` | 17-12–17-13 | Dynamic acceptance context and physical pager pressure |
| `17g-quality-oracles` | 17-14 | Exact/native and concurrent correctness oracles |
| `17h-full-benchmarks` | 17-15–17-17 | Full-context quality, paired speed, and pressured soak |
| `17i-full-context` | 17-18 | Full 256K context capacity and residency |
| `17i-summary` | 17-19 | Revised phase-17 evidence summary |
| `18a-benchmark-review` | 18-01 | Benchmark-only overall-goal assessment and remediation chaining (Sol High) |
