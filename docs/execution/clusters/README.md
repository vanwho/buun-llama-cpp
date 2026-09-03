# Cluster context contract

`WORK_STATE.json` assigns each task to one session cluster. The clustered runner reuses a Codex thread
only while consecutive tasks have the same cluster, and its prompt tells the agent to read the matching
file in this directory.

Clusters are intentionally smaller than phases. Two or three tasks share a session only when they use
the same source ownership and intermediate mental model. Repository provenance, CUDA kernel work,
server lifecycle, benchmarks, exact reference work, and upstream slicing rotate to fresh sessions.

All task recommendations use `gpt-5.6-luna`: Luna Medium is the default, Luna Low is reserved for
checklist/documentation/pure-arithmetic tasks, and Luna High is reserved for cross-repository placement,
VBR/residency transactions, CUDA/operator integration, concurrent server/speculative lifecycle, CUDA
acceptance, and exact online-softmax work. A Luna task that blocks twice receives one final Terra/high
recovery attempt before the runner preserves a real block; Terra is not a normal task recommendation.

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
