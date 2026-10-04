# Cluster forward102-full-256k

Revision: `hotpath-v10-20260914`. Amendment: `repo-context-scale-20261004`.

Tasks in this context area: `102-06`.

Purpose: Prove full 256K history, hot paging and full-L GPU Turbo4 MTP using
the deterministic tracked-repository corpus built and validated in 102-02.
Start at L=262144 with target C=250,000 committed repository tokens (decimal
250K), leaving 12,144 tokens for the measured final query/output/replay/MTP
reserve. Use a filled H=51,200-token hot window (200 pages), matching 102-05.
Do not substitute 16K or any smaller hot window. If the measured reserve does not fit, stop at the
safe C and record the shortfall; never trigger compaction to reach the target.
CPU target-KV is
omitted; compare selected mode with all-GPU target KV only if a read-only
capacity check shows the latter fits alongside weights, full-L draft, scratch
and reserve. On clean-start OOM/scratch pressure diagnose identity, stale
processes, configuration and accounting. Keep H=51,200, L, B/U, batching and
all other server settings unchanged; genuine inability to admit all 200 hot
pages is a not-admitted result, not authorization to lower H.

Reuse the freshly built, identity-matched binary from task 102-05 unless the
engine source changed. The exact first-row effective server command is in
`tasks/102-06.md`; the 128K/H=50K process must be fully stopped before the
L=256K/H=50K process starts. This is a fresh context, not a continuation.

Read the current packet and forward/OVERVIEW + forward/TESTING only, plus
its explicit source regions and compact immediate-predecessor handoff.
Do not read the full Sol source handoff, REPAIR85/old cadence design, retired
93-12/93-13 packets, raw JSONL or every historical gate. This fresh cluster
separates the current source/lifecycle/risk boundary from prior noisy sessions.
Cluster size follows shared context and evidence, with no fixed task-count cap.

Keep the existing pager/host/selector/Turbo4/GDN/MTP substrate. Selection occurs
at the final-user query boundary, changed history triggers query-only replay,
then history freezes. Generation recycles only eligible output-owned pages.
Full-L GPU Turbo4 MTP remains the first production design. All geometry derives
from the actual model/device admission. Stage counters may not add CUDA fences.
Use one managed Qwen process and preserve its verified candidate on success.

Each packet specifies minimal tests. Broader canonical performance/scaling
work belongs only to phases101/102. Bad measured speed creates a concrete
source repair/retest before scaling; missing/setup-invalid rows do not pass.
New candidate semantics are validated by the task's executable named proofs.
Handoffs contain current result/symbols/raw pointers, never an appended diary.
