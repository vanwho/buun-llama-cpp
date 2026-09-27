# Cluster forward94-turn-epoch

Revision: `hotpath-v10-20260914`. Amendment: `forward-turn-retrieval-20260927`.

Tasks in this context area: `94-01`, `94-02`.

Purpose: Add turn retrieval state and derived R/G admission; Freeze historical selection during generation and native MTP.

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
work belongs only to phases100/101. Bad measured speed creates a concrete
source repair/retest before scaling; missing/setup-invalid rows do not pass.
New candidate semantics are validated by the task's executable named proofs.
Handoffs contain current result/symbols/raw pointers, never an appended diary.
