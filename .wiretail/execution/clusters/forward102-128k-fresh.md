# Cluster forward102-128k-fresh

Revision: `hotpath-v10-20260914`. Amendment: `repo-context-scale-20261004-50k-hot-fresh-run`.

Tasks in this context area: `102-05`.

Purpose: obtain one candidate-bound L=131072 selected-GPU Turbo4 load with the
full H=51200 (200-page) hot window filled; capture prefill speed as occupied
context grows, VRAM headroom, and canonical MTP results. Attempt 01 proved the
candidate and effective argv were correct but admission stopped at 66 pages.
The old ledger charged hypothetical packed/F16 workspaces proportional to H.
The explicit-H source correction now omits those prospective charges and
unmeasured nominal target/draft compute asks; actual allocations still apply.
Retry 1 must clean-build this changed source and measure actual memory through
startup, H crossing, the occupied load, and canonical GPU-MTP generations.
Do not infer that 200 pages fit or that lazy workspaces are unnecessary.

On attempt 01, start from task 102-05's exact first step: force a clean
configure/build into `build-102-05` (`cmake --fresh`, then
`cmake --build ... --clean-first --parallel 16`). On retry 1, preserve
attempt-01 artifacts and clean-build the accounting/measurement source change
into the same path. Do not reuse the old attempt-01 candidate, a loaded server or
older build. Verify candidate hash and source HEAD before managed reload. Follow the
packet's exact effective argv, including explicit `--no-context-shift`; verify
the flag and all geometry from the live process argv. Any change to candidate,
model, L/H, B/U, codec, pager, or MTP configuration requires a clean stop,
restart, and identity verification before more requests. Never run two Qwen
processes. Run one short 80-token prompt-1 native-MTP startup smoke before the
expensive fill; it is diagnostic only and has no acceptance threshold. A crash
or failed request must be diagnosed from its raw log and exact candidate
identity; do not mask it by changing task geometry or routing without
source/runtime evidence.

Preserve and resume the same occupancy checkpoint on valid progress; do not
replay/refill committed history. Host-residency telemetry is optional and is
not a gate. Keep the successful candidate loaded for 102-06.

The compact source assessment is engineering-notes/pager-admission-memory-accounting.md.
H=200 target storage alone is 825 MiB; the old combined model charged 2675
MiB. Packed owners remain lazily allocated with safe draining lifetimes;
routes and row capacity are unchanged. Target and MTP schedulers are distinct,
so two nominal estimates were not proven duplicate allocations. Use
forward/sample-pager-memory.py once across the actual workload; persist raw
samples and its compact summary outside Git, then update the engineering note
with observed peak/free bytes. Missing optional fields are unknown, not zero,
and must never gate loading. Keep H=200 and L/B/U/MTP unchanged. Do not add
another diagnostic-only stress campaign, speculative route rewrite, or repeat
an unchanged admission refusal. An actual OOM needs a named allocation-owner
repair placed before 102-06, not an evidence-only retry loop.

Read only the current 102-05 packet, this cluster, and its explicitly listed
benchmark code/fixture files. Dependency 102-04 is a sequencing prerequisite
only; do not load its packet or handoff, and do not import geometry or findings
from previous attempts. For the next run, use a fresh Codex context and a new
`attempt-02` result directory while preserving `attempt-01` as immutable
evidence. Do not load old transcripts or large raw logs into task context; use
only the compact handoff diagnosis and named metric/source excerpts in the
task packet. The archived interrupted-run pointer and other prior logs are
audit history only and must not be resumed or included in task context.
