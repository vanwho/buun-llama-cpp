# Cluster forward102-128k-fresh

Revision: `hotpath-v10-20260914`. Amendment: `repo-context-scale-20261004-50k-hot-fresh-run`.

Tasks in this context area: `102-05`.

Purpose: perform one fresh, candidate-bound L=131072 selected-GPU Turbo4 load
with the full H=51200 (200-page) hot window filled; capture prefill speed as
occupied context grows, VRAM headroom, and canonical MTP results. This is a
procedural live benchmark on one freshly built candidate, not a source repair.

Start from task 102-05's exact first step: force a clean configure/build into
`build-102-05` (`cmake --fresh`, then `cmake --build ... --clean-first`). Do
this before service actions. Do not reuse a previously loaded server or older
build. Verify candidate hash and source HEAD before managed reload. Follow the
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

Read only the current 102-05 packet, this cluster, and its explicitly listed
benchmark code/fixture files. Dependency 102-04 is a sequencing prerequisite
only; do not load its packet or handoff, and do not import geometry or findings
from previous attempts. Begin a new task attempt with a fresh Codex context and
an empty `attempt-01` result directory. The archived interrupted-run pointer,
old transcripts, and prior run logs are audit history only and must not be
resumed or included in task context.
