# Cluster forward102-128k-repeat

Revision: `hotpath-v10-20260914`. Amendment: `repo-context-scale-20261004-50k-hot`.

Tasks in this context area: `102-05`.

Purpose: repeat a fresh L=131072 selected-GPU Turbo4 load with the full H=51200
(200-page) hot window filled, capture
prefill speed as occupied context grows, measure VRAM headroom, then measure the
three canonical MTP prompts while preserving the loaded history. This is a
procedural live benchmark on one freshly built candidate, not a source repair.

The first work step is the forced-clean configure/build in task 102-05, using
the stable output path `build-102-05` (`cmake --fresh`, then
`cmake --build ... --clean-first`). Do not use the stale loaded 102-03 server
or an older build. Verify candidate hash and source HEAD before managed reload.
Then follow the exact effective argv in task 102-05, including explicit
`--no-context-shift`; verify the flag and all geometry from the live process
argv. Any change to candidate, model, L/H, B/U, codec, pager or MTP
configuration requires a clean stop/restart and identity verification before
more requests. Never run two Qwen processes. Run one short 80-token prompt-1
native-MTP startup smoke before the expensive fill; it is diagnostic only, with
no acceptance threshold. A crash or `unsupported_shape` is a real issue to
diagnose, not a setup result to relabel or bypass by changing routing/split-KV.
Preserve/resume the same occupancy checkpoint on valid progress; do not
replay/refill committed history. Host residency telemetry is optional and not
a gate. Keep the successful candidate loaded for 102-06.

Read only the task packet, this cluster, forward/TESTING, and handoff 102-04.
Do not load the archived prior 256K receipt/handoff or raw task logs.
