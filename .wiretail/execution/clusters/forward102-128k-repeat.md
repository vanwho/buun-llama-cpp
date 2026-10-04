# Cluster forward102-128k-repeat

Tasks in this context area: `102-05`.

Purpose: repeat a fresh L=131072 selected-GPU Turbo4 load with the full H=51200
(200-page) hot window filled, capture
prefill speed as occupied context grows, measure VRAM headroom, then measure the
three canonical MTP prompts while preserving the loaded history. This is a
procedural live benchmark on one freshly built candidate, not a source repair.

Use the fixed build and exact effective server argv in task 102-05. Any change
to candidate, model, L/H, B/U, codec, pager or MTP configuration requires a
clean stop/restart and identity verification before more requests. Never run
two Qwen processes. Preserve/resume the same occupancy checkpoint on valid
progress; do not replay/refill committed history. Host residency telemetry is
optional and not a gate. Keep the successful candidate loaded for 102-06.

Read only the task packet, this cluster, forward/TESTING, and handoff 102-04.
Do not load the archived prior 256K receipt/handoff or raw task logs.
