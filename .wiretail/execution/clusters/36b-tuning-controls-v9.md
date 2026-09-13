# Cluster 36b-tuning-controls-v9

Revision: hotpath-v9-20260913. Tasks: `36-03`, `36-04`.

## Purpose

- 36-03: Tune B/U/A and safe H with a small adaptive comparison.
- 36-04: Measure the small three-prompt hot-offload speed against real controls.

## Context boundary

Start a fresh session for this V9 cluster; never resume an old V8/phase-31
thread. Reuse only relevant work from tasks in this cluster. Read the current
packet's explicit context_files, repository instructions and named source.
Do not load historical plans, whole WORK_STATE/WORK_LOG, all predecessor
handoffs, archived evidence or recursive links. Old design facts needed for
this work have been folded into V9. Scheduling dependencies are not reading
instructions. Current packet and V9 contracts override old comments/diaries.

All tasks: Luna Medium initially, opt-in Luna High first retry; later separate
assessments follow the shared runner policy. Hardware is available; candidate
replacement is authorized with sudo -n. Keep a successful current candidate
loaded, but replace it sequentially when testing a new binary or control.
