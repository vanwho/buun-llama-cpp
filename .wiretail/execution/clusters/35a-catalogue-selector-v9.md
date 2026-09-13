# Cluster 35a-catalogue-selector-v9

Revision: hotpath-v9-20260913. Tasks: `35-01`, `35-02`.

## Purpose

- 35-01: Build generation-safe GPU summaries for all logical Turbo4 K pages.
- 35-02: Implement cooperative current-Q page selection as an independent graph op.

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
