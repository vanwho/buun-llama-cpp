# SPEED26_01_FULL_CONTEXT

Result: `not_complete` (`incomplete_clean_build_live_repair_still_fails`)

The 262144 allocation and native GPU Turbo4 MTP readiness smoke passed. The
readiness case used L=262144, H=73216 rows, A=2304, B=256, UBATCH=128, 128
committed output tokens, and 80/94 accepted MTP tokens. Canonical host
publication was 2048 rows / 34.6 MB with 3136 seal calls.

The automatic-H full request reached 12288 processed tokens before a 737 MiB
dequant scratch reserve failed. The H=8 changed retry reached 188416 processed
tokens (90% of the exact 210031-token admitted prompt) before the same scratch
reserve grew to 737 MiB and the server returned HTTP 500. Neither produced
generation, so neither is a completed full-context proof.

Source owner: `llama_kv_cache::prepare_with_slots()` in
`src/llama-kv-cache.cpp`. The repair caps VMM selective-attention scratch to
resident mapped cells plus the current batch, avoiding charging cold logical
history as materialized f16 attention data. Focused pager, telemetry, and
resume tests passed. A complete CUDA build (100%) loaded cleanly and passed
isolated readiness. Its automatic-H full request failed at 13312 processed
tokens on a 26.5 + 26.5 MiB scratch reserve; a clean H=8 retry reached 157952
processed tokens before HTTP 500 context exhaustion. Both generated zero
tokens. The H=8 raw root is `/srv/ai/paged-kv/results/26-01-retry2-full-q0-h8`;
its case-state log retains the timed-out and resumed attempts.
A final H=1 run failed immediately with `KV pager batch write reservation
failed: all_pinned`, confirming that this residency is below the writable
batch minimum and cannot satisfy the task.
The final B=64/UBATCH=64 H=8 run using the rebuilt async host-seal repair
reached the full 210031-token admitted prompt, with 820 changed pages,
216481 seal calls, and 3.55 GB host D2H publication, but ended with HTTP 500
context exhaustion and zero generated tokens. Its raw root is
`/srv/ai/paged-kv/results/26-01-retry3-async-full-q0-h8-b64`.
The H=16/B=64 async run progressed steadily to 143360 tokens without an
observed runtime error before being manually stopped to restore production;
it is retained as incomplete, not as completion evidence.

Recovery assessment found that the V6 driver's fixed neutral corpus rendered
only 210031 tokens for its requested 261504-token prompt. That under-filled
case is not a near-full 262144 proof; `_fit_prompt()` now grows the neutral
corpus from the rendered-token result, with a focused regression test, before
retrying the remaining pager frontier fault at the intended occupancy.

Retry 4 fixed the benchmark boundary: the fitted `messages` are now sent to
the chat endpoint, so the server applies the template exactly once. The live
request recorded `append=261504` before the H=8/B=1024 diagnostic stalled. A
budget retry accidentally supplied a row count as `--kv-hot-pages` (pages),
causing an oversized allocation and OOM before generation. Both raw roots are
retained outside the repository:
`/srv/ai/paged-kv/results/26-01-retry4-full-q0-fitted` and
`/srv/ai/paged-kv/results/26-01-retry4-full-q0-fitted-auto`.

Raw roots and complete details are in `SPEED26_01_FULL_CONTEXT.json`.
