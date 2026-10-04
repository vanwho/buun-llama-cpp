# Selector readback and pager lifecycle investigation

Revision: `hotpath-v10-20260914`. Source inspection: 2026-10-04.

## What is established, and what is not

103-02 repeatedly rebuilt `build-102-06`, filled roughly 249,500 tokens with
B/U=256/64, and queried an anchor on logical page 500. The page was cold,
host-backed and summary-ready. The query returned slash filler; no selector
readback or promotion was recorded. These were diagnostics, not comparable
B/U=1024/256 benchmarks. No testing or compilation was performed by the outer
code investigation that produced this note.

`attempt-03-sync-fence/candidate-server.log` shows the final-user graph fenced
and `apply_pager_live_policy` entered in provisional phase with refresh=1 and
16 outputs. Therefore the claim that a missing scheduler fence alone explains
the failure is not supported. Generation later overwrites the diagnostic graph
gate; its final `invalid_query_or_ubatch` value is not the final-user failure.

The earlier successful 101-18 natural-promotion proof used H=4096 (16 pages),
not H=51200 (200 pages). Successful short proof does not cover scaled selector
transport. The defects below predate that proof; do not blame commit 477999 or
103-01 without an actual causal comparison. 103-01's position-aware route fix
must be retained: it repaired independently observed nonfinite target logits.

## Concrete source defects and implemented remedies

1. **Readback width ceiling.** `build_kv_page_select` emits admitted resident
   attention capacity plus up to five cold IDs *per attention layer*.
   `llama_kv_pager::prefetch_candidate_mailbox_` was two slots of 128 records.
   The synchronous path rejected outputs above 128; the async path broke before
   submitting a segment larger than slot capacity. At 200 resident IDs this
   disables the producer-to-policy bridge without changing the CUDA selector.
   Completion also used a 128-ID stack copy. Startup now sizes the two host
   slots from admitted resident width, logical capacity, shared cold width and
   layer count, with checked arithmetic. Completion/fallback copy the actual
   bounded count. No additional target/draft GPU KV pool is allocated.

2. **Owner replacement during scratch reserve.** `sched_reserve()` calls
   `init_kv_pager()` on sampler/graph changes, not just construction. The latter
   unconditionally created a new owner and rebound memory. The direct log shows
   the same target context changing pager pointer during the first base request
   (old `0x5584d843fc10`, new `0x5584e0de6f00`), then operating in idle phase.
   Replacing the owner loses its catalogue, turn state and selection while the
   KV rows survive. `init_kv_pager()` now returns if an owner already exists.
   New L/H needs a fresh context, not a scratch-reserve lifecycle reset.

3. **Readback/commit race.** Poll-only async completion can leave the current
   query's selector pending when the server commits/freezes history. A later
   graph advances query generation and cannot safely consume that old result.
   At the existing provisional final-query policy fence, queue all compact
   layer results, finish the readback backend once, poll/consume them, then
   release completed descriptors before history commit. Do not add waits to
   ordinary page/token/generation boundaries. The synchronous fallback is only
   used when async transport is unavailable, avoiding duplicate reads.

4. **Logical ID versus compact inventory ordinal.** CUDA catalogue inputs are
   addressed by `record.id.logical_page`; captured descriptor pages previously
   used `push_back` order. A logical gap would decode an ID through the wrong
   page. Capture now uses a logical-capacity-indexed immutable descriptor;
   unused entries still fail full identity/content-version authentication.

The worker's sealed partial-tail identity repair and empty trailing-subblock
range normalization are valid, separate fixes and are retained. The former
bug (`state == filling_gpu` as a tail test) also existed at source a0086f.
Keep accumulator lifetime and generation-ring repairs from c0102f688 and the
query-boundary work from 228387e01; no blanket route rollback is justified.

## A definite test defect

`server_slot::release()` resets `query_replay_count`,
`pager_frozen_history_generation` and `pager_history_frozen` at completion.
The attempt-3 `run-changed-history-rich-query.py` checks those fields in its
post-response slot snapshot. A zero there is normal cleanup, not proof that
replay/freeze failed. Capture active-request samples or the existing
`kv pager generated span` request log. Unchanged history correctly needs zero
replays. Do not require arbitrary exact formatting of an answer or read a
later generation gate as if it were the selector's provisional-query gate.

## Next validation, not another giant diagnostic loop

Task 103-02a owns compilation and targeted testing. First test 205 IDs x 16
layers with a deterministic transport fixture (including late-layer cold IDs,
logical gaps, pending-event completion and stale identities). Then one small
natural-promotion/replay request with 400 output tokens. Only after that, one
200-page transport row with C just above H; no 249K refill for a descriptor
bug. Keep source/binary/DSOs/model/argv bound to each row. Distinguish transport,
ranking, admission, transfer, target use, semantic output and MTP acceptance.
Exact long-context semantic reliability remains unproven until later evidence.

All current changes are uncompiled/unverified by this investigation. Detailed
progress instrumentation is opt-in and must be off for throughput measurements.
