# Cluster 05b — server lifecycle and speculative rollback

Tasks: `05-03`, `05-04`.

Purpose: make asynchronous pages safe across real request slots, prompt reuse, checkpoints, and draft
accept/reject transitions.

Carry forward:

- initial pager authority is single-slot/single-sequence unless later evidence explicitly expands it;
- slot reuse mints a new session/sequence generation before old completions can publish;
- cancellation drains/fences transfers without leaking pins, slots, rings, or catalog charge;
- immutable prompt artifacts may seed host backing only after complete identity validation;
- target accepted frontier, recurrent companions, MTP carry/cache, and page write metadata advance or
  roll back atomically;
- speculative rejection must not evict unrelated sealed target pages.

Read: plan sections 5.2–5.4, 8.9, 12.3, 17; cluster 05a and scheduler handoffs; server slot/cache-yield,
prompt artifact, checkpoint, speculative restore, and generation sources.

Exit artifacts: server lifecycle and checkpoint/speculative tests covering stale completions, slot reuse,
clear/rewind/copy/remove, cancellation, and teardown.
