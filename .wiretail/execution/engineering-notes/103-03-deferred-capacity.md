# Deferred exact-capacity boundary and source disposition (103-03)

## Why stop this campaign

103-03 pursued exactly committed C=L=262144, while useful ranking/MTP tests
need space for generation, replay and speculative work. It completed 22
requests up to C254393 with H51200 before a real planning failure. Repeating
token padding/reserve adjustments does not diagnose its memory owner and
does not advance cold-page ranking. Phase104 proceeds independently using
small L8192/H4096 recall and one L131072/H51200 row around C59392.

## Observed failure versus inference

Raw root `/srv/ai/paged-kv/results/forward/103-03/attempt-01/`:

- `capacity-final/incremental-state.json`: 23 records (22 committed plus
  one failure), next_request_index23, next_turn_index22. It contains large
  conversation bodies: extract needed fields, never load the complete file.
- `capacity-final/occupied-frontier.json`: last committed C254393; allocated
  target bytes approximately865075000 as rendered by metrics; H200.
- `capacity-final/request-000022.json`: rendered prompt259020, max_tokens400;
  `raw-000022.sse`: error400/context exceeded, zero generated output.
- `request-22-server-window.log`: hybrid `init_batch` attention planning
  failed; server repeatedly reduced batches and could not find memory slots.

The error message alone does not prove total logical capacity was exhausted
by that rendered input; prompt259020+400 is below L. Nor does it prove VRAM
OOM or selector failure. Cache/replay positions, target/draft cell reservations,
and planning rollback need source-level isolation before assigning a cause.
Final snapshot C may not be a safe resumable frontier after a partial failing
decode; preserve last committed and observed state separately.

## Code to keep, and where

Archive commit `7c162ca08` on `codex/task-103-03` keeps:

- `tools/server/bench/run-occupancy-frontier.py`: exact token-padding fitter,
  exact finalA2/ignore-EOS mode, separate capacity accounting, reserve changes,
  and same-commit source-fingerprint rebind.
- `tools/server/bench/validate-occupied-frontier.py`: opt-in exactC=L checks.
- `tools/server/bench/test_occupancy_frontier.py`: exact planner/accounting test.

No production llama/ggml/server source changed. Preserve these as unfinished
research only; do not merge them into main/phase104. Reasons:

1. Rendered prompt+completion need not equal durable KV positions; generated
   final token, templates, rewind and speculative reservations differ.
2. Replacing replay/safety reserves by only400 output to reach C=L is not a
   demonstrated valid native-MTP runtime contract.
3. Same-commit dirty-fingerprint changes are accepted without independently
   checking each selected source range at that rebind; retained source identity
   therefore needs stricter validation before reuse.
4. Exact-fit punctuation searches repeatedly tokenize near-full histories;
   they are a poor default for fast iteration and semantic retrieval.

Useful existing stable ordinary occupancy helpers remain on the plan branch.
Bulky artifacts stay local; neither raw evidence nor binaries are committed.

## Future investigation only if justified

After the ranking verdict, a separately scheduled task can reproduce the
boundary with small deterministic geometry instead of refilling256K:

- `src/llama-memory-hybrid.cpp::init_batch`: locate the failed attention-plan
  return separately from recurrent-state/draft planning.
- `src/llama-kv-cache.cpp::plan_slots` and pager prompt-plan reservations:
  inspect committed/temporary positions, retained query and generation pages,
  full-span versus rolling-hot planning, and rollback after a rejected batch.
- `tools/server/server-context.cpp`: follow request prompt reuse/query replay,
  target/draft reservations and the `server_memory_failure_is_logical_capacity`
  error-classification path around `failed to find free space`.

First determine whether safely reserved real use is affected. Do not prescribe
new ranker behavior, lower H/B/U, relax identity, or bypass planner invariants
as a cure without that evidence. Exact terminal occupancy remains unproven;
deferral is not a successful capacity claim.
