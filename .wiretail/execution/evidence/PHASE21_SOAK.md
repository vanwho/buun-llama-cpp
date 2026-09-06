# Phase 21 bounded paging pressure and lifecycle — task 21-09

## Result

The segment is **incomplete and resumable**. The immutable 20-07 bundle,
Turbo4 target, native GPU Turbo4 MTP, and four-page hot budget started
successfully. All three semantic sentinels passed. One 6,393-token warm-focus
request completed in 348.29 seconds, then the next equal-work request caused a
checkpoint-save assertion and systemd restarted the service.

Machine-readable evidence is [`PHASE21_SOAK.json`](PHASE21_SOAK.json). Raw
artifacts are under:

`/srv/ai/paged-kv/results/21-09-soak-20260906T0550Z/`

## Observed pressure

The requested diagnostic context was 6,401 tokens; the engine rounded its
working context to 6,656 tokens, or 26 logical 256-token pages. The physical
hot capacity was four pages. Across 90 resource samples, selected/resident
pages rose from 1 to 4 while host backing rose to 3 pages / 768 valid rows.
The target valid rows rose to 780. H2D/D2H useful and aligned bytes, transfer
submissions/completions, faults, evictions, and prefetch hits all remained
zero in the exported counters. That is an observed zero-movement result, not
proof that paging is correct or unnecessary.

The completed warm-focus response was exactly `Focus marker stable-a.` with
four draft and four accepted tokens. Its prompt phase took 344.41 seconds;
the six-token decode took 3.79 seconds.

## Failure and lifecycle boundary

After the second equal-work request disconnected, the service journal recorded
an assertion in `ggml_backend_tensor_get` during
`llama_io_write_host::~llama_io_write_host` / `state_seq_get_data` /
`server_slot::prompt_save`. The process exited with SIGABRT/status 6. The
harness then could not construct the cancellation payload because the prompt
endpoint was unavailable, so cancellation, drain, page promotion correlation,
checkpoint restore, slot reuse, and focus/churn segments were not measured.

systemd automatically restarted, and the exact 262,144-token selective,
four-hot-page, native GPU Turbo4-MTP profile was explicitly restored and left
healthy on port 8080. Port 8092 was not touched.

## Deferred verification

- Repair or isolate the checkpoint-save out-of-bounds assertion.
- Resume the retained case ledger with the same 6,401-token/four-page
  diagnostic identity.
- Complete cold-page promotion/checksum/fence correlation, focus shifts,
  sustained churn, cancellation/drain, checkpoint restore, slot reuse, and
  post-soak restart.
- Capture after-state resources and keep movement counters separate from
  paging correctness claims.
