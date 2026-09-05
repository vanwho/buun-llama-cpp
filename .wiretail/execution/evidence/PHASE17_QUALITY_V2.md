# Phase 17 quality V2 — task 17-15

Decision: **blocked for acceptance**, with complete raw evidence.

The resolved V4 ceiling was 22,016 tokens. The bounded startup gate passed for
Turbo4 target KV plus native Turbo4/GPU draft KV: 90.578 seconds of stable
health, process identity, and systemd state across 94 samples, with zero
restarts. The gate and all raw records are at
`/srv/ai/paged-kv/results/17-15-quality-20260905T052900Z/`.

| Mode | Configuration | Records | Result |
| --- | --- | ---: | --- |
| dense | pager off, MTP off | 24 | 22 pass; 2 HTTP 400 context-overflow failures |
| selected-all | selective pager, 86 hot pages, native Turbo4/GPU MTP | 24 | 18 HTTP-200 answer mismatches, 4 timeouts, 2 HTTP 400 overflows |
| exact | exact pager, native Turbo4/GPU MTP | 24 | 22 HTTP 501 supported exact-page-wave refusals, 2 HTTP 400 overflows |
| selective | selective pager, automatic hot pages, native Turbo4/GPU MTP | 24 | 18 HTTP-200 answer mismatches, 4 timeouts, 2 HTTP 400 overflows |

The dense run embedded the expected fact in 22/24 responses. Neither paged
mode embedded an expected answer in its responses, so no selected-mode quality
score is accepted. No checker, threshold, missing case, or observed record was
rewritten. The exact 501 response is the repaired supported refusal from the
prior task, not a successful quality result.

The requested parity, routing, attention, MTP acceptance, resident/host-page,
and transfer measurements are not claimed. `/metrics` returned HTTP 500 with
`json.exception.type_error.302`; the resource timeline consequently contains
an empty pager object. Startup command lines and journal entries do confirm
Turbo4 target KV and native Turbo4/GPU draft placement. The concurrent oracle
remains blocked by the service's single independent slot.

Raw records and per-mode provenance/checksums:

- `quality/dense/records.jsonl`, `quality/selected-all/records.jsonl`,
  `quality/exact/records.jsonl`, and `quality/selective/records.jsonl`
- `quality/*/provenance.json`, `summary.json`, `command.txt`, and `SHA256SUMS`
- `raw/startup.json`, `raw/startup.journal.txt`, `raw/startup.systemd.txt`,
  `raw/startup.kernel.txt`, lifecycle records, and `raw/SHA256SUMS`

The previous native startup SIGSEGV did not reproduce under this gate. A
debugger/core backtrace was unavailable because `coredumpctl` is not installed
and the prior service had core collection disabled. This is recorded as a
local deferred diagnostic, not as a claim that the old crash was repaired.
