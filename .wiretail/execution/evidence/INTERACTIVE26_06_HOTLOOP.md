# INTERACTIVE26_06_HOTLOOP

## Outcome

The highest avoidable maintenance owner identified by the permitted historical
attribution was the pager lifecycle: 2,343 seal calls and 4,672 summary-build
calls in a 2,048-token native-MTP run. The source repair makes maintenance
content-versioned and queue-driven. Historical pages are no longer visited on
every synchronization; allocation failure falls back to a complete scan.

The focused fixture measured 2 queued pages for the initial two-page content
wave, 0 pages for an unchanged follow-up, and 1 page for a one-page tail wave.
Its summary provider made 32 initial calls and 16 more for the changed tail.
These are deterministic fixture counts, not a whole-model throughput claim.

## Critical-path attribution

| Owner | Before attribution | After / decision |
| --- | --- | --- |
| pager seal/dirty maintenance | 2,343 seal calls; historical page walk | queued candidates: 2, 0, 1 in the focused wave sequence |
| summary construction | 4,672 builds; historical 2,048-token run | only changed-page summaries; 32 initial, +16 tail fixture calls |
| host/event and scheduler spans | 22 D2H calls / 95,160,000 B; 1.498 s wait and 2.508 s queue | no D2H or queue claim from the unavailable fresh native-MTP run; spans overlap and are not additive |

The historical timing is retained only for attribution. A fresh post-change
native-MTP profile was not invented: port 8080 remains PID 1238402 with
`--spec-type none`, and `sudo -n -v` reports interactive authentication.
Port 8092 was not touched.

## Changed files

- `src/llama-kv-pager.cpp`
- `src/llama-kv-pager.h`
- `tests/test-kv-pager.cpp`

The pager queues each page once per maintenance wave, requeues mutable tails
and asynchronous host completions, and rebuilds queue state after residency
mutations. The regression asserts no scan on unchanged synchronization and a
single scan for a changed tail. Speculative rollback and all page mutation
paths remain covered by the existing lifecycle test.

## Verification

- CPU focused CTest (`pager`, `routing-summary`, `routing-retrieval`,
  `attention-telemetry`, `attention-execution`): 5/5 pass.
- CUDA focused CTest: 5/5 pass; CUDA device detected as RTX 4080.
- `test-kv-pager-model`: pass on CPU and CUDA builds.
- `git diff --check`: pass.
- Candidate CUDA server built in `build-cuda`; it was not loaded over the
  existing service because passwordless lifecycle authorization is unavailable.

## Raw artifacts and next command

Machine-readable evidence is in `INTERACTIVE26_06_HOTLOOP.json`. The allowed
historical attribution source is `SPEED25_13_WHOLE_MODEL.json`.

Next command after authorization: replace only the authorized Qwen 8080
service with the candidate native-MTP bundle, then run the V7 primary case at
L=8192/H=4096 with 128 committed output tokens and capture the fresh profiler
and unprofiled pair. Keep the successful candidate loaded afterward.

## Deferred verification

Native-MTP handoff/acceptance, natural cold recall, fresh whole-model timing,
CUDA graph capture/launch attribution, target/MTP kernel attribution, and
actual post-change transfer counters remain deferred solely because the
required candidate service lifecycle needs passwordless `sudo -n`.
