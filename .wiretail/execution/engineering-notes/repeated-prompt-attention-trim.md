# Repeated long prompts: hybrid checkpoint attention suffix leak

## Root cause (2026-10-11)

The 72K occupied prompt-1 benchmark completed its warmup and two measured
requests, then failed partway through the third with `KV append is waiting
for a canonical page copy`. This was not a new CUDA allocation failure or
evidence that cold-page selection should run on every generated token.

`llama_memory_hybrid::seq_pos_max()` in `src/llama-memory-hybrid.cpp` returns
the minimum of the attention and recurrent frontiers. A PARTIAL_ONLY restore
installed the recurrent checkpoint at position 29, while attention still held
the preceding request's rows through position 72484. The server's suffix trim
at position 30 consulted that minimum, decided there was no suffix, and skipped
both target and draft attention removal. Replay then allocated duplicate
logical cells. Three roughly 72K prefills consumed roughly 216K cells; the
following prefill exhausted the 262144-cell logical table after 45056 rows.

`llama_kv_cache::record_slot_failure()` incorrectly relabeled logical-cell
exhaustion as `pending_copy` if any unrelated resident page was sealing.
The journal reports attention ubatch planning failure, not a transfer-lease
failure. The original request artifacts are local under
`/srv/ai/paged-kv/results/forward/long-query-no-victim-20261011/attempt-02/`.
The two completed measured requests are partial evidence, not a three-run
median: 43.16/41.60 decode tok/s, 508.33/508.11 prefill tok/s, 89.16% MTP.

## Production corrections

Source commits: `c6d71b44d` and `f5478ee12`.

- `tools/server/server-prompt-trim.h` and `server-context.cpp`: after a
  checkpoint restore, always execute the attention-only suffix removal for
  target and draft. The hybrid minimum is not an attention-frontier bound.
  Preserve paired recovery if either trim fails; do not reset valid recurrent
  state or silently continue with leaked attention cells.
- `src/llama-kv-pager.cpp`, `mutate()`: drain/wait for affected captures before
  copying page state into the mutation transaction. The old pre-wait snapshot
  could resurrect `host_inflight` and the capture pin after completion retired
  them. Apply the same ordering to remove, rewind, clear, keep and relevant
  copy/shift operations. Wait only at the mutation boundary when an affected
  capture is actually pending, not once per token or for unrelated captures.
  `f5478ee12` explicitly retains unrelated capture leases when a transient
  MTP rollback releases turn pins; their completion remains asynchronous.
- `clear_turn_state()`: a sealing current page owns a capture pin, not a write
  pin. Do not decrement or relabel it before capture completion.
- `src/llama-kv-cache.cpp`: logical `find_slot()` exhaustion remains
  `logical_capacity`. Explicit unresolved transfers alone report the new
  pager `pending_copy` status. Generic transaction failure is not proof of
  a transfer wait. Check destination removal before inserting restored cells.

These corrections do not change L/H, batch/microbatch, Turbo4 codecs, MTP
n-max, ranking policy, scratch sizes or graph routes. Retrieval selection and
publication remain at the request's provisional-query/replay boundary. Frozen
retrieved history stays protected during generation; the separately repaired
eviction path may recycle authenticated, unprotected completed query/output
pages when the bounded physical pool requires space.

## Focused verification

Incremental CUDA build used `--parallel 16`. `test-server-prompt-cache`,
`test-kv-pager` and `test-kv-policy` passed. New coverage exercises the hybrid
minimum-frontier trim and pending asynchronous capture followed by partial
remove/reappend, rewind, clear and keep, plus unrelated suffix rollback and
turn-end capture retirement. The initial added CUDA fixture used
a fake buffer handle with a real transfer adapter; that fixture bug was fixed
to allocate a real tiny CUDA buffer before the final passing test run.

Build/test logs and subsequent live verification are local-only under
`/srv/ai/paged-kv/results/forward/long-query-no-victim-20261011/repair-02/`.
Do not ingest the raw request/transcript files at task startup. Use this note
and the compact result summary; inspect bounded error excerpts only if needed.

The decisive live check is the unchanged saved 72K low-reasoning prompt-1
warmup (40 output tokens) plus three measured 400-token requests on one
restarted repaired process. All four must finish without logical-cell
exhaustion, transfer/pin refusal or SSE errors. Record real timing/MTP values;
poor values are findings, not a reason to alter the configuration or rerun
completed rows. The full scored matrix remains owned by task 105-15 onward.

### Repaired managed-service result

Build 14516, embedded commit `f5478ee12`, was loaded by a single managed
Qwen process (PID 2601912) without changing the qwen38-aco profile:
L=262144/H=65536, P=256, B=1024/U=256, Turbo4 target/draft KV,
GPU draft-MTP n-max=2, CUDA0, no context shift. Saved request bytes from the
failed attempt were replayed unchanged, with low reasoning, temperature 0,
seed 42, streaming, one 40-token warmup and three 400-token measured runs.

| Run | Evaluated prompt tokens | Prefill tok/s | Decode tok/s | Accepted/drafted | MTP % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warmup | 72086 | 728.77 | 39.66 | 24/28 | 85.71 |
| Measured 1 | 72056 | 760.06 | 40.39 | 248/301 | 82.39 |
| Measured 2 | 72056 | 758.73 | 40.52 | 248/301 | 82.39 |
| Measured 3 | 72056 | 758.71 | 40.18 | 247/304 | 81.25 |
| Measured median | — | 758.73 | 40.39 | — | 82.39 |

All four completed with `finish_reason=length` and no SSE error. Measured
requests each reused only 30 cached tokens, so this is not an almost-empty
cached-prefill measurement. The sequence evaluated 288254 prompt tokens in
total across requests, exceeding the 262144-cell capacity without leaking
the preceding requests' attention rows. Each measured request logged the
paired attention trim at p0=30; bounded journal checks found no trim rejection,
mandatory recovery reset or runtime failure. Source/loaded-DSO identities
remained unchanged throughout the test. Sampled GPU usage peaked at 15549 MiB.
The managed process remained idle and healthy afterward; 8080/8091 passed.
Port 8092 was not touched.

Repeated-run prefill is higher than the previous partial run's 508 tok/s;
decode and MTP values are reported as measured, not claimed improvements.
The prior leaking candidate was not rerun as a valid performance control.
This verifies the identified repeated-request failure, not every possible
future selection/promotion/cancellation scenario or the remaining full matrix.

Compact result:
`/srv/ai/paged-kv/results/forward/long-query-no-victim-20261011/repair-02/live/summary.json`.
SHA-256: `a809f5bca4914d8a1b7e7c0c21b4828b087615ac9327d294f4a31582c2710642`.
Loaded server SHA-256:
`18959b1444c6890ea8fa4bfd3db95375140807f522725717749a2f4b6ba8e822`.
Loaded libllama SHA-256:
`049c798f0c8ef653085a64c2607d4ff061ce9b237f1a5088bd0838589dbc6cee`.
The local replay script, saved requests/SSE, GPU samples and full identities
remain external raw artifacts, not Git content or startup context.
