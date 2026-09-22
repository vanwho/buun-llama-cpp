# repair93 — fast MTP diagnosis under attention-aware paging

Revision: `hotpath-v10-20260914`. Amendment: `repair93-mtp-fast-20260922`.

Phase 93 is a short diagnostic phase, not a 256K occupancy or throughput
campaign. Its purpose is to determine whether the low native-MTP acceptance is
caused by draft/target state divergence, page-table/route divergence, rollback
errors, Turbo4 dequantization/logit mismatch, or merely a bad benchmark setup.

Every live run must use a small bounded prompt and generation budget so that a
diagnostic completes in seconds or a few minutes. Do not run the old full-L
frontier, long context ramp, 20K+ prefill, or repeated 30-second polling loop
in this phase. Use the same Qwen3.8-27B UD-IQ4_XS model and one immutable CUDA
binary for each comparison, with target and draft K/V explicitly Turbo4 and
draft K/V explicitly GPU-resident.

The canonical benchmark lifecycle owns the service. For every rung, invoke the
configured `CANONICAL_BENCHMARK_RUNNER` through the existing pager benchmark
adapter and pass `BENCH_SERVER_BIN` for the candidate executable. A process
already listening on port 8080 is not evidence that the requested rung is
loaded: reuse it only after verifying its PID/start time, `/proc/<pid>/exe`
realpath and SHA-256, model realpath/SHA-256, and all relevant command-line
flags. Otherwise reload/activate the requested profile before sending a
request. Do not create an ad-hoc endpoint or silently test the old selective
service. A separate diagnostic process is allowed only when the canonical
lifecycle cannot express the rung and it emits the same complete identity and
restoration manifest; it is never a reason to score an unverified endpoint.
The scored harness must receive a non-empty `BENCH_SERVER_BIN` (or equivalent
explicit binary argument); a healthy endpoint without candidate identity is a
negative setup check, not a usable fallback.

The RTX 4080 has room for only one Qwen3.8-27B process. Before each rung,
construct the expected identity from the requested binary, model, and command
line. If the currently loaded process matches that identity and the prior
handoff/lifecycle manifest has `continue_loaded=true`, reuse it as-is. In every
other case, load the expected binary through the managed lifecycle, wait for
the old PID and CUDA allocation to disappear, and verify the new PID/executable
/model before sending requests. Never leave one 27B process running while
launching another on a second port.

The required comparison ladder is:

1. MTP-off dense/all-GPU target control.
2. MTP-on dense/all-GPU target plus GPU Turbo4 draft control.
3. MTP-on selected/paged with every test page resident (no cold promotion).
4. Only after 1–3 are recorded, one bounded cold-page promotion/rollback
   probe with a deliberately tiny hot set.

The harness must capture request/response SSE, per-step draft and accepted
counts, target/draft positions, rollback and rewind events, page-table epoch,
route identity, and MTP placement/types. A run is invalid if it reports only
aggregate counters without the request-level evidence. The first failing rung
localizes the defect and determines the code inspection scope.

`selected_reference` is diagnostic-only and cannot be used as a production
success or performance result. If an automatic run selects it, stop, record
the route-policy failure, and repair the dispatch/configuration before doing
the comparison again.

Phase output must state whether the defect is in MTP itself, paged attention,
or the harness/configuration, identify exact source functions and state fields,
and either implement the smallest justified repair or create a directly
actionable successor task with file/function/test pointers. Do not proceed to
large-context or speed acceptance until the short MTP control reaches the
expected high acceptance behavior and the paged resident case matches it.
