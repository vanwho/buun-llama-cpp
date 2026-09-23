# repair93 — short Turbo4 MTP and pager correctness

Revision: `hotpath-v10-20260914`. Amendment: `repair93-mtp-fast-20260922`.

Phase 93 resumes from the measured 93-04 failure. Dense GPU Turbo4 MTP works
on its bounded control. The Q=1 selected direct-attention stall and layer-page
stride issue have deterministic CUDA coverage, and the latest cold A→B→A run
observed the cold page's H2D completion, table publication, and target use.
Cold promotion is **not yet accepted**: the run lacks request-correlated
transfer event ordering and page-specific draft consumption proof. The next
steps are one bounded hot-resident speed geometry check, then resume that
exact proof gap before the paired placement screen.

## Execution model lock

Every remaining phase-93 task uses the exact implementation model
`gpt-6-luna` at High reasoning. Every separate recovery assessment for these
tasks also uses `gpt-6-luna` at High reasoning; task metadata pins both the
implementation and assessment model, so Wiretail's generic family retry
escalation does not apply. Use only `gpt-6-luna` for this project; do not use
another model family. Any successor tasks created by 93-09 or later reviews
must copy these model fields into `WORK_STATE.json` and retain this policy in
their packet/cluster context. The shared Wiretail default is also
`gpt-6-luna`.

## No-skip live-test policy

The missing `CANONICAL_BENCHMARK_RUNNER` in 93-08 was a correctable invocation
omission, not an unavailable-hardware result. The canonical script is
`/srv/ai/benchmarks/run-profile-benchmark.sh`. For live tasks, explicitly set
that environment variable together with the candidate binary, endpoint,
active-profile file and API-key file listed in the task packet. A mismatched
healthy service is expected to be replaced through that managed lifecycle;
it is never valid test input and is not a reason to end the task.

Configuration, permissions, service identity and safe test geometry must be
repaired in the owning task and the bounded run repeated. Do not mark a task
done or proceed to its dependent task when no candidate-bound request was sent
or an expected benchmark remained `not_measured`. A runtime defect may be
handed to its explicit repair task only after a request against the verified
candidate reproduces it with hashed raw evidence. Completion checks enforce
this for 93-10; local deterministic tests alone cannot close live MTP or
promotion gates.

## Goals and bounds

- Keep the target and MTP draft on CUDA with Turbo4 K and V for every MTP-on
  control. A route that silently moves draft state to CPU, changes KV type, or
  uses `selected_reference` is a setup failure.
- Every speed benchmark keeps native MTP enabled with GPU Turbo4 draft K/V.
  Use the exact original three prompts, one 40-token warmup and three measured
  400-token generations. Batch geometry 1024/256 is primary; 512/128 is the
  paired lower-scratch comparison. MTP-off is allowed only for explicitly
  named functional controls, never for a speed row. Set reasoning mode off
  for both warmups and measured requests. Per-prompt MTP acceptance is a hard
  gate, evaluated as the median of that prompt's three measured requests for
  each B/U geometry and placement: prompt 1 >=75%, prompt 2 >=40%, prompt 3
  >=70%. Do not average prompts or configurations together. These floors are
  below the supplied reasoning-off reference values (94.12%, 57.04%, 82.69%).
- Do not reserve more than 48 Ki tokens of target hot KV in VRAM in any phase
  93 test (`48 * 1024 = 49152` tokens, or 192 pages at 256 tokens/page). This
  is a test ceiling, not a production hot-set constant. Request only the
  smaller hot-page count needed for each test; the allocator's byte budget
  and `--kv-safety-headroom auto` remain authoritative and may admit less.
- Keep diagnostic context small: use 4096 for trim/MTP iteration and at most
  8192 for the cold-promotion sequences. Task 93-11f starts at 8192 total
  context with 4096 target hot tokens. Task 93-11g configures the actual
  llama-server at `-c 8192` and its ordinary automatic target-KV hot budget at
  4096 tokens, then appends exact 1K-token file fixtures as normal same-slot
  user context. Keep the A→B→A sequence below the full 8K context so server
  context shifting cannot discard A; natural pager pressure above the 4K hot
  budget must evict A, and the later ordinary A query must cause promotion.
  Do not manually force eviction, page selection, or promotion. Any later
  phase-93 speed row may use a matched context only when safe, never above
  49,152 tokens (48 Ki tokens); no phase-93 task may exceed that ceiling.
- No request used to measure prefill/input speed may contain more than 16384
  rendered input tokens, including the retained prompt prefix. Preflight the
  exact rendered prompt with the canonical tokenizer before sending it; reject
  over-limit input instead of truncating or reporting it as a valid speed row.
  Prefer 4096/8192-token measurements while iterating; use 16384 only for a
  final bounded point after the smaller setup is stable.
- Keep generation at 16 tokens or less and `draft_n_max=2` during functional
  diagnostics. The canonical 93-11f/93-12 speed probes are the only exception:
  use 400 output tokens and the explicitly paired B/U geometry. Use one slot,
  the existing managed lifecycle, and never launch a second Qwen3.8-27B CUDA
  process.
- Read only this cluster, the active packet, the compact 93-04/93-05 evidence,
  and the exact source functions named below. Do not load old V9/phase-85
  planning documents as task context; consult historical material only when
  one named code contract cannot be understood from current source.

## Candidate identity and service lifecycle

Use the canonical Qwen3.8-27B UD-IQ4_XS model, one immutable candidate binary,
and the existing `CANONICAL_BENCHMARK_RUNNER` via
`tools/server/bench/run-mtp-diagnostic.py`. Before any request, require the
expected managed process identity: PID/start time, `/proc/<pid>/exe` realpath
and SHA-256, model realpath and SHA-256, and exact context/batch/ubatch/pager/
Turbo4/MTP command-line options. If it matches and the previous lifecycle
manifest says `continue_loaded=true`, continue with that process. Otherwise
reload the expected binary through the managed lifecycle and verify it before
requesting. A healthy endpoint by itself is never candidate evidence.

Every raw rung gets its own directory and request-local counters. Preserve the
candidate identity, command, prompt hash and exact token count, server log,
request/response, `/slots`, metrics before/after, page identities, VRAM and
scratch/headroom values, and lifecycle manifest. Missing telemetry remains
null and fails the relevant proof; do not infer zeros from missing fields.

## Required diagnostic order

1. MTP-off dense/all-GPU control.
2. MTP-on dense/all-GPU control with GPU Turbo4 draft KV.
3. MTP-on selected/paged with all pages resident for the short control.
4. Only after the first three complete, run the bounded two-document cold
   promotion sequence with a four-page hot set: ingest A; append B and query
   B; query a unique fact from A again. These must be prefix-preserving
   requests in the same slot so A can actually become cold. The 93-01 filler
   prompt is not a promotion proof.

For the live controls, pair prompt, seed, sampler, generation limit, model,
binary and geometry across dense and selected routes. Report `draft_n_accepted /
draft_n` from each request; `accepted_target_tokens` is not the denominator.
Compare selected-resident acceptance against the same dense MTP prompt. For
promotion, require the same logical page identity to be observed cold, selected
by the query, H2D-completed and published before attention, then consumed by
the target and draft. A page allocation, route name, or successful response
alone does not prove promotion.

## Scope boundary

Repair the selected trim/recovery defect and cold-sequence harness before any
large-context run. A prefill timing captured by a functional diagnostic is
only a diagnostic unless it uses paired canonical prompts and exact input
token counts no greater than 16384. Do not claim practical speed, 256K
occupancy, production capacity, or quality from this phase. Phase 93-09 must
schedule the next small, paired performance measurement only after these
functional proofs pass; otherwise it schedules the concrete repair supported
by the failing evidence.

## 93-09 decision

93-08 sent no live request: the canonical runner was unset, and the active
service executable/configuration did not match the bounded candidate. Dense
MTP remains supported by the earlier 93-05 8/10 control. The only speed work
allowed before cold-promotion closure is the bounded 93-11f 8K/4K hot-resident
batch-geometry comparison; do not treat it as offload or promotion evidence.
Next, 93-11g runs file-backed natural promotion at that same 8K/4K geometry.
The paired placement screen and all large-context work remain gated on the
candidate-bound cold proof and its explicit prerequisites. If a canonical
MTP acceptance floor is missed, keep the owning benchmark incomplete, use its
dense MTP control to localize general versus selected/offload behavior, and
repair or schedule the exact fix before proceeding.

## 93-11c decision

The candidate-bound four-rung diagnostic passed the MTP-off dense, native MTP
dense, and selected-resident controls. The same-slot cold A request passed, but
the 1760-token B append timed out after checkpoint restore and
`CHECKPOINT_ATTN_ONLY_TRIM p0=27 target=1 draft=1`. With CUDA graphs disabled,
one run reported an asynchronous illegal access at the generic CUDA kernel
launch check; with launch blocking enabled it still timed out without naming a
kernel. The source owner is not identified and no safe fix or seam regression
can be justified from these artifacts. Task 93-11d is scheduled immediately
before 93-12 to isolate the CUDA operation, add the regression, repair it, and
repeat all four gates. Task 93-12 remains todo and depends on 93-11d.

## 93-11d decision

The candidate-bound per-node synchronization diagnostic identified the first
failing operation during cold B as `FLASH_ATTN_EXT` (`node_232`, output shape
`[256,24,64,1]`) after checkpoint restore and attention-only trim. Earlier
nodes synchronized successfully. The current trace does not identify which
K/V/page backing identity is invalid, so it does not justify a safe source
repair or deterministic kernel regression. The execution order is now
93-11f resident speed check, 93-11g file-backed A→B→A promotion campaign,
93-11e repair/proof closure, then 93-12 paired placement performance.

## Current 93-11e evidence boundary

The latest A fixture spans multiple pages. A-again selected logical page 0;
the live snapshot recorded generation/content version 9, 4,325,376 useful H2D
bytes, completed transfer, mapping publication, and target graph use. The
receipt still fails closed because it does not record request-local H2D event
identity/order or page-specific draft consumption. Preserve the Q=1 dispatch
and layer-stride repairs. The 93-11g campaign uses actual fixture file bodies,
not page IDs or the prior short inline fact. 93-11e remains blocked/incomplete
until the fixture run has exposed and closed any remaining proof gaps and the
canonical gates pass; neither 93-11f nor 93-11g replaces the source repair and
request-correlated proof work.
