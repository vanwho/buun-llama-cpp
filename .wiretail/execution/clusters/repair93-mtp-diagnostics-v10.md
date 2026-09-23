# repair93 — short Turbo4 MTP and pager correctness

Phase 93 resumes from the measured 93-04 failure. Dense GPU Turbo4 MTP now
works at 8/10 accepted draft tokens on its bounded control. Selected/paged
requests still crash after `memory_seq_rm [p0, end)` is rejected, and the
existing cold rung has not proved document-driven page promotion. The phase
must resolve those specific gaps before a speed or context-capacity campaign.

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

## Goals and bounds

- Keep the target and MTP draft on CUDA with Turbo4 K and V for every MTP-on
  control. A route that silently moves draft state to CPU, changes KV type, or
  uses `selected_reference` is a setup failure.
- Do not reserve more than 48 Ki tokens of target hot KV in VRAM in any phase
  93 test (`48 * 1024 = 49152` tokens, or 192 pages at 256 tokens/page). This
  is a test ceiling, not a production hot-set constant. Request only the
  smaller hot-page count needed for each test; the allocator's byte budget
  and `--kv-safety-headroom auto` remain authoritative and may admit less.
- Keep test context small: use 4096 for trim/MTP iteration and at most 8192 for
  the two-document cold-promotion sequence. Never configure more than 16384
  context tokens in this phase.
- No request used to measure prefill/input speed may contain more than 16384
  rendered input tokens, including the retained prompt prefix. Preflight the
  exact rendered prompt with the canonical tokenizer before sending it; reject
  over-limit input instead of truncating or reporting it as a valid speed row.
  Prefer 4096/8192-token measurements while iterating; use 16384 only for a
  final bounded point after the smaller setup is stable.
- Keep generation at 16 tokens or less and `draft_n_max=2` during diagnostics.
  Use one slot, the existing managed lifecycle, and a small `ubatch` (start at
  64, lower it if the measured scratch/headroom preflight requires it; record
  actual `-b` and `-ub`). Never launch a second Qwen3.8-27B CUDA process.
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
