# V7 — fast incremental hot/cold KV measurements

Historical: V8 replaces this for unfinished work from27-02.
Read BENCHMARK_PROTOCOL_V8.md and GPU_HOT_PATH_REDESIGN_V8.md.

Authority: PHASE26_INTERACTIVE_KV_STRATEGY.md, current task, then this protocol.
Supersedes V6's full-256K-first and final six-coordinate obligations for active
work. Keep original three canonical questions; tiny deterministic fixtures are
additional correctness diagnostics, not replacements for speed questions.

## 1. Reusable coordinates and proof stages

| Fixture | Logical capacity L | Hot tokens H / pages at P=256 | Occupancy and purpose |
| --- | ---: | ---: | --- |
| primary | 8192 | 4096 / 16 | about 6144 occupied before decode; true cold history exists |
| scale | 32768 | 16384 / 64 | about 24576 occupied; incremental continuation |
| maximum | 131072 | runtime safe maximum | first allocate+short smoke, then C>H, finally approach L minus generation reserve |

H is allocated physical capacity, not just a smaller A. Prove host-only pages
actually exist and the GPU target store is bounded by H. If auto H would cover
all L, report the naturally all-fit result and separately cap H to create a
clearly labeled pressure experiment. Never call all-fit generation offload.

Normal initial experiment: B=128, U=64, native MTP n-max=2, target/draft Turbo4,
draft GPU rows=L. These are benchmark coordinates, not universal defaults.
Use one sequence/slot. Pin recent/current/mandatory rows within H while leaving
append and promotion slack; validate page-aligned quotas before starting.
Keep selected A fixed across a batch-only A/B; log actual per-layer A because
changing selection work invalidates a claimed batch speed effect.

## 2. Exact requests and incremental history

Extend run-final-curve.py and existing test_resume_contract.py; do not write
another anonymous /dev/fd corpus loop. Use the server tokenizer/chat template
for the actual model. The fitted messages are sent unchanged to chat exactly
once. Count template, BOS, assistant prefix, prior output, and new input.
Record requested versus rendered versus processed versus cached tokens.
Validate C + new uncached input + output reserve + speculative boundary slack
against resolved L; no double addition of already retained prefix.

For primary/scale setup, accumulate durable neutral/tool-like turns in 512–2048
new-token increments with a stable fact/topic in an early page and distractors
later. Use the server's supported slot/prefix-reuse contract, preserving actual
assistant outputs in subsequent messages. Prove reuse from cached-prefix and
processed-token deltas. If chat rendering changes earlier tokens, fix that
driver boundary; do not silently re-prefill the entire history every turn.
Compare one contiguous prefill versus incremental equivalent token sequence
in a small deterministic fixture where possible; different chat templates are
not numerical-parity pairs. Chunking one request is not evidence of multi-turn
reuse, and multi-turn appended-only pp/s is not cold full-prefill pp/s.
For this hybrid/recurrent model, validate GDN state and draft frontier reuse as
well as attention KV. A bounded last-turn checkpoint/carry may be needed for
the server's needs_reeval path; charge its buffers/transfers and report replayed
tail tokens. Blanket disabling checkpoints/prefix facilities is not a valid
normal-use speed optimization if it forces every turn to reprocess history.

After C>H, query the early topic. Record page identity/layer/content version,
host-clean/nonresident state before recall, live query-summary ranking, policy
selection reason, H2D completion, mapping publication, and subsequent attention
use. Separate rank-driven recall, rotating exploration, and forced test hooks.
Explore fallback may be useful but does not by itself prove attention ranking.
No HTTP call or huge queue count alone proves that payload reached the GPU.

Generate >=128 committed output tokens for speed checks when EOS permits.
Record actual EOS and short samples; rerun once with a predeclared long-answer
prompt or a clearly labeled diagnostic ignore-EOS option for a stable decode
sample. Do not present 1–25-token generations as a reliable speed comparison.
Include generation/verification slack in the near-full capacity fixture; use
>=512 token reserve unless actual MTP/server requirements require more.

## 3. Cheap iteration versus checkpoint evidence

Per code change: focused deterministic regressions, then original q0 on primary
at one fixed occupied-history coordinate, 128 output tokens, one measured run.
One warm-up only where needed for kernel/graph warmup; warmup must not silently
fill the prompt cache for a cold-prefill measurement. Retain cold first-use and
warm steady-state separately. For a prospective improvement, use three matched
q0 trials and report median/range. Do not automatically run all three questions,
all batch combinations, or all context sizes per patch.

Checkpoint 26-08 and summary 27-03 run the canonical three questions once each
per successful required coordinate, with q0 repeats for the winning primary
profile. Reuse matching successful case records instead of regenerating them.
Identity key includes source diff, executable+DSOs, model, prompt/template,
L/C/H/A/B/U, route, MTP, reuse condition and generation settings. Any identity
change invalidates reuse for the changed claim, not every unrelated unit test.

Required controls at primary: feature-off all-GPU target with the SAME GPU
Turbo4 native MTP, and ordinary CPU target KV offload with GPU Turbo4 native
MTP. Discover supported launch flags from --help/current adapter; assert
actual placement/acceptance. A CPU-weights server is not a CPU-KV control.
Use one q0 trial for the expensive CPU control, with the same occupied history
and a progress-aware budget derived from measured processing slope. If it
cannot complete, keep the partial rate labeled and ratio null; do not substitute
a different prompt/context or disable MTP. Do not repeat the expensive CPU
control at every larger coordinate. GPU control comparisons may be unavailable
at higher L; state that rather than reducing L invisibly.

## 4. Rate definitions and required evidence

Keep raw final SSE timing object unchanged. Report server prompt/decode metrics
with their original count convention (some decode metrics use n-1), PLUS:

- uncached processed prompt tokens / measured prefill seconds;
- committed output tokens / measured decode interval, with start/end defined;
- client TTFT, request elapsed, actual output count, optional inter-token tails;
- MTP proposed/accepted/rejected and target verify tokens, never proposals as
  output throughput;
- initial C, final C, newly processed rows, draft consumed/committed frontier;
- request-delta H2D/D2H, host seal bytes/calls, promotions, evictions, recall
  reasons, sampled attention count and stale/drop reasons;
- requested/effective B/U and query tile, route per phase, A range per layer;
- memory by owner before load, after KV allocation, first/max-U prefill, MTP
  verification, promotion, and settled decode; requested/physical scratch,
  mapped versus reserved VA, pack/ring bytes, headroom and external GPU users.

Do not add overlapping queue/wait/CUDA timings and claim a wall-time total.
Use unprofiled speed runs; collect a separate short profiler run for critical-
path attribution. Missing telemetry is null with an instrumentation task, not
zero. Distinguish confidence in measurement, functionality, and speed outcome.
Speeds are findings, not arbitrary 3x/5x/70% gates. Never assert a speedup from
different output counts, startup-only runs, stale binary hashes, or cold versus
cached input mismatch. Keep answer text for small quality diagnostics only.

## 5. Progress, interruption and privilege

The phase26-01 driver work must add a V7 wrapper/validator around existing raw
records (keep old receipts readable). Required top-level sections and units:

```text
schema: interactive-speed-v7
case_id; stage: setup|primary|scale|maximum|control|diagnostic
provenance: source_commit, source_diff_sha256, bundle_manifest, model_sha256,
            request_sha256, template_sha256, argv_redacted, live_pid, slot_id
configuration: logical_capacity_tokens, page_size_tokens, hot_capacity_pages,
               hot_capacity_tokens, batch_tokens, ubatch_tokens,
               effective_batch_tokens, effective_ubatch_tokens,
               target_k_type, target_v_type, draft_k_type, draft_v_type,
               target_compute_device, draft_kv_device, draft_capacity_tokens
history: occupied_before_tokens, occupied_after_tokens, rendered_tokens,
         cached_tokens, processed_tokens, committed_output_tokens,
         draft_processed_frontier, draft_committed_frontier, truncated
timing: raw_server_timing, prefill_seconds, decode_seconds,
        rate_count_conventions, ttft_seconds, elapsed_seconds
movement: host_only_pages_before, valid_attention_samples_delta,
          rank_promotions_delta, exploration_promotions_delta,
          forced_promotions_delta, evictions_delta, h2d_useful_bytes_delta,
          h2d_actual_bytes_delta, d2h_actual_bytes_delta, trace_path
memory: stage_snapshots[{stage, owner_id, context_role, category,
                        requested_bytes, physical_bytes, peak_bytes,
                        reserved_virtual_bytes, aliases_owner_id}],
        peak_global_bytes, minimum_headroom_bytes
outcome: request_completed, measurement_valid, natural_joint_proof,
         allocation_only, failure_category, failure_trace
raw: request_path, response_path, progress_path, metrics_before, metrics_after
```

Use null plus reason for unavailable optional measurements. Use integers for
counts/bytes, not floats copied from a formatted metrics display. Required
identity and units cannot be null in an accepted speed case. A successful
request is not automatically a natural_joint_proof. That flag requires C>H,
actual host-only prestate, valid natural rank selection with traced completed
H2D/use, bounded GPU storage, correct Turbo4 target/draft placement, draft
capacity=L, native-MTP activity and no truncation/reset. A forced or exploration-
only promotion is explicitly a different result. Validators should test each
missing condition independently and never infer success from a filename.

For setup/diagnostic stages natural_joint_proof may be false without making
the raw measurement invalid; their scope is not final capability. The primary
checkpoint and final summary require a separate matching joint-proof case.
Keep mutable configuration C/A ranges in per-request records so aggregate
summaries cannot silently combine unlike runs.

Durable roots: /srv/ai/paged-kv/results/<task>-<fixture>-<UTC>/, not /tmp.
Persist config/provenance before launch, raw request before send, stream data
and progress during execution, terminal status and response afterward. Track
owned PIDs/service MainPID and endpoint, not the first llama-server process.
Never identify/kill clients by the shared Python /dev/fd command name.

Separate startup, active-prefill no-progress, active-decode no-progress, total
elapsed budget, and detached client failure. Existing defaults can be initial
watchdogs but not hard 240s full CUDA rebuilds or arbitrary long-prefill kill
limits. Poll live progress in short intervals; a working request with increasing
C is not idle because it has not emitted the first output token. Estimate
remaining time from recent per-chunk slope and report it. On genuine stall,
capture frontier, outstanding events, allocation failure and owned PID stack
before bounded termination. A client timeout is not a model capacity result.

Withdrawn heuristic: frontend zero graph replay and overlapping queue/wait
timers do not establish CUDA recapture or host overhead. No absolute rate
threshold replaces profiling. V8 specifies a two-append pilot, ETA/useful-work
budget and matched kernel/control comparisons before long population.

Resume completed cases by exact identity. Resume incremental history only if
live slot/model/context/frontier identity matches; otherwise reconstruct the
setup once from durable requests and label the extra setup time. Never append
an unknown already-committed turn twice or use a generic HTTP200 as proof of
resume identity. Keep one benchmark owner; do not launch another while an
earlier request remains active.

Use sudo -n for systemd and protected files. Capture original identity safely,
stop only authorized Qwen service when replacing it, pass the candidate binary
and actual B/U explicitly through site overrides, and verify model+bundle+argv
after readiness. Keep the successful tested candidate loaded for next task;
restore only for explicit comparison, failed startup or requested teardown.
Never print API key values or capture secret environment contents.

## 6. Completion and next task

Every implementation handoff names its working primary case ID or one precise
failed invariant. Do not advance a live-proof task with “ready” alongside false
movement/placement readiness. The final compact summary indexes only this
campaign's measurements and negative findings; old phases are attribution
references, not gates. If 128K fails, report the largest actually populated
working L/H and failure owner, not an endless ladder of blind retries.
