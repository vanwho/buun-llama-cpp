# V9 fast iteration and honest benchmark evidence

This replaces V5–V8 benchmark gates. No 24/48-case matrix, ten-trial minimum,
mandatory long prompt, historical phase-14 acceptance audit, or 3x threshold
is required to complete a measurement task. Speed is a finding. Broken
capability is still a failure and must produce executable remediation.

## B1. One reusable harness, actual candidate identity

Use `/srv/ai/benchmarks/run-profile-benchmark.sh` for site lifecycle and the
existing portable adapters in `tools/server/bench/` (`run-pager-profile-benchmark.py`,
`run-incremental-recall.py`, `run-incremental-scale.py`, `prompt_sizing.py`,
`cache_plan_common.py`). Add small options/fixtures there instead of creating
another all-purpose acceptance framework. Site-only orchestration belongs in
`/srv/ai/benchmarks`, portable logic/tests in the fork. Do not import old
acceptance manifests or require their telemetry schema to run a new q0 probe.

Before collecting rates, verify endpoint -> managed service MainPID -> resolved
`/proc/PID/exe` -> loaded CUDA/ggml DSOs -> candidate hash. Record the model's
resolved symlink/hash and effective argv: L/H/A/B/U, target K/V types, native
MTP type, draft types/context/device and page count. An 8092 process selected
by `pgrep llama-server` is not the 8080 identity. Health alone does not prove
candidate identity. Redact credentials/headers; use the configured key file.
Record active PID/config and keep a successful candidate loaded between tasks.

A baseline replaces the current service sequentially; do not load a second
27B GPU process. Return to the selected candidate only for the next selected
measurement, not reflexively after each successful task. Use full CUDA weights
for **CPU-main-KV** control too. CPU model-weight execution is a different
experiment and must not be mislabeled RAM-KV offload.

If the all-GPU baseline cannot fit the same L, report the largest same-C
comparison with both L values and memory effect, or null for unavailable
same-L comparison. The CPU-KV control uses `--no-kv-offload` with independent
`--spec-draft-kv-device gpu`, `-ctk/-ctv/-ctkd/-ctvd turbo4`, native MTP and
GPU model weights. Inspect actual residency and operator placement; if the
combination is unsupported, repair the generic placement boundary rather than
invent a ratio from historical 12 tok/s anecdotes.

## B2. Canonical workloads and measurement modes

Preserve the original three exact prompts:

0. `write a python function that merges two sorted lists into one sorted list, with docstring.`
1. `explain the difference between mmap and read for loading large files, one paragraph.`
2. `write a bash script that watches a directory and prints new files as they appear.`

Use the existing token-counted padding/retained-history generators and tokenizer
endpoint, not character-count estimates. Freeze prompt token IDs/hash per paired
run. Chat template, reasoning mode, sampling/seed and generation limits match.
Choose an actual generated length of 128 committed tokens for primary decode;
record EOS/stop-short rather than inflating a 2-token sample into a speed result.
No forced answer-content modifications just to equalize MTP acceptance.

| Mode | Initial workload / repetition | Purpose |
| --- | --- | --- |
| code smoke | C256–1024, output 16–32, one request + append | Startup/correctness only; not speed evidence |
| kernel fixture | Small synthetic multi-head shapes; event-timed repetitions | Select same-work route/copy implementation, not user throughput |
| `quick` | L8192/H4096/A2048/B128/U64, token-counted C6144, q0 once, output128 | Default speed check after a material code change |
| `repeat` | Same quick q0 three times with clean/defined slot state | Median/variability and repeated-request crash detection |
| `cold` | Populate C>H incrementally, recent question about a sealed evicted page | Prove real current-Q cold promotion/use with native MTP |
| `compare` | Original three prompts once each per selected / CPU-KV / all-GPU profile; repeat q0 if noisy | End-to-end small comparison |
| `incremental` | Fresh L32768/H16384, modest 512–2048-token append chunks until C>H | Cached append scaling, never giant one-shot prefill by default |

Initial B/U and A are reproducible test settings, not recommended universal
production constants. If ledger disallows H4096 on this model/backend, report
the actual reason before altering it; the matched test identity must change
explicitly. Keep L fixed and native draft full-L while fitting auto H/U.

Always separate:

- cold prefill: all uncached input tokens / prefill duration;
- cached append: **new evaluated tokens**, cached count, prefill duration;
- committed decode tok/s and inter-token times (proposal rate is separate);
- TTFT and total request wall time, including setup where explicitly measured.

Prefill dominated by CPU weights or full-H custom attention is not a legitimate
test of the new packed hot path. Check dispatch before spending hours.

## B3. Bound runs by purpose, not by fabricated success

Persist each completed request immediately to JSONL, with a deterministic
request ID and phase/case key. Resume by validated binary/DSO/model/config/
prompt hashes and retained service sequence state; no blind re-posting an
in-flight request. If a POST times out, cancel/poll that exact request/slot
and confirm completion/cancellation before retrying. Preserve partial results.

Live startup: use a measured/progress-aware bound (default 180s for an already
built local model); after one actual OOM collect the ledger and lower auto H
by page units, then U as F5 specifies. Bound to three distinct capacity fits.
Never loop systemd restart or restore indefinitely. Build waits are separate:
keep incremental compilation while object count/logs advance; poll commands
at <=60s intervals and send updates. A long build is not a prefill timeout.

Default quick request budget 180s, with a no-progress detector based on
authenticated server/request evidence. **Do not apply that fast-candidate
deadline unchanged to the deliberately slower CPU-KV denominator.** Calibrate
the CPU-KV control with a small unscored request, estimate the matched C6144
cost, then allow a documented per-request budget up to 900s and a 30-minute
control-campaign budget. Measure q0 first; if q1/q2 will exceed the budget,
retain q0's real paired ratio and mark the other CPU rows not_run. Never
compare a smaller-C CPU pilot against the larger-C selected run as a matched
speedup. A long baseline is not a reason to run the candidate for hours.
Larger-context pilots get a declared
30-minute **campaign** budget, not unlimited extensions while tokens trickle.
After the first two append chunks, estimate remaining time from recent new-token
rates and C-dependent slope, record ETA. If it exceeds the declared budget,
stop at the valid checkpoint with `too_slow_to_scale` and its rates, then
proceed to the summary/review that schedules repair. Forward progress at
16 tok/s alone is not a reason to continue a multi-hour capacity campaign.

Do not require completion of all larger coordinates to discover a small-path
failure. The 20K/40K/60K/100K/175K/256K curve is a **future final findings**
campaign after capability works, not a task validation matrix. Up to128K is
the active live ceiling in this revision; record 256K ledger feasibility only.

## B4. Minimal authoritative receipt

Each compact evidence JSON uses `schema: hotpath-v9`, task/revision, UTC, and:

```
provenance: code_sha, source_diff_hash, server_sha, loaded_dso_hashes,
            model_hash, prompt_hash, raw_root, effective_config_hash
config: L, C_before, C_after, H, A_by_layer, B, U, page_tokens,
        target_k, target_v, draft_k, draft_v, draft_L, draft_device, mtp_type
execution: target_attention_route, target_layer_count, resident_bytes,
           canonical_host_bytes, selected_rows_by_layer, fresh_or_restored
rates: prefill_new_tokens, prefill_seconds, cached_tokens,
       committed_output_tokens, decode_seconds, ttft_seconds,
       pp_tok_s, tg_tok_s, proposed_tokens, accepted_tokens
work: pack_bytes, host_copy_bytes, h2d_bytes, selected_cold_pages_used,
      graph_builds, graph_replays, router_refreshes, queue/copy/wait samples
resources: peak_bytes_by_category, safety_headroom_bytes, CPU/RSS
decision: pass|fail|not_run; capability_proven: boolean;
          reason; next_action; partial_checkpoint
```

Unavailable counters are null with a reason, never zero substituted for unknown.
Use integer byte counters from authoritative local records for exact byte
proof, not rounded Prometheus strings. `hot_tokens` is a row count, not VRAM
bytes. Schema validation checks units and dimensions (selected<=A<=H<=L for
this single-sequence fixture), required native-MTP types and per-layer IDs.
Raw logs can be large under `/srv/ai`; receipts/handoffs stay compact. Use one
versioned receipt per benchmark run, never append an entire stderr transcript
to the task handoff. A reported 3x ratio must have actual paired denominators.

Detailed CUDA timings are opt-in profiles: use events or Nsight/CUPTI if locally
available. NVTX/op trace host time is not kernel duration. Measure with and
without diagnostic collection once; keep the lower-overhead production setting.
Counters must not force custom FA, full softmax matrices, per-token device
synchronization, graph rebuilds or per-layer filesystem logging.

## B5. Scale eligibility and comparisons

32K requires a stable small selected route, successful natural cold proof,
three repeat requests, and actual small paired controls. Rates below target
are recorded; clear regressions/slow pilot go to remediation, not a huge run.
128K requires a useful 32K run and a ledger fitting full-L GPU Turbo4 MTP with
headroom. First allocate and issue a short request, then an incremental pilot;
only proceed to occupancy beyond H and near L if budget allows. Occupancy
means retained native tokens, not `-c` or a single 512-token prompt.

Compare original prompts at chosen occupied frontiers with matched fresh or
paired target+draft state. Never restore target alone and pretend full historical
MTP is present. Report MTP loss separately from attention-kernel cost. Paired
CPU baseline at large C can be budget-limited: retain partial rate findings
with sample count and do not extrapolate a precise final ratio.

Minimal quality: deterministic selected-set numerical tests, one early/cold
fact, coherent committed output and MTP acceptance. Do not demand broad exact
CPU/GPU or long soak before seeing speed. Do not suppress wrong results: they
inform selection tuning and prevent meaningless fast-but-invalid claims.
