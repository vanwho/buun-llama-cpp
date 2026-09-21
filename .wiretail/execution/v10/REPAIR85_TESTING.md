# Fast, reliable testing for repair85

Revision: `hotpath-v10-20260914`. Supersedes old universal fixed-coordinate
requirements for tasks85-09 onward. Test geometry is chosen for the invariant;
large speed curves are final findings, not per-change gates.

## Common setup (reuse, do not create another benchmark stack)

1. Use the current incremental CUDA build and focused named targets. Preserve
   a build-time receipt via existing `write_build_receipt`/bundle helpers in
   `tools/server/bench/run-pager-profile-benchmark.py`. Snapshot executable
   and project DSOs together. If worktree changes are tested, record base SHA
   plus exact source-diff hash, build flags and loaded DSO hashes; don't stamp
   an arbitrary current HEAD on an older binary. Do not rebuild all templates
   from scratch for each smoke or repeatedly kill a progressing compile at240s.
2. Model is the installed profile's Qwen3.8-27B UD-IQ4_XS, resolve
   `/srv/ai/models/text/current.gguf` and recorded GGUF metadata. Use existing
   credential-file references without printing key contents. Verify the PID
   owning the tested endpoint, `/proc/PID/exe`, loaded DSOs, model and actual
   flags. `pgrep llama-server | head` is never process identity.
3. Hold existing lifecycle lock. `sudo -n true` and `sudo -n systemctl` operate
   the authorized Qwen service; passwordless sudo is expected. If denied,
   inspect `id`, `sudo -n -l`, command path and sandbox/no-new-privileges;
   use the correct authorized execution environment. Do not silently switch
   to all-CPU27B. Do not alter sudoers or unrelated services. Never touch8092.
4. Reuse canonical profile runner/portable adapter and the installed
   `/srv/ai/scripts/start-primary-llama-profile.sh`. Verify observed `-b/-ub`,
   resolved L/H/A, `-ctk/-ctv turbo4`, native `-ctkd/-ctvd turbo4`, full-L
   draft and explicit GPU draft placement. Fix ignored launch overrides at
   their boundary, regression-test and rerun. A same-bundle direct isolated
   launcher is allowed for a quick diagnostic while repairing the adapter,
   but a different model/binary/placement is not an equivalent result.
5. Keep a working candidate loaded and hand off its identity. Restore only
   for an explicit control, unsafe failed candidate, or teardown. Kill only
   the recorded child PID/process group that this run owns, not generic
   python/Codex/renderers. Record service change/recovery once, compactly.

## Geometry ladder and recovery

- Kernel/address tests use 1–3 small pages, no27B model, real CUDA, nonzero
  head-distinct data. F32 CPU math is an oracle, not a CPU Turbo4 inference run.
- First real inference: **L4096/H2048/A1024, B128/U128, one slot**, 512–1024
  actual input tokens and32 output tokens; all four KV types Turbo4, native
  draft n-max2. Derive head/layout sizes from model. For actual cold crossing
  use C around3072 at the same L. H/A are fixture choices, not product rules.
- Small cold test requests `--kv-hot-pages 8 --kv-attention-tokens 1024
  --kv-pin-recent 512 --kv-page-size 256 --kv-prefetch-depth 2`. Verify actual
  admission and mandatory/current pins leave cold candidates and recyclable
  slots; an inherited large recent window must not pin the whole tiny pool.
  Use `AI_BENCHMARK_KV_HOT_PAGES`/`AI_BENCHMARK_KV_PIN_RECENT` and existing
  launcher options where supported, then inspect observed command/runtime.
  At H4096 start recent1024, A2048 or4096 only when the ledger can reserve
  required transfer/write slack. Report physical slots, occupied resident
  rows, A and reserve separately; do not hide extra staging inside H's label.
- The standard small speed test is **L8192/H4096**, actual C around6144, then
  a matched65-token cached append and64–128 committed output tokens. Do not
  rerun it for every C++ edit. Only end-of-phase work advances to32K/16K hot,
  128K and eventual256K with dynamically safe H.
- Reserve measured scratch and headroom before H. For small fixture OOM,
  stop the owned restart loop, ensure only the intended GPU model is loaded,
  reduce **H/A first** in page increments, then U128→64 and B consistently.
  If needed run the same invariant at L2048/H1024/C1536. Rebuild matched
  control rows at the new observed geometry and retain the failed request.
  Never reduce full-L draft relative to L or move it to CPU to make a pass.
- For a fixed-L scaling test, keep L and full-L draft, shrink H/A/U first.
  Only after the measured minimum viable layout cannot fit may that L be a
  capacity finding. Distinguish allocation from actual occupied C.
- Measure high-water memory during prefill AND draft/verify/rollback, not
  just startup free VRAM. Initial site reserve: at least max(512MiB, twice
  the observed incremental scratch growth), adjustable from measurements;
  keep this site policy out of generic kernel code.

## Short test and timeout policy

Per implementation edit: named CPU invariant where useful + named CUDA
fixture; at most one short real-model differential request when specified.
No24/48-case corpus, 10-repeat campaign, CPU27B loop or256K frontier here.
Use existing test selection or add narrowly scoped fixture options; don't run
all timing sizes as part of the default correctness executable.

Track token progress for the exact request (not a cumulative metric, polling
message, checkpoint or table epoch). For tiny live smokes start with a120s
request deadline and15s no-token-progress watchdog after readiness. If a
healthy equivalent baseline needs longer, calibrate once from its measured
time. After warmup, a >4x latency/prefill slowdown vs the *matched* control is
a profiling trigger: retain that small trace and repair it rather than start
the next larger context. These are iteration guards, not acceptance targets.
Do not abort a CUDA build as if it were an inference stall. Permit a single
bounded extension for clearly advancing work; never renew indefinitely.

Config failures trigger deterministic repair/retest in the same task:
OOM→measured budget ladder; HTTP401→existing key-file setup; ignored U→launcher
fix; stale PID/DSO→reload candidate; context overflow→tokenize rendered request,
budget output and resize fixture; occupied port→owned lifecycle isolation.
After each change verify actual geometry again. Illegal access/nonfinite
attention/output mismatch is a code failure: reproduce tiny, don't hide it
by shortening generation, enabling reference fallback or relabeling the row.

## Numerical and MTP assertions

- Compare the *same encoded Turbo4 bytes, native positions, selection and
  scale*, not unquantized K/V. Use combined abs/relative tolerances justified
  by mature CUDA-vs-reference error on the identical fixture. Preserve the
  existing valid tolerance; do not widen until a wrong address passes.
- Fast-path fixtures must record actual kernel variant with a test-only
  hook; route `selected direct` is not proof of MMA. Do not attach page-mass
  or split-state options when verifying the normal MMA dispatch predicate.
- Reject nonfinite values in valid K/Q/output/logit rows. Padding can be
  poisoned deliberately to prove it is never read; padded vocab exclusions
  are separate from valid vocabulary logits. Argmax diagnostic `-1` is not
  a NaN detector. All-head output canaries expose unwritten/overlapping heads.
- Greedy MTP verification must match non-speculative target continuation
  under the same frozen selected table. Count draft attempts/acceptances per
  request, committed outputs, bonus token, rejected suffix and real restore
  success. Native accepted/drafted>0 alone is not success; compare with the
  repaired all-GPU native control on original prompts.98% is an observation
  on a prompt, not a universal threshold. No grammar/logit_bias to force
  acceptance. Diagnostic hashes/full-vocab reads off during timing.

## Minimum phase-end campaign

Original three prompts from `/srv/ai/benchmarks/run-profile-benchmark.sh`:
merge-two-sorted-lists Python; mmap-vs-read paragraph; watch-directory Bash.
Use unchanged wording, same chat template, thinking mode, seed, sampling and
output budget. One warmup per loaded profile and one measured trial per
prompt initially; repeat only noisy/contradictory rows (up to3). Keep prompt
cache off for fresh pp and deliberately on for separately labeled append.
No huge padding masquerading as the original short prompt benchmark.

Report fresh evaluated prompt tokens/sec, newly evaluated cached-append
tokens/sec, committed decode tokens/sec, elapsed/TTFT, request MTP acceptance,
L/C/H/A/B/U and observed placements. A separate context-bearing workload
crosses H and triggers an actual cold promotion; do not claim the short three
prompts themselves prove offload. Compare all-GPU MTP off/on, CPU-main-KV+
GPU-native-MTP and selected native/off at matched small C. Same binary/model
and prompt per comparison, no concurrent GPU benchmark clients.

Use on-demand CUDA events/Nsight for one warm interval to separate host setup,
packing/materialization, GPU work, canonical seal, selector, H2D, waiting and
checkpoint work. No permanent event sync per layer/token just for statistics.
Minimal counters sampled before/after are enough for nonprofiled timing.
Ratios are findings, not inherited3x/5x gates or permission to sacrifice
numerical correctness. Larger campaigns wait until this small path is usable.
