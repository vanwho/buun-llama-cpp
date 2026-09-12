# V8: short experiments that decide implementation work

Read GPU_HOT_PATH_REDESIGN_V8.md sections1–4/10 and the current task.
This replaces V7's active scaling instructions. Keep its precise token-count,
canonical-question, privacy and service identity contracts where consistent.

## A. Scope and controls

One binary and its loaded DSOs, model Qwen3.8-27B UD-IQ4_XS, all Turbo4.
Weights/compute/GDN GPU; native full-L GPU Turbo4 MTP in all speed controls.
Primary L8192,H4096 (16 pages at256), C approximately6144, so cold pages exist.
Scale L32768,H16384,C about24576. Larger L131072 only after the small path
works; H is the budget solver's measured safe result, not a fixed value.
Keep A explicitly fixed across implementation and B/U-only comparisons.
Starting test A=2048 total rows per layer inclusive of mandatory rows at
primary; compare4096 only with enough append/promotion space or adjust H and
label it. At H4096 reserve workspace independently; A cannot consume pinned
and spare slots twice. These are experiments, not production defaults.

Use existing original q0/q1/q2 from /srv/ai/benchmarks. Keep question text and
generation settings fixed; separate neutral-history setup from each question.
Small fixtures may use synthetic tensors and a known cold fact for proof.
Do not substitute generated short questions for the final canonical questions.

Primary controls: feature-off all-GPU main KV + same GPU MTP; selective with
resident selection; selective with a real cold promotion; ordinary CPU MAIN
KV + same GPU MTP (weights stay GPU). One CPU control after production works,
not on every patch. A missing slow CPU control leaves its ratio null; it
does not prevent fixing the GPU kernel or reporting paired GPU differences.

## B. Development inner loop

1. Focused CPU/unit test then CUDA fixture: Q=1,2,3,16,64,128 where supported,
   D/head counts read from model; selected rows2048/4096, page permutation,
   native gap, partial page, causality. Separate timing from sanitizer.
2. Kernel comparison: same Q/K/V/indices/output domain, CUDA-event timing,
   5 warmups and 20 timed iterations initially. Small repeat count adjustable
   for timer resolution; never hundreds of full model requests. Report kernel,
   pack/unpack, split/merge separately and combined. Time resident inputs.
3. Live one q0 at primary with128 committed output if EOS permits. Report
   native SSE counts as-is and independent committed-token/wall rate.
   A one-token response is smoke only; early EOS is not a128-token rate.
4. Compare three matched q0 trials only to choose a candidate improvement.
   Changing A/C or cold/warm state changes the experiment; do not credit that
   as a kernel speedup. Clear/recreate prefix state consistently between pairs.
5. End of an integration task: one real query-ranked host-only promotion and
   subsequent layer attention use, same run as native MTP. Capture identities
   only around this diagnostic window; no per-token heavy production audit.

Profile one short replay with nsys CUDA trace and, if useful, ncu on the
selected kernel. Available tools: /usr/bin/nsys,/usr/bin/ncu, compute-sanitizer
(discover exact path). Launch the exact candidate/DSOs with tracing for this
diagnostic; do not time the profiler as production speed. Use existing
launcher or one documented temporary foreground candidate, not another
general harness. Bound capture duration, save trace under /srv/ai results.
Small synthetic kernel timing remains useful if process tracing is unavailable.

Classify prefill vs target verify vs draft generation from actual dispatch.
Existing route counters and backend actual graph counters can be reused.
No new mandatory telemetry collector, endpoint or trace schema just to pass.
A counter unavailable on one route means “not instrumented”, not “no work”.
A snapshot profiler and endpoint timings suffice to diagnose before more
instrumentation is justified.

## C. Scaling admission and anti-waste

No context curve per code change. At29-02 first test32K/16K, then128K allocation
plus short request. Fill only after source/kernel fixes and measured primary
cold use. Add512–2048 new tokens per turn, retaining exact assistant output.
Small U chunks input; it does not mean the user can submit only U tokens.

After every completed turn, atomically persist final SSE, exact messages,
rendered/cached/new/committed tokens, L/C/H/A/B/U, binary+DSO/model/config
identity and the next turn index. Existing run-incremental-scale.py currently
writes its aggregate at the end; fix it to resume safely, with no unconditional
slot clear on resume. Raw SSE and completed requests are authoritative.
A warm process with a different generation/frontier must not accept the
saved “resume” state. Reconstruct missing state at most once, disclose setup.

Pilot two appends and one decode at each larger coordinate. Derive remaining
ETA from NEW-token rates and planned remaining occupancy. Long campaigns
need a declared wall-time budget; initial developer budget15minutes for setup,
not a universal product timeout. If ETA exceeds it, save a partial finding,
profile the smallest affected shape and schedule/perform a source repair.
Progress is not permission to spend hours proving the same slowdown.
Do not restart from zero or increase timeouts merely because progress exists.

No absolute30seconds/1000tokens rule and no frontend graph_replay=0 abort.
Instead require an identifiable change/hypothesis, paired kernels/controls,
and a useful next measurement. An avoidable measured regression routes back
to its specific implementation symbol, not a generic final-audit retry.
Insufficient128K evidence stays partial, never called success.

Discovery of H: reserve actual weights/GDN, full-L draft cache, ALL possible
normal query-shape scratch, summaries, hot layer slabs, rings and headroom.
Try a short normal-U and MTP verification request before long input.
At OOM fix dimensions/late growth first; lower H in page-priced increments,
not L or MTP capacity silently. Record batch/scratch/H Pareto alternatives.
No Cartesian sweep of every context, H, A, U and speculative length.

## D. Final compact evidence

29-01: canonical three questions once per primary mode and viable32K fixture,
q0 repeats for chosen primary; genuine natural cold-page movement/use.
29-02:32K/16K continuation and128K findings with biggest SAFE H, including
allocation-only, partial occupied, and full occupied labels. Stop at source
failure; no forced256K/six-point curve.
29-03: assemble these without rerunning them.

Each row: source/tree+binary/DSOs/model/prompt hash, mode, actual L/C/H_l/A_l,
B/U, full-L draft placement/type, cached/new tokens, pp/tg definitions,
TTFT, output count/EOS, native-MTP draft/accepted counts, setup vs measured
time, memory ledger/peak, and sampled cold source/copy/publication/use proof.
One raw pointer per result, matched ratio only for matched conditions.
Classify controls, diagnostics, partial findings and capability proof explicitly.
Summarize bottleneck from trace, not from overlapping “queue/wait” counters.
Production timing has diagnostics off; the short proof run is labelled separately.

Keep the successful candidate loaded. Use sudo -n on authorized Qwen/8091
lifecycle operations; never touch unrelated8092. Never print credentials.
Store site settings/results in /srv/ai or this execution package, generic code
only in source. Do not mark implementations done by deferring available CUDA.
