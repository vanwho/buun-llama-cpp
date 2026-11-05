# Comparison completion — current handoff, 2026-11-05

## Current result and remaining work

R8/R9 primary queues ended with runtime findings. Do not resume an old queue
blindly. The quality-filtered report retains **805 complete cells / 2,415
measured repetitions**. Both current ACO curves are complete: 264 cells.
Historical controls remain retained; there is no blanket rerun instruction.

There are **116 feasible selected gap-fill cells**, separate from 24 verified
GPU-capacity N/A cells:

- Fork CPU-Q4/MTP-on: 33. C12288: 6; C28672: 9; C61440: 9;
  C126976: 8; C196608: 1.
- Master CPU-Q4/MTP-on: 41. C6144: 3; C12288: 9; C28672: 9;
  C61440: 9; C126976: 9; C196608: 2.
- Master GPU-Turbo4/MTP-off: 42. C6144: 12; C12288: 10; C28672: 10;
  C62464: 1; C88064: 6; C120832: 3.
- Three independent master CPU/MTP-on C28672/off/prompt-3 rechecks each have
  two passing measured repetitions followed by a runtime error. None is a
  complete triplet; do not combine repetitions across independent cohorts.

Fork GPU/MTP-off mapped fill completed through C120832. Master CPU/MTP-on
L262144 clean startup failed before readiness on a 1730 MiB GPU allocation,
with 15900 MiB free and no other compute owner before launch. That is a
confirmed configuration limit, not an unrun measurement or general CPU limit.

The response audit excludes 31 whole fork CPU/MTP-on triplets: R8=27, R9=4.
R8 contributes only C12288/low/P1-P3; R9 adds C126976/medium/P1. HTTP 200 and
SSE completion did not make the slash-only responses valid results. The
classifier is narrow, not a semantic accuracy certification; unavailable
historical response bodies remain explicitly unassessed.

## Required benchmark method and frozen identities

One continuous mapped repository fill per configuration series, with ordinary
cumulative questions and answers. All three prompts share the fill across
reasoning modes. Keep prior content and reasoning verbatim. No benchmark disk
KV save/restore/erase, per-mode refill, context shifting, or shortened output
budgets. Group by nominal C and record actual drift per repetition. Keep
backend, net-new, and HTTP-wall prefill rates separate.

B1024/U256, sixteen threads, Turbo4 GPU K/V, CPU target Q4_0, native MTP n-max2,
the original three prompts and four modes, warmup40 + 3 measured400-token caps.
Exact argv, binaries/DSOs, template and chunk map are pinned in R9 JSON;
its frozen source plan is an archive, not a ready queue after these failures.

- Master binary: `/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server`.
- Fork binary: `/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server`.
- Raw R9 root: `/srv/ai/paged-kv/results/comparison-completion-20261104-r9`.
- Detailed live inventory: `COMPARISON_PROGRESS_20261105.md/.json`.

## Blockers and bounded diagnostics

Fork CPU/MTP-on trace proves target and draft NaNs after several finite
verification steps. Do not assume EOS alone or carry reset is the cause.
Master CPU/MTP-on throws an argmax error that can also mean an invalid
candidate, not necessarily a missing row. Master GPU/MTP-off CUDA-GDB proves
a Turbo4 Flash Attention kernel out-of-range address after a valid warmup.

Engineering notes:
`cpu-mtp-nonfinite-control-20261105.md`,
`normal-append-master-cuda-fault-20261105.md`, and
`checkpoint-retention-prefill-reprocessing-20261105.md`.
They live in `../engineering-notes/` and bind the diagnostic roots.

Paired master/fork CUDA traces now import successfully. Their main bulk-prefill
attention dispatch differs (master F16 compute, fork fused Turbo4). Whole-profile
kernel totals are diagnostic and overlap API waits; do not replace benchmark
rates with them. See `dense-turbo4-prefill-cohort-difference-20261104.md`.

Runtime owner `/root/comparison_runtime_prep` completed the bounded diagnostics;
both CPU/MTP and master GPU faults persisted with CUDA graphs disabled.
All owned diagnostic clients stopped and managed services were restored healthy.
Frozen primary inference builds, flags and six pinned helpers remain unchanged.
A user has authorized separately versioned repaired builds and Luna runner
tests. The frozen-build results remain the historical baseline. Repaired
identities are not yet assigned; use the R10 draft as a campaign outline only,
and do not treat it as an executable or active plan until root pins the repaired
artifacts and argv.

Use one model and `/tmp/ai-pager-benchmark.lock`; protect Jellyfin/ffmpeg and
8091, never touch8092. Restore the managed model and gateway after each run.
Do not render charts during primary CPU timing. Raw payloads, traces and
binaries stay external; commit compact reports and tools to the plan branch.
