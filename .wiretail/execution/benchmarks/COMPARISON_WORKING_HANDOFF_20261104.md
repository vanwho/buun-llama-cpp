# Comparison completion — current handoff, 2026-11-05

## Current retained result

R8/R9 remain historical baseline campaigns. Quality audit retains **805 usable complete cells / 2,415 measured repetitions** across the ten comparison families. The ACO ON/OFF curves remain complete at 264 cells total. The response artifact filter excludes 31 whole fork CPU/MTP-on triplets; 189 source triplets lacked saved raw SSE and remain unassessed. The narrow filter is not semantic answer validation.

There are **116 feasible selected gap cells** and **24 separate GPU capacity N/A cells**:

- Fork CPU-Q4/MTP-on: 33 gaps — C12288: 6; C28672: 9; C61440: 9; C126976: 8; C196608: 1.
- Master CPU-Q4/MTP-on: 41 — C6144: 3; C12288: 9; C28672: 9; C61440: 9; C126976: 9; C196608: 2.
- Master GPU-Turbo4/MTP-off: 42 — C6144: 12; C12288: 10; C28672: 10; C62464: 1; C88064: 6; C120832: 3.
- Master/fork GPU/MTP-off each have 12 capacity-N/A cells at C130048; these are not missing runnable measurements.

See `COMPARISON_PROGRESS_20261105.md/.json` and compact result snapshot at `../evidence/COMPARISON_20261105/`. R9's three independent master CPU/MTP-on C28672/off/P3 cohorts each contain two measured repetitions plus a runtime error; never combine them into a triplet.

## Latest source/build direction

Use pristine upstream master source commit `958b444c677ad9b02058d5fbf8fd0e13b7d24286` for the next master build. No source patches, hooks, overlays, or benchmark-specific inference edits are allowed in that build; approved host-toolchain compatibility is allowed. The pristine worktree is `/srv/ai/paged-kv/worktrees/master-pristine-958b444c677a`. Its new binary and loaded DSO identities are pending build completion. The prior master repair worktree/build is retired.

Fork temporary finite-diagnostic edits have been reverted. The fork source is again `8a79008b5473676bb99a54cc1505d11a4e107357`. Wait for pristine-master startup and short-check results before deciding whether a fork repair is warranted. Do not change ACO code without a concrete demonstrated defect.

Start with pristine-master startup and narrow checks, then address the missing master controls. Defer fork CPU/MTP-on work until the master findings guide a source-directed fix. Root owns build identity and launch pinning. The user has already authorized separately versioned repaired-build investigation and Luna runner tests; no additional user approval is pending.

R8/R9 late traces, runtime faults, capacity observations, and punctuation-only outputs remain evidence tied to their recorded builds. Treat them as historical/provisional symptoms, not a proven single cause or a repair. Do not replay retired master diagnostic launches against the new binary. Preserve the 805 retained-cell and 116-gap baseline until new quality-qualified measurements are indexed.

R10 outline: `COMPARISON_REPAIRED_R10_DRAFT.md/.json`. It is non-executable pending root pinning of actual binaries/DSOs and exact argv. It sequences pristine-master startup/short checks before master gaps, then gates fork work on master findings. Preserve the current normal cumulative conversation, one mapped fill per configuration, same prompts/modes, warmup/repetition/output schedule, actual-C recording, and whole-triplet response-quality gate.

## Historical measurement identities and method

R9 master binary `/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server` and R9 fork binary `/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server` are historical identities only. Raw R9 root: `/srv/ai/paged-kv/results/comparison-completion-20261104-r9`.

Comparison protocol: B1024/U256, 16 threads, Turbo4 GPU KV, Q4_0 CPU target, native MTP n-max2, T=0/seed42, original three prompts and four modes, one 40-token warmup plus three measured requests capped at 400. Use ordinary cumulative Q/A history, retain reasoning, one fresh mapped 32K chunk-map fill per configuration, and group by nominal C while recording actual C per request. Keep backend, net-new, and HTTP-wall prefill rates separate. No context shifting, output shortening, or benchmark-owned slot save/restore/erase.

The quality-filtered external full report and chart files are referenced by path/hash in the snapshot manifest. Raw SSE, answer text, binaries, and model files remain external. Do not render charts during timed CPU benchmark work; restore managed services and protect Jellyfin, 8091, and 8092 under the runtime lifecycle procedure.
