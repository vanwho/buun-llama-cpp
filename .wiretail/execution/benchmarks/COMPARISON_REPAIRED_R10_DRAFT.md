# Repaired comparison campaign R10 — draft, not executable

The user authorized separately versioned repaired builds and Luna runner tests. Root owns the repair decisions and must pin repaired artifact identities and argv before this draft can become executable. This draft does not claim source, binary, DSO, template, or argv identities and launches nothing.

## Fixed comparison protocol

Preserve the R9 request schedule and core configuration: B1024/U256, 16 threads, Turbo4 GPU KV, Q4_0 CPU target, same native MTP draft/quantization, T=0/seed 42, three prompts × four reasoning modes, one 40-token warmup and three measured requests capped at 400 tokens. Use normal cumulative Q/A history, one fresh mapped 32K-chunk fill per configuration series, and actual C accounting beside nominal C. Keep the response-quality gate; no slash-only triplet enters speed summaries.

## Ordered work

1. **Narrow source-fault reproductions.** Use the original frozen R9 chat template for diagnosis. For fork CPU/MTP-on, fresh-fill through C=6,144 then C=12,288; run low P1/P2/P3 in order, then medium P1 through measured repetition 2, stopping at the first confirmed punctuation-only completion. The known baseline failed at medium P1 repetition 2 after a substantive 42-token EOS response in repetition 1. For master GPU/MTP-off, fresh-fill through C=6,144 then issue off/P1: 40-token warmup followed by the first measured 400-token request; the baseline CUDA fault occurred before generated tokens. The master CPU/MTP-on gate remains root-selected against its first-fault record; pin its precise sequence before execution.

2. **Affected full control curves.** After passing the gates, run one fresh mapped fill per family and recover only the listed feasible gaps. Fork CPU/MTP-on uses its R9 L=262,144 primary profile and reviewed startup fallbacks. Master CPU/MTP-on uses L=200,704 primary and L=131,072 alternative; do not retry the known clean-start L=262,144 failure without a separately reviewed startup gate. Master GPU/MTP-off uses L=122,880. Do not rerun complete historical cells.

3. **Template policy.** Keep the frozen R9 template during the initial source-fault reproductions. If root later integrates the prepared append-stable template repair, pin that single reviewed template identity and use it consistently across all repaired full curves. Do not switch templates within a family or between repaired family curves.

## Quality-audited feasible gaps

| Family | Feasible cells | By nominal C |
|---|---:|---|
| fork-cpu-mtp-on | 33 | 12288: 6; 28672: 9; 61440: 9; 126976: 8; 196608: 1 |
| master-cpu-mtp-on | 41 | 6144: 3; 12288: 9; 28672: 9; 61440: 9; 126976: 9; 196608: 2 |
| master-gpu-mtp-off | 42 | 6144: 12; 12288: 10; 28672: 10; 62464: 1; 88064: 6; 120832: 3 |

**Total: 116 selected feasible missing cells.** The 24 C=130,048 GPU capacity exclusions are not in this gap count. The three independent master CPU/MTP-on C=28,672/off/P3 partial cohorts remain separately reported; do not merge repetitions across sessions.

The exact 116 family × nominal-C × mode × prompt cells are listed in this draft JSON. Admit only whole, complete triplets that pass finite-state and narrow response-artifact auditing; retain raw failures separately.

## Unpinned fields

Repaired source commits, binaries, DSO manifests/hashes, any post-diagnostic template, executable paths, literal argv, and final L/H profiles are unset. User authorization is in place; root’s artifact and command pinning is the remaining prerequisite to activation.

## Provenance

Quality audit source: `/srv/ai/paged-kv/results/comparison-final-20261104-quality-audit/comparison-final-20261104.json` (SHA-256 `f3f9bfef2b39d0945c87ad43e357638553e310fd52b5b7eb9343cab34b99af2b`). Frozen reference: `COMPARISON_COMPLETION_20261104_R9.json` (SHA-256 `d94223ec4357e3d9de6fda0adfb18bec4de1a4aef3c5d3bc3793dabf7787fa6e`). Diagnostic sequence source: `/srv/ai/paged-kv/results/candidate-short-repros-r9-20261105/candidate-short-repros.json` (SHA-256 `accc075633bc391167919f4bd1a6457a6f382cfde2c902b881ccbe67983326ff`).
