# ACO preview release handoff — 2026-10-15

## Release identity

- Clean release branch: `release/aco-preview-20261015`.
- Release commit before tag: `f7d78258afa73fb925597683cfb82df4d9033b4f`.
- Intended preview tag: `aco-preview-20261015` (annotated; experimental ACO
  preview, targeted validation only, no full-CI or human-review claim).
- Upstream base: `d2407c33f9595d1badccfd9a82a33c4c5ceda490`.
- Tested production source: `794edc21151d2841add685da0b41e3534cf88e94`;
  source tree `a59a52ac1474fde7836757a2a52fb23894655c42`.
- CUDA server SHA-256:
  `c9041ca96611ed4c9aa06be7b9a23443aa11d4b3bb30983d7dd2558c582659ec`.
  Later release commits are docs/evidence only; production delta from the
  tested build source is empty.
- Public docs: `docs/aco.md`, `docs/aco-validation.md`.
- Public evidence: `tools/server/bench/release_evidence/RESULTS.md` and
  `tools/server/bench/release_evidence/results/fixed/` (JSON/CSV result files,
  summary and long-context export).
- Integration branch preserving upstream merge and ACO changes:
  `integration/attention-aware-kv-paging-upstream-20261015` at
  `21658fa45` (including the narrow dirty-victim repair).

## Build and focused validation

All builds used parallelism 16. The CUDA Release build used `GGML_CUDA=ON`,
CUDA architecture 89, tests enabled, and ccache. The production fix retains
non-droppable physical KV owners outside attention-selection/generation
budgets, then assigns cold pages only to genuinely free or clean-droppable
slots. The transaction's dirty/pin/version safety guard is unchanged; the
focused policy regression passed.

- CPU-only Release CTest: 8/8 focused pager, prefetch, page-select, router,
  server lifecycle and attention tests passed at `bc294606c`; synthetic pager
  contract and speculative MTP state checks passed.
- At `794edc211`, the authoritative dirty-owner admission policy fixture
  passed in CPU and CUDA builds. CUDA-enabled host CTest passed 5/5 focused
  pager, attention, cache-budget and prompt-cache tests.
- Three CUDA fixtures passed: fused paged Turbo4 attention (including long
  prefill regression), CUDA KV page summary, and associated host CUDA checks.
- The portable offline long-curve/evidence-file protection regression has
  four passing cases; it requires no model or service.

This is targeted validation, not the full project test matrix, a model-quality
evaluation, or a human-review certification.

## Runtime evidence and limitations

- The fixed candidate completed all 16 long-fill frontiers through actual
  `C=258,048` without a runtime fault. The earlier pre-fix run failed at
  `C=234,592` with `dirty_victim`; those pre-fix measurements are diagnostic
  only and excluded from the fixed cohort.
- The synthetic long canonical workload rewrites a growing system-source
  prefix and omits prior user/assistant replies; it is a history-rewrite/
  rewind test, not an append-only full-conversation turn. Only 4/12 planned
  canonical rows completed (p0 warmup plus p0 r1/r2/r3); they all passed but
  each reported `cache_n=149,668`, `prompt_n=108,424`, and about 143.4–143.8 s
  measured wall time for the 258,071-token rendered prompt. A p1 warmup was
  interrupted and is not a pass; no p1/p2 rows completed. Do not claim a
  complete 12-row matrix or generalize this replay behavior to ordinary
  append-only turns. The evidence identifies no cause.
- A separate append-only follow-up preserved the full prior transcript and
  passed the 400-token input plus 64-token reserve check. Three single
  observations (not medians): p0 native context 258,508, `prompt_n=54`,
  `cache_n=258,475`, 6,023 ms wall, 84 output tokens; p1 258,620,
  `prompt_n=47`, `cache_n=258,591`, 11,078 ms, 208 output tokens; p2 258,857,
  `prompt_n=49`, `cache_n=258,827`, 13,306 ms, 328 output tokens. A separate
  recovery warmup was 132,749 ms and is not one of those append observations.
  These are not comparable 400-token actual-output trials or a broad
  performance characterization.
- The 3-placement small matrix has 36 measured/warmup rows and remains
  unchanged. Natural cold recall passed: target page 4 [1024,1280) was
  cold-to-resident and selected with the same identity; useful H2D was
  34,603,008 bytes. The singleton witness was page 1 rather than the target;
  this mismatch is reported, with direct inventory/H2D evidence retained.
- For long fill `C=247,694` (frontier 15), metrics and server timing remain,
  but its raw request/SSE was overwritten by the final point. It is explicitly
  marked `raw_artifact_retained=false`; do not imply complete raw provenance.
  The interrupted p1 warmup is also not a successful/pass row.
- Managed ACO profile is healthy on the tested binary at `L=262144`,
  `H=65536`, MTP=2, batch=1024, ubatch=256. Deployment identity, aliases,
  loaded DSOs, and health checks passed. No second model allocation occurred
  during failed restart attempts.

Public claims and the preview tag must preserve these limits. The next
optimization investigation is checkpoint selection/retention for the
synthetic history-rewrite workload; the current evidence does not attribute
the observed latency to cache RAM, pager selection, or transfer behavior.

## Plan/integration preservation

- Private plan branch: `plan/attention-aware-kv-paging`; preserved the
  upstream integration, engineering note, portable tooling/results, and both
  production dirty-victim fix commits (`06d6d48a9`, `4f50d38ca`).
- The clean release ancestry is based on upstream master and contains no
  `.wiretail` paths. No large raw binaries or host-specific campaign payloads
  are included.
- Structural state validation: `PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp
  python3 /srv/wiretail/task_state.py validate` passed (`Valid state: 556
  tasks`). Re-run after this note is committed.
- Push the plan branch and clean release branch/tag normally (no force). Do not
  post an article or PR as part of this handoff.
