# Execution evidence index

This is the foundation-series evidence index for phases 00–07. It separates
deterministic implementation evidence and bounded CUDA checks from the model-
backed acceptance that was deferred in Phase 06. Production-completion phases
08–15 are now queued in `WORK_STATE.json`; they must produce `INDEX_V2.md` and
`FINAL_ACCEPTANCE_V2.*` after the live selective pager, placement ledger,
transfer telemetry, and frozen expected-answer stream exist.

No result below is a release claim. External result directories are named for
reproduction and remain outside this repository.

## Source, state, and review package

- [Execution state](../WORK_STATE.json) - task order, status, and dependencies.
- [Execution plan](../ATTENTION_AWARE_KV_PAGING_PLAN.md) - architecture,
  contracts, gates, and non-goals.
- [Semantic contract](../design/SEMANTICS.md) - off/observe/selective/exact
  behavior, identity, publication, and fallback rules.
- [Upstream slice map](../upstream/SLICE_MAP.md) - source SHAs, exact hunk
  ownership, destinations, tests, risks, and rebase procedure.
- [Phase 00 provenance](../baseline/PROVENANCE.md) - source, model, build,
  geometry, hardware, and benchmark identity.
- [Phase 00 baseline results](../baseline/BASELINE_RESULTS.md) - valid controls
  and their measured summaries.
- [Salvage matrix](../baseline/SALVAGE_MATRIX.md) - prior implementation and
  community references with generic/Buun ownership decisions.
- [Work log](../WORK_LOG.md) - chronological task outcomes and commands.

## Baselines and raw result directories

The four canonical raw result directories are outside Git under
`/srv/ai/paged-kv/results/`:

| Run | Directory | Status | Use |
| --- | --- | --- | --- |
| Fast/default all-GPU target and MTP | [`profile-benchmark-default-fast-20260903-055158/`](/srv/ai/paged-kv/results/profile-benchmark-default-fast-20260903-055158/) | available control | 77,824-token Turbo4 target/MTP comparison |
| Fast/large scratch control | [`profile-benchmark-large-fast-20260903-055411/`](/srv/ai/paged-kv/results/profile-benchmark-large-fast-20260903-055411/) | available control | approximately 44K input, five measured trials |
| Ordinary full-context CPU-KV | [`ordinary-cpu-kv-full-20260903/`](/srv/ai/paged-kv/results/ordinary-cpu-kv-full-20260903/) | available control | 262,144-token `--no-kv-offload` comparison |
| Big/spec-off feature control | [`profile-benchmark-default-big-20260903-065012/`](/srv/ai/paged-kv/results/profile-benchmark-default-big-20260903-065012/) | available control | same-build rollback control; not a pager-off pair |

Each valid profile result contains the canonical `run-config.json`,
`records.jsonl`, summaries, raw responses, and service/profile metadata as
described in [BASELINE_RESULTS.md](../baseline/BASELINE_RESULTS.md) and
[FINAL_ACCEPTANCE.json](FINAL_ACCEPTANCE.json). The external paths are not
staged or copied into this repository.

## Deterministic CPU and fake-backend evidence

- [CPU test matrix](CPU_TEST_MATRIX.md) - 16 focused targets and the complete
  requirement-to-test mapping.
- [Task 06-01 handoff](../handoffs/06-01.md) - fake transfer, rollback,
  lifecycle, and feature-off results.
- [Task 02-01 handoff](../handoffs/02-01.md) - bounded selected-page capture.
- [Task 02-02 handoff](../handoffs/02-02.md) - canonical host catalog.
- [Task 02-03 handoff](../handoffs/02-03.md) - asynchronous transfer seam.
- [Task 02-04 handoff](../handoffs/02-04.md) - transactional publication and
  rollback.
- [Task 02-05 handoff](../handoffs/02-05.md) - derived geometry and fixed
  304-page window.
- [Task 03-01 handoff](../handoffs/03-01.md) - compact selected view.
- [Task 03-02 handoff](../handoffs/03-02.md) - backend-neutral metadata.
- [Task 03-04 handoff](../handoffs/03-04.md) - graph and prefill/decode
  lifecycle.
- [Task 04-01 handoff](../handoffs/04-01.md) - all-page routing summaries.
- [Task 04-02 handoff](../handoffs/04-02.md) - bounded attention telemetry.
- [Task 04-03 handoff](../handoffs/04-03.md) - hot-set policy.
- [Task 04-04 handoff](../handoffs/04-04.md) - prefetch and backpressure.
- [Task 05-01 handoff](../handoffs/05-01.md) - hybrid/recurrent atomicity.
- [Task 05-02 handoff](../handoffs/05-02.md) - MTP fit and placement boundary.
- [Task 05-03 handoff](../handoffs/05-03.md) - server slot generations.
- [Task 05-04 handoff](../handoffs/05-04.md) - checkpoint and speculative
  rollback.

The local build directories named by the handoffs (`build/`,
`build-cuda-test/`, and task-specific `/tmp` directories) are generated
verification outputs, not source evidence to stage.

## CUDA and exact-reference evidence

- [CUDA test matrix](CUDA_TEST_MATRIX.md) - direct Turbo4 page kernel, VMM,
  graph-key, sanitizer, and fault dispositions.
- Raw CUDA logs: [`cuda-fixture.log`](cuda-fixture.log),
  [`cuda-toolkit-version.log`](cuda-toolkit-version.log),
  [`cuda-sanitizer.log`](cuda-sanitizer.log),
  [`cuda-racecheck.log`](cuda-racecheck.log),
  [`cuda-vmm.log`](cuda-vmm.log),
  [`cuda-vmm-sanitizer.log`](cuda-vmm-sanitizer.log),
  [`cuda-graph-key.log`](cuda-graph-key.log),
  [`cuda-server-faults.log`](cuda-server-faults.log),
  [`cuda-resource-usage.log`](cuda-resource-usage.log), and
  [`cuda-kernel-resource-summary.log`](cuda-kernel-resource-summary.log).
- [Exact reference](EXACT_REFERENCE.md) - online-softmax `(m,l,o)` merge,
  page coverage, cold staging, and partial CUDA output.
- [Exact live checkpoint](EXACT_LIVE_CHECKPOINT.md) - canonical host-page
  inventory, bounded live preflight ledger, verification, and deferred graph
  binding checks.
- [Exact CUDA fixture log](exact-cuda-fixture.log) - bounded partial-state
  execution and memcheck evidence.
- [Task 06-02 handoff](../handoffs/06-02.md) - CUDA scope and deferred live
  model integration.
- [Task 06-05 handoff](../handoffs/06-05.md) - exact route and page-wave
  executor scope.

The CUDA results are bounded fixture results on the recorded RTX 4080 sm_89
environment. They do not prove full Qwen3.8 graph execution, 304-page live
allocation, or production transfer ownership.

## Acceptance decision and frozen controls

- [Final acceptance decision](FINAL_ACCEPTANCE.md) - selective capacity,
  quality, and speed acceptance is deferred.
- [Machine-readable acceptance](FINAL_ACCEPTANCE.json) - frozen sampling,
  corpus identity, control paths, required runs, gates, and next inputs.
- [Task 06-03 handoff](../handoffs/06-03.md) - canonical profile benchmark
  adapter and explicit `not_configured` pager counters.
- [Task 06-04 handoff](../handoffs/06-04.md) - acceptance decision and health
  gate disposition.

Frozen controls use greedy sampling (`temperature=0`, `seed=42`) and
`pager-corpus-v1`. Expected needle answers are not configured. The target
acceptance thresholds remain: warm-focus median decode at least 2x ordinary
CPU-KV and within 20% of comparable 77,824-token all-GPU after warmup; report
the 3x stretch result; and meet the predeclared needle/conversation quality
thresholds. None of these selective gates has been promoted to passed.

## Deferred verification register

| Deferred check | Evidence/status | Required later input |
| --- | --- | --- |
| Live `observe` and selective pager runtime | No production pager route or telemetry endpoint; deferred in [FINAL_ACCEPTANCE.json](FINAL_ACCEPTANCE.json) | Pager-enabled model server exposing mode, placement, epochs, counters, and resource ledgers |
| 262,144 logical target with about 304 hot pages | Model metadata and deterministic 304-page arithmetic only; no live pager ledger | Qwen3.8 model-backed run with target allocation and page ledger |
| Canonical Turbo4 host backing for all cold pages | CPU/fake capture/catalog tests passed; model-backed run absent | Live D2H/H2D transfer binding, host catalog, and checksum reconciliation |
| Full-length Turbo4 MTP in GPU memory under target pressure | 77,824-token control and deterministic fit checks recorded; full pressure run absent | Native MTP CUDA run with placement and teardown inspection |
| Warm-focus, cold-needle, focus-shift, and churn throughput | No selective trials; no claim | Frozen `pager-corpus-v1`, external runner, and pager telemetry |
| Needle and conversation quality | Expected answers and selective outputs not configured | Frozen expected-answer stream and threshold harness |
| Feature-off paired regression | Existing spec-off control is not a paired pager-off run | Same-build paired pager-off/selective comparison |
| Full exact 256K page-wave execution | Bounded CPU/CUDA reference passed; graph/catalog integration absent | Production graph callbacks, host catalog, CUDA streams, and raw latency/transfer evidence |
| Generic upstream extraction | llama.cpp fork is 11 commits behind current upstream and generic draft port has no tip SHA | Clean fast-forwarded worktree, maintainer direction, and human-authored upstream communication |

Deferred status is intentional. It must not be changed to passed by changing
documentation alone.

## Operator documentation

- [Server operator guide](../../../tools/server/README.md) - VBR, internal pager
  boundary, memory accounting, failure behavior, and telemetry limitations.
- [Speculative decoding guide](../../../docs/speculative.md) - draft K/V device
  placement, Turbo4 aliases, native MTP scope, and fail-closed behavior.
- [Pager benchmark adapter guide](../../../tools/server/bench/README.md) - dry
  runs, variant names, manifest contract, and `not_configured` counters.
- [07-01 handoff](../handoffs/07-01.md) - upstream slicing and rebase state.

The generated option tables in the CLI and server guides are parser output;
future option changes must regenerate them. The pager mode and page-tuning
switches are not currently public CLI options.

## Task handoffs

The complete task handoff chain is available here:

- [00-01](../handoffs/00-01.md), [00-02](../handoffs/00-02.md), [00-03](../handoffs/00-03.md), [00-04](../handoffs/00-04.md), [00-05](../handoffs/00-05.md)
- [01-01](../handoffs/01-01.md), [01-02](../handoffs/01-02.md), [01-03](../handoffs/01-03.md), [01-04](../handoffs/01-04.md)
- [02-01](../handoffs/02-01.md), [02-02](../handoffs/02-02.md), [02-03](../handoffs/02-03.md), [02-04](../handoffs/02-04.md), [02-05](../handoffs/02-05.md)
- [03-01](../handoffs/03-01.md), [03-02](../handoffs/03-02.md), [03-03](../handoffs/03-03.md), [03-04](../handoffs/03-04.md)
- [04-01](../handoffs/04-01.md), [04-02](../handoffs/04-02.md), [04-03](../handoffs/04-03.md), [04-04](../handoffs/04-04.md)
- [05-01](../handoffs/05-01.md), [05-02](../handoffs/05-02.md), [05-03](../handoffs/05-03.md), [05-04](../handoffs/05-04.md)
- [06-01](../handoffs/06-01.md), [06-02](../handoffs/06-02.md), [06-03](../handoffs/06-03.md), [06-04](../handoffs/06-04.md), [06-05](../handoffs/06-05.md)
- [07-01](../handoffs/07-01.md), [07-02](../handoffs/07-02.md)

The task packet and state file remain authoritative for acceptance and status.
