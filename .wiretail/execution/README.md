# Current hot-KV execution package

The active forward contract is [OVERVIEW](forward/OVERVIEW.md),
[PREFILL101](forward/PREFILL101.md) and [TESTING](forward/TESTING.md).
Every build on this host uses CMake `--parallel 16`, for CPU and CUDA alike.
Do not change the job count without operator direction; diagnose slowness or
resource failures first. See [CONTEXT_POLICY](CONTEXT_POLICY.md) and
[TESTING](forward/TESTING.md).
Continue the first unfinished task in WORK_STATE, never a historical phase ID.
105-01g through105-05 are completed. The current sequence is105-06 through
105-20: freeze the benchmark harness and exact server configurations once,
then run the unchanged old-profile GPU, old-profile CPU-KV, ACO and context
matrix;105-21 aggregates the three median tables per setup and fresh-prefill
curve. Read only the `forward105-matrix` cluster and current task packet for
that sequence. No source, benchmark, build or performance-config changes are
allowed between measured configurations.105-01a/b/c/d are historical;
105-01d explicitly declined adoption. Read current packet/cluster, immediate
handoff and the compact task-specific engineering note only.105-02 onward
uses [selected generation](engineering-notes/selected-generation-contract.md).
No unchanged256K semantic retry. Correct response-local MTP accounting and
coherent same-map generation precede new ranking tuning or scale work.

Preserve B1024/U256, Turbo4 target/draft,
frozen GPU history and inclusive host storage. Never run two Qwen models or
change the production kernel for diagnostic counters. Preserve historical
usage/raw artifacts without treating old failed oracles as release evidence.
Before model-backed testing, take `/tmp/ai-pager-benchmark.lock`, verify the
managed Qwen slot is idle, and inspect GPU compute owners with
`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.
The operator authorizes stopping the Immich machine-learning Docker worker
when it competes for GPU space. Resolve its exact container from Docker's
name/image and PID before `docker stop`; do not stop unrelated containers.
Do not restart a worker the operator already stopped. Stop the one managed
Qwen owner before a standalone model fixture, wait until its CUDA allocation
is gone, then run only one Qwen process. Restore the matching managed argv
and verify actual port8080 health afterward. A leftover external GPU owner
or second model is a setup issue, not permission to alter paging routes,
quantization, batch sizes or admitted-H calculations.
Main adoption is a reviewed source slice, never a blanket dirty-worktree merge.
Unstarted93-12/93-13 are removed from the runnable graph;
their old packets are archived for provenance. Forward tasks supersede the
old accepted-token historical refresh and universal attention-route preference.
Only current packets/selected source regions and compact handoffs are loaded.

Read the [long-horizon context and usage policy](CONTEXT_POLICY.md) first. For
execution, trust the current task in `WORK_STATE.json`, its packet/cluster, and
its explicit `context_files`; those identify the active design slice. Do not
assume an older V10/V9 plan or phase summary is current merely because it is
linked from historical evidence. Wiretail injects `CONTEXT_POLICY.md` into
each task and assessment prompt, while task-specific context remains bounded.

Raw run evidence under `evidence/raw/` is host-local and must never be added
to Git, even if a receipt references it. Keep compact handoffs, summaries,
manifests, and receipts in the repository; put bulky request logs, profiler
captures, and candidate binaries in the external results area and record their
path and hash in the handoff. Do not force-add ignored raw evidence.

```bash
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
PROJECT_BRANCH=plan/attention-aware-kv-paging /srv/wiretail/wiretail.sh
```

The project model defaults, family IDs, retry assessors, and final repair
attempt are configured in `.wiretail/models.json`. Wiretail uses GPT-6.1 Sol
high to assess a failed attempt 2, then GPT-6 Luna xhigh to assess a failed
attempt 3; the final attempt runs on GPT-6 Luna high. The task prompt prefix is
carried into each assessment and retry. Task/state execution-model locks may
override those project defaults; the new phase101 task metadata pins Luna
High for code/state work and Medium for the procedural benchmark. Existing
project-wide model-map edits are preserved by this planning correction.

Each bounded task explicitly lists `context_files`. Read that list, the current
packet/cluster, repository instructions, and the current task handoff if
present. Do not recursively read old plans, the full state/log, old acceptance
gates, or every dependency handoff. Every task has its own named-proof
completion contract; follow the contract linked by that task.

V9 and older plans/packets/handoffs/results remain historical, not active.
Do not PR any execution metadata/history upstream. The historical cleanup
inventory remains in `v9/OPERATIONS.md` for a future publication-only task;
do not load it in current implementation sessions. The V10 repair/testing
documents are historical design references unless a current task names them.
Portable code commits remain separate.

Task and phase token totals are Codex-reported `turn.completed` usage
aggregates, not the size of one prompt or active context window. See
[`CONTEXT_POLICY.md`](CONTEXT_POLICY.md) for field semantics, compaction, and
the handoff format.

Automatic selective paging is fast-route-only. `selected_reference` is a
temporary explicit correctness oracle, never a production fallback or valid
speed result. Unsupported automatic shapes must be refused with a reason until
their GPU-native dense, direct Turbo4, or bounded packed route is implemented;
see task [92-01](tasks/92-01.md).

The final review reads only the new benchmark summary. If the goal is unmet,
it must append actual remediation/benchmark/summary/review task entries before
completing, not merely recommend another action in prose.

## Local-only benchmark assets and tooling cleanup before upstream PR

The phase-102 repo-context fixture and the local GPU101 release-validation
helpers are intentionally kept in separate commits so they can be excluded
cleanly from a later upstream code PR. These paths are campaign/test data or
server/project-specific validation tooling, not portable Buun product
changes. Remove them from the upstream PR commit range (or relocate them to
external benchmark tooling) during final cleanup:

- `tools/server/bench/fixtures/repo-context-v1/README.md`
- `tools/server/bench/fixtures/repo-context-v1/manifest.json`
- `tools/server/bench/fixtures/repo-context-v1/prompts.md`
- `tools/server/bench/canonical_result_check.py`
- `tools/server/bench/test-gpu101-release.py`
- `tools/server/bench/validate-gpu101-release.py`
- `tools/server/bench/verify-gpu101-retake.py`
- October7 promotion-harness-only commits, including `cae9cc06e` and the
  subsequent query/page attribution repair: `run-pager-promotion.py` and
  `test_pager_promotion.py`. Keep these local campaign fixes separate from
  portable attention/MTP/kernel/residency changes when selecting upstream code.

Also remove or rewrite references to these assets/helpers in phase-102
packets, clusters, overview, testing plan, and evidence before preparing the
upstream PR. Preserve the local commits/history until that publication
cleanup is performed; do not include these paths in an upstream PR.
