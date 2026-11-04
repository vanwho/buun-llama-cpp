# Comparison completion — active working handoff

The user requests completion of all comparison graphs and the remaining valid
measurements, with unexplained outliers investigated. Keep validated historical
cells; never repeat whole families just to align old/new checkpoint coordinates.

## Active run

- Branch: `plan/attention-aware-kv-paging`; newest tool fix `7e360cf9b`.
- Frozen plan: `COMPARISON_COMPLETION_20261104.json`, revision 5, pending launch.
- Output: `/srv/ai/paged-kv/results/comparison-completion-20261104-r5`.
- Luna High runtime owner: `/root/comparison_runtime_prep`. It started the
  detached queue with `nohup setsid`. One model, one lifecycle lock.
- R4 was stopped safely; no test-owned model/queue remains. R5 must never clone
  target-only slot files before an MTP-on question. It keeps paired resident
  checkpoints, unchanged model/binaries/quants/B/U/threads, and the same map.
- The R4 fork CPU-Q4/MTP-off series completed all seven points through C256000.
  C196608/off/prompt3 now has three full 400-token responses: min/median/max
  3.6813 / 3.7312 / 3.7395 tok/s. Its 13 append groups capture the fresh curve.
- R4 fork/master CPU-MTP-on used target-only files that reset native draft
  history; both are quarantined from the normal resident-MTP comparison. Their
  raw observations remain external as diagnostic data; do not republish the
  R4 8.136 tok/s value as the normal full-prefix MTP control.
- R5 order: fork CPU MTP-on, master CPU MTP-on, master CPU MTP-off, fork GPU
  MTP-off conditional endpoint, master GPU MTP-off checkpoint-free recovery.
- The queue now pauses for assessment before a whole-family rerun; use continue
  for evidenced statistical causes, full-curve for an unexplained anomaly.
- B1024/U256, t16/tb16, GPU model weights, target Turbo4 on GPU and Q4_0 on CPU,
  one persistent new-repo-chunk fill per configuration; no refill per mode/prompt.
- Protect Jellyfin/ffmpeg, 8091, and 8092. Restore managed gateway/model services.

## Pristine-master failure, still unresolved

C6144 crashes with CUDA illegal memory access on the second independent
question, around creation of a recurrent context checkpoint. The same failure
exists in historical L8192 runs and new L122880/L131072 runs. It is **not an
established allocation limit**. Both native `resume` and `legacy` slot formats,
and resident-prefix rewind, reproduce it. No inference binary was changed.
The harness now calls a truncated HTTP-200 SSE stream a runtime fault rather
than falsely reporting a prefix-cache miss.

Next narrowly justified master recovery: disable optional checkpoint creation
and prompt-cache retention with `--ctx-checkpoints 0 --cache-ram 0
--no-cache-idle-slots`, retain full legacy base-state restore between independent
questions, and leave CUDA graphs, FA, quants, B/U, threads, model and binary
unchanged. Do not mutate the active queue's hashed driver files; prepare this
after the current queue is finished or deliberately paused safely.

Source justification in pristine worktree
`/srv/ai/paged-kv/worktrees/master106-d2407c33f959`:

- `tools/server/server-context.cpp:21485`: periodic capture requires a positive
  checkpoint count. Typed SWA capture is also disabled when this count is zero.
- `tools/server/server-context.cpp:19468`: native legacy restore independently
  applies the full saved target sequence state.
- `tools/server/server-context.cpp:21904`: checkpoint partial-state serialization
  synchronizes CUDA. The crash surfaces there; this is a plausible interaction,
  not proof that serialization itself caused the faulty kernel access.
- `src/llama-context.cpp:7039`: full sequence-state read remains available.
- There is no public switch to purge only the CUDA graph cache. Do not disable
  CUDA graphs and silently publish that as the full-speed control.

## Additional master MTP / supervisor findings

Master CPU-Q4/MTP-on emits `DFlash target argmax row lookup failed` in some
requests; the message belongs to a shared native-MTP sampling shortcut, not a
wrong loaded draft architecture. Failed SSE streams often end at 38 of a
40-token warmup or 396/398 of the 400-token measured budget. Keep the original
budget; do not raise it to conceal a runtime error. `logprobs:true` with
`top_logprobs:1` bypasses the shortcut (`common/sampling.cpp` requires n_probs=0)
but adds sampling/transfer overhead, so those would be separate diagnostic
results, not a silent replacement of the requested baseline.

Both supervisor issues were fixed for R5:

- `run-context-matrix-cell.py` aggregates every stage, not only the last one.
- `run-comparison-series.py` pauses on flags before an assessed full rerun.
- Audit existing queue receipts against per-stage/per-cell results; repair any
  false `complete` projection from evidence, retaining original status/history.

## Reports and outliers

- Complete current ACO MTP-on/off curves: `evidence/ACO_FULL_CURVES_20261104.json`.
  Both have 11 points, 12 prompt/mode cells per point, three measured repeats.
- Tracked historical control index: `COMPARISON_COVERAGE_20261104.json`.
- Outlier audit: `/srv/ai/paged-kv/results/comparison-outlier-audit-20261104.json`,
  SHA256 `d7e2a93f1897c22784aa9e78a660532a55c0df8b0f2a3e0e0540c5eac71bf50a`.
- No >30% repeat spreads or >35% adjacent decode jumps in the 264 current ACO
  cells. Small prefill dips at C62465 and C196608 are 1009/993-token checkpoint
  tails, not comparable 32K bulk groups; label them.
- Master CPU/MTP-on C28672/off/prompt3 has a 41% repeat spread correlated with
  MTP acceptance 63% versus 30%. R4 explicitly rechecks this one historical cell.
- Fork CPU MTP-on/off C196608/off/prompt3 previously produced only two tokens;
  each variant has a targeted four-request recheck already in R4.
- Report generator (new, not yet committed):
  `report-comparisons-20261104.py`, owned by `/root/aco_docs_refresh_20261104`.
  Draft outputs: `/srv/ai/paged-kv/results/comparison-final-20261104`.
  Do not render plots/install plotting dependencies during CPU timing runs.
- Expected historical cells before fresh additions: 731 total across ten
  families; GPU MTP-on master=72, fork=60. Earlier 12/15 counts were a parser
  bug that lost modes. CPU legacy null mode is verified off in saved requests.
- Final plots must use a standard scientific plotting library, actual occupancy,
  min/median/max, separate backend/net-new/HTTP-wall ingest metrics, provenance,
  and explicit unsupported/missing cells. No interpolation or mixed-sample bests.

Local benchmark-tool commits are separate from portable inference code:
`f130f74a4`, `367a32050`, `98b9c8b23`. Raw runs remain outside Git.
