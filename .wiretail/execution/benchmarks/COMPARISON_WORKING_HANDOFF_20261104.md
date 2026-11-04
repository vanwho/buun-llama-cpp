# Comparison completion — active working handoff

The user requests completion of all comparison graphs and the remaining valid
measurements, with unexplained outliers investigated. Keep validated historical
cells; never repeat whole families just to align old/new checkpoint coordinates.

## Active run

- Branch: `plan/attention-aware-kv-paging`; latest plan commit `7c4954c06`.
- Frozen plan: `COMPARISON_COMPLETION_20261104.json`, revision 4.
- Output: `/srv/ai/paged-kv/results/comparison-completion-20261104-r4`.
- Luna High runtime owner: `/root/comparison_runtime_prep`. It started the
  detached queue with `nohup setsid`. One model, one lifecycle lock.
- Step 1 (pristine master GPU/MTP-off) is `finding_recorded`, **not complete**.
  Its missing cells remain gaps. Other five steps continue.
- Step 2 (fork CPU-Q4/MTP-off) completed all seven points through C256000.
  C196608/off/prompt3 now has three full 400-token responses: min/median/max
  3.6813 / 3.7312 / 3.7395 tok/s. Its 13 append groups capture the fresh curve.
- Step 3 (fork CPU-Q4/MTP-on) completed its C12288 checkpoint with all 36
  selected low/medium/xhigh requests passing. It is continuing toward C28672.
  Do not restart or interrupt it.
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
