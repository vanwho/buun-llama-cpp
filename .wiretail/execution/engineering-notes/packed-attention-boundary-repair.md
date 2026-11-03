# Packed-attention hot-boundary repair — 2026-11-03

The independently repeated H=65,536 boundary cliff is a consumer regression,
not evidence of per-token ranking or repeated host promotion. Selection,
query replay and history freezing already happen at the user-turn boundary.
Current-token writes still have to become visible to attention during generation.

## What changed and why it was expensive

Commit `a6fcbebdb80e0e5ccd812644fda05df58a6d9833` replaced the persistent,
per-layer packed duplicates with one shared layer workspace to avoid the extra
H-sized K/V allocation in every attention layer. Keep that VRAM saving.
However, `llm_graph_context::build_attn()` then emitted one K and one V copy
for every selected page in every layer. At 256 selected pages this is 512 copy
operations per layer, plus their planning and dependency views.

`llm_graph_input_attn_kv::refresh_selected_data()` also compared the baked copy
row count with each page's new valid length. A changing partial tail therefore
rejected graph reuse even without new selection or a host transfer. Changes to
physical membership and compact offsets had further structural rejection paths.

The baseline repeat measured approximately 42.2 to 15.0 tok/s without MTP,
and 78.85/56.26/72.25 to 43.09/30.98/40.09 tok/s with MTP (three prompts),
when moving from C=62,465 to 78,849 at fixed L=262,144/H=65,536. Some slow
requests had zero H2D promotion bytes. See `context-speed-deviation-audit-20261103.md`
and its JSON for hashes, raw paths, all repeated values and sampling limits.

## Implemented execution change

- `ggml_kv_pack()` / `GGML_OP_KV_PACK` copies native encoded rows on CUDA in
  one batched operation for K and one for V per layer. It does not dequantize
  Turbo4, transfer history through CPU RAM, or introduce a host fence.
- One small mutable I32 descriptor contains `{source_row_begin, valid_rows}`
  per selected page. Destination page buckets have fixed size. Appends,
  rollback, and physical remaps update descriptors and causal-mask values,
  rather than creating per-page graph nodes. A changed page-count capacity or
  query shape can still legitimately require another graph.
- `llama_kv_attention_packed_page_layout()` generates both the pack descriptor
  and native-position/validity vectors in that same bucket order. It validates
  source and destination bounds before upload. Invalid bucket rows remain
  masked, and the GPU pack operation explicitly zeroes their encoded bytes.
  Canonical source tails are not assumed initialized and are never read.
- Graph edges explicitly connect `SET_ROWS -> KV_PACK -> attention`. Keep the
  existing current-row dependency, finite-tail and absolute-layer-base repairs
  from `52d04942c`; reading raw cache aliases or copying arbitrary full source
  pages would reintroduce their correctness failures.
- Keep one shared encoded staging pair for all layers, bounded by selected
  physical pages rather than logical L. The graph arena is bounded per layer,
  eliminating the old page-count multiplier on CPU graph metadata as well.
- CPU implementation is a byte-copy oracle. It does not enable TurboQuant CPU
  attention. A new RPC minor version refuses incompatible older peers.

Ranking, admission, eviction, turn freezing, host page storage, MTP policy,
precision, hot size and production route selection are unchanged. The packed
consumer still gathers its selected encoded rows on-device; it now does so
without a per-page executable copy plan. No persistent per-layer H duplicate
has been restored.

## Validation protocol

Focused CPU and CUDA tests exercise multi-head byte-exact packing with poisoned
source tails, changed descriptors on the same graph, remaps, growth, rollback,
and zero invalid tails. Native causal-mask tests include multiple page buckets
and partial tails. Rebuild with `--parallel 16`.

The live comparison uses the same original three prompts, reasoning off,
L=262,144/H=65,536, Turbo4 K/V, B=1024/U=256, MTP off/on, and C=62,464 then
78,848. Each prompt has one 40-token warmup and three 400-token measurements.
Only fresh repository-file chunks grow the context. Do not change settings,
force routes, lower H, or mix binaries between the paired checkpoints.

The old binary/DSOs are preserved outside Git at
`/srv/ai/paged-kv/builds/fork-before-packed-cliff-20261103T164201Z/bin`.
Use a new frozen manifest/output root for the repaired candidate; preserve the
old raw results and classify them as measurements of the old implementation.

## Measured outcome

Runtime fix: `ca54933a7c43c02c1f6c56b39279cea08ece032a`.
Rebuilt server SHA-256:
`f35732397edd9ba1ee7b18087ef7f17e0b6a774fdc47a0137e695c65787067a7`.
All 48 live requests passed (36 measured and 12 warmups). The CPU execution
fixture, CUDA byte-exact execution fixture, and native CUDA paged-attention
fixture passed. No batch, hot capacity, precision, MTP or route override was
changed during the paired runs.

| C / MTP | Prompt 1 decode | Prompt 2 decode | Prompt 3 decode |
|---|---:|---:|---:|
| 62,465 / off, old | 42.36 | 42.37 | 42.22 |
| 62,465 / off, repaired | 42.35 | 42.34 | 42.21 |
| 78,849 / off, old | 14.91 | 15.00 | 15.00 |
| 78,849 / off, repaired | 37.42 | 37.64 | 37.57 |
| 62,465 / on, old | 78.85 | 56.26 | 72.25 |
| 62,465 / on, repaired | 78.79 | 56.39 | 72.25 |
| 78,849 / on, old | 43.09 | 30.98 | 40.09 |
| 78,849 / on, repaired | 70.56 | 50.40 | 65.97 |

Values are three-repetition median output tok/s. Actual base occupancy is
one token above each requested checkpoint, identically in both candidates.
At the post-H point the repair is about 2.50x faster without MTP and 1.63–1.65x
faster with MTP. A smaller, approximately 9–12% pre/post-H difference remains;
this repair does not make packed attention a zero-copy dense view. The shared
workspace still gathers encoded rows on-device.

Post-H MTP acceptance medians are 90.12% / 49.37% / 80.92%, unchanged from the
old paired run. Before H they are 91.67% / 51.74% / 81.25%. Sampled peak GPU
usage was 14,569 MiB without MTP and 15,685 MiB with MTP. At the post-H point,
request-window GPU utilization samples rose from median 55% to 96% without
MTP; sample counts differ because the repaired requests finish faster.
This is evidence of recovered utilization, not a precise kernel-time profile.

Raw build/test and live artifacts, including a frozen manifest with model,
template, executable and DSO hashes, are under
`/srv/ai/paged-kv/results/forward/packed-decode-cliff-20261103/`.
The compact `packed-attention-boundary-repair.json` records exact per-prompt
medians and hashed artifact references for both builds.

The full larger-context/four-reasoning-mode matrix is pending on this fix. The
queue reruns every ACO checkpoint, including points previously measured on the
old implementation; those full-matrix results are historical and superseded for
current repaired-build comparisons. The checkpoint ladder stays unchanged.

## Delivery

The benchmark queue pins the stable runtime bundle at
`/srv/ai/paged-kv/builds/aco-packed-repair-ca54933a7/bin`, rather than mutable
compiler output. It uses a fresh output root and slot state. Both ACO variants
now queue all four modes and all three prompts at all ten established checkpoints
(120 mode/prompt pairs and 480 requests per variant; 960 total). These full-matrix
runs are pending. All earlier ACO full-matrix rows are superseded for current
repaired-build comparisons; their raw artifacts remain archived as historical
results of the prior implementation. The 48-request focused boundary comparison
above remains a valid paired old/repaired result for C=62,464 and C=78,848.

The four Open WebUI ACO aliases now resolve to that same tested bundle on their
next profile selection, with L262144/H65536/B1024/U256/Turbo4/GPU-MTP2 unchanged.
The previous profile is backed up under
`/srv/ai/paged-kv/builds/aco-profile-backups/`. The active `qwen38-fast` profile
was not switched to ACO. Its managed server and authenticated gateway were
verified healthy after benchmarking; an earlier port-bind failure was cleared.

## Full-matrix prefix-reuse repair

The first full four-mode run exposed a separate server retention defect: the
server recognized a shared 6,149-token prefix, but automatic checkpoint FIFO
eviction left only a later 6,178-token recurrent-state checkpoint. It correctly
refused to rewind that image and replayed the prefix. The harness stopped on
`prefix_cache_miss`; those partial runs are excluded from current results.

Commit `1d74e9a315acaf0379eb2af1245202b820cff1e1` keeps the newest successfully
restored/current-task image at or before `stats.n_prompt_cached` when the
automatic two-entry ring makes room. The other entry can become the newest
MTP rollback image. The successful restore already authenticates target, draft
and speculative state. No extra snapshot, checkpoint capacity, template, batch
or hot-size change is introduced; configured/lifecycle-authority policy is
untouched. This fixes genuine recurrent-state reuse rather than relaxing the
no-replay measurement guard.

The existing prompt-cache suite passed. The model-backed 6K-to-12K proof then
passed all 96 requests across all four reasoning modes and three prompts on
one server/incremental fill. Minimum cached tokens per mode off/low/medium/xhigh:
6K: 6,145/6,148/6,148/6,148; 12K: 12,289/12,292/12,292/12,292.
Raw proof: `/srv/ai/paged-kv/results/forward/packed-cache-anchor-repair-20261103/live/`.

The current full replacement now pins the combined-fix server SHA-256
`189473719da99f8c391593c61a4c928769c80f72df8ceb6e527e707f95f35794` in
`/srv/ai/paged-kv/builds/aco-packed-repair-1d74e9a31/bin`. All ten ACO
checkpoints, all modes and all prompts are rerun for both MTP settings, not
only the post-H points. Current output root is
`/srv/ai/paged-kv/results/forward/final-after-packed-repair-20261103-v2`.
The first speed-pair proof remains identified above as the ca54933a7 build;
it is not substituted for any cell in the new full matrix.
