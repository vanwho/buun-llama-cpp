# 20-04 profile receipt

## Result

Local implementation and verification pass. Live Qwen/MTP and forced-paged GPU
checkpoint remain deferred because the RTX 4080 was occupied by the existing
qwen38-fast-turbo4-mtp service; no live process was disturbed.

## Measured deterministic fixture

The expanded KV-attention matrix covered prefill, decode, and MTP verify. It
recorded 9 submissions, 5 graph captures/rebuilds, 4 replays, 48 bytes of
capture-time direct page-table upload, and 2 residency-epoch changes. Three
content-only cases (page permutation, physical-slot remap, and query-position
change) reused the layout; tail growth and query-shape/configuration changes
rebuilt it. The raw machine-readable line is emitted by
`test-kv-attention-execution` as `kv-attention-profile`.

## Implementation findings

- `graph_layout_key()` now contains only bounded tensor/configuration shape and
  representation identity; residency epoch and descriptor values remain data
  generations/content keys.
- Graph parameter reuse no longer rejects a table epoch or content-key change.
  Reused selected inputs refresh fixed-capacity host/device descriptors only
  when the content key changes; unchanged submissions skip those transfers.
- Direct page host descriptors are updated in place, preserving the pointer
  retained by the GGML direct kernel node. Reference physical row IDs and the
  selected causal mask follow the same update boundary.
- Context page-ID, query-position, and row-ID workspaces retain bounded
  capacity. Direct physical-slab submissions skip the duplicate reference row
  lookup.

## Verification

- CPU and CUDA-configured builds of the changed focused targets passed.
- CPU CTest regex `test-kv-attention-(view|execution|telemetry|exact)`: 4/4.
- CUDA-configured CTest regex `test-kv-attention-(view|execution|telemetry|exact)`: 4/4.
- `PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp python3 /srv/wiretail/task_state.py validate`: valid, 122 tasks.
- `git diff --check`: pass.

## Deferred verification

The named live protocol is not claimed: no isolated Qwen35 model/corpus run,
forced-paged direct-kernel hardware parity, matched selective/observe/off wall
timings, native-MTP acceptance/output parity, or true warm-pressure result was
available. The exact later action is to provision GPU isolation and the Qwen35
fixture, run forced-paged parity first, then run the matched three-prompt and
warm-pressure protocol while recording route fractions, graph counters,
acceptance/checksums, and wall-time tails.
