# 19-03 live residency movement evidence

Classification: implementation boundary passed locally; the named live
Qwen/CUDA cold-page checkpoint is deferred because the reference RTX 4080 was
occupied. No live model workload was started or disturbed.

## Production boundary

The real decode/ubatch path records the current sequence, seals completed
pages, and invokes the pager policy only after scheduler synchronization and
attention graph completion. The pager binds its upload ring, canonical host
catalog, layer-major GGML adapter, synchronous completion mode, transfer
identity recheck, clean-drop/restore hooks, and final table recheck. Promotion
therefore writes the same cache slab read by attention; it does not maintain a
duplicate dense cache.

The adapter exposes per-layer K/V byte views over the cache-owned slab. The
deterministic fixture verified a layer-1/value run at its actual layer offset,
and pool reconciliation verified stale mapping removal plus replacement slot
publication. Existing lifecycle tests cover map/copy refusal, cancellation,
backpressure, and no-victim behavior.

## Verification

| Command | Result |
|---|---|
| `cmake --build build-cuda --target llama test-kv-pager test-kv-residency test-kv-policy -j2` | pass |
| `ctest --test-dir build-cuda --output-on-failure -R '^test-kv-(pager\|residency\|policy)$'` | 3/3 pass |
| `build-cuda/bin/test-kv-pager` | pass |
| `build-cuda/bin/test-kv-residency` | pass |
| `build-cuda/bin/test-kv-policy` | pass; submitted=4, completed=1, cancellations=3 |
| `git diff --check` | pass |

Machine-readable details and hashes are in
`19-03_MOVEMENT.json`. The focused binaries are under `build-cuda/bin/`.

## Deferred verification

Observed hardware was an NVIDIA GeForce RTX 4080 with 16,376 MiB total and
1,128 MiB free; the active process was
`/srv/ai/paged-kv/results/18-01-runtime-bundle-20260905-v3/bundle/bin/llama-server`
using 14,660 MiB. The model is present at
`/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf` (14,252,845,984 bytes), but its
hash and live results are intentionally unmeasured.

After GPU isolation, run the runtime-supported Qwen invocation discovered from
`--help` with a deliberately small observed hot budget. Capture the original
compressed page checksum, canonical host checksum, eviction slot, completed
H2D event and useful/aligned bytes, promoted slot, attention-consumed checksum,
failed replacement rollback, and a warm repetition proving zero redundant
upload. A profiler copy trace may cross-check direction and event size.

## Exact next action

Repeat only the isolated live cold-page checkpoint above; do not infer it from
graph epochs or the local fake/CPU receipt.
