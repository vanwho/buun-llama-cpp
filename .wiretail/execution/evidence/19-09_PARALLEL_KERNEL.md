# 19-09 parallel split-KV evidence

Classification: local CUDA correctness and graph build passed. Kernel timings
are observations, not a performance gate. The live native-MTP, populated
pressure, same-prefix parity, and end-to-end overhead checkpoints are deferred
because the shared RTX 4080 had 1128 MiB free and the existing workload was
preserved.

The implementation adds bounded graph-owned F32 scratch for per-partition
unnormalized `[m,l,o]` states and optional per-page `[m,l]` states. The CUDA
dispatcher selects no more than the admitted capacity, logical page count,
shape partition count, and a runtime device limit. Partition states are merged
on device before output normalization; page mass is rescaled using the same
global denominator. The serial query-tile implementation remains the
capacity-one/small-shape fallback and correctness oracle.

## Local measurements

The final fixture shape is 530 selected rows, four logical pages, three query
tokens, four query heads, and head width 256. It exercises a 17-row tail, a
one-row tail, logical/physical permutations, a gap-free compact table, masks,
causal native positions, cold upload, exact partial-state merge, and page-mass
telemetry. The split arena is 38,304 bytes for capacity three.

Warm CUDA-event observations from the same fixture shape:

| Path | Device time |
|---|---:|
| Serial control, three query tokens | 0.734 ms |
| Split requested capacity 1 | 0.736 ms |
| Split requested capacity 2 | 1.463 ms |
| Split requested capacity 3 | 1.556 ms |
| Split control, capacity 3 | 1.514 ms |

The first printed serial timing was excluded from comparison because it
included first-use CUDA compilation. The narrow 530-row/4-page control is
still serial-faster; the production split selection is intended for the
larger admitted-row shapes, while this task does not claim a native-MTP
end-to-end speedup without the isolated live checkpoint. No compatible
page-addressing tiled/MMA alternative was exposed for a controlled local
comparison.

## Verification

| Command | Result |
|---|---|
| `cmake --build build-cuda --target test-cuda-fattn-paged-turbo4 -j2` | pass |
| `./build-cuda/bin/test-cuda-fattn-paged-turbo4` | pass; split capacities 1/2/3 and serial oracle matched |
| `cmake --build build-cuda --target llama-server -j2` | pass; graph integration compiled and linked |
| Focused `ctest` for pager, routing, telemetry, attention view/execution/exact | 7/7 pass |
| `python3 -m unittest discover -s tools/server/bench -p 'test*.py' -v` | 46/46 pass |
| `git diff --check` | pass |
| `PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp python3 /srv/wiretail/task_state.py validate` | valid state: 122 tasks |

The machine-readable receipt is
[`19-09_PARALLEL_KERNEL.json`](19-09_PARALLEL_KERNEL.json). Its source diff
hash excludes execution metadata, and its raw output records the observed
RTX 4080 device and CUDA-event measurements.

## Deferred verification

On an isolated GPU, use the supported native-MTP exact/pressure procedure with
the same model, corpus, context, hot set, and telemetry settings. Verify
target/draft row parity, all native GPU Turbo4 draft buffers, same-prefix
logits and acceptance, serial/split/tiled dispatch at the actual verify shape,
kernel occupancy/throughput/synchronization, end-to-end decode share, and one
populated pressure case. Do not disturb the existing workload.
