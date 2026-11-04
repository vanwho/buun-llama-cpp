# Contiguous selected-mask repair — 2026-11-04

Implemented in `8a79008b5`. The matched 32K runs show only a small observational difference. **No material prefill speedup is established.** ACO remains materially slower than master; the mask repair must not be presented as resolving that gap.

## Implementation

`llama_kv_attention_fill_contiguous_selected_mask` validates that the current selected native positions form one consecutive valid prefix with an invalid tail. It then fills each query row’s visible and blocked intervals in bulk, matching the ordinary contiguous cache-mask strategy. The destination is fully initialized, including padding.

The graph uses this helper for selected non-direct masks with at least 32 query rows, F16/F32 host storage and contiguous 2D shape. ALiBi, sparse/irregular positions, validity holes, unsupported layouts, and small decode/MTP verification batches retain the prior scalar implementation. Validation completes before any write. Native-position arithmetic widens before subtraction. No cache, descriptor registry, GPU buffer, new mode, server setting or attention route was added.

## Validation

- Incremental CUDA build: `cmake --build build-cuda --parallel 16 --target llama-server test-kv-attention-execution test-kv-cache-mask` — passed.
- Exact focused CTest selection — 2/2 passed. The execution target keeps assertions enabled.
- `test-kv-attention-execution --cuda` — passed. New differential checks compare F16/F32 output against the existing scalar visibility oracle, including padding, sparse refusal, missing vector rows, nonzero origins, extreme positions, noncausal masks and unsupported shapes.
- Independent code review found no semantic issue.

## Matched live comparison

Identical frozen 32K repository payload and A1 file optimization question. Each build has three fresh-server samples. Native template, low reasoning, GPU Turbo4 target/draft, MTP2, L/H=65,536, B1024/U256 and 16 threads are unchanged. Every fill evaluated 32,832 new tokens; every A1 reused all 32,832 and evaluated only 91 new tokens. All six rendered A1 token hashes match.

| Median metric | ACO before | ACO repaired | Master control, current run |
|---|---:|---:|---:|
| 32K fill backend throughput | 1039.52 tok/s | 1096.36 tok/s | 1630.56 tok/s |
| 32K fill HTTP wall | 31.609 s | 29.972 s | 20.158 s |
| Question to first token | 434.42 ms | 427.07 ms | 117.43 ms |
| Question processing | 406.66 ms | 400.33 ms | 90.06 ms |
| Decode | 87.27 tok/s | 87.39 tok/s | 91.75 tok/s |
| MTP acceptance | 87.04 % | 87.04 % | 86.61 % |

HTTP file-fill time fell **1.637s / 5.18%**. Repaired ACO backend fill rates ranged 1091.63–1096.73tok/s. The current master rate is essentially unchanged from the earlier paired control, The small ACO difference is insufficient to establish a meaningful speed gain; the remaining gap is the actionable finding.

The remaining backend prompt-time gap is **9.811s** at 32K. The scalar mask was a real contributor, not the entire cause. Page-view construction and backend attention execution remain candidates; this run does not isolate their costs. No extra runtime instrumentation was added to attribute the residual.

Generation lengths varied within each build. Decode/MTP medians are descriptive follow-ups; whole response wall differences do not isolate a fixed ACO cost. No cold-page timing claim is made from this all-hot workload.

## Identity, preservation and artifacts

- Completed live run: `/srv/ai/paged-kv/results/matched32k-latency-20261104T090659598313303Z`. The old baseline remains in [the original matched report](MATCHED_32K_MASTER_ACO_LATENCY_20261104.md).
- Build/test/diff records: `/srv/ai/paged-kv/results/contiguous-mask-build-20261104`. The runtime was built from base `ab10f84e5` plus the exact four-file diff; file hashes match the subsequently committed `8a79008b5` source.
- Pre-repair binary and complete colocated loader closure were frozen outside Git at `/srv/ai/paged-kv/builds/aco-before-contiguous-mask-6068d4a/` before rebuilding. Runtime and per-process loaded-library hashes are retained in the JSON companion.
- Managed server and gateway restored; Jellyfin PID 1000914 and port 8091 PID 76881 preserved. Port 8092 untouched. Full benchmark suite remains paused.
- Only runtime code/tests and compact reports are committed. Raw payloads, logs and binaries remain external.

## ACO-only scope and discarded experiment

The small mask timing difference is not a demonstrated material prefill speed
improvement. This investigation did not establish a fix for the broader gap.

A diagnostic found that the old coalescer excluded Qwen’s four-axis M-RoPE text
and normal appends after the short system prefix. An experimental correction
removed 96 intermediate page fences, but the waits moved into graph
preparation and the fill stayed approximately 29.95 seconds. The experiment
was therefore **reverted**, not retained as a performance fix. Its patch and
results remain outside Git for audit. The earlier hypothesis that the partial
tail was absent from residency was not established: `complete_write` publishes
mutable `filling_gpu` records too.

One profiled same-binary pager-toggle comparison measured 26.243 seconds with
paging off and 29.950 seconds with ACO on. Backend submission totals were
nearly identical (15.865/15.814 seconds). This narrows ACO-specific investigation
to graph preparation and host-page maintenance; it does not justify treating
the entire older-master comparison as ACO overhead. It is one diagnostic pair,
not a statistical benchmark, and its component scopes overlap.

The comparison raw directories are:

- `/srv/ai/paged-kv/results/allhot32k-hotpath-diagnostic-20261104T095405144309261Z`
- `/srv/ai/paged-kv/results/allhot32k-nopager-diagnostic-20261104T095831971727958Z`

The final source and `build-cuda` were restored to `8a79008b5`, retaining only
the selected-mask repair. The final incremental build used 16 jobs, both
focused CTests passed again, and runtime-source hashes match that commit.
Final build identity is under
`/srv/ai/paged-kv/results/contiguous-mask-build-20261104/final-source-restore/`.
No global scheduler/kernel changes or server performance-setting changes are
kept. Do not add general optimizations or complicated protocols based on these
counters; any further patch must address an identified ACO-specific cost.
