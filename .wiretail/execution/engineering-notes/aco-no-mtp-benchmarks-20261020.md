# ACO without MTP: replay repair and benchmark protocol

## Replay defect

The first no-MTP run at L=8192/H=4096 failed after processing its
6144-token repository prefix. The streamed error was `Unable to restore
final-user query for replay`; the server identified
`partial_remove_unsupported` before attempting checkpoint restoration.

`server_query_checkpoint::restore()` in `tools/server/server-context.cpp`
incorrectly required `llama_memory_can_seq_rm_partial()` for both contexts.
For hybrid Qwen memory, `llama_memory_hybrid::can_seq_rm_partial()` requires
both attention range removal and recurrent token rollback. The latter is
disabled when `n_rs_seq=0`, which is the normal non-speculative configuration.

Query replay does not use recurrent token rollback. It captures and restores
the recurrent image through `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`, then
truncates the provisional attention suffix with
`llama_memory_seq_rm_attn_transient()`. The hybrid implementation delegates
that operation only to its attention child. The repair removes the unrelated
general rollback capability gate and keeps checkpoint identity/frontier
checks, undo capture, recurrent restoration, attention truncation checks,
and draft quarantine/recovery.

No attention kernels, selection budgets, KV types, hot capacities, or batch
sizes are changed by this repair. No speculative context is enabled to work
around the defect.

## Frozen benchmark protocol

Use the known successful release bundle at
`/srv/ai/paged-kv/builds/aco-preview-20261015-release/bin/`, whose server
executable SHA-256 is
`c9041ca96611ed4c9aa06be7b9a23443aa11d4b3bb30983d7dd2558c582659ec`.
Its replay defect was reproduced. The original bundle is retained, and a
separate copy contains only a replaced server implementation DSO. Original
CUDA, GGML, and llama DSOs are retained and hashed. The changed
implementation DSO and source diff are recorded explicitly; an unchanged
wrapper executable hash alone does not establish an unchanged build.

The main plan branch's existing uncommitted MTP argmax correction is
preserved. If present in the implementation DSO, record it in source
provenance; it is inactive with `--spec-type none`.

Keep the successful release's Turbo4 target KV, GPU weights, Flash Attention,
selective/probe-rerank pager, 256-token pages, `--kv-pin-recent 0`, B=1024,
U=256, one slot, no context shift, cache RAM 12288 MiB, eight checkpoints,
8192-token checkpoint minimum step, idle-slot caching, and unified KV.
Change speculative mode to `--spec-type none` and allocate no MTP draft.
Use an isolated slot-state directory for every context setup.

| Server L | Hot H | Occupied prefix target C |
| ---: | ---: | ---: |
| 8192 | 4096 | 6144 |
| 16384 | 8192 | 12288 |
| 32768 | 16384 | 28672 |
| 65536 | 32768 | 61440 |
| 131072 | 65536 | 126976 |
| 196608 | 65536 | 192512 |
| 262144 | 65536 | 258048 |

Start one fresh server for each geometry. Freeze repository source bytes and
native tokenizer counts. Retain the prefill curve at approximately 16K
increments, including the final native occupied count. These source-prefix
extensions follow the release workload; report actual processed/cached
tokens, because checkpoint rewind can process more than the new source
increment. Do not describe this as a full-transcript append-only workload.

Backend prompt throughput divides processed tokens by the server's prompt
processing time. It is not the net ingest rate when old tokens are replayed.
For every curve point, also report `(C_current - C_previous) / wall_seconds`;
for the complete fill report `C_final / sum(fill_wall_seconds)`. Canonical
requests report their actual processed/cached counts and full request wall
time. Do not call tiny cached-query prompt rates bulk prefill performance,
or silently treat a large rewind as a tiny cached request.

Run the three original canonical prompts with reasoning off, greedy
sampling, seed 42 and streaming. For each prompt run one 40-output-token
warmup and three measured requests with a 400-output-token ceiling. Natural
EOS may produce fewer than 400 tokens. Report medians using actual server
decode timing and token counts, together with wall time, prefill timings,
cache counts, GPU memory, process RSS and swap. MTP must remain disabled and
have no proposals. Preserve raw responses even when content quality is poor;
an execution error is a separate failed measurement.

## Evidence

Replay fix commit: `218f5ab53` on `plan/attention-aware-kv-paging`.
The corrected bundle is
`/srv/ai/paged-kv/builds/aco-no-mtp-replay-20261020/bin/`.
Its implementation DSO SHA-256 is
`82534d219ed68e96789efbda319645a4c87861242efc6f96638a96749a4d5a48`.
The executable and all remaining released DSOs are unchanged.

Campaign root:
`/srv/ai/paged-kv/results/release/aco-turbo4-no-mtp-replay-20261020/`.
All seven geometries completed: 128 successful requests comprising 44 fill
frontiers, 21 warmups, and 63 measured responses. Request identifiers are
unique within each context (128 context-qualified identifiers). Independent
record inspection confirmed counts, success status, zero MTP proposals, and
all three measured decode medians at every geometry. Frozen prefix bytes
and source-manifest hashes match the CPU control at all seven sizes.

### Final measured results

One K means 1024 tokens. Decode columns are medians of three requests with
reasoning off and MTP disabled, in original Python / mmap / Bash prompt order.
The actual occupied prefixes are listed in the protocol table above.

| Server L / hot H | Net fill tok/s | Backend fill first → last tok/s | Decode prompt 1 | Decode prompt 2 | Decode prompt 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K / 4K | 1459.1 | 1486.4 → 1486.4 | 43.02 | 43.19 | 43.02 |
| 16K / 8K | 1492.0 | 1508.2 → 1508.2 | 40.12 | 40.39 | 40.30 |
| 32K / 16K | 1369.2 | 1415.7 → 1374.0 | 35.81 | 35.74 | 35.55 |
| 64K / 32K | 1001.7 | 1470.5 → 935.6 | 25.56 | 25.25 | 25.17 |
| 128K / 64K | 830.3 | 1442.0 → 816.8 | 14.14 | 14.18 | 14.10 |
| 192K / 64K | 639.0 | 1410.9 → 1033.8 | 13.89 | 13.92 | 13.85 |
| 256K / 64K | 390.0 | 1393.4 → 1077.1 | 13.97 | 14.00 | 13.98 |

The 256K backend's final 1077.1 tok/s is not 1077.1 newly added tokens/s:
that fill request processes 108402 rows after reusing 149668. The final
frontier's net growth rate is about 102.5 tok/s. The overall fill takes
661.7 seconds for C=258048, hence 390.0 net tok/s. The following canonical
requests process approximately 108420 rows each, with measured wall-time
medians of 124.6 / 114.2 / 129.4 seconds. These are the same synthetic
history-rewrite/cache-rewind semantics documented for the earlier MTP-on
release, not evidence of an attention-kernel regression caused by disabling
MTP. An append-only transcript is a different workload and is not measured
by this matrix.

For reference, upstream-master CPU-Q4/no-MTP control decode medians at 256K
are 2.990 / 2.996 / 2.988 tok/s: ACO is about 4.67x as fast in the server's
decode phase. This does not include the large replay costs just described.
CPU controls use cache RAM/checkpoints disabled and one-shot fill; do not
claim their fill timings are an equivalent ingest workload.

At 256K the three measured output lengths are 334 / 190 / 400 tokens;
responses contain ordinary Python, mmap/read prose, and Bash rather than
slash filler. Completion is not a new answer-quality or natural-promotion
certification. Peak benchmark-process swap is zero at every geometry; this
does not assert that the whole host has no unrelated swapped processes.

Full per-frontier backend/net curves, actual prompt/cache/output counts,
wall-time medians, memory peaks, source equality, and bundle hashes are
preserved in `../evidence/ACO_NO_MTP_20261020.json` (a manual benchmark
summary, not a V10 task receipt). Raw reports remain outside Git at the
campaign root's `aggregate.json`, `aggregate.md`, and `aggregate.csv`.
The source aggregate SHA-256 is
`4228790e8cf06b3b91b6d0b1d7f5053cfa0f4c6263731cc0f286aac85cffab77`.

Managed `llama-server.service` was restored active; health checks on 8080
and 8091 returned HTTP 200. Port 8092 was not accessed. The original
managed deployment was restored unchanged, not replaced with the benchmark
candidate.

The original release reproduces the no-MTP replay rejection. A first
corrected 16K attempt also suffered an illegal CUDA memory access during
the second measured Bash response (NVIDIA Xid 13/43, no reported OOM).
A synchronous diagnostic and then a fresh normal asynchronous rerun both
completed all 13 requests; the CUDA fault's cause was not established.
A bounded source audit did not identify a justified additional kernel fix.
Keep the interrupted and diagnostic attempts separate from the completed
normal-mode cohort; do not describe the fault as repaired or diagnosed.
