# Current hot-path audit — 2026-09-12

Post-replan health: Qwen8080 HTTP200, candidate still loaded. Companion
ai-long-memory/8091 was inactive at the final check; this turn did not change
its service lifecycle. Do not report both endpoints healthy or treat this as
an explanation of the GPU kernel slowdown. Unrelated8092 not modified.

Decision: stop the maximum campaign and repair GPU execution. Not goal success.
Source audit:9173600e2b50fd0257b57186bbd6af3eef75023d. Live executable and
loaded DSO hashes/authenticated metrics are in the adjacent JSON.
Loading libggml-cpu by itself does not prove CPU attention.

Raw: /srv/ai/paged-kv/results/27-02-maximum-20260912T163151Z.
L131072,H319pages=81664tokens,B128/U64,page256, native full-L Turbo4 GPU MTP.

| SSE | Cached tokens | New tokens | pp/s | Output tokens | server tg/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw-00.sse | 0 | 1848 | 413.961 | 128 | 56.188 |
| raw-01.sse | 1975 | 1928 | 192.499 | 59 | 48.917 |
| raw-17.sse | 34753 | 1997 | 17.526 | 56 | 9.518 |
| raw-23.sse | 47073 | 1997 | 12.399 | 104 | 5.940 |
| raw-27.sse | 55260 | 2015 | 10.629 | 12 | 5.033 |

Rates retain SSE conventions; short outputs are not robust decode benchmarks.
Prefix reuse works: raw-23 did not replay47073 old tokens. No completed128K
population or physical cold-promotion speed claim follows.

At audit: prefill-direct899,dense0,packed0; selected pages226; faults/evictions0.
The path deteriorates before H pressure. Precise time attribution needs trace.

Confirmed source mechanisms:

1. fattn.cu:ggml_cuda_fattn_turbo4_paged_query_tile_kernel has row-serial
   decode/barriers, scalar dot-product/shuffle/exp updates, unlike mature MMA.
2. Its reductions area is8*4floats; dispatcher shared_floats counts8. Shared
   memory undercounts24floats/96bytes. Fix and run sanitizer.
3. planned_route sends non-contiguous Q<=128 to direct, including U64 prefill.
   Why the live prefix rejected dense eligibility remains to reproduce.
4. Context fallback selects all resident pages; host policy clears/unions
   layer selections. Nominal A can expand to H as C grows.
5. Exact row/page-shaped graph inputs and changing diagnostic-node presence
   can prevent reuse. Backend counters/trace, not frontend counts, decide cost.

Correction: pager graph_capture/rebuild1416,replay0 are frontend bookkeeping.
graph_construction_us3386 is not full graph-building cost. queue_time_us
1.16652e9 may include GPU execution, not1166seconds of controller work.
Attention D2H470729000bytes reported33489us, with coverage still to verify.
None independently attributes the slow appends. V7's absolute rate/zero-replay
heuristics are withdrawn in favor of V8 pilot/ETA and actual kernel profiling.

Operational action: verified runner1016377, agent1306674 and owned incremental
client1311837 stopped for the user's replan. Candidate service1311735 stayed
loaded; unrelated8092 untouched. Raw inputs/SSE and all reported task usage
preserved. Interrupted task had no completed provider usage event; totals were
not invented. New27-02 starts a fresh cluster/session.

Additional concrete shell defect fixed: unescaped command backticks in
Wiretail's unquoted build_prompt heredoc would execute privileged examples
during prompt construction. Replaced them with inert quoted prose; bash
syntax checked. Model/retry defaults unchanged.

Next:27-02 kernel bounds/profile;27-03 mature FA; phase28 bounded GPU selection,
stable inputs, optimized paged kernels, layer copies and async promotions;
phase29 measurements;30-01 summary-only review. No runtime speedup is claimed
by this research/planning update. V8 contains source-specific tasks/references.
