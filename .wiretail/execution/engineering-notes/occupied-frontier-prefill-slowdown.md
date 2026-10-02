# Occupied-frontier prefill slowdown: measurements and next diagnosis

Recorded 2026-10-02 from the 102-04 128K occupancy run. This is a diagnostic
note, not an accepted speed result or a conclusion that the selected-attention
architecture is fundamentally too slow. Keep it distinct from later 102-05
measurements, which use a different logical/hot geometry.

## Run identity and observed curve

The measurements below came from L=131,072, H=16,384 tokens (64 pages at 256
tokens/page), B=1024/U=256, GPU Turbo4 target KV and full-L GPU Turbo4 MTP. The
task reduced H from its initial 59,904-token request because measured target
allocation and scratch reserve could not admit that hot pool; it kept all other
server settings fixed. The raw request journal is
`/srv/ai/paged-kv/results/forward/102-04/attempt-01/occupancy-resume/request-journal.jsonl`.

| occupied after request | fresh tokens | timed prompt tokens | prefill time | reported prefill rate |
| ---: | ---: | ---: | ---: | ---: |
| 20,303 | 15,615 | 15,615 | 13.73 s | 1,136.9 tok/s |
| 36,319 | 15,617 | 15,617 | 21.42 s | 729.2 tok/s |
| 52,334 | 15,616 | 15,616 | 29.91 s | 522.1 tok/s |
| 68,342 | 15,609 | 15,609 | 40.18 s | 388.5 tok/s |
| 84,223 | 15,482 | 15,885 | 52.58 s | 302.1 tok/s |
| 100,228 | 15,606 | 31,201 | 133.34 s | 234.0 tok/s |
| 116,238 | 15,611 | 15,611 | 79.55 s | 196.2 tok/s |

These are occupancy-campaign request timings, not the prescribed final three-
prompt speed benchmark. The later 145-second request at C=100,228 is especially
important: the campaign counted 15,606 newly committed tokens, but the server
timed 31,201 prompt tokens. Its API usage reported 84,223 cached plus 15,606
new prompt tokens. This suggests extra replay/re-evaluation work in that
request; investigate target versus draft/replay accounting before treating its
234 tok/s denominator as fresh-input throughput. The row also reports roughly
twice the usual summary-build and host-seal counts. A later row returned to
15,611 timed prompt tokens, so the double-work event is intermittent rather
than a simple smooth context-scaling effect.

## What the route and graph counters establish

- The candidate used selected direct/packed prefill routes; reference-route
  counts were zero. Thus this slowdown is not explained by an observed
  `selected_reference` fallback.
- Ordinary 15.6K chunks report about 59–60 direct plus 4–5 packed prefill route
  submissions, near the roughly 61 subbatches expected at U=256. Total graph
  completion counts also include output generation and MTP verification, and
  must not be mistaken for prompt graph count.
- Reported graph-construction time was below 1 ms per request. These counters
  do not support “graph compilation/rebuild itself accounts for tens of
  seconds” or “one graph is rebuilt for every prompt token.” Stable CUDA graph
  reuse may still help, but it is not the leading evidenced cause.
- Scheduler wait totals were material (about 14–26 seconds/request), while
  reported aggregate token time also rose with frontier size. Waits alone do
  not explain the full prefill time.
- Normal chunks showed roughly 4,000 summary builds / 134–138 MB and about
  6,000 host-seal D2H calls / 270–279 MB. The host copy is part of maintaining
  canonical RAM backing; counters give volume/count, not elapsed-time
  attribution. Determine whether these are efficiently batched and overlapped
  before changing the required backing behavior.
- H2D promotion and eviction deltas were zero in these request rows. They
  establish continued occupancy and selected-route execution, not that these
  particular turns naturally promoted a cold historical page.

## Interpretation and open question

The observed prefill rate keeps falling after the 16K hot-page capacity should
be saturated. If each selected graph really consumes a fixed bounded set of
resident rows, attention work per new token should become much less dependent
on total logical context. The counters above do not reveal the actual selected
row count per graph, so they cannot yet distinguish among:

1. selected views or attention kernels accidentally processing more than H as
   C grows;
2. target or MTP draft prompt replay/cache invalidation that reprocesses
   previously cached tokens;
3. per-boundary pager inventory/summary work that grows with host-backed
   history;
4. fragmented host sealing, CPU staging, event fencing, or backend waits; or
5. genuine GPU attention/kernel work over a larger-than-expected KV view.

Do not infer the culprit from `selected` route labels alone. Capture actual
selected logical rows/pages and route shape per prefill graph.

## Next narrow diagnostic

Using one candidate binary, the existing L/H/B/U/codec/MTP settings, and the
same-size deterministic fresh chunk, compare a few occupied frontiers (for
example C near 20K, 52K, and 100K). Preserve the same slot and do not run the
full context curve here. Record:

- actual target and draft prompt tokens processed separately, cache-hit/replay
  rows, MTP rewind/replay rows, and fresh committed tokens;
- selected page IDs and actual K/V row count per prefill graph, proving whether
  the view stays at or below H;
- per-operation elapsed time for attention kernels, graph execution, CUDA
  events/waits, policy/inventory scans, summary construction, host-seal packing
  and D2H copies;
- effective prefill batch/subbatch sizes, graph capture/reuse/rebuild counts
  separated for prefill versus decode, and GPU utilization/power over each
  request.

First reconcile the C=100,228 row's 15,606-versus-31,201 prompt-token
discrepancy. If actual selected rows grow with C, fix the bounded-view owner.
If selected rows remain fixed, profile replay and CPU/page-management work and
repair the measured dominant owner. Keep the regular route unchanged while
profiling; avoid full Nsight traces or per-token logging unless a short bounded
capture is needed to resolve attribution.

## Performance opportunities, conditional on measurements

- Preserve the fast selected direct/packed path and a fixed hot-row bound.
- Avoid replaying already-cached target or draft prefix. Attribute any MTP
  draft rebuild/rewind separately; do not fold it into fresh target prefill.
- Build or refresh routing summaries only for changed page content versions;
  avoid rescanning/rebuilding unchanged host history at every graph boundary.
- Batch host-seal D2H work by layer/page and reuse pinned staging; overlap it
  with compute where page ownership and scheduler dependencies permit. Do not
  remove canonical host backing.
- Reuse a bounded set of graph shapes and update page-table/data inputs without
  rebuilding topology when safe. Do not pursue “graph paging” as a substitute
  for KV paging or add a synchronization after every input token.
- Defer synchronization only across dependency-safe prompt subbatches, with
  page sealing, output consumption, and policy publication fenced before use.
  Never read an unfinished KV write or publish a mapping before its transfer
  completes.
- Preserve asynchronous promotion through the ordinary pager transaction;
  do not add a synchronous host-to-GPU page fault to the per-token path.

## Source pointers

- `src/llama-context.cpp`, `llama_context::prepare_kv_attention_graph()`:
  selected-view construction and bounded K/V row shape.
- `src/llama-context.cpp`, `llama_context::synchronize()` (currently around
  lines 2454–2567): scheduler fence timing, host-page sealing, attention
  telemetry publication, and live pager-policy application.
- `src/llama-kv-attention-execution.cpp`: selected route and graph
  capture/rebuild/replay metrics.
- `src/llama-kv-cache.cpp`, `apply_kv_pager_policy()` and the live-policy
  boundary: inventory, candidate, summary and promotion/eviction work.
- `src/llama-kv-pager.cpp`: canonical host sealing and routing-summary build
  counters.
- `tools/server/server-context.cpp`: JSON export of the counters used by the
  request journal. Confirm counter semantics before using them as elapsed time.

Raw evidence root: `/srv/ai/paged-kv/results/forward/102-04/attempt-01/`.
This note is a handoff for a later measured diagnosis, not authorization to
change batch geometry or claim a speed gate passed.
