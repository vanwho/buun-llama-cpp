# C256000 prefill variation — fixed candidate v2

Source: MTP-on C256000 `campaign/schedule.jsonl`, fixed candidate source 1d74e9a315acaf0379eb2af1245202b820cff1e1. The 196,609→256,001 stage adds 59,392 net-new tokens across 30 chunks. Backend prompt work was 118,421 tokens / 267.290 s; wall time 273.936 s. Weighted rates: 443.0 backend prompt tok/s, 222.2 net-new tok/s per backend prompt second, 216.8 net-new tok/s per wall second.

For prior occupancy 230,000–<250,000 (15 chunks, actual 230,133→252,018), OLS `T_prompt = a + b × fresh_tokens` gives `a=3.344 s`, `b=2.887 ms/token`, R²=0.997. `cache_n` matched prior occupancy for all 15 (zero cache misses within 8 tokens); backend prompt tokens were 1.97–2.00× fresh-token deltas. Median wall-minus-backend-prompt overhead was 0.226 s (0.219–0.234 s).

Matched sizes show both scale and noise: 648-token chunks at 243,798→244,446 and 245,861→246,509 took 5.171 and 5.195 s backend prompt time (248.3 and 247.1 backend tok/s). 4,081-token chunks at 232,140→236,221 and 247,937→252,018 took 14.799 and 15.079 s (550.7 and 540.5 backend tok/s). Thus fixed overhead is about 3.34 s in this subrange, while backend work varies with context; the matched pairs differ by 0.5% and 1.9% respectively. This supports a real context-dependent backend cost, with no cache-miss/reprocessing event in the sampled subrange.

## Code path and observed fixed work

In the measured historical protocol, `tools/server/bench/repo_context.py`
built one corpus item per tracked file. `fill_context()` in
`run-master-context-matrix.py` sent one such item or its
remaining suffix per append, capped by `FILL_REQUEST_BUDGET=4080`. It splits a
large file but does not combine small adjacent files. The 453–4,081-token
request sizes therefore pay very different per-token shares of fixed turn work.
The reported 210–550 range was a mix of those sizes, not repeated identical work.

With `probe-rerank`, the runtime first evaluates the new user span to capture
query probes, then selects/promotes pages and replays that span with the final
history. This is the production query-replay path in `server-context.cpp` and
`server_prompt_query_capture_reuse_prefix()` in `server-task.cpp`. The client
sends each new file piece once; cached older repository tokens are not
re-evaluated. The second evaluation is confined to the current user span.

For task 51410 (648 new tokens), saved logs record approximately 146 ms and
147 ms for two recurrent-state checkpoint captures, each about 420 MB, and
1.274 s `pre_decode_us` on the first batch after replay. For task 51458
(4,081 new tokens), the corresponding first replay batch has 1.278 s
`pre_decode_us`; the near-end checkpoint capture is 151 ms. These are real
fixed contributors inside prompt time. They do not account for the complete
fitted 3.34-second floor. Per-fill phase profiling was disabled, so assigning
the remaining time to router finalization, restore, CUDA waits or another owner
would be speculation. Query checkpoints contain partial recurrent state and
MTP carry; this path does not copy the entire historical K/V as a checkpoint.

## Interpretation and next measurement

The 2.6x spread is predominantly a chunk-size effect with a large fixed backend
cost, not 2.6x repeat-to-repeat jitter. Same-sized saved examples at nearby long
frontiers are within about 0.5% (648 tokens) and 1.9% (4,081 tokens). An exact
same-frontier, equal-token repetition has not yet been run. The fitted floor is
a descriptive fit, not a profiler attribution.

The completed decode/MTP results remain valid. Treat the existing ingest curve
as performance of variable-sized per-file turns; do not label its fastest
backend counter rate as net ingestion throughput. A clean bulk-ingest comparison
should coalesce file fragments into fixed token-sized batches, retain all source
paths/byte ranges, and report final short boundary batches separately. Any
controlled phase timing must be a separate run; do not change the frozen
completed decode candidate or silently blend ingestion protocols.

## Mapping prepared; final retests paused

The replacement protocol now groups frozen repository bytes into native-token
payloads. 8K/16K/32K/64K sizing requests passed with the same immutable candidate,
L262144/H65536, B1024/U256 and GPU Turbo4 MTP. The largest tested payload, 64K,
added 52MiB process VRAM and left a directly sampled minimum 375MiB free.
Four exact 65,536-token groups now map 262,144 content tokens from 74 files;
checkpoint tails are labeled separately. See
[the sizing and mapping handoff](repo-prefill-chunk-map-20261103.md).
These are allocation checks, not replacement prefill performance results.
Jellyfin remains running. The operator requested a pause before final clean
prefill retests, so no such retest has started.
