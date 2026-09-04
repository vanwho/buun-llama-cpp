# Decode checkpoint — task 10-03

This is the first graph-epoch checkpoint for live selected Turbo4 decode. It is
an implementation checkpoint, not the final speed gate. The machine-readable
record is [`DECODE_CHECKPOINT.json`](DECODE_CHECKPOINT.json).

## Result

The epoch and lifetime contract passes in
[`test-kv-attention-execution.cpp`](../../../tests/test-kv-attention-execution.cpp):

- unchanged metadata replays one graph;
- ordered page content, physical slot, representation, table, query, shape,
  tail, and route changes each force a rebuild;
- scratch and metric ownership are ignored by graph topology;
- clearing a current key preserves old immutable-view leases until completion,
  while a later table is admitted alongside them.

The test records eight capture/rebuild decisions, one replay, nine submissions,
and 96 bytes of direct page-table payload across the two unchanged direct
submissions. The production boundary now exposes counters for graph capture,
replay, rebuild, submission/completion, page-table bytes, descriptor preparation,
kernel time, total token time, waits, and scratch high-water.

## Route and timing disposition

The Qwen3.8 GGUF is present, but this repository has no task-local harness that
can run dense, selected-reference, and selected-direct routes with the same
prompt/output stream and expose all new counters. Consequently no pp/tg,
TTFT, inter-token latency, or output-parity claim is made here. Those route
comparisons are deferred to the model-backed integration harness.

The direct CUDA fixture remains valid bounded evidence. Its raw five-trial
record is [`decode-direct-fixture-trials-10-03.txt`](decode-direct-fixture-trials-10-03.txt);
it measures a four-head, one-query, 529-row Turbo4 kernel and is not an
end-to-end model speed result. Existing sanitizer, VMM, and resource logs are
linked from the JSON checkpoint.

## Bottlenecks and next inputs

The current graph input path rewrites the immutable page-table descriptors on
each submission. The graph key prevents stale topology reuse, and the fence
fix prevents old-page reclamation, but amortized live capture/replay cost still
needs the shared route-control harness. Task 10-04 should add that harness
around prefill-to-decode, then record two warmups and five measured trials per
route plus a bounded H<L allocation/timing run with no quality claim.

No final 3x performance decision is made by this checkpoint.
