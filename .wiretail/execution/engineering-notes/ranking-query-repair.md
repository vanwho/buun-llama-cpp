# Query ranking and retrieval repair — 2026-10-06

Revision: `hotpath-v10-20260914`.

## Conclusions from source, not another diagnostic campaign

The fast GPU/host storage architecture is worth retaining. Keep encoded Turbo4
on host, GPU hot target K/V, full-L GPU MTP, GPU prefill/attention, once-per-query
retrieval/publication/replay, and frozen historical attention during generation.
Do not replace this with CPU attention or per-token host retrieval.

The legacy selector was `dot(mean(user Q), mean(page K))`. This is not an exact
attention ranker: signed, position-dependent Q can cancel across a long file
message; one relevant key is diluted by unrelated keys in the page mean. There
is no mathematical guarantee that eight mean-K winners contain the correct
page at 256K. Exact reranking cannot recover a page already excluded upstream.

The isolated ranking experiment has separate concrete lifecycle defects:

1. `llm_graph_context::cb` uploads positions/control/catalogue before graph
   allocation. Legacy positions/control have no buffer then. This explains
   its `tensor buffer not set` assertion; another model instance is irrelevant.
2. `build_kv_page_select` keys persistent probe state by `state.rows`. A partial
   final microbatch therefore selects another capture owner. Probes across
   batch shapes must share sequence/layer/dimension/head identity; only graph
   inputs depend on row count.
3. `execute_query` treats an all-nonfinite Q layer as a successful empty exact
   transaction. That masks corrupt execution, does not repair it, and must not
   advance a successful-turn cache or publish a selection.

The main trace also inspected only two cold candidates, including across
multiple layer segments. A trace-only `ranked_out` conclusion was not reliable
for ranks 3–8 or later layers. Do not infer numerical rank from that old label.

## Implemented in main source during this assessment

- `ggml_kv_query_accumulate`: backward-compatible 2D mean; optional bounded
  four-plane GPU state preserves span mean and actual Q rows at the first,
  middle and final positions of the final 32-token user tail. It survives
  ubatch splits and resets on query generation. Missing probes are NaN,
  not fictitious zero-score measurements.
- `ggml_kv_page_select` mode 2: independently scores probes and takes their
  maximum mean-K response. CPU and CUDA reject nonfinite Q consistently.
  This repairs mean-Q cancellation, not the remaining mean-K approximation.
- `llama-graph.cpp`: intermediate capture graphs can reuse compatible shape,
  sequence and non-final boundary; final and intermediate graphs cannot alias.
  Final selectors have one input uploader, after allocation. Removed duplicate
  accumulator uploads, construction-time capture and per-tensor mismatch traces.
- `llama-kv-cache.cpp`: reject unallocated sideband uploads; nomination traces
  inspect all segments, independently of bounded displayed IDs.
- Direct attention no longer allocates unused split/page-mass diagnostic
  scratch during production prefill/decode. Page-mass observation requires
  `LLAMA_KV_PAGER_DIAGNOSTIC_PAGE_MASS=1`, small direct geometry and telemetry.
  Ordinary telemetry must not change the timing/MTP consumer. Leave this env
  unset for production and all performance comparisons.
- `refresh_direct_telemetry` no longer copies the logical residency catalogue
  per token when the page-mass observer is disabled. This is a real CPU hot-path
  cost, not a numerical ranking measurement, and is removed rather than timed.

These changes do not read Q on CPU, change Turbo4 encodings, change K/V page
mapping, introduce per-token scoring, or alter MTP acceptance counters.

## Executable repair supplied for the isolated experiment

`../fixes/ranking-query-lifecycle.patch` removes construction-time input writes
and duplicate upload registration, shares probe owners across batch shapes,
moves shape-specific controls/positions to graph inputs, rejects infinite Q,
and removes the false-success nonfinite fallback. `git apply --check` passes
against the current dirty experiment source. This patch is NOT applied there:
the current session can write main and /tmp, not the experiment directory.
Task 105-01a applies it in the authorized runner environment, preserving dirty
experiment work; no wholesale experimental merge is implied.

## Remaining implementation decisions (not open-ended diagnostics)

Separate coarse key shortlist width from the eight-page full-K/V promotion
transaction. Use at most 64 cold key candidates per attention layer, plus
resident competition, with mean/bound candidate diversity. Preserve the union
of independent per-head/per-probe candidates before global aggregation.
Exact GPU encoded-key LSE reranking decides which pages justify promotion;
stage K only for coarse candidates, V only for final promoted pages. Retain
the stable valid map when there is no useful cold winner. All GPU readers must
finish before staging-slot reuse. Publication/replay and MTP carry happen once.
Do not silently promote 64 full K/V pages or increase H.

A finite but weak coarse rank is a retrieval approximation. Nonfinite Q,
missing buffers, invalid masses, publication failures and replay failures are
execution defects. Keep these categories distinct. MTP cannot be repaired by
changing its accounting when target execution produces slash filler/NaNs.

## Verification boundary and observed results

The focused CPU selector/probe test passed (split capture, generation reset,
mean-Q cancellation, missing/empty probes and nonfinite rejection); the modified CUDA
translation unit compiled. GPU execution is unavailable in this session:
`nvidia-smi` cannot communicate with the NVIDIA driver. No live speed, retrieval
or MTP improvement is claimed for this edit.

Previous tiny fresh prompt-1 controls were coherent: dense 82.01 tok/s decode,
75.78% MTP; legacy 76.02, 74.81%; experiment 76.39, 74.81%. The larger paired
experiment yielded slash filler and failed/inconclusive recall. Its ~1,300
tok/s prefill and ~34–37 decode are not valid successful-retrieval benchmarks.
Do not mix these identities or claim the experiment's semantic/MTP goal passed.

Next proofs: one compiled identity; small coherent dense/selected request;
one natural cold ABA sequence with dense answer control; then the prescribed
three-prompt MTP/speed row. Large occupancy only follows those results.
