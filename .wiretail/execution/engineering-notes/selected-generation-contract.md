# Selected generation: current source and measured findings

Revision: hotpath-v10-20260914. Updated October7 for105-02 onward.
Current summary, not an attempt diary. Raw runs remain external.105-02 is
in_progress; no cold-recall success or complete numerical-parity proof.

## Architecture and preserved repairs

GPU Turbo4 hot target K/V, canonical Turbo4 host storage, full-L GPU Turbo4
MTP. GPU coarse shortlist -> exact encoded-key rerank -> bounded common R ->
atomic promotion -> query-only replay -> frozen generation. CPU RAM stores
pages, not historical attention computation. B1024/U256, automatic fast
routes, one retrieval decision per user query. No per-token rerank/H2D,
F16 draft, global reference route or expected-answer injection.

- `52d04942c`, `src/llama-graph.cpp::build_attn`: packed copies consume the
  completed layer SET_ROWS result. ggml nested offsets are absolute in the
  slab; rebase relative to the write result and preserve owner/bounds/type.
  Zero encoded K AND V padding once per shared-slab execution.519/768 leaves
  249 rows;4089/4096 leaves7. Negative-infinity masks cannot sanitize NaN
  norms/V. CUDA uses stream-ordered captured memset, not F16 history/fences.
  Reuse compares flattened allocation owners and absolute offsets.
- `538b32727`, `common/speculative.cpp::common_speculative_rollback_dft`:
  successful paired accepted-prefix replay retains authenticated carry and
  its idempotent guard. The old target_restored_without_draft event cleared
  the full draft prefix on the next process. Real unpaired restores still
  invalidate carry. Accept precedes target/draft rollback.
- `230a366fb`, `llama_kv_cache::apply_pager_live_policy`: reusable streaming
  chunk staging is not entire-transaction storage. Keep hot/history/slot,
  eight-page/event/max-transfer admission; do not cap by ring_bytes/page_bytes.
- `06e6b9473`, `llm_graph_input_mem_hybrid::can_reuse`: run the recurrent
  child's existing identity/binding guard. Do not copy an unconditional
  false can_reuse implementation or globally disable CUDA graphs.
- `llama_context::prepare_kv_attention_graph` consumes ALL admitted frozen
  R pages plus required query/current/resident-output rows. Nomination is
  provenance, not another filter. No additional all-layer union allocation.

## Confirmed I32 -> I64 position defect

`include/llama.h` defines llama_pos=int32_t; direct CUDA query/native tensors
are GGML_TYPE_I64 and the paged kernel reads int64_t. Four ordinary direct/
exact-wave uploads sent sizeof(llama_pos), not the tensor element size.
Actual device query0 became4344+(4345<<32)=18661632905464. Only7 of14 I64
positions were filled; query7..13 used zero/stale bytes. Correct host masks
hid this from host-only snapshots and synthetic attention fixtures.
The old first mismatch at row7 was the halfway point of the malformed
upload, NOT evidence of a4351/4352 page-boundary kernel bug.

Root correction in `src/llama-graph.h/.cpp`:

- direct_query_positions_host/direct_native_positions_host are persistent
  vector<int64_t>, widened from metadata/ubatch positions before upload.
- ordinary direct query/native and exact-wave query/native paths send8bytes
  perposition. Host *_uploaded vectors remain I32 comparison keys only.
- No new kernel, full-H/L copy, fence, F16 representation or diagnostic work
  is introduced into generation. Ordinary direct uploads only its Q vector.

The built comparison reads exactly4344..4357, maps all14 current rows and
captures7392bytes each of K/V, byte-equal across replay/control. SAME-question
logits/hidden and all emitted layer taps are exact0/0. This producer failure
is fixed, not bypassed. Compiled libllama SHA256:
`a78b2e0924374e43c0a4c8da9260f56c44d2d4e8b60c25803e741533e71c42a1`.

## Post-fix measurements

Exact original prompts, one40-token warmup and three400-output-token maxima
each, temperature0/reasoning off. All12HTTP200, coherent, fresh slot and
cache_prompt=false/cached input0. Normal EOS below400 is valid.

| Prompt | Median decode tok/s | Median request MTP | Aggregate accepted/proposed |
| --- | ---: | ---: | ---: |
| Python sorted merge/docstring |104.88|91.86%|712/772|
| mmap vs read paragraph |82.32|61.04%|283/460|
| Bash watcher |102.28|86.59%|426/492|

Inputs30/27/28; outputs365/171/225. These tiny prompts are NOT bulk-prefill
measurements. Do not relabel prior candidates' benchmarks or claim an
isolated patch speedup without an old/new ablation.

One post-fix PY_MERGE_03 A/B/A completed3HTTP200. Requested page5 was cold,
host-backed/summary-ready and not promoted. Final400-token output repeated
the wrong Bash comment, accepting258/282drafts(91.49%). High acceptance is
not correct retrieval or useful speed. Page6 completed a real promotion/use
chain; it is not page5 proof. Current trace was OFF, so default zeros and
cumulative rejection counters do not localize the page5 miss. An older
query104 shortlist[6,7,8] must not be attributed to the current query.

## Remaining numerical boundary and exact next seam

CPU prefetch/residency/speculative-state and production-shape/padding CUDA
checks pass. CUDA covers actual24-Q/4-KV, Q1/Q3/Q256, causal gaps/noncontiguous
slots, generation roll and repeated NaN-poisoned padding. Cancellation,
native frontiers, frozen map and next-MTP2/2 pass. Full teacher-forced
scalar-packed vs width3-direct continuation parity still fails over12rows:
max logits6.82667/hidden8.40555. This is separate from fixed question inputs.

First continuation layer0-2 and layer3 normalized Q/K agree. First layer3
attention delta0.00167859 is inside the existing0.003 CUDA route bound;
layer4 differences are downstream. Full-softmax row0 KL(scalar||batch)
0.000472/TV0.01044 with same top1; row3 (next width3 group) increases to
KL0.1017/TV0.1629. First top1 mismatchrow5; maxKL0.9742/TV0.5029 atrow7.
Do not dismiss late drift as rare-logit roundoff or assume a kernel bug.
Source audits found no confirmed stale second-group upload/row-ticket or
recurrent-plane indexing defect. No speculative production edit follows.

`tests/test-server-query-replay.cpp` now has OPTIONAL fixture-only captures
of layers0/4 current recurrent+conv planes at scalar token3 vs width3 group1
endpoint (position4360, plane0), and scalar token4 vs width3 group2 first
snapshot (position4361, planeK-1). Read actual hybrid recurrent owner rows,
checked tensor strides, missing reasons, byte hashes and numeric differences
before draft catch-up; the draft decode does not write target state.
Optional scalar token4/second-group taps expose device positions/current K/V
and first changed layer. Captures default off, capped24MiB/branch, outside
production. Rebuild passed; the new capture has NOT been model-run yet.
The earlier sidecar's object-closing/comma serialization defects are fixed;
original raw artifacts are retained, not silently overwritten as valid JSON.

## Cold ranking: avoid false bottlenecks

`llama_kv_query_cold_rank_width` is min(cold_pages,64). L16K/H4K has coarse
cold width48, not eight. `retain_pager_coarse_shortlist` retains those records
and `prepare_router_query_layers` processes the full authenticated list.
Eight is a separate later full-K/V H2D transaction limit. Default encoded
query/key domains agree by source audit; no compensating transform needed.

Use current sequence/query/generation-bound existing opt-in selector trace
to distinguish page descriptor eligibility -> coarse shortlist -> exact-mass
rank -> admission -> transfer/publication -> target use. Current page5 cause
is unknown. Trace retains only a bounded raw-ID prefix; absence from that
prefix is not absence from a48-page coarse list. Do not widen imaginary
budgets, invent zero scores or special-case a fixture/expected answer.

## Artifacts and continuation

Compact hashes/findings: `evidence/PRODUCER105_02_FINDINGS.json`, not a V10
success receipt. External raw root:
`/srv/ai/paged-kv/results/forward/105-02/source-review-fixes/`.
Relevant runs: `generation-parity-int64-current-kv-20261007T183000/`,
`continuation-first-capture-20261007T193000/`,
`canonical-mtp-matrix-corrected-int64-fresh-20261007T190000/`,
`promotion-corrected-int64-20261007T184000/`. Do not load whole transcripts.

Keep single Qwen under lifecycle lock; exact argv/next fixture in105-02.
Immich ML remains operator-stopped; stop only that authorized worker if it
returns. Never restart it automatically, never8092; leave8091 unchanged.
105-02a owns ONE32K/H16K trajectory and bulk-prefill curve, then only missing
matched controls. No unchanged400-token campaign retries or blind250K fill.
Benchmark tooling/compact execution metadata are excluded from upstream PR;
raw tensors/binaries stay external and uncommitted.
