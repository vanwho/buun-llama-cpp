# Phase 104: bounded query-aware ranking experiment

Revision: `hotpath-v10-20260914`. Amendment: `ranking104-probe-rerank-v1`.

## Purpose, scheduling, and ownership

Goal: improve useful cold-page recall while retaining fast GPU prefill/decode,
opaque Turbo4 host backing, full-L GPU Turbo4 MTP, and a small additional
memory footprint. This is an experimental deviation, not a change to the
accepted main-branch attention, storage, or MTP implementation.

Current recovery:104-04e/f/g/h and104-05 completed actual cache-owned CUDA
execution and model-backed publication/replay.104-06 is deferred after real
HTTP500 query-finalization failure. Next is104-06a/b/c/d, then104-07/08.
[RANKING104_LIVE_REPAIR](RANKING104_LIVE_REPAIR.md) supersedes earlier startup
order and campaign scope. This longer design is a reference, not required
whole-file startup context. Typed producer diagnosis precedes any repair;
fresh target/MTP sanity precedes natural recall. Do not relabel coarse scores
or isolated fixtures as production exact mass. Preserve passing owner work.

Order: 103-03 is deferred with C=254393 measured, not exact-full accepted.
Execute 104-01 through 104-08 next; then 103-04 and 103-05 consume the
experiment decision. 104-01 depends on completed 103-02a, not completion of
103-03. The unregistered 103-02b packet is retired in favor of this phase.
Do not resume the exact-fill campaign or add a backward capacity gate.
Read 103-03's compact disposition only if capacity is relevant; its raw
campaign/checkpoints are not ranking-task startup context.

103-03's request at rendered prompt 259020 failed attention-ubatch planning
after 22 committed requests. The error is not proof of CUDA OOM or a ranking
cause. Exact-capacity fitting/ignore-EOS/resume-rebind changes are retained
only in archive commit 7c162ca08 on codex/task-103-03, excluded from the plan
and experiment base. Do not cherry-pick those changes into this experiment.
Use the unchanged ordinary occupancy helper with the explicit reserves in
104-06/07. Better ranking is not asserted to fix that allocation/planning edge.

Wiretail continues in the main project and integrates coordinator metadata
normally. 104-01 creates a separate worktree at
`/srv/repos/vanwho/buun-llama-cpp-ranking-v1`, on branch
`experiment/attention-aware-ranking-v1`, from the then-current local
`plan/attention-aware-kv-paging` commit. Experimental product code, test code,
and benchmark helpers are edited and committed only in that worktree. The
worktree is now created; task agents follow Wiretail's no-Git rule and edit
that existing worktree. The outer owner preserved incomplete code in426e34562;
commit absence must not become an implementation blocker. Wiretail owns
main Git, the outer owner owns experiment milestones. Never switch the active
main worktree's branch, run a nested
Wiretail, modify the experiment's inherited WORK_STATE, or merge the experiment
back automatically. Record experiment SHA and build provenance separately.

## Verified reasons for the experiment

The 103-02a handoff reports current-query transport completion and ranking
misses at L8192/H4096 and at H200. It also records admitted/queued H2D without
publication in an earlier scaled request: ranking is not the only unproven
transition. Keep admission, transfer, publication and consumption separate.

Current code locations, relative to either source worktree:

- `src/llama-kv-cache.cpp::llama_kv_cache_context::build_kv_page_select`:
  applies router-only `ggml_turbo_wht(..., 2)` to post-RoPE target Q; accumulates
  the entire final-user span; selects using scorer_mode=1 (page Mean-K).
- `ggml/src/ggml-cuda/kv-page-select.cu::query_accumulate_*` and
  `page_select_scores`: turn arithmetic mean; per-page score loops through
  grouped query heads and reduces to the best raw head score.
- `ggml/src/ggml-cuda/kv-page-summary.cu::kv_page_summary_kernel`: catalogue
  carries min, max, mean of represented Turbo4 K per page/KV head; summaries
  are refreshed only when marked dirty and retained after eviction.
- `src/llama-kv-prefetch.cpp::llama_kv_prefetch_expand_selector_ids` and the
  synchronous path in `llama-kv-cache.cpp::apply_pager_live_policy` substitute
  `1/(1+rank)` for the actual numerical score.
- `apply_pager_live_policy`: bundles layer nominations by logical-page identity;
  takes maximum rank-derived priority, then final-layer/layer-count/ID ties.
  `LLAMA_KV_QUERY_COLD_SELECTOR_PAGES=5` and transaction maximum=8 are separate
  candidate and transfer limits. A reported two-page bundle is not proof of a
  universal hardcoded top-two selector.
- `tools/server/server-context.cpp`: actual rendered user-span detection,
  final-user synchronization, `commit_kv_pager_query`, restore/query replay,
  and historical freeze establish the safe boundary. `release()` clears fields.

Mean(Q) dot Mean(K) averages all query/key pairs. A sparse useful key can lose
to generic background even with correct transport. Post-RoPE queries at
different positions can cancel when averaged. This arithmetic establishes a
plausible loss of information, not proof that every miss has this cause.
No claim of exact dense equivalence follows from a better router.

## Chosen implementation, not open algorithm selection

Implement one opt-in `--kv-router legacy|probe-rerank` option, default `legacy`.
Legacy uses the current computation and existing I32 compact output unchanged.
`probe-rerank` uses the following algorithm. Do not add an ANN library, learned
router, file-name lookup, full-L GPU key duplicate, subblock catalogue, or new
page-storage format in this first experiment. These would confound attribution
or add persistent context-sized storage. All numerical work runs on CUDA in
production. A scalar CPU kernel is only a tiny mathematical test oracle.

### A. Four independent query probes

For each target attention layer retain actual transformed Q rows at user-span
positions `query_end - {1,2,4,8}` that are >=query_start. Deduplicate clipped
positions; use however many real rows exist. Preserve each row/head, never
average them together. The server's already located final_user_token_end is
the authority; do not include the assistant header or generated answer.
Capture rows incrementally across microbatches into reusable owner storage;
reset validity on turn/query generation, restore/cancel, slot clear/reuse.
Maintain the existing post-RoPE/Turbo4 InnerQ transform. No inverse rotation,
new RoPE setting, or MTP query substitution is part of this change.

### B. GPU candidate shortlist from existing metadata

Read the existing min/max/mean catalogue, without duplicating it. Score each
eligible page separately for each probe and grouped query head. Produce both
Mean-K dot products and sign-aware min/max estimates using the same transformed
Q. Normalize each source within its own head/probe candidate population with
stable softmax; never compare raw logits across different heads or layers.

For each layer derive shortlist width
`K=min(eligible_cold_pages, max(8,min(32,ceil(H_pages/4))))`.
These are bounded search tuning constants, not fixed context/model geometry.
Include up to ceil(K/2) pages from each scoring source, order each source by
its largest normalized head/probe probability, tie by mean probability then
logical ID, and deduplicate. If duplicates leave fewer than K, continue down
the two rankings alternately. Retain actual probabilities and identities.
Resident pages remain candidates for the later resident-vs-cold comparison.
The shortlist is deliberately wider than the eventual promotions.

Launch score work across `(page, query-head, probe)` rather than serializing
every head in one page block. Use warp/block reductions and existing CUDA
stream/CUB primitives; allocate scratch once per owner/geometry. Page validity,
causality, host source/content/summary versions remain mandatory. Scan metadata
once at the final-user boundary, not per prompt token or assistant token.

### C. Accurate key-only reranking

For each attention layer compute actual token QK logits for all current
eligible resident history pages and shortlisted cold pages. Resident K is read
directly from its existing GPU tensor/mapping. Only shortlisted cold K units
are copied from canonical host storage into bounded ping-pong staging; V is
not copied for ranking. Reuse `host_page.page.units`/logical_unit_id/key-side
descriptors and `unit.bytes->read` from the existing host-summary and H2D paths.
Gather an entire contiguous key unit, not one host call per token. Derive
row_bytes from `ggml_row_size(type_k,tensor->ne[0])`, valid rows and checked
offsets; never assume raw model layer equals compact attention ordinal.

The CUDA kernel reads Turbo4 coefficients/norms directly and applies the same
query transform/calibration and attention scale/softcap as mature target FA.
Do not materialize a complete F16 key matrix. Compute per-page/per-head/probe
`log_mass = logsumexp(valid causal token logits)` with stable running max/sum;
mask invalid/padded rows, partial pages and incompatible identities. Streaming
chunks combine using max/sum, never average page logits. A tiny decoded-key
CPU oracle verifies represented-key arithmetic; it is not an inference route.

Normalize page masses across the resident plus shortlisted-cold pool within
each layer/head/probe. This is exact on that pool, not on omitted cold pages.
For each page output peak normalized probability and mean probability. Avoid
publishing a full `[L,heads,probes]` matrix to the CPU. Across layers choose
logical bundles by maximum peak probability, then mean probability averaged
over participating layers, then support count, then logical ID. This replaces
rank-derived strength while preserving a head with a distinctive retrieval.

### D. Bounded promotion and historical commit

Compare resident and reranked cold candidates through the existing policy;
reserve query/generation-owned/mandatory pages first. Cold candidates must win
space in the available historical target, not be force-promoted just because
they are cold. Preserve logical-bundle transfers and existing per-layer views.
At most the admitted atomic H2D budget (currently eight pages), actual history
space, upload-ring capacity, and available candidates may be promoted in one
turn. Shortlist width is independent of that limit. Report a cold winner cut
off by capacity as budget-limited, not as a scoring miss.

Integrate coarse completion, key uploads/rerank completion, final IDs, H2D
publication and commit before the existing query-only replay/freeze boundary.
Attach each event/result to full identity + turn/query/content generations.
Do not keep pointers into recyclable graph allocator memory. Replay does not
start a second ranking pass for the same query. After freeze, decode and MTP
verification use the existing immutable historical set and fast attention
routes. No historical H2D/reselection per accepted token.

## Memory and latency limits

Extra persistent storage is probes, compact candidates and owner staging,
not another L-sized K/V image or another summary catalogue. Use exactly two
key-staging slots, default at most 4 MiB each, plus two matching pinned host
slots; derive rows/chunk from encoded bytes. Score processing is one layer at
a time, bounded by 2 MiB default score workspace; tile pages/heads if needed.
Probe bytes are `attention_layers * valid_probe_capacity * query_heads * D *
sizeof(float)`; all geometry comes from tensors/model metadata. Include GGML
tensor overhead, allocator alignment, CUDA/CUB temporary memory and events
in an explicit byte ledger. Defaults are tunable experiment budgets, never
claims that all models require the same bytes. If one encoded key row cannot
fit, return a concrete configuration error before a request, not CPU fallback.
Admission charges actual allocations once; preserve explicit H and full-L MTP.

Queue uploads/kernels on owner streams with event dependencies and reuse a
staging slot only after its reader completes. One final boundary drain is
allowed before commit; no cudaDeviceSynchronize, no per-layer host wait loop,
no extra fences inside prefill/microbatch/decode. Bounded owner-local duration
and copied-byte accounting is enabled only for diagnostic rows. Keep default
speed rows free of full page/Q score dumps and progress tracing.

## Test policy and decision

Implementation tasks use small production-seam mathematical/identity tests,
compiled once where changed. Central live proof is 104-06; paired performance
is 104-07. Use one Qwen process and the existing managed lifecycle, CUDA0,
B1024/U256, Turbo4 target/draft, GPU full-L MTP, reasoning off, temperature0,
400 output tokens, no newline/16-token stops. Maximum fresh input chunk16K.
8K/4K is the first A/B/A row, using the existing two-Python/three-Bash fixtures
and a tokenizer-frozen schedule crossing H while leaving >=1536 output/replay
reserve. If it cannot fit, reduce unrelated pressure input before any run.
One scaled row is L131072/H51200, C about H+8192; do not refill 120K/249K or
repeat the historical 24-case/48-run campaigns to test a rank change.

Compare legacy and experimental modes with the SAME final binary and frozen
input seed/schedule. Dense GPU controls at small geometry explain semantic
misses; a dense failure is an inconclusive fixture, not a router failure.
Use actual token QK ranks/mass in deterministic fixtures to assess selection;
do not demand exact formatting from natural answers. No forced nomination of
the answer page, answer text fed back, fabricated telemetry, or speed-based
inference that a page was promoted. A completed semantic miss is a finding.
Record nomination, admission, transfer, publication and use when available;
missing optional diagnostics do not rerun otherwise valid speed rows.

Canonical speed rows: three original prompts, warmup1x40 and measured3x400
per prompt, reasoning off and MTP enabled. Compare medians/tails and report
prefill fresh-token denominator separately from cached prompt length. Preserve
MTP acceptance target40% as a finding/repair signal, not an infinite retry
gate. Record extra memory and turn retrieval/replay latency as well as pp/tg.

104-08 produces adopt/reject/inconclusive with explicit evidence. Recommend
adoption only when accuracy improves across distinct targets, canonical pp/tg
are within5% of same-binary legacy medians (or absolute speed improves), and
memory stays within the allocated ledger. The5% is a decision tolerance, not
a task completion gate. Slow or unsuccessful experiments still finish with a
truthful recorded decision. No automatic main merge/default flip. 103-05 may
schedule explicit adoption or one narrowly measured correction; it must not
repeat the same unsuccessful campaign. Setup failures are fixed at their
owner and resumed, source/correctness failures require repair.

## Context and research pointers

Read the current packet and named section of this design, plus its immediate
compact predecessor handoff. Source paths are symbol pointers, not directions
to ingest whole 20K-line files. Main handoffs/receipts remain <120 lines where
practical; raw records, request bodies, binaries and DSOs live outside Git.

Primary inspiration, not claims of Qwen/Turbo4 performance:
[Quest](https://arxiv.org/abs/2406.10774) uses query-aware min/max page estimates;
[SparQ](https://arxiv.org/abs/2312.04985) selectively fetches keys/values;
[RetrievalAttention](https://arxiv.org/abs/2409.10516) explains why generic
vector search on keys/queries can be inaccurate. The chosen first version
reuses our existing catalogue/host machinery and avoids a new index dependency.
