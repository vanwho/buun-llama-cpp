# Phase104 production recovery: executable route, not isolated oracles

Revision: `hotpath-v10-20260914`. Amendment: `ranking104-production-recovery`.

Current scheduling authority: RANKING104_PIPELINE.md. A/B/C work is preserved;
104-04d is deferred and new104-04e/f/g/h explicitly implement its absent
cache-owned executor. SectionsD/E below are historical background, not a
request to repeat missing-symbol audits or older retries. Revised104-05 owns
publication/replay after the new cache-seam proof.

## Authority, failures and preserved work

Implementation root EXP_ROOT=/srv/repos/vanwho/buun-llama-cpp-ranking-v1.
Main receipts/state/handoffs remain /srv/repos/vanwho/buun-llama-cpp.
The experiment is still opt-in; never merge it into main or change legacy.
Existing implementation was preserved as experiment commit `426e34562`.
That commit is an incomplete implementation snapshot, not an accepted build.
The remaining copied experiment handoff is not a second state ledger.

Read only this document's task-named sections and targeted symbol regions.
Do not load prior retry JSONL, the complete work state or old phase103 fills.
No ranking implementation task should turn into another missing-symbol audit:
the absent production methods below are work to implement, not an external
blocker. Build failures require actual code correction within that task.

The original plan left too many connected systems to 104-05, and completion
of CPU oracles/compilation did not imply a working CUDA production path:

- `ggml/src/ggml-cuda/kv-page-select.cu::ggml_cuda_op_kv_page_rank` only calls
  page_rank_clear; `ggml-cuda.cu::supports_op` returns false for PAGE_RANK.
- `src/llama-kv-cache.cpp::build_kv_page_select` still returns legacy
  ggml_kv_page_select even when independent probes have been captured.
- `llama_kv_router_owner_state` exists as a data type/test only. Cache owns
  no rerank job, buffers or call to ggml_kv_page_rerank.
- Rerank returns log masses, not normalized packed exact-mass records.
  Packed shortlist decoder labels its output probe_softmax; policy rightly
  rejects those in probe-rerank mode. Never remove that provenance check.
- ggml_kv_page_rerank's builder mistakes K.ne[2] (streams) for KV-head count;
  the fixture uses two heads AND two streams and masks this error.
- Rerank compares all pages to one identity pair, although page generations
  and content versions differ. A multi-page job needs per-page expectations.
- The row-serial CUDA rerank synchronizes the block for each key row; replace
  it before using H200 so the new query-boundary work is not an idle-GPU loop.
- Staging ring poll retires failed events indistinguishably from successful
  events. Publication must never adopt scores from a failed key reader.
- Model-required test-server-query-replay was run with no --model; exit2 is
  usage, not execution or proof. Give exact commands and one model lifecycle.

Historical104-02/03/04 receipts remain scoped to their executed unit work;
they are not rewritten as production proof. New104-04a/b/c/d supply the
missing CUDA/coarse/math/owner/boundary implementations before revised104-05.

## A. Production coarse shortlist — owner104-04a

Sources: ggml/src/ggml-cpu/ops.cpp::ggml_compute_forward_kv_page_rank is
the existing mathematical oracle; kv-page-select.cu is the CUDA implementation;
src/llama-kv-cache.cpp::build_kv_page_select, capture_kv_routing_query,
pager_selector_complete and synchronous apply_pager_live_policy own wiring.

1. Implement page_rank_score/normalize/aggregate/select CUDA launches on
   ctx.stream(). Reuse bound layout [D,3,KVheads,P] and metadata eligibility.
   Each head/probe computes dot(Q,meanK) and sign-aware min/max upper score.
   Use stable max-subtracted normalization independently for each source and
   head/probe. Aggregate page peak/mean without averaging probe vectors.
   Handle +inf ties as equal finite probabilities; NaN/-inf cannot win.
2. Tile heads/probes to keep temporary score+sorting memory <=2MiB. Reuse
   backend pool allocations; do not create a full-L-by-query logits tensor.
   Keep full-P reduced page strength arrays on GPU. Use bounded GPU top-K
   selection/sort, not CPU fallback or D2H of all scores.
3. Cold width K=min(eligible_cold,max(8,min(32,(H_pages+3)/4))). Shape can
   reserve min(logical_capacity,K_bound); invalid slots carry logical=-1.
   Use ceil(K/2) from mean and remainder from bound, deduplicate and refill.
   Resident output contains eligible incumbent history, not current/user/tail.
   Width is distinct from max_h2d_pages8. Remove legacy five-candidate caps
   in experiment-only shortlist/mailbox sizing, not legacy production mode.
4. In build_kv_page_select branch on router_mode. Legacy path unchanged.
   Probe mode calls ggml_kv_page_rank with probes_op as src5 so GGML has an
   actual capture->rank dependency; persistent probe leaf alone is NOT enough.
   src6 is persistent validity. Do not run mean accumulation solely to feed
   ranker: keep transformed routing_q for shape/source0 and independent probes.
   Nonfinal ubatches root capture only. Query replay/generation root neither.
5. Store explicit output format/byte stride/offset in pager_routing_output
   and submission segment; do not derive mixed byte offsets from a global
   raw_count*16. Legacy segment uses4 bytes, packed uses16; immutable copied
   descriptors authenticate logical IDs and page/content/query versions.
   Coarse results enter the job shortlist, NEVER policy as exact_mass.
6. CUDA supports_op becomes true only for implemented shape/type constraints.
   An unavailable probe pipeline must refuse opt-in clearly, never place
   PAGE_RANK on CPU and move probes/catalogue back to host.

Example production branch (adapt variable names, not its semantics):

```cpp
if (pager.router_mode() == llama_kv_router_mode::legacy) {
    return ggml_kv_page_select(/* existing unchanged inputs */);
}
return ggml_kv_page_rank(ctx, routing_q, summary, metadata, membership, query,
    probes_op, state_it->probe_validity, resident_width, cold_width,
    page_tokens, -1, 0, target_attention_scale);
```

Execute the real CUDA op against CPU oracle for 1024 logical pages and
H16/H200 metadata, mixed byte formats and late-layer IDs. Tiny tensors only.
Do not defer actual CUDA execution to104-06 or use source scans as proof.

## B. Correct encoded-K math and GPU normalized records — owner104-04b

Sources: ggml.c::ggml_kv_page_rerank; ggml-cpu/ops.cpp scalar implementation;
ggml-cuda/kv-page-rerank.cu; ggml-cuda.cu support/dispatch;
tests/test-cuda-kv-page-summary.cpp::run_rerank.

1. Actual cache K shape is [D*KVheads,physical_rows,streams]. Derive
   kv_heads=K.ne[0]/Q.ne[0]; require exact division and Qheads%kv_heads==0.
   K.ne[2] is streams, never heads. Validate staged K packed width matches,
   ne[1] is staged row capacity and ne[2] valid streams. Fix builder, CPU and
   CUDA support checks together. Fixture must have heads2/streams1 and a
   separate streams3 case: head count and stream count must differ.
2. Replace global identity[2] by I64[2,P+1]. Column0 contains query_generation
   and job_serial; column p+1 contains expected page_generation/content_version.
   Descriptors remain [10,P]; compare desc5/6 against identity[2*(p+1)+0/1].
   Also require probe validity0==query_generation and per-probe valid flag.
   Other sequence/session/rollback/epoch authentication remains owner-side;
   integers never travel through floats. Mixed page versions are valid.
3. Attention math uses transformed coefficient-domain Q exactly once. GQA
   maps qhead/(Qheads/KVheads). API attention_scale is the ordinary target
   dot scale; if softcap>0 use softcap*tanh(dot*scale/softcap). Mature
   fattn-common.cuh internally divides scale before tanh; mirror the
   mathematical result, not its adjusted scale PLUS another division.
   Pass actual target hparams, not fixture0.25 or default1 into production.
4. Replace serial-row/block-wide synchronization with warp-parallel rows.
   Start one block(page,kvhead) with8 warps; warp processes row=warp+i*8,
   lanes stride through D and shfl-reduce dot. Reuse encoded row tiles over
   grouped Qheads and four probes, accumulate warp-local LSE, then one block
   reduction per output tile. No __syncthreads for every one of256 key rows.
   Preserve norm/centroid representation and causal absolute probe positions.
5. Resident K is read in place. Each cold chunk descriptor points at staging
   slot rows (not original physical slot), carries absolute first position
   advanced by row_offset, and writes its persistent candidate state index.
   State [2,P,Qheads,4] initializes (-inf,0) once per layer/job. Chunk LSE
   combines in that state; do not initialize for every chunk. Keep completed
   page output until all chunks of the layer have been consumed.
6. Add GGML_OP_KV_PAGE_MASS / ggml_kv_page_mass(ctx,state,descriptors,identity,
   validity). Output I64[2,P] is the existing16-byte probability record.
   CPU oracle + CUDA implementation, op names/counts/meta/RPC/dispatch updated
   exactly beside rank/rerank. Normalization reads final (m,s) for ALL resident
   history+cold shortlist candidates per head/probe, then peak and mean valid
   channels per page. All-empty channels contribute nothing; +inf/NaN are
   rejected in exact represented-key masses, not silently converted to wins.

Example with two pages/channels: logmass [log3,log1] -> probabilities [.75,.25];
another channel [log1,log3] -> peak [.75,.75], mean [.5,.5]. Omitting resident
pages from denominator would give incorrect cold priority. Output is exact
only within this candidate pool, not dense equivalence across omitted pages.

Tests run real CUDA: mixed versions, stale single page, heads!=streams,
negative/empty/tail/causal rows, nonzero softcap and split chunks. Use double
scalar reference; compare logmass and peak/mean within stated tolerances.
Do not dequantize whole pages to F16 or read full logmass matrices to CPU.

## C. Persistent cache owner, key staging and terminal events — owner104-04c

Sources: llama-context.cpp::initialize_kv_pager and destructor;
llama-kv-cache.h pager private members; set_kv_pager, destructor,
end_kv_pager_turn/invalidate_pager_query_probes; llama-kv-prefetch.{h,cpp} ring;
llama-kv-pager.h resources and host_catalog; ggml-backend.h async/event API.

1. Add one persistent dedicated router ggml_backend_ptr to llama_context,
   created on the same target CUDA device only for probe-rerank. Do NOT share
   pager_transfer_backend with its existing host capture worker. Expose
   non-owning backend + pinned buft via new ranking_backend/ranking_host_buft
   fields in llama_kv_pager_resources and corresponding pager accessors.
   Use ggml_backend_dev_init and ggml_backend_dev_host_buffer_type; no CUDA
   headers/raw streams in portable llama cache source, no extra model.
2. Cache owns unique_ptr<llama_kv_router_job> (forward declare in cache.h;
   implementation in llama-kv-router-job.{h,cpp}, add existing src CMake list).
   It owns ring, two host pinned buffers, two device encoded-K tensors,
   immutable query/page descriptors, graph contexts, persistent per-layer
   score state, packed result staging and terminal events. No graph-arena
   pointers across scheduler reserve. Probes remain the existing persistent
   cache buffers. Serial per-layer scratch keeps score workspace <=2MiB.
3. Allocate slots once when attaching admitted probe pager: each host/device
   slot<=4MiB, two pairs; use allocation ledger with actual buffer sizes and
   alignment. No extra L-sized tensor/catalogue. Failure unwinds all owned
   allocations and clearly refuses opt-in; never lower H/B/U or fake admission.
4. Extend ring submission with a by-value chunk task (candidate_index, compact
   layer_index, row_offset, rows, query_generation, content_version). Enqueue
   callback receives it explicitly, not a mutable ambient current-page index.
   Store page holder/chunk before enqueue; failed enqueue joins/releases any
   started work before clearing. Ring completion returns success/failed/cancelled
   plus ticket identity; current poll() counts retirement and hides failures,
   so add a typed poll_completed API for production. Keep compatibility wrapper
   for existing tests if desired. Latch job failure; never normalize incomplete K.
5. Host source resolves canonical bundle using host_catalog()->find_page,
   fallback id.attention_layer=UINT32_MAX as existing policy. Unit key is
   compact_layer*2 with side key, type TURBO4, exact row_bytes and content
   version. Unit.bytes->read copies contiguous complete rows into pinned slot;
   hold shared page/unit chain until reader terminal. Recheck before/after read
   and at adoption. No per-token reads, V read or GPU->CPU resident-K copy.
6. Callback enqueues tensor_set_async(router_backend,device_slot,host,...),
   descriptors, the actual ggml_kv_page_rerank graph, then event_record on
   same backend. Only that event's terminal state permits slot reuse. Use
   preallocated graph tensors/contexts; do not rebuild buffers per chunk.
   When both slots are busy, bounded producer waits/polls only the oldest owned
   terminal event; never cudaDeviceSynchronize or model-graph fence per page.
7. set_kv_pager(nullptr), end/reset/cancel and context destruction retire job
   events before free. Detach while router backend is still alive: context
   destructor already detaches memory before kv_pager_owner.reset; reset router
   backend AFTER detach. Destructor guard is idempotent. Preserve published
   residency mappings; job cancellation invalidates only unpublished job data.

New unit test test-kv-router-job invokes the SAME production class with fake
event adapter, not a copied state machine. Cover reader failure propagation,
pending last chunk, two-slot reuse, stale identity, cancel and repeated detach.
CUDA fixture then constructs the real owner buffers/events, runs tiny encoded
keys and normalization on its backend, and checks actual allocation release.
No Qwen model required for these lifecycle tests.

## D. Real boundary-to-exact-record pipeline — owner104-04d

The job methods must be called from production, not just available as helpers.

```cpp
// In commit_kv_pager_query, AFTER final-user target synchronization:
if (pager_->router_mode() == llama_kv_router_mode::probe_rerank) {
    const auto result = complete_router_query_job(sequence_id, turn_id);
    if (result != router_job_result::ready) return false;
    // Only completed owner-authenticated PAGE_MASS output becomes exact_mass.
    apply_pager_live_policy();
}
// Existing retrieval_commit/query_replay transition follows publication gate.
```

1. Coarse callbacks/sync decoder authenticate packed PAGE_RANK output and
   place it in cache-owned job per-layer shortlist, not prefetch policy mailbox
   as exact evidence. Keep copied descriptor/content/query/rollback identities.
   All resident eligible history descriptors join the shortlist pool; reserve
   current/query/generation structural pages separately. No full-L K reads.
2. At final-user commit, lock a new job identity (sequence/session/turn/query/
   rollback/epoch) once. Require all participating attention-layer captures
   valid for that query. Finish coarse events; collect bounded cold sources;
   run resident scoring, staged cold scoring, then PAGE_MASS per layer on
   the dedicated backend. One job owns each sequence, no replay duplication.
3. After the last compact packed D2H event, validate every record against
   immutable per-layer descriptors and current content identity. Factory
   make_exact_rerank_candidates(completed_owner_result) alone assigns
   exact_mass; a coarse caller cannot set that tag. Reject partial/stale/failed
   jobs, nonfinite scores, unknown IDs. Do not reuse coarse probabilities.
4. Add pager_exact_rerank_candidates_ owned by cache/job or an equivalent
   typed completed batch. apply_pager_live_policy consumes it only for current
   provisional query; coarse mailbox never masquerades as that batch.
   Aggregate canonical logical bundle peak=max across layers, mean=mean of
   participating-layer means, support-count and logical-ID ties. Keep all
   resident winners in that same score space. Bound host sort to H+K bundles,
   not L pages; candidates may be per-layer but bundle sorting is small.
5. Remove experiment-only cold-budget min(query_cold_selector_pages5).
   Promotions limited by available historical slots/max_h2d_pages8/ring bytes,
   not shortlist width or old selector5. Cold rejected by real budget is
   budget_limited, not a scoring miss. No forced answer-page nomination.
6. Run this job ONLY at final-user commit. synchronize/apply_policy during
   ordinary prefill can maintain existing pages but must not stage cold rerank
   or consume incomplete probes. query_replay/generating/MTP verify do neither.
   Completed/failed/cancelled stage/result is explicit; no generic bool that
   treats pending as success. Repeated commit is idempotent for same query.

This task's tiny fixture drives the production job, real PAGE_RANK->RERANK->
MASS calls and candidate factory; it ends with authenticated exact candidates.
Publication/replay is next104-05, so do not claim full history commit yet.

## E. Publication then replay/freeze — revised owner104-05

Keep existing transaction design. Rerank does not justify bypassing its gates.
Apply policy with the exact candidates, remember admitted transaction IDs and
expected winning bundle identities/content versions. Poll/complete existing
transfer events and reap completed transactions at the final-user boundary;
do not freeze just because transfers were queued or an old resident map exists.
Only successful table publication at the expected generation makes the job
ready. Missing publication/failed H2D is a failed query, not unchanged success.

Relevant seams: llama-kv-cache.cpp::apply_pager_live_policy transaction code,
commit_kv_pager_query, freeze_kv_pager_history; server-context.cpp's final-user
server_query_replay_transition callback, checkpoint restore and replay count.
Use one replay only after changed mapping, none after unchanged mapping;
ensure the replay does not restart the completed retrieval job. Target/draft
checkpoint/GDN carry use existing production restore, not separate callbacks
asserting what production should do. Cancel before ready cannot freeze.

Small no-model transaction tests first. One integrated Qwen production-seam
fixture is required only here, with --model and --router probe-rerank added
to test-server-query-replay if needed. Stop/free the managed model before
the fixture; use the canonical model path, L8192/H4096, B1024/U256, codecs
Turbo4, native GPU MTP, tiny query/generation. Restore or retain managed
candidate according to successful lifecycle. Never run a second Qwen.

Implementation gates demand that the requested code and tiny numerical/
lifecycle fixtures actually execute; semantic natural misses remain findings
in104-06/07, not endless repairs. A test invoked incorrectly must be corrected
and rerun in-task, not passed/deferred with an exit2 usage artifact.

## Git/worktree rule and completion scope

Task agents follow Wiretail's no-Git rule. Edit implementation only in existing
EXP_ROOT and coordinator handoffs/receipts only in MAIN; do not recreate the
experiment, switch branches or load its copied execution state. Planner/outer
owner snapshot426e34562 preserves prior code. Later outer owner checkpoints
experiment milestones; lack of an agent commit is NOT a code implementation
blocker or reason to omit a truthful receipt. Each receipt records actual
experiment HEAD PLUS dirty source diff hash and build/DSO identity as needed.
Do not mix a main coordinator SHA with a claim the experiment was merged.
