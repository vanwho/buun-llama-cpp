# Selected generation: source/evidence correction, October 7

Revision: hotpath-v10-20260914. Current for105-02 onward. No test, build or
service change was performed for these source-only edits.

## 105-01f facts

Selected request-03/record.json under
/srv/ai/paged-kv/results/forward/105-01f/selected/cases/probe-rerank/PY_MERGE_03/
reports374 drafts /37 accepted (9.893%),400 output tokens,38.002 decode tok/s.
Zero /slots counters do not prove MTP absent. The coherent opening becomes
slash filler: not useful speed/recall. A1/B bulk prompt processing was about
1316.6/1398.0tok/s. Cached83-token A2 at130.52tok/s is not bulk prefill.
Requested fact page8 was resident; unrelated page5 promotion stays separate.

parity-cuda.log maxima: direct-packed0.002927, direct-oracle0.000016,
packed-oracle0.002934. Only direct matches oracle within0.000016. The2:1 GQA
materialized control disables fused Turbo MMA. Actual Qwen6:1 production
dispatch/current writes/graph reuse/speculative rollback need105-02 proof.
Do not loosen0.003 or claim a kernel fixture certifies natural retrieval.

## Consumer and accounting edits

apply_pager_live_policy stores nominating-layer provenance, but
llama_kv_attention_committed_pages already appended ALL frozen R members.
prepare_kv_attention_graph builds a common selected view. No evidence of R dropped
because a layer did not nominate it; do not invent another union repair.

October7 src/llama-context.cpp::prepare_kv_attention_graph removes redundant layer
copies/union/intersection from replay/decode/verify, consuming frozen R
directly with identical ID/version/residency checks. Current query/tail and
resident Q/output stay visible. Provisional prefill retains old routing.
No new transfer/workspace/rerank/fence; reduced H-bounded host work, not a
claimed semantic cure or measured speedup.105-02 must build/verify it.

MAIN run-pager-promotion.py records validated completion-local
draft_n/draft_n_accepted pairs. Missing/invalid pairs stay unknown; drafted0
has undefined acceptance. Slot snapshots are diagnostics, not cross-request
delta evidence. Generation proposal proof uses the final response. Native
sampler/acceptance is unchanged.

## Invariants and ordered owners

Sparse target may legitimately disagree with full-L draft, but SAME-MAP
scalar target vs batched verification must agree within numerical bounds.
Replay/rollback/carry must share one committed frontier. Verify those before
retuning ranking. If they pass and sparse generation still degenerates, fix
bounded structural/query/history membership at the turn boundary, not
counters/transport/per-token retrieval. Fact matches cannot rescue slash tails.

105-01g integration remains unchanged.105-02 builds/proves edits, actual-Qwen
numerics and generation state;105-02a paired recall/canonical MTP/speed precedes
105-03 scale. Misses insert targeted implementation before scale.105-04/05
review current measurements. Load this note/current packet/immediate handoff,
not transcripts; preserve raw history/hashes.
