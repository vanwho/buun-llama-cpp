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

## October7 setup repair and immediate longer row

Wiretail already used --yolo; root-owned mode700 result directories caused
the Permission denied errors, not a Codex sandbox. Both named hotpath29
directories now have a ninja-only read/traverse ACL (including inherited
read access), retaining private other-user permissions. Protected authorized
reads use scoped sudo -n; do not recursively ingest historical results.
Tool launches now spell out --dangerously-bypass-approvals-and-sandbox in
assessment/fresh/resume paths. This does not elevate the Unix user.
`/srv/wiretail` is not a Git worktree; its live tool, README and regression
are updated in place. The portable runtime patch is archived under
`.wiretail/execution/tooling-fixes/wiretail-permissions-20261007.patch`, outside
normal task startup context. No Git commands were run against `/srv`.

Main promotion driver previously hardcoded L8192 despite105-02's explicit
L16384 command. CLI --context/--hot-pages now bind all identity, prompt-sizing
and allocator checks; --print-server-command emits matching argv without
launch. Tracing is optional. Validate the repository fixture manifest before
service access; /srv/ai/paged-kv/fixtures is not this fixture root.

Latest105-02 handoff: production direct24/4 Q1/Q3/Q256 numerical errors are
0.000010/0/0 vs oracle. The materialized ordinary-FA control atQ256 differs
by0.003883 vs oracle; it is NOT proof that the selected packed production
dispatcher passes. Tokenizer-derived generation parity is still pending.
No source sampler repair or new coherent selected/MTP proof is established.
The completed selected recall remains slash filler,374/37 proposals/accepts.
Preserve the active fixture work; do not repeat its failed patch anchors.

105-02a starts with one32K/H16K repository occupancy+recall row toC~26000,
before the small paired canonical matrix. Its completed goal_miss is useful
ranking/speed evidence, not a reason for an unchanged retry.105-03 retains
L262144/H51200 after any demonstrated correctness repair. Existing ordering
is unchanged; no active task/status reset or server action is needed here.

Validation of this setup repair:20 offline promotion-harness tests pass;
Wiretail shell syntax, unrestricted-launch regression, task-state structure
and active-plan validation pass. No model/build/service action was run.
The broader Wiretail options suite has five stale model-policy test failures
(removed family resolver/old assessment signature/default assignments); these
are separate from this passing launch regression, not new runtime evidence.
Benchmark-only harness/README changes are isolated in commit d6f6f7c55 and
must remain excluded from an upstream product-code PR alongside other local
tools/server/bench campaign tooling. The next runner owns the unfinished
105-02 CUDA/replay fixture changes and its live proof/receipt; do not stage
those partial edits as completed proof during this planning checkpoint.

## October7 first-row producer and MTP rollback source repairs

The new short model fixture reports nonfinite target logits immediately after
`llama_decode`, before speculative processing/publication/restore: initially
three selected pages/519 rows, later the first provisional query token with
4089 causally visible rows. Neither a nonempty mask nor the passing direct
kernel fixture proves the production packed producer is safe.

Code review found three independent defects:

1. Packed copies read raw cache views with no graph dependency on this
   layer's `GGML_SET_ROWS` write. The active task's dependency patch is retained.
   Nested ggml views flatten to the allocation owner and absolute `view_offs`.
   The copy's view of the write result must use a relative layer offset;
   the repair checks common owner/base/type and byte bounds. It does not
   change selected membership, row IDs, promotion or route policy.
2. Packed staging copies only valid rows into a page-rounded tensor. The
   ordinary Turbo4 MMA loader reads every capacity row, including padding,
   before masking, unlike the direct-paged loader's invalid-row zero path.
   519/768 leaves249 uninitialized rows;4089/4096 leaves7. Arbitrary half
   norms may be NaN, and a -inf mask cannot sanitize them (or NaN V operands
   multiplied by zero). The shared slab's tail is now zero-filled once per
   graph execution, not once per layer or across all H rows. CPU/CUDA
   `GGML_FILL` now supports only Turbo4 zero via encoded norm-zero blocks;
   CUDA uses captured stream-ordered memset, without a host fence or expanded
   F16 cache. All layer copies preserve the initialized padding.
   `refresh_selected_data` also wrongly compared nested views' flattened
   owners with intermediate views, rejecting graph reuse. It now checks
   actual allocation owners and exact absolute byte offsets while retaining
   physical/layout/row-count compatibility checks.
3. After `replay_accepted_prefix` repaired the draft's sampled+accepted rows,
   `common_speculative_rollback_dft` sent `target_restored_without_draft`.
   `sequence_transition` invalidated pending carry AND reset the applied
   rollback guard; the following `process` took its target-only path and
   cleared the ENTIRE draft sequence. This is not rejected-suffix cleanup.
   Successful paired repair now retains accepted carry and idempotence;
   native accepted frontier/hidden-buffer sizes are checked before replay.
   Genuine unpaired checkpoint restores continue to invalidate carry.

These are source-confirmed defects; their individual contributions to the
observed model failure and acceptance are not yet measured. The repeated
37/374 is response-local data (9.893%), not a hardcoded acceptance count.
The frozen deterministic temperature0 workload repeats the same poor token
trajectory. The erroneous carry reset additionally creates no-draft recovery
cycles and loses draft attention history; corrupt target rows can independently
poison hidden handoff and verification. Do not retune ranking or fabricate
acceptance as a repair for either defect. Sparse target/full-history draft
disagreement may still limit acceptance after finite/state parity is restored.

105-02 owns the short real-production finite/parity and two-transaction
partial-rejection check before its one saved live outcome.105-02a then runs
the bounded32K finding and matched canonical prompts. Stop only genuine
execution failures; coherent semantic/MTP misses are findings for a measured
producer repair, not repeated unchanged large campaigns. Keep diagnostics
fixture-only and preserve full-L GPU Turbo4 draft, B1024/U256, automatic
production routes and once-per-turn retrieval. No task state was reset by
this review; unfinished active fixture edits are retained separately.
