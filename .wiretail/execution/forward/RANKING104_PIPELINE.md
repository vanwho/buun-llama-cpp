# Ranking104: finish the executable cache-owned pipeline

Revision: `hotpath-v10-20260914`. Current amendment: `ranking104-pipeline-20261005`.

## Why104-04d failed

Source inspection on2026-10-05 found an incomplete call chain, not unavailable
CUDA. `llama_kv_cache::complete_router_query_job` authenticates captures and
then requires `pager_router_job_->owner_state().stage == ready`. It never
calls `submit`, builds RERANK/MASS graphs, or calls `complete_query`. Production
can therefore never manufacture the terminal result that this gate expects.

`test-kv-router-job --cuda` stages four arbitrary bytes, supplies a fabricated
probability-one record to `complete_query`, then invokes two independent
kernel executables through `std::system`. Those are useful lower-level tests,
not a test of cache -> actual keys -> GPU masses -> exact candidates.
Attempt4 repeated these tests without implementing the absent executor.
The old packet demanded too many connected components while providing only
a conditional that checked their result. This amendment supplies the missing
implementation contracts and separates them into104-04e/f/g/h.

One further lifetime hazard is concrete: `enqueue_copy` records its event
immediately after H2D. `llama_kv_rerank_stage_ring::poll_completed` frees the
slot and its page holder when that event completes. Adding rerank AFTER
polling would let the next copy overwrite keys while the kernel reads them.
The event must cover H2D AND its key-reader kernel before slot reuse.

Policy also reserves resident slots for cold candidates before comparing
their exact strengths. New probe-mode policy must actually compare resident
and cold winners together; mere transport completion does not improve ranking.

Another integration trap: probe validity uses accumulator/capture generation,
while candidate identity uses the separate route generation. GPU RERANK/MASS
compare validity[0] to identity[0]; plan must carry capture generation there
and preserve route identity separately. Standalone equal-number fixtures did
not exercise this distinction. PAGE_RANK also currently receives literal1.0
scale; derive real attention scale instead, with exact softcap in rerank.

## Scope and source ownership

Implementation only in existing worktree
`/srv/repos/vanwho/buun-llama-cpp-ranking-v1`, branch
`experiment/attention-aware-ranking-v1`. Main `.wiretail` owns state, task
packets, compact handoffs and receipts. No task-agent Git, no automatic merge
of experimental code into the main plan branch, no flip of legacy default.
Read this compact document, the current packet, and its immediate handoff.
Older RANKING104_IMPLEMENTATION is historical A/B/C guidance, not the startup
authority for new pipeline work. Do not read raw retry JSONL as task context.
Preserved source snapshot: `83e5cbfdaa76e57ccc7716131e0f1b2a55202e20`.
This includes04a/b/c/d groundwork; it is not a completed integration claim.

Preserve completed CUDA coarse shortlist, transformed multi-Q probes,
heads/streams math, per-page identity checks and bounded staging allocations.
Use actual Turbo4 encoded keys; never decode a full key image on CPU or upload
V solely for ranking. Promotion still uploads the winning complete K/V bundle.
Keep mature attention routes, asynchronous host sealing, full-L GPU Turbo4
native MTP and B1024/U256 unchanged. Rank once at final-user boundary; no
historical ranking/transfers on each generated or MTP-verification token.

## Executable order and contracts

1.104-04e: cache supplies an immutable per-layer execution plan: GPU probes,
   borrowed resident K, exact resident descriptors and bounded canonical cold
   key readers. Empty cold shortlist is a valid resident-only plan.
2.104-04f: SAME cache-owned `llama_kv_router_job` executes that plan:
   resident RERANK once -> staged cold chunks and RERANK -> GPU MASS -> compact
   packed readback -> authenticated owning exact batch. Reader events own slots.
3.104-04g: final-user cache commit invokes executor; bounded common resident/
   cold ordering feeds real policy. No early-ready assumption or forged masses.
4.104-04h: one tiny CUDA fixture enters the actual cache adapter and executes
   that chain, including a host-key needle outranking a resident distractor.
5.104-05: transaction publication, replay and frozen target/draft state.
6.104-06/07/08: measured natural recall, paired speed and adoption verdict.

104-04d is deferred, not passed; its absent integration proof is owned by
104-04h now. Keep its old artifacts and usage, but never retry its broad audit.
Do not turn missing planned methods into external blockers: implement them.
Low-level fixtures cannot be relabeled as integrated production proof.
Implementation milestones prove their own API with tiny fixtures, not Qwen
semantic answers; semantic success/failure is evaluated later as a finding.

## Memory/performance boundaries

Two pinned encoded-K slots and two GPU slots, each <=4MiB, reused across
layers/chunks. Existing GPU hot keys are borrowed, not duplicated. Per-layer
state/scores bounded to2MiB and reused; account CUDA MASS temporary pool bytes
too, not just persistent buffers. Tile query channels if necessary. No L-sized
F16 K copy or CPU softmax. CPU handles small descriptors/readers and bounded
H+K winners. Slot backpressure may wait for the oldest reader event; not a
whole-model synchronization after every page. Final readback/event is waited
once per query; events/stream dependencies order the rest.

The production pipeline may expose scoped counts (plans, key readers, RERANK,
MASS, records, transactions) for one integrated fixture. These report executed
operations, not counters incremented merely to satisfy a receipt. Disable
detailed timing by default. Final speed testing belongs104-07, not every task.

## Evidence/iteration rules

Build incrementally with `--parallel 16`. No fresh full rebuild per retry.
One model in VRAM only for later server tests; tiny new fixtures need no model
and need not stop the managed service. Exit77 is CUDA unavailable, not pass;
fix setup within the task, or record an actual external blocker once.
Receipts bind experiment HEAD plus dirty-source hash, executable/DSO hashes,
executed command and raw output. Raw output goes under
`/srv/ai/paged-kv/results/ranking104/<task>/`, never Git.
Main receipt proof name is specified in each task, not invented afterwards.
Keep handoff <=100lines: implemented functions, exact next seam, test result,
source/build identity. A missing full model benchmark does not block a tiny
implementation milestone; it also cannot count as final promotion proof.
