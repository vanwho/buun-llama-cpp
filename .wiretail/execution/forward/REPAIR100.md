# Active repair after the incomplete 100-02 campaign

Revision: `hotpath-v10-20260914`. Amendment: `forward-generation-admission-20260929`.
This overrides the stale request ceiling and completion statements in 100-02.
Read this compact document, not the old retry journals or original source plan.

## What is actually known

- Preserve implementation checkpoint `ccc61f52545ddca9fe6b17a258119b04bd44ad8c`:
  current ubatch query pages are included in selected attention, and a packed
  capacity change invalidates graph reuse. Focused CUDA attention test passes.
- 100-02 attempt-02 rendered 5,242 input tokens with L=8192/H=4096,
  page=256, G=512, B/U=1024/256, all-Turbo4 GPU-native MTP. Prefill crossed H;
  generation failed after 133 outputs with `begin_write_batch: no_victim`.
  Latest local MTP counters were accepted/drafted=76/92 (82.61%). No terminal
  usage/timing: this is a useful failure trace, NOT a completed speed row.
- Attempt-01 p0 warmup: 201.74 fresh prompt tok/s, 43.86 decode tok/s,
  MTP=25/26. One measured row: 137.46 server prompt tok/s, 32.67 decode,
  MTP=44/80. Replay adds work; record useful fresh/effective and server
  processed-token rates separately. These isolated rows are not medians or
  proof of a matched CPU-offload advantage.
- Latest run queued 20 canonical host seals, but H2D useful bytes=0. Thus
  no new natural cold-promotion proof exists on that candidate. A post-error
  cleared slot cannot explain the allocator's first refusal.

Raw provenance: `/srv/ai/paged-kv/results/forward/100-02/attempt-01/` and
`attempt-02/`. Inspect only the necessary structured row/first-error fields.
100-02 is deferred, not passed; its usage ledger and raw evidence are retained.

## Decisions already made for the implementer

1. Keep Buun's existing pager, inclusive Turbo4 host store, selector and
   transactions. Retrieval is once at the final-user boundary. Replay only
   that query on a changed history, then freeze historical selection.
2. First fix **actual generation admission**, not a speculative new spill
   subsystem. The live failure is after prefill, and the one-row generation
   fixture does not exercise production `begin_write_batch` preflight.
3. R/G/H are unique-page budgets. Guarantee generation capacity by removing
   dispensable prior-history residents at query commit, not by freezing all
   populated physical slots. Query/tail pages must not become accidentally
   frozen historical pages. G is a minimum, not an output limit.
4. Batch reservation must finish a prior frontier before testing its victim
   eligibility, protect every page written by the batch, and use one consistent
   eligibility/ownership plan. Never unpin an in-flight graph/speculative page.
   Seal asynchronously; wait only at an actual affected capacity boundary.
5. GPU performs all target/draft/GDN compute. Host handles encoded backing.
   No CPU/F16 gather, synchronous per-token catalogue scan, full-L checkpoint,
   selected_reference production route, or per-output historical H2D.
6. Noncontiguous selected packed FA is presently faster than direct paged FA
   in matched 99-01 fixtures. Preserve shape-aware dispatch; optimize its
   delta-copy/graph ownership, not replace it with the slower kernel by fiat.
   Concrete policy repair: committed frozen history must be the attention
   view's base. The current half-H default/first-layer advisory fallback can
   discard admitted pages.100-02c reconciles A at admission and builds the
   exact committed union plus mutable tail, without mid-generation reselection.
7. Latest MTP acceptance disproves a universal zero-acceptance claim. Verify
   changed-query replay/carry/rollback on the repaired candidate and distinguish
   state defects from genuinely different sparse-target versus dense-draft
   distributions. Do not page or truncate full-L draft KV to raise acceptance.
8. Use L8192/H4096 for small live pressure; B1024/U256 for comparable speeds.
   Normal output budget, EOG allowed. No exact filename/YES/NO gates and no
   lifetime 36-request cap. Full canonical campaign = 36 valid rows across
   three placements on one candidate; smoke/repaired rows are separate.

## Ordered work and the next-run contract

`100-02a` budget/query ownership -> `100-02b` batch/ring recycling ->
`100-02c` measured hot-path cost reduction -> `100-02d` MTP state integration ->
`100-02e` natural promotion/frozen-decode proof -> `100-02f` staged speeds ->
`100-03` cost-based repair scheduling -> `100-04` final parity/cancellation ->
`101-01/02/03` scale -> `102-01` final findings and concrete successors.

Every implementation task uses one named focused regression, not the entire
old benchmark matrix. Config/identity/auth failures are repaired in-task.
On a real defect, retain the smallest reproduction and implement the owner
repair. If a substantial prerequisite must become a separate task, the plan
owner records the original as deferred with explicit replacement ownership,
inserts the repair and revalidation before its consumer, and reconciles the
current pointer. Never leave a test in_progress waiting for a later repair
that Wiretail cannot reach. Missing proof is never `done`.

Run a receipt's exact completion check **before** `task_state.py complete`.
The state CLI refreshes scheduling but does not prove executable behavior.
If a runner/agent prematurely marked done, inspect the receipt and use the
supported `reconcile-current` operation for derived pointers; a deliberate
planning correction of an invalid terminal status must preserve all usage,
implementation and historical evidence. Do not rerun useful source work.

Production geometry remains model/device-derived. Test H <=49,152 is a local
ceiling, not a compiled constant. Draft context always equals logical L. Do
not tune G by shrinking L, turning off MTP or changing the canonical B/U.
