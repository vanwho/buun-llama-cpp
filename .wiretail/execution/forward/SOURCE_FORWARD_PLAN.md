# Buun Attention-Aware KV Paging — Recommended Forward Architecture and Task-Plan Amendments

**Purpose:** This is a planning/engineering handoff for an LLM maintaining `vanwho/buun-llama-cpp`, branch `plan/attention-aware-kv-paging`. It is intended to be ingested before revising the current Wiretail execution plan and creating the next implementation tasks.

**Current branch reference when this document was prepared:** `e5e5f5c2bb32a5e44f447c2449cb7bc2ad678b55` (`merge(93-11m)`).

**Primary goal:** Keep the existing Buun implementation and finish it into a fast, dependable, configurable **bounded GPU hot target-attention KV cache backed by the full target-attention KV history in CPU RAM**, optimized for Qwen3.8-27B hybrid Gated DeltaNet + full attention, Turbo4 target K/V, and native MTP. Decode/thinking should not read historical target KV from CPU RAM. Cold historical KV should be selected and promoted at a safe turn boundary, then the historical attention view should remain stable throughout the generated turn.

**Design reference:** current `kvmem/kvmem-llama.cpp`, especially `docs/architecture.md` and its query-replay/tiering design. Do **not** import KVMem as a second pager or replace Buun's current subsystem. Reuse the best KVMem semantics on top of Buun's existing Turbo4-native substrate.

---

## 0. Executive directive

Do **not** restart the implementation and do **not** layer KVMem's library beside the current pager.

The current Buun branch already owns the hard low-level pieces needed for a superior Turbo4-specific implementation:

- bounded logical→physical target KV residency;
- canonical host backing and page identity/versioning;
- async H2D/D2H transfer machinery;
- transaction-safe publication/rollback;
- GPU-resident routing summaries;
- GPU selection/mailbox path;
- Turbo4 page geometry and representation metadata;
- corrected page-table Turbo4 attention path;
- mature Turbo4 FA path for layouts that qualify;
- Qwen hybrid/recurrent integration;
- native MTP/speculative state handling;
- extensive lifecycle and proof instrumentation.

The best forward architecture is therefore:

1. **Keep the existing Buun pager/residency/transfer/selector/attention implementation.**
2. **Finish the current natural cold-promotion proof (`93-11n`) before broad redesign.**
3. Immediately after that proof, **change the execution policy** from periodic mid-generation reselection to:
   - retrieve at the real final-user-query boundary;
   - checkpoint before the query span;
   - use the query to choose the historical working set;
   - promote missing pages;
   - restore/replay only the current user query if the selected historical view changed;
   - freeze the selected historical view for the entire assistant turn.
4. Preserve unlimited/very-long generation by using the existing shared-pool eviction capability as a **generation ring**, but protect the turn-frozen retrieved-history set. Evict only old, completed, host-backed generation pages when generation pressure requires space.
5. Keep CPU RAM as an **inclusive canonical backing store** for completed target-attention pages: promotion must not destroy the host copy; clean eviction should normally be a mapping drop, not another D2H.
6. Benchmark the existing direct paged Turbo4 attention path against the mature Turbo4 FA/contiguous or packed path with **identical selected pages**. Pick per-shape execution based on evidence rather than ideology.
7. Keep fixed Turbo4 throughout the first complete production implementation. Treat VBR/mixed Turbo tiers as a later optimization.
8. Keep full-L Turbo4 MTP on GPU initially. At 256K it is only about 264 MiB for this model geometry. Convert MTP to a follower of the target slot pool only after target paging is proven and if reclaiming that VRAM materially improves the useful target hot set.
9. Scale only after the short-path semantics and performance are correct: 8K/4K → 32K/16K → 128K → 256K.
10. Preserve measurable fallbacks and reference paths, but never silently use a slow reference/gather/CPU-attention path as production success.

The single most important architectural correction is:

> **Historical retrieval should be a turn-boundary operation, not an every-N-accepted-tokens decode operation.**

The single most important correctness addition is:

> **If the final user query causes a different historical working set to be selected, restore the pre-query hybrid state and replay the user-query span against that final selected history before generation.**

The single most important long-generation rule is:

> **Pin/freeze retrieved historical pages for the turn, but recycle old generation-owned pages through CPU RAM so there is no KVMem-style hard `gen_reserve` output ceiling.**

---

# 1. Existing work that should be preserved

Do not rewrite these subsystems unless a narrowly demonstrated defect requires it.

## 1.1 Pager and residency ownership

Keep the current `llama_kv_pager`, residency table/pool, page identities, epochs, content versions, physical-slot ownership, host-valid state, and transactional publication model.

Important existing behavior to preserve:

- logical page identity is independent of physical GPU slot;
- page identity/version survives residency changes;
- graph consumers see immutable table generations;
- in-flight pages are pinned;
- a dirty GPU page cannot be declared canonical on host until its bounded D2H completes;
- a clean host-backed page can be evicted while preserving its CPU copy;
- stale generations/epochs fail closed;
- current mutable write-page ownership is protected.

Relevant files include, at minimum:

- `src/llama-kv-pager.{h,cpp}`
- `src/llama-kv-residency.*`
- `src/llama-kv-residency-transfer.*`
- `src/llama-kv-residency-transaction.*`
- `src/llama-kv-live-policy.*`
- `src/llama-kv-cache.*`

## 1.2 Canonical Turbo4 host representation

Keep the design where target attention K/V is stored in CPU RAM in its canonical Turbo4 representation rather than expanding the entire history to F16.

For the current Qwen3.8-27B geometry:

```text
16 full-attention target layers
4 KV heads
K width = 256
V width = 256
Turbo4 = 4.125 effective bits/value

target bytes/token
= 16 * 4 * (256 + 256) * 4.125 / 8
= 16,896 bytes/token
= 16.5 KiB/token
```

Useful payload figures:

```text
4K target KV       ~ 66 MiB
16K                ~264 MiB
32K                ~528 MiB
40K                ~660 MiB
48K                ~792 MiB
56 Ki-token hot    ~924 MiB
64K                ~1.031 GiB
256 Ki-token full  4.125 GiB
```

The single MTP attention layer is approximately 1,056 bytes/token, so full 256K Turbo4 draft K/V is about **264 MiB**.

These are encoded KV payloads only. Always account separately for allocator granularity, routing catalogue, page metadata, graphs, scratch, transfer staging, recurrent state, MTP state, CUDA workspaces, model weights, and safety headroom.

## 1.3 Inclusive host backing

The desired host semantic is **inclusive**:

```text
CPU RAM:
    canonical copy of every completed/sealed historical target-attention page

GPU:
    selected subset of those pages
    + current generation tail
```

Promotion:

```text
CPU canonical Turbo4
        ↓ H2D
GPU Turbo4

CPU copy remains valid
```

Clean eviction:

```text
GPU mapping/page
      ↓ remove residency
CPU copy already exists

No new D2H required.
```

A GPU-written page is the exception while it is still dirty/current. It becomes clean/inclusive only after async canonical host capture completes.

Do not turn the host tier into a move-only/exclusive cache unless measurements show a compelling reason. RAM capacity is not the scarce resource on the target machine.

## 1.4 Current write-path slot recycling

The existing `llama_kv_pager::begin_write()` already contains valuable machinery that KVMem's current llama port is planning to add for generation:

- when crossing to a new logical page, release the previous write-frontier pin;
- seal completed pages;
- when no physical slot is free, find a clean, host-backed, unpinned page;
- drop its GPU residency while preserving host backing;
- reuse the physical slot for the new logical page.

Preserve this mechanism. Refine **which page is eligible to be sacrificed during a generated turn** rather than replacing the allocator.

## 1.5 Turbo4 attention implementation

Preserve both:

- corrected page-table/direct Turbo4 GPU attention;
- mature Turbo4 FA/contiguous legal fast path.

Do not pre-decide that either one must win universally. The direct page-table implementation may avoid packing/rearrangement, while mature FA may have much better kernel efficiency for some prefill shapes.

Make route choice empirical and shape-aware.

## 1.6 Native MTP work

Preserve the existing native MTP/DFlash/speculative integration, acceptance instrumentation, rollback semantics, and Turbo4 draft-KV support.

Do not simultaneously redesign target paging and draft paging while target paging is still stabilizing.

---

# 2. Architecture to converge on

Use the following conceptual model.

```text
                         LOGICAL TARGET HISTORY L
                  (up to 262,144 initially; later optional >L)

                                  │
                   canonical Turbo4 K/V for all
                         completed attention pages
                                  │
                                  ▼
                         CPU RAM backing store
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
            GPU routing catalogue        selected-page H2D
             for all sealed pages              │
                    │                           │
current user span Q ─┘                           │
                    │                           │
                    ▼                           ▼
                retrieval                 GPU hot pool H
                                             │
                         ┌───────────────────┴──────────────────┐
                         │                                      │
                turn-frozen historical                    generation tail
                     attention set                         rolling/ring
                         │                                      │
                         └───────────────────┬──────────────────┘
                                             │
                                    GPU-only target FA
                                             │
                        48 Gated DeltaNet layers remain exact,
                        recurrent, GPU-resident and see every token
```

Define explicitly:

- **L** = logical target-attention context capacity/history.
- **C** = currently occupied logical tokens.
- **H** = total target-attention GPU physical capacity in tokens/pages.
- **R** = turn-frozen historical retrieval allocation inside H.
- **G** = generation-tail guarantee/reserve/ring capacity inside H.
- **A** = attention execution set/workspace, if the execution implementation uses a subset or packed view distinct from H.
- **B/U** = logical batch / physical microbatch.
- **Host** = canonical full-history CPU RAM backing.

Recommended invariant:

```text
R + G <= H
```

but implement **G as a guaranteed minimum**, not necessarily a permanently wasted hard partition. If history selection uses fewer than R pages, generation may temporarily use spare physical slots. Conversely, history selection must never consume the minimum number of slots required to guarantee G.

The product-facing configuration should eventually expose or derive at least:

```text
logical context L
target hot capacity H (or VRAM budget from which H is derived)
generation tail G
attention execution A if separately meaningful
page size
retrieval policy/scorer
```

All geometry must remain derived/tunable. Do not hardcode 56K, 77K, 4K, 256-token pages, etc., except in explicit test fixtures.

---

# 3. Replace periodic decode reselection with a turn retrieval epoch

The current V10 policy calls for refresh on first request, committed page crossing, and every eight accepted target tokens. That was useful while proving the selector/promotion chain, but it should **not be the final production semantic** for Qwen's hybrid architecture.

## 3.1 Why mid-generation historical reselection is undesirable

Qwen3.8 interleaves recurrent Gated DeltaNet and standard attention. The recurrent state evolves causally through every token.

If the full-attention historical set changes from A to B halfway through a generated turn:

```text
tokens 1..N:
    recurrent state evolved while attention exposed A

token N+1:
    full attention suddenly exposes B
```

the recurrent state cannot be retroactively recomputed as if prior generated tokens had seen B.

This creates a mismatched causal trajectory.

It also adds:

- selector work during steady decode;
- possible H2D transfers during decode;
- publication synchronization during a latency-sensitive path;
- additional MTP mapping/state complexity;
- harder reproducibility.

## 3.2 Required final semantic

Introduce a **turn retrieval epoch**:

```text
new final user turn
     ↓
capture pre-query checkpoint
     ↓
provisional user-query processing / Q capture
     ↓
select historical pages
     ↓
promote missing pages
     ↓
publish final historical map
     ↓
restore/replay user query if map changed
     ↓
freeze historical map
     ↓
decode/thinking
```

During the generated turn:

- no historical selector refresh;
- no historical H2D promotion;
- no historical working-set eviction;
- no CPU historical KV reads by target attention;
- speculative/MTP transactions operate against a stable historical mapping.

Generation pages may roll inside their own policy described later.

Keep the old cadence mode only as an explicit experimental/A-B mode if it is useful for research. It should not remain the default target architecture.

---

# 4. Add query-boundary checkpoint and replay

This is the most important KVMem semantic to adopt.

## 4.1 Problem being solved

The current natural-promotion proof can establish:

```text
user query runs
    ↓
query nominates cold historical page
    ↓
page is promoted
    ↓
a later target graph sees the promoted page
```

That proves transport and routing, but the **user query itself** was initially processed without the newly retrieved historical page.

For Qwen's recurrent hybrid, that matters because the query has already mutated the Gated DeltaNet recurrent state.

## 4.2 Required algorithm

At the real final-user-message boundary:

1. Identify the exact token span belonging to the final `role=user` message after chat templating.
2. Immediately before that span, checkpoint:
   - Gated DeltaNet recurrent state;
   - convolution/recurrent companion state;
   - target logical position/frontier;
   - pager/current write-page state needed to undo provisional query writes;
   - MTP/carry state that must round-trip at this boundary;
   - any other model-specific state changed by query prefill.
3. Process the user span provisionally against the previous stable historical selection while capturing retrieval Q.
4. Run retrieval using that user-span Q.
5. Diff new historical selection against current residency.
6. H2D only missing selected pages. Retain already-resident selected pages in place.
7. Wait only at the necessary safe boundary for required promotions.
8. Publish the final mapping.
9. If the historical attention selection actually changed:
   - restore the pre-query checkpoint;
   - remove/truncate provisional query KV/state;
   - replay **only the user-query span** against the final historical view.
10. Freeze the resulting historical selection and enter generation.
11. If the selection did not change, skip replay.

Do **not** replay the entire 200K/256K history.

## 4.3 Acceptance tests

At minimum:

- selection unchanged → replay count = 0;
- selection changed → exactly one user-span replay;
- replay restores query-start recurrent state before recomputation;
- after replay, recurrent state/logits match a control that processed the same query once with the final selected mapping already installed, within appropriate Turbo4 numerical tolerance;
- provisional query pages do not survive twice or duplicate logical positions;
- MTP state/carry is not duplicated or advanced twice;
- table epoch and query-replay epoch are explicit and observable;
- query replay does not force CPU target attention.

---

# 5. Use the entire final user span as the retrieval query

Do not make final production retrieval depend only on the final causal query row.

KVMem's current approach of aggregating query information over the last user span is a stronger default.

Implement retrieval query capture with these principles:

- source span = actual final `role=user` tokens after template rendering;
- preserve per-layer/per-head identity;
- use all 16 standard-attention layers initially where practical;
- MTP/draft Q does not define the historical target selection;
- GDN layers continue to process the query normally.

Keep final-row scoring as a diagnostic baseline.

Potential query reductions to benchmark:

1. mean across user-span Q rows;
2. max/robust maximum contribution;
3. mean + final-row weighted blend;
4. all-row scoring without collapsing first, if GPU cost is acceptable.

The goal is not to maximize selector sophistication. The goal is robust recall without adding meaningful TTFT.

---

# 6. Keep the current transformed-domain selector, but add a KVMem-style retrieval baseline

Do not throw away the current GPU routing-summary work.

The existing implementation has a useful Turbo4-specific property: summaries and query transforms can operate in the same transformed K/Q domain used by the real attention path.

However, do not assume the current min/max/representative method is superior merely because it is GPU-native.

Add a selectable KVMem-style content baseline, preferably Mean-K or an equivalent content-position-independent representative compatible with the actual Buun/Turbo4 transform.

Then compare:

```text
A. existing transformed-domain min/max / representative selector
B. Mean-K-style selector
C. optional two-stage hybrid
```

A possible hybrid:

```text
all historical pages
     ↓
very cheap coarse summary bound
     ↓
top candidate set
     ↓
Mean-K / richer scoring
     ↓
final retrieval pages
```

Evaluation should separate:

- recall quality;
- selector time;
- catalogue memory;
- catalogue update cost;
- query capture cost;
- number of false promotions;
- selection overlap between adjacent turns.

Do not replace the current selector until the comparison demonstrates a better quality/latency point.

---

# 7. Turn-frozen history + rolling generation ring

This should combine the strongest property of KVMem with the strongest property already present in Buun.

## 7.1 Historical pages

After query retrieval/replay:

- mark selected historical pages as **turn-pinned**;
- do not evict them to make room for generation;
- do not reselect them during the turn;
- keep them address-stable for target/MTP transactions where possible.

## 7.2 Generation pages

Generation owns a rolling tail of GPU target-attention KV.

When a generation page is complete:

1. asynchronously seal/capture its Turbo4 target K/V to canonical CPU RAM;
2. generate/update its routing summary;
3. remove its current-write pin once safe;
4. keep it resident while there is generation-tail capacity.

When generation needs a slot and the guaranteed generation region is full:

1. select the **oldest completed generation-owned page** from this turn;
2. require host-valid canonical bytes;
3. never choose:
   - current incomplete write page;
   - turn-frozen historical pages;
   - in-flight pages;
   - speculative transaction pins;
4. drop the old generation page's GPU residency;
5. reuse its physical slot;
6. continue generation.

This removes KVMem's current hard `--kvmem-gen-reserve` length limit.

## 7.3 Semantics of G

`G` becomes:

> the amount of exact full-attention access retained to the model's **recent current-turn output**

rather than:

> maximum generation length.

Example:

```text
R = 40K frozen historical retrieval
G = 16K generation tail
H = 56K total target GPU KV
```

After generating 40K tokens:

- full attention sees the same frozen ~40K retrieved historical evidence;
- full attention sees the most recent ~16K generated tokens;
- older current-turn generated tokens remain in CPU RAM;
- the 48 Gated DeltaNet recurrent layers have still processed all 40K generated tokens in order.

On the next user turn, those spilled generated pages become ordinary historical candidates and can be retrieved again.

## 7.4 Do not let current generic eviction cannibalize frozen history

The current `begin_write()` fallback can evict any safe, clean, host-backed, unpinned page. That is useful infrastructure but too permissive for final turn semantics.

Add page ownership/classification sufficient to prefer/require:

```text
during active generated turn:
    victim class = completed generation-owned page
```

Only a deliberate fallback mode should permit sacrificing turn-frozen historical retrieval, and production should normally refuse rather than silently alter the turn's historical evidence.

## 7.5 Tests

Required:

- generate >2×G tokens without `no_victim`/hard-reserve failure;
- historical needle page stays GPU-resident for entire turn;
- oldest generation pages spill in chronological order;
- current write page never evicted;
- one host seal per completed generation page, not per token;
- no historical H2D during decode;
- spilled generation page remains host-valid and becomes retrievable next turn;
- MTP rollback/acceptance does not publish rejected draft pages into canonical generation history.

---

# 8. CPU backing and transfer pipeline

Highest speed does **not** require permanently pinning the entire 4.125 GiB 256K host target store.

Prefer:

```text
pageable canonical host store
+
bounded pinned transfer slabs/ring
+
async CUDA copy streams/events
```

unless measurement proves that keeping a much larger permanently pinned tier materially improves the target workload without unacceptable system impact.

## 8.1 Prefill D2H

During long initial ingest:

```text
GPU computes chunk N
    │
    ├── completes/seals target pages
    ├── creates routing summaries
    └── queues D2H of completed pages

GPU computes chunk N+1 while D2H proceeds
```

The host copy should normally exist before a page becomes an eviction candidate.

## 8.2 Turn-boundary H2D

At retrieval:

- diff selection first;
- retain hits in place;
- batch only misses;
- combine pages into large transfers/staging slabs;
- use `cudaMemcpyAsync` / existing backend abstraction;
- overlap host gathering and copy preparation;
- avoid device-wide synchronize;
- publish only after required copies complete.

## 8.3 Decode

After historical selection is frozen:

- historical H2D = zero;
- historical CPU KV reads = zero;
- CPU control should be negligible;
- D2H occurs only when a completed generation page is sealed, normally once per page (e.g. once every 256 generated tokens in the current geometry);
- no per-token CPU synchronization.

Expose counters that can prove this without adding permanent hot-path synchronization.

---

# 9. Decide direct paged Turbo4 attention versus mature Turbo4 FA empirically

This decision has large performance consequences.

Do not delete the current `fattn-paged-turbo4`/page-table MMA path merely because KVMem avoids changing FA. Conversely, do not keep it as the default merely because significant work has already been invested.

Use the **same exact selected page set and encoded Turbo4 bytes** and compare execution strategies.

At minimum compare:

### Route A — direct paged Turbo4

```text
physical page table
      ↓
corrected paged Turbo4 MMA/FA
```

Advantages:
- avoids one-time/repeated packing;
- arbitrary physical residency;
- natural fit with current pager.

Risk:
- custom kernel may have lower occupancy/tile efficiency.

### Route B — mature/contiguous Turbo4 FA

Construct a bounded compatible selected view using GPU-resident operations only, then use the mature Turbo4 kernel.

Advantages:
- known optimized kernel;
- potentially much faster prefill.

Risk:
- D2D packing/materialization cost;
- more view maintenance as generation appends.

### Route C — hybrid by phase/shape

Potentially:

```text
turn-boundary query replay / larger prefill:
    packed/contiguous mature FA

single-token or short-query decode:
    direct paged Turbo4
```

This may be the best answer if the custom paged kernel is strong at decode but weaker at multi-query prefill.

## 9.1 Benchmark requirements

For identical model, selected pages, B/U and Turbo4 data, measure separately:

- selected-view preparation/packing;
- actual FA kernel time;
- query replay prefill;
- steady decode;
- generation-page append;
- GPU utilization/power;
- transfer waits.

Do not count cached input as fresh prefill.

Do not make a route decision from a route label or one aggregate throughput number.

Suggested decision heuristic:

- if direct paged path is within roughly 10–15% of mature FA at the important shape and avoids material complexity/copy cost, keep direct;
- if direct paged path is dramatically slower (e.g. ~25%+ slower, especially at prefill), use mature/packed for that shape;
- allow shape-aware dispatch rather than forcing one universal route.

The numeric percentages are decision heuristics, not acceptance law. Correctness is mandatory.

---

# 10. Keep MTP simple until target paging is done

For the first complete architecture:

```text
target attention KV:
    bounded H + CPU full-history backing

MTP KV:
    full-L Turbo4 on GPU
```

This costs roughly 264 MiB at 256K for this specific model geometry and is a reasonable price for reducing state-machine risk.

Only after the following are proven:

- query replay;
- turn freeze;
- generation ring;
- cold promotion;
- 256K target host backing;
- selected prefill/decode speed;
- stable MTP acceptance;

should a new optimization task test:

```text
MTP follower pool
```

where target and draft share selected block identities/slot geometry.

The benefit is reclaiming most of that ~264 MiB for larger H or headroom. The cost is substantial extra complexity around draft writes, follower eviction, rollback and generation-ring slot reuse.

Do not make follower MTP a blocker for the primary goal.

---

# 11. Keep fixed Turbo4 before reintroducing VBR

Do not combine:

- attention-aware paging;
- query replay;
- turn freeze;
- generation ring;
- mixed per-layer/per-side dynamic VBR codecs

in the same stabilization phase.

First make:

```text
target K = Turbo4
target V = Turbo4
draft K  = Turbo4
draft V  = Turbo4
```

fully correct and fast.

Afterwards, VBR can become a separate performance/capacity experiment.

If VBR is later added, every host page must carry sufficient representation metadata to guarantee that a promoted page is interpreted using the exact codec/transform/version with which it was stored.

---

# 12. Recommended changes to the current Wiretail task graph

## 12.1 Finish `93-11n` first

Do not invalidate or broaden `93-11n`.

Its job is valuable and narrowly defined:

```text
cold host-backed fact page
→ natural selector nomination
→ mailbox
→ live-policy admission
→ H2D
→ publication
→ later real target attention consumption
```

Complete it on the current architecture. Fix only narrow defects necessary to prove this physical chain.

This gives a trusted baseline before policy semantics change.

## 12.2 Insert architecture-correction tasks before the current `93-12`

Do **not** run the full current `93-12` speed campaign and then optimize an architecture that will immediately change.

Create a new ordered sequence after `93-11n`, before the paired speed campaign.

Suggested task breakdown:

### Task A — Turn retrieval epoch and frozen historical map

Implement:

- explicit turn/query generation identity;
- historical-selection freeze after query-boundary commit;
- disable accepted-token historical refresh while a generated turn is active;
- retain old cadence as experimental-only;
- enforce stable mapping across speculative transactions.

Acceptance:
- generate at least several page crossings;
- selector historical refresh count remains zero after turn freeze;
- no historical H2D/publication during decode;
- MTP transactions observe stable historical table generation.

### Task B — Final-user-span checkpoint and query replay

Implement:

- exact templated final-user span discovery;
- checkpoint immediately before span;
- provisional query processing and Q capture;
- retrieval/promotion;
- restore + replay current query only when selection changes;
- replay skip when unchanged.

Acceptance:
- changed-view replay equals one-pass-final-view control;
- recurrent state and target logits agree within justified tolerance;
- MTP/carry and logical KV frontier round-trip exactly;
- no duplicated query KV.

### Task C — User-span retrieval query and scorer comparison

Implement:

- aggregate Q over final user span;
- all standard-attention layers or justified subset;
- current transformed-domain selector baseline;
- Mean-K-style optional baseline;
- optional two-stage hybrid only if data justifies it.

Acceptance:
- deterministic selector fixtures;
- existing natural-promotion fixture still works;
- needle/page-recall matrix;
- measured selector latency/catalogue overhead;
- choose default based on quality + latency.

### Task D — Protected history + generation ring

Implement:

- page ownership class;
- guaranteed G capacity;
- turn-pinned retrieved history;
- oldest-completed-generation eviction;
- host-backed generation spill;
- no hard generation-length limit below L.

Acceptance:
- generation >2×G;
- historical retrieved needle remains pinned;
- old generation pages spill;
- current page never spills;
- next turn can retrieve a spilled generation page.

### Task E — Transfer-path audit/optimization

Verify and optimize:

- inclusive host copies;
- one D2H per completed page;
- clean eviction with no redundant D2H;
- selection diff;
- batched H2D;
- bounded pinned slabs;
- no decode-time historical PCIe.

Acceptance:
- event/counter trace proves intended transfer topology;
- promotion transfer bandwidth and wait time recorded;
- no global device sync on steady decode path.

### Task F — Direct-paged vs mature-FA execution A/B

Use identical selected pages and encoded bytes.

Test:
- multi-query prefill/query replay;
- one-token decode;
- short MTP verification shapes.

Allow shape-aware final dispatch.

Acceptance:
- numerical parity against existing reference oracle;
- measured kernel/view-preparation times;
- production route selected from evidence.

### Task G — Revised paired speed campaign

Then run the intent of current `93-12` on the finalized architecture.

Keep its strongest existing discipline:

- B=1024/U=256;
- fixture-backed C>H;
- exact CPU-RAM vs selected-hot vs dense-GPU placements;
- Turbo4 target and draft;
- fresh-token accounting;
- 500 tok/s selected-prefill floor;
- 750 tok/s preferred target;
- per-prompt MTP acceptance floors;
- actual placement/byte/headroom proof.

Add:
- query-replay time;
- retrieval scoring time;
- H2D promotion time;
- view/packing time;
- steady decode free of historical PCIe;
- generation-ring events if output crosses a page.

### Task H — staged capacity scaling

Only after the short path passes:

```text
32K L / 16K H
→ 128K L with measured H
→ 256K L with dynamically admitted H
```

At every step report:

- target logical bytes;
- target GPU hot bytes;
- host canonical bytes;
- MTP bytes;
- recurrent state;
- summary catalogue;
- graphs/scratch;
- transfer slabs;
- peak prefill/verify usage;
- safety headroom.

Do not call 256K feasible merely because allocation succeeds.

### Task I — optional MTP follower

Only if worthwhile after the 256K byte ledger.

### Task J — optional first-attention-Q one-pass retrieval

After query replay is working and measured, experiment with Qwen's first full-attention layer as an early selector source.

Because three Gated DeltaNet layers precede the first full-attention layer, its Q can potentially be obtained before historical full-attention selection has influenced the query at any standard-attention layer.

This might allow:

```text
first 3 GDN/query Q
→ retrieve/promote
→ full user-query prefill once
```

instead of provisional query + replay.

Treat this strictly as an optimization. The replay implementation remains the correctness reference. Adopt the one-pass path only if:

- retrieval quality is close enough to the all-layer query method;
- final outputs/quality remain acceptable;
- TTFT improves materially.

---

# 13. Revised performance model

Performance must be separated into phases.

## 13.1 Initial long-context ingest

Measure:

```text
model prefill compute
+ routing-summary creation
+ overlapped D2H seal
```

D2H should largely overlap later prefill chunks.

Do not count full rendered tokens as freshly evaluated when prompt cache is involved.

## 13.2 New-turn retrieval boundary

Measure separately:

```text
provisional user-span prefill
retrieval scoring
selection diff
host gathering
H2D
publication
checkpoint restore
user-span replay
```

This is TTFT overhead, not steady decode cost.

The objective is to make this bounded by query span and changed pages, not by L.

## 13.3 Decode/thinking

Steady decode should consist of:

```text
GPU target attention over frozen history + recent generation tail
GPU recurrent state
GPU MTP
```

Expected CPU/RAM interaction:

```text
none for historical reads;
occasional async D2H when a generation page seals/spills.
```

There should be no H2D historical page fetch every token or every eight tokens.

## 13.4 Subsequent turn

Spilled generation pages are now historical CPU-backed candidates and may be promoted during the next turn's retrieval boundary.

---

# 14. Correctness and dependability gates

A production candidate should not be considered complete until it proves all of the following.

## 14.1 Identity/fallback

When:

```text
H >= occupied target context
```

the paging path should reduce to an identity-equivalent/full-resident case within Turbo4's ordinary numerical behavior.

No unnecessary host movement.

## 14.2 Natural cold promotion

Prove, using immutable page identity/version:

```text
GPU resident
→ host canonical
→ GPU cold/nonresident
→ selected from current user query
→ H2D completion
→ mapping publication
→ target attention consumption
```

Answer correctness is separate evidence.

## 14.3 Query replay

Prove the user query itself is recomputed against newly retrieved history when necessary.

## 14.4 Turn freeze

Prove the selected historical set does not change during decode.

## 14.5 Long generation

Generate substantially beyond G with:

- no hard reserve failure;
- no eviction of frozen retrieved history;
- generation-ring spill;
- stable recurrent state;
- stable MTP behavior.

## 14.6 Host authority

Every evictable completed page has a valid canonical host copy.

Corrupt/stale/missing copies fail closed.

## 14.7 No decode PCIe dependency

For a turn that does not cross a generation-page boundary:

```text
historical H2D = 0
historical D2H = 0
```

For a long generation, D2H should be page-grained, not token-grained.

## 14.8 MTP

Verify:

- target-equivalent greedy behavior at the same selected table;
- request-local draft attempts;
- acceptance floors on canonical prompts;
- rejected draft suffix never becomes committed canonical history;
- rollback restores pager/recurrent/draft state to the accepted frontier.

## 14.9 256K memory proof

For current model geometry, target host KV should converge around 4.125 GiB encoded payload at a full 256K target history, independent of H.

Target GPU KV should remain proportional to H, not L.

Current full-L MTP Turbo4 should be ~264 MiB payload.

Peak observed allocation must include compute/scratch/graphs, not just KV payload arithmetic.

---

# 15. Quality evaluation

Do not judge retrieval solely by a single needle.

Use several categories:

1. exact fact/needle pages;
2. relevant code/document retrieval;
3. questions requiring two old pages;
4. recent-history questions;
5. long conversational/agent continuation;
6. retrieval of an earlier page generated by the model itself;
7. distractor-heavy history.

Compare:

- dense/full-resident reference where feasible;
- current transformed-domain selector;
- Mean-K baseline;
- optional hybrid.

Track:

- required-page recall;
- unnecessary promotion count;
- selected-set stability;
- final answer quality;
- query-replay rate;
- selector latency.

Do not require exact full-attention equivalence from a deliberately sparse system.

---

# 16. Observability required in production/debug builds

Keep counters cheap and request-scoped.

Useful fields:

```text
L, C, H, R, G, A, B, U
logical pages
resident pages
turn-frozen historical pages
generation-tail pages
host-valid pages
dirty pages
H2D pages/bytes/time
D2H pages/bytes/time
selector calls/time
query capture rows
query replay count/tokens/time
selection hits/misses
promotion count
clean mapping-drop evictions
generation-ring evictions
historical evictions during active turn (must normally be 0)
FA route/kernel variant
view/packing bytes/time
MTP drafted/accepted
peak VRAM/headroom
```

Do not add synchronization merely to populate metrics.

---

# 17. Explicit non-goals / traps

The planning LLM should reject proposals that regress into any of these unless a task is explicitly experimental.

## Do not:

- import KVMem as a second pager beside the current Buun subsystem;
- restart from KVMem merely to gain semantics already implementable here;
- stream historical CPU KV layer-by-layer during decode;
- perform CPU attention in selective production;
- expand full history to F16;
- reselect historical pages every few generated tokens by default;
- use answer correctness as proof that cold-page promotion occurred;
- use route labels as proof that a GPU kernel ran;
- count cached tokens as fresh prefill;
- treat a 4K/8K benchmark as proof of 256K capacity;
- make VBR a dependency of the first stable pager;
- redesign MTP target/draft residency simultaneously with core target query replay;
- silently fall back to selected-reference/gather paths and report success;
- discard host copies after promotion if RAM capacity permits inclusive backing;
- let a long generation silently evict the turn's retrieved evidence;
- impose a KVMem-style hard generation maximum when a generation ring can recycle old output pages;
- run broad 256K campaigns before the short correctness/performance chain passes.

---

# 18. Recommended source ownership boundaries

Keep responsibilities clear.

### Pager/residency

Owns:

- logical page identity;
- physical target slots;
- page class/turn ownership;
- host-valid/dirty state;
- pinning;
- write frontier;
- generation-ring eviction;
- publication epochs.

### Host store

Owns:

- canonical Turbo4 page bytes;
- content/version authentication;
- full-L retention;
- pageable/pinned staging accounting.

### Retrieval catalogue

Owns:

- summary per logical page/layer/head/version;
- no dependence on current GPU residency;
- invalidation when content version changes.

### Selector

Owns:

- query representation;
- ranking;
- structural anchors/recent guarantees;
- final historical selection.

It does **not** directly mutate GPU residency.

### Live policy/transaction

Owns:

- selection diff;
- target/victim admission;
- H2D plan;
- safe publication;
- rollback.

### Attention execution

Owns:

- direct paged vs contiguous/packed route;
- correct transformed Q/K/V semantics;
- causal logical positions;
- no CPU fallback in production.

### Hybrid/recurrent

Owns:

- every-token GDN processing;
- query-boundary checkpoint/restore;
- turn replay semantics.

### MTP

Owns:

- speculative checkpoint/rollback;
- accepted frontier;
- draft cache semantics.

It must consume the same stable historical target view during one speculative transaction.

---

# 19. A proposed final product configuration model

Names can change; semantics should be explicit.

Example:

```text
--ctx-size 262144

--kv-pager selective
--kv-page-size 256

# Either direct:
--kv-hot-tokens 57344

# Or derived:
--kv-vram-budget <bytes/MiB>

--kv-gen-tail 16384
--kv-retrieval user-span
--kv-retrieval-scorer <turbo-summary|mean-k|hybrid>

--ctk turbo4
--ctv turbo4
--ctkd turbo4
--ctvd turbo4

--spec-type draft-mtp
```

Result:

```text
H = 56K target GPU physical KV
G = 16K generation-tail guarantee
R ≈ up to 40K historical retrieval
full target history = CPU Turbo4
MTP = full-L GPU Turbo4 initially
```

This example is not a mandated default. Admission must derive safe H from measured VRAM.

---

# 20. Definition of the final goal

The project is successful when, on the canonical Qwen3.8-27B UD-IQ4_XS deployment:

1. A configurable large logical target context (initially 256K) can exist with complete canonical target Turbo4 KV in CPU RAM.
2. Target-attention GPU KV is bounded by a configurable/admitted H independent of L.
3. Relevant cold pages are discovered using the current final-user query and promoted before generation.
4. If promotion changes the historical view, only the current query is replayed from a correct hybrid-state checkpoint.
5. The selected historical view is frozen for the assistant turn.
6. Decode/thinking uses GPU-resident target KV only.
7. Long generations can exceed the generation-tail size by spilling old generation pages to CPU without evicting the frozen retrieved history.
8. Gated DeltaNet recurrent state remains exact and processes every token.
9. Native MTP remains fast, correct and GPU-resident, initially with full-L Turbo4 KV.
10. Target host storage is inclusive so clean GPU eviction does not require another D2H.
11. The production Turbo4 attention route is chosen from direct-paged versus mature-FA measurements and is GPU-native.
12. Selected-mode prefill is at least the existing 500 tok/s floor on every canonical fresh-input prompt and preferably >=750 tok/s, while beating CPU-main-KV at matched geometry.
13. Decode speed is materially better than CPU-KV streaming/offload and does not scale with PCIe traffic over total L.
14. A full 256K run passes memory, promotion, replay, MTP, quality and speed gates with measured scratch/headroom.
15. Feature-off behavior remains unchanged.

---

# 21. Immediate instruction to the planning LLM

When updating the current plan:

1. Read the current `93-11n` packet and complete it first unless a concrete blocking defect requires a narrow repair.
2. Treat `.wiretail/execution/v10/REPAIR85_PLAN.md` and current repair93 packets as the authoritative current implementation history, but supersede the **periodic accepted-token historical refresh** as the intended final architecture.
3. Insert the architecture-correction tasks described in Section 12.2 **before the present full `93-12` paired performance campaign**.
4. Preserve existing task/evidence IDs; do not rewrite historical receipts.
5. Amend dependencies so the revised speed campaign measures:
   - query replay;
   - frozen historical retrieval;
   - generation-ring behavior;
   - the chosen GPU attention execution route.
6. Keep implementation tasks on Luna High where current project policy requires it.
7. Give every new task:
   - bounded source-context list;
   - exact source owners/symbols;
   - one narrow objective;
   - deterministic correctness test;
   - live CUDA proof where required;
   - candidate identity;
   - fail-closed acceptance;
   - raw artifact path;
   - concise handoff.
8. Avoid large benchmark matrices in implementation tasks.
9. Do not start 128K/256K performance work until the 8K/4K and 32K/16K paths prove the corrected semantics and useful speed.
10. Keep the current codebase as the implementation substrate. KVMem is evidence for policy choices and a source of test ideas, not a dependency.

---

# 22. Source references for the planning LLM

Current Buun branch/reference:

```text
repo: vanwho/buun-llama-cpp
branch: plan/attention-aware-kv-paging
observed tip: e5e5f5c2bb32a5e44f447c2449cb7bc2ad678b55
```

Important current Buun files:

```text
.wiretail/execution/v10/REPAIR85_PLAN.md
.wiretail/execution/v10/REPAIR85_TESTING.md
.wiretail/execution/tasks/93-11n.md
.wiretail/execution/tasks/93-12.md
.wiretail/execution/tasks/93-13.md

src/llama-kv-pager.h
src/llama-kv-pager.cpp
src/llama-kv-live-policy.h
src/llama-kv-live-policy.cpp
src/llama-kv-routing-retrieval.h
src/llama-kv-routing-retrieval.cpp
src/llama-kv-routing-summary.h
src/llama-kv-prefetch.h
src/llama-kv-attention-execution.h
src/llama-kv-attention-view.h
src/llama-kv-cache.h
src/llama-kv-cache.cpp
src/llama-context.cpp
src/llama-graph.cpp
src/llama-memory-hybrid.cpp
common/speculative.cpp
ggml/src/ggml-cuda/fattn-paged-turbo4.cuh
ggml/src/ggml-cuda/kv-page-select.cu
ggml/src/ggml-cuda/kv-page-summary.cu
```

KVMem design reference:

```text
repo: kvmem/kvmem-llama.cpp
file: docs/architecture.md
```

Particularly useful KVMem ideas to adopt semantically:

```text
bounded GPU working set
host-backed completed blocks
query-boundary retrieval
last-user-span query policy
selection diff / resident reuse
query replay
pinned historical selection during generation
generation-tail ring proposal
MTP follower as optional later optimization
```

Do not copy KVMem's current hard `budget + gen_reserve` maximum-generation behavior. Buun's existing shared-pool host-backed eviction machinery permits a better generation-ring implementation.

---

# 23. Short architecture statement suitable for inserting into a future OVERVIEW

> The production selective target-KV path uses a bounded Turbo4 GPU pool backed by an inclusive canonical Turbo4 CPU-RAM history. Retrieval occurs once at the final-user-query boundary. The system checkpoints hybrid recurrent state before that query, captures user-span target Q, selects and promotes cold historical pages, replays only the user span when the historical view changes, and freezes the selected historical mapping for the assistant turn. Decode and native MTP then use GPU-resident target KV only. A configurable generation-tail ring shares the bounded pool but may evict only old completed generation-owned pages, never the turn-frozen retrieved-history set; spilled generation pages remain canonical in RAM and become retrievable history on the next turn. The 48 Gated DeltaNet layers remain exact and GPU-resident and process every token. The implementation retains Buun's existing pager/residency/transfer/selector infrastructure and Turbo4-native attention paths; direct paged versus mature/packed Turbo4 FA dispatch is selected from matched GPU benchmarks. Full-L Turbo4 MTP remains GPU-resident until the target pager is stable and a follower-pool optimization is separately justified.
