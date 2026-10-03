# Selected-attention stale routes: diagnosis and follow-up

Recorded 2026-10-02 for later implementation/review. This note captures the
source-level diagnosis from task 102-03 and the intended division of
responsibility between attention graph construction and the pager policy.

## Finding

There was a real runtime defect in the selected-attention graph path, separate
from the benchmark finalizer issue. An earlier A2 request failed before
attention execution with `selected attention refused: stale routing refresh
page is no longer resident`. The old recovery path responded to one invalid
entry in the saved routed set by rebuilding optional pages from pager inventory
that could include host-only pages. It then attempted to append a page without
a current GPU physical slot and refused graph construction.

The later finalizer error about recomputing a generic threshold before loading
the frozen schedule is a benchmark-harness/resume defect: it sent no generation
request and is not evidence of another llama runtime fault.

The 102-03 source change filtered stale-refresh candidates against the
current resident snapshot. Its rebuilt A/B/A trajectory completed without the
earlier runtime refusal. This is evidence that the crash path was addressed;
it is not, by itself, proof that a cold page nominated by a new query is
promoted and used by the next attention graph.

## Relevant code and ownership

- `src/llama-context.cpp`, `llama_context::prepare_kv_attention_graph()`:
  obtains the current `pager.residency(sequence_id)` snapshot, reads the saved
  per-layer route through `selected_attention_page_layers()`, and constructs a
  bounded selected-attention view. `append_page()` requires the page to be in
  that snapshot, have a physical slot, and be in a GPU-resident state.
  The stale-route refresh is in the branch following `if (!routed_valid)`.
- `src/llama-kv-cache.cpp`, `selected_attention_page_layers()`:
  returns the last published per-layer selections. The accessor does not
  reconcile them with a newer residency snapshot; graph preparation must
  validate them before use.
- `src/llama-kv-cache.cpp`, live policy boundary around
  `pager_->apply_live_policy(boundary)`:
  authenticates selector candidates, constructs the bounded query target and
  H2D promotion plan, and runs the residency transaction. After a successful
  transaction, the per-layer routes are rebuilt from `result.target_pages`,
  admitting only records with a physical slot.
- `src/llama-kv-live-policy.cpp`,
  `llama_kv_live_policy_prepare_query_target()`:
  retains mandatory pages and authenticated retrieval candidates, requires
  host backing for cold candidates, and chooses reusable physical slots.
- `src/llama-kv-residency-transaction.cpp`:
  reserves and executes transfers, handles eviction/reuse, and publishes the
  resulting residency table transactionally. The graph builder is not the
  place to perform an H2D promotion or mutate this mapping.

## Correct behavioral contract

1. A selected-attention graph uses one immutable current residency snapshot.
   It must never dereference or pack a host-only page.
2. On a stale advisory route, retain still-valid resident routed pages and
   remove only entries absent from the snapshot or no longer resident. Keep
   mandatory query pages; fill remaining bounded capacity using the established
   current/pinned/sink/recent fallback order.
3. A stale route entry alone is not a fresh promotion request. Do not blindly
   reload every evicted old route entry: that can undo intentional eviction and
   cause page ping-pong.
4. A cold page becomes a promotion candidate only through the normal current
   selector/mailbox path with authenticated page identity and content version.
   The live policy must admit it, require canonical host backing, select an
   eligible victim/slot, transfer it, and publish the mapping atomically.
5. The graph already being constructed continues with its valid hot set. A
   later graph may use the newly promoted page after the transfer and mapping
   publication are complete and it obtains the newer snapshot. Do not add a
   synchronous per-token host-to-device wait to graph construction.
6. If a committed/frozen replay requires a page that is no longer resident,
   retain the existing strict refusal/retry semantics; do not silently replace
   required committed history with a selective approximation.

## Performance implications

The stale-refresh branch is exceptional; the normal valid-route graph path
should not pay for it. The refresh in `prepare_kv_attention_graph()` iterates
the pager residency snapshot, not the complete host catalogue/logical 250K
context. Its inventory work is therefore bounded by admitted hot-page capacity
(plus routed entries across layers), and does not scale linearly with total
context when hot capacity remains fixed. Increasing the hot set increases this
bounded metadata work; increasing only host-backed context does not.

The current helper `llama_kv_attention_resident_page_ids()` uses linear
membership searches and allocates vectors, but only in the stale-route branch.
Do not optimize or move that work onto every token without profiling evidence.
If stale refresh is frequent, treat that frequency as a route/snapshot
consistency defect first. A reused logical-page index/bitmap can be considered
later if measurements show the exceptional-path CPU work is material.

## Follow-up implementation and proof

- Add/retain a focused regression with an input route containing resident,
  host-only/evicted, and resident entries. The resulting graph selection must
  preserve the valid resident entries, omit only the cold one, preserve
  mandatory query pages, and avoid refusal.
- Preserve individually valid advisory entries rather than clearing the
  selected set when one route element is stale. This source repair is now
  implemented as described below; its regression/live proof remains unrun.
- Add a separate bounded integration proof for genuine promotion: current
  selector nominates an authenticated host-backed page; policy admits it;
  transaction records an eligible eviction and H2D completion; publication
  binds the promoted page to a slot; the next graph snapshot contains and uses
  that page. Do not infer this chain from a no-crash A/B/A trajectory or from
  aggregate counters.
- If needed, bind published route data to a residency/table epoch or selection
  generation so snapshot mismatch is explicit. Epoch validation must not force
  a full host-inventory scan or a global synchronization on the normal graph
  path.
- Keep route/promotion telemetry bounded and diagnostic. No full inventory
  dump or per-token synchronous copy is required.

## Evidence pointers

- `.wiretail/execution/handoffs/102-03.md` records the old-binary A2 failure,
  the resident-only refresh change, the rebuilt A/B/A run, and the remaining
  proof boundary.
- Source symbols listed above are the implementation locations to re-open
  before changing this behavior; line numbers may move as the branch evolves.

## Source review and implementation — 2026-10-04

Reviewed and implemented on `codex/task-102-05`, source commit `f28588e7b`.
**No build, test, benchmark,
generation request, server reload, or service change was performed.** Existing
102-03 runtime evidence belongs to its earlier resident-filter repair, not to
these new changes. Do not mark a task or promotion proof complete from this
source review.

The resident-only filter was already in place and did not need replacing. Three
remaining defects were identified in `prepare_kv_attention_graph()`:

1. The advisory route loop stopped at its first invalid entry, and the refresh
   branch then cleared the entire selected set. Valid historical entries both
   before and after the stale entry could therefore be displaced by recent
   fallback pages, even when their exact identities were still resident.
2. The missing-exact-ID fallback accepted any dirty/filling page with the same
   sequence, logical number and start position. Despite its comment, it neither
   restricted rebinding to current/query frontiers nor checked session,
   sequence/page/representation generations and codec/model identities. That
   was too broad to authenticate a harmless extension of the same allocation.
3. `committed_generation_view` excluded every prefill-shaped ubatch, including
   `turn.phase == query_replay`. Native multi-token query replay is itself a
   prefill-shaped forward pass. It could therefore enter advisory recovery
   rather than the existing exact identity/content-version frozen-history
   branch. This conflicted with `llama_kv_cache::apply_pager_live_policy()`,
   which explicitly preserves final committed history during both replay and
   generation, and with the freeze/commit transition contract.

Implemented changes, limited to graph-view selection:

- Process all advisory route entries and keep every successfully validated
  resident entry that fits the bounded view. A stale entry does not terminate
  processing of later valid entries.
- Do not clear `selected_pages` during refresh. Seed its optional priority list
  from that already-validated set, then consider current/pinned, sink, request
  boundary and recent/prior resident pages for unused capacity. The existing
  helpers still prioritize every mandatory query page, filter out host-only
  pages and enforce admitted attention capacity using the same snapshot.
- Rebind a missing exact ID only for a current/query dirty or filling frontier
  whose old identity was a tail and whose extent has not shrunk. Compare the
  entire identity with only `position_end` normalized to the old extent; every
  other authenticated field must match. A new generation at the same logical
  position is not a match.
- Treat `query_replay` as a committed view independently of whether its kernel
  phase is prefill, decode or MTP verification. Missing, changed or over-capacity
  frozen history still uses strict refusal, never advisory substitution.

For the illustrative route `[resident A, evicted B, resident C]`, the revised
source retains A and C (subject to mandatory-query/admitted-capacity limits),
omits B and fills only spare capacity from current residents. This is a
source-level expected behavior, not a newly executed test result.

Promotion ownership remains unchanged: a fresh authenticated selector candidate
must reach policy admission, eligible slot selection, H2D completion and mapping
publication before a later graph can consume the page. No graph-build H2D copy,
global fence, host-catalogue scan, forced promotion, per-token diagnostic counter
or route/precision/batch configuration change was added. Existing valid advisory
routes do not allocate the stale-refresh vectors; the extra fallback identity
checks run only after an exact snapshot lookup misses.

The follow-up test should cover a stale entry between two valid entries, a
same-allocation tail extension versus a changed generation, and multi-token
replay with missing frozen history. Run genuine promotion proof separately;
neither this source fix nor a faster/no-crash request establishes that proof.
