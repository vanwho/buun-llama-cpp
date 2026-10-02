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

The current 102-03 source change filters stale-refresh candidates against the
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
- Prefer filtering individual stale entries over clearing the entire selected
  set when one route element is stale. The current `selected_pages.clear()`
  fallback is crash-safe but can discard other still-valid attention-selected
  history for that graph; assess and repair this without changing the normal
  fast path.
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
