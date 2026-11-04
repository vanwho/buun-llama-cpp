# Automatic attention reuse implementation

Current continuation (2026-11-04): `6068d4a00` adds automatic early routing for
bulk user spans and preserves all-layer selection for short recall questions.
The fresh 256K fill completed at C261121 with 764.22 wall tok/s versus 448.61
previously, bounded 9.35 MiB prefix state and no measured VRAM increase. See
[the detailed design](early-routing-prefix-20261104.md) and
[current validation](../evidence/EARLY_ROUTING_PREFIX_20261104.md).
The full decode/reasoning suite remains paused.

Earlier milestone (2026-11-04): commit `f12d3ff14` fixes the all-hot replay
failure found by the first live run. A production build passed an exclusive
fresh prefill ladder through C62465/H65536 with zero replay and backend work
identical to net-new input. See `../evidence/ALLHOT_REUSE_REPAIR_20261104.md`
for current evidence; historical results below remain historical. The operator
then authorized another prefill-only curve through the 256K allocation, now
complete at C261121. The full decode/reasoning suite remains paused. Jellyfin must remain running;
authorized exclusive test runs restore the managed server/gateway afterward.
The implementation adds no client mode, filename heuristic or file registry.

## Code direction and invariants

- `server-context.cpp` supplies an internal planned endpoint: actual task
  prompt size + bounded n_predict + speculative lookahead + one row. An
  unbounded output never claims all-hot admission. This is no client API field.
- `llama-kv-cache.cpp::begin_kv_pager_turn` preserves the plan through empty
  inventory/deferred activation. The automatic path requires the full plan
  to fit both H and the configured attention bound. Existing prefix coverage
  is proven by positions/valid rows/owner identity, not a resident count.
- `recheck_full_attention` re-evaluates eligibility after cache restoration
  before a query graph has run. A safe miss downgrades to ordinary retrieval
  with query capture restored. Once query state exists, unexpected data loss
  remains a real refusal; no incomplete view becomes a successful reuse claim.
- `llama_context::prepare_kv_attention_graph` uses complete causal coverage
  for eligible all-hot requests. Otherwise, first query pass, replay, decode
  and MTP use the same initial/final R plus bounded recent Q construction.
  Earlier incoming messages remain ordinary causal prefill, retaining sink
  and available prior selection, before the final-user seam.
- `llama-kv-attention-view.*` owns pure coverage and replay-certificate logic.
  Successful selected prefill submissions record what was submitted; the
  existing query-commit scheduler fence precedes certification. No generation
  token collects a certificate. Storage is O(H pages), not O(total context
  or number of batches). The query frontier tracks contiguous completed work.
- The certificate compares full historical identities/content versions and
  expected recent-Q coverage. Physical slots, harmless epochs and invisible
  future pages are irrelevant. Q and its seam page may roll out normally;
  full old R must remain identical. A partial seam page may grow by append,
  but sequence edits, copies, shifts, state imports, pre-seam writes and VBR
  representation changes invalidate the proof before a skip can be allowed.
- `llama_kv_pager::publish_page` reports whether valid rows form a contiguous
  prefix, computed inside its existing row scan. Interior holes cannot pass
  the new coverage check merely because valid_length names a later row.
- Commit builds the same bounded R+recent-Q view as forward execution. Failed
  proof requires the original one-replay transaction and joint recurrent/draft
  carry restore. Exact/observe/off retain their prior commit behavior.
- `kv_full_attention` is an internal graph-topology discriminator: crossing
  between selector-free full coverage and query routing rebuilds once; stable
  graph geometry continues to reuse normal CUDA graphs/descriptors.

The existing 256K repository map and checkpoint ladder are unchanged. These
changes require a new immutable candidate/cohort after the hold is released.
The old opt-in ingestion build is withdrawn and must not be used. Historical
1d74e9a31 results remain historical, including their paid query replay.

## Validation and next bounded live check

Build incrementally with parallel16; do not clean the CUDA tree or alter B/U.
Targets: llama-server, test-server-prompt-cache, test-kv-attention-view,
test-kv-pager and test-server-query-replay. Run the three model-free prompt-cache,
attention-view and pager executables. External build/test logs are at
`/srv/ai/paged-kv/results/forward/automatic-attention-reuse-20261103/`.

The fixtures cover coverage holes/cold pages/mixed owner generations, versions,
physical remaps/epochs, invalid/incomplete query work, causal future-page
exclusion, exact and partial seams, two-page rolling windows with and without
old R, explicit invalidation, graph topology transitions and cached-query seams.
The pager fixture checks that the actual owner reports an interior row hole.

When the operator resumes live testing, do one bounded functional check before
the final curve: one fresh candidate/process at the unchanged B1024/U256,
Turbo4 target/draft, GPU MTP2. Use the frozen repository map and a small prompt
whose prompt+400-output+lookahead fits H; expect full coverage and zero query
replays. Then exercise C>H with a short follow-up, confirming that a stable
actual view skips replay and a genuinely changed selection still replays once.
Keep original three-prompt output budgets/sampling and normal reasoning setup.
Include mixed source/question, source in earlier messages, repeated/updated
filenames and a rewind to a shorter prefix. Do not impose answer-format or
promotion oracles on this speed check. Record paid backend prompt work and
net-new/wall throughput separately, plus request-local MTP counts and actual
memory. No live speed or accuracy improvement is claimed until those runs.

Implemented runtime commit: `d2d9c7291`. The five CUDA build targets and all three
model-free fixtures passed in the `final3` logs. A real-owner test first exposed
that interior removal published its new valid_length without recomputing the
contiguous-prefix flag; that mutation path was fixed and the assertion now
passes. No live server was replaced, no final prefill test ran, and Jellyfin
remained healthy. The runtime source slice is separate from these plan notes
and earlier development-only benchmark commits.


## Authorized prefill-only result (2026-11-03)

The operator authorized one fresh L262144/H65536/B1024/U256/Turbo4 target+draft,
GPU native MTP2 prefill curve, without canonical decode tests. Candidate binary
SHA-256 was `da58c02c2c9a9cbc7b1879e07fa96fa86b02f4ca155feab09a0a2c1b12587dc2`;
no runtime rebuild or source change occurred during the attempts. Full suite
remains paused. See `../evidence/AUTOMATIC_REUSE_PREFILL_20261103.md` and JSON.

The completed 32K-group attempt made 15 successful disjoint-source appends and
reached C261121, leaving 1023 tokens in the configured 256K allocation. Old
prefix cache counts matched the previous occupied frontier exactly. It did
515930 backend prompt-token evaluations for 261121 new occupied tokens:
790.36 backend tok/s, 400.01 net-new/prompt-second tok/s, 398.97 net-new/request-
wall-second tok/s. No decode or MTP acceptance conclusion follows from the
one-token fill ACKs. GPU reported-free minimum was 357 MiB, sampled used peak
15589 MiB, and process VmSwap stayed zero. Prior managed services were restored;
Jellyfin stayed running and healthy. Actual admitted physical page telemetry
was not sampled; 256 pages is the configured pool.

**Historical pre-fix result (all-hot issue resolved on 2026-11-04):** the initial request had
zero replay, but all 14 later new-user spans replayed once, including requests
below H. At that point the logs did not distinguish the full-candidate,
prefix-coverage or actual-view certificate rejection. The 2026-11-04 diagnostic
resolved this to a commit snapshot lifetime bug and supersedes this all-hot
conclusion; the historical measurements themselves are retained unchanged.

A separate 64K-group attempt failed after C130049 while finishing an approximately
65550-token new user span: exact-history publication returned transaction0,
generation0 and winners0. It was not an OOM. Code explains a likely path:
`llama_kv_query_history_budget` (`src/llama-kv-live-policy.cpp`) returns zero
when the mandatory Q-page count consumes the retrieval pool; a span crossing
257 pages exceeds this 256-page hot configuration. In
`llama_kv_cache::apply_pager_live_policy` the zero candidate budget makes
history_capacity zero and yields no selected winners. The commit fallback
only accepts empty publication when `exact_candidates().empty()`, so nonempty
ranked candidates with no budget can leave publication incomplete. The exact
failed predicate was not instrumented, so this remains code-supported rather
than a fully traced root cause. Smaller 32K requests completed; this is not a
runtime fix for large Q spans. Preserve authentication and commit validation.

The all-hot gap below is resolved by the snapshot-lifetime fix; the large-Q
zero-budget publication gap remains. Do not attribute all older mapped-chunk
speed gains to runtime code alone or automatically start a final suite.
The old variable per-file cohort and this mapped32K cohort have different
request widths; their comparison is useful operational evidence, not an
isolated code A/B. No new API/request mode was introduced.


## All-hot root cause and repair — 2026-11-04

The live diagnostic passed the original eligibility checks: actual physical
capacity256pages/65536tokens, attention_tokens0, verify1, candidate1, full1,
planned_end12292 for C12288. No prefix coverage rejection occurred. Replay
was requested later by the commit certificate, so changing admission limits,
turn geometry, prefix coverage rules or model placement was unnecessary.

The new all-hot commit loop used
`for (const auto & page : pager_->residency(seq).pages())`. The filtered
snapshot is a temporary; its owning shared state dies before iteration in
C++17, leaving a dangling vector reference. The compiler warned at that line.
This could corrupt selected history identities, which pruning then removed;
the actual-view certificate consequently saw changed history. Keep the
filtered snapshot in a local variable across selection and pruning. This also
removes a redundant filtered snapshot copy. No replay check was disabled.

The controlled two-request reproduction changed from 12276 backend evaluations
for 6144 new tokens/replay1 to 6144 evaluations/replay0. The final uninstrumented
build then passed 5 append operations through C62465: backend and net-new
rates matched at1457.53/1208.00/950.13/610.40 tok/s by checkpoint. Cached prefix
counts matched the prior occupied frontier. All temporary ACO-REUSE diagnostic
code was removed. Executable SHA alone is insufficient because this fix changes
shared libllama; retain the loaded DSO hashes in future candidate identities.

Release had disabled the assertions in test-kv-attention-view. Its target now
uses -UNDEBUG (or /UNDEBUG); actual checks exposed a stale unsupported-F16
expectation and an out-of-range slot in a two-slot fixture, now corrected to
test supported F16/unsupported I32 and valid slot reuse. An existing duplicate-
generation case also exposed early acceptance in covers(); it now checks all
intersecting records before returning success. The assertion-enabled fixture
passes. No test shape, acceptance threshold or telemetry gate was loosened.


The operator-authorized fresh 256K prefill rerun is complete at C261121. It
made 15 successful appends with exact cached-prefix reuse, no truncation or
request errors: 5 initial requests through C62465 had zero replay; the next
10 requests beyond H replayed once. Weighted rates were792.05 backend /
449.94 net-new /448.61 request-wall tok/s, versus790.36/400.01/398.97 before
the repair. The full-run net-new gain is12.5%; under-H interval gains were
89.3%/92.9%/93.9% at12K/28K/61K. Beyond-H interval rates were essentially
unchanged. Do not call beyond-H replay necessary solely because C>H: the
remaining actual-view/certificate decisions were not diagnostically traced
in this run. The full decode/reasoning suite remains paused. The fixed
candidate was tested, then the prior managed service was restored. See the
new compact evidence for exact configuration, DSO identities and raw pointers.
