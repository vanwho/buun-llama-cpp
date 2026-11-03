# Automatic attention reuse implementation

The three requested improvements are implemented without a request mode,
filename heuristics, a file registry, or a change to the benchmark workload.
Final clean prefill retests remain paused. Jellyfin and the managed services
must remain running; this work does not deploy the rebuilt candidate.

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
