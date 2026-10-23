# Canonical incremental repository-context fill method

This is the only accepted method for repository-context fill and prefill-curve
tests in this project. A test turn must introduce only one fresh, disjoint
repository-file chunk. Never send a single cumulative user prompt containing
all earlier file chunks again.

## Request sequence

1. Start a fresh server process with the exact frozen model, binary, ACO mode,
   context, hot-page request, batch, micro-batch, and MTP settings for the run.
   Start with an empty slot/context.
2. Freeze a deterministic ordered manifest of tracked repository source spans.
   The manifest must record file path, source SHA-256, byte range, and the turn
   where each span is introduced. Chunks must be disjoint; no file span may be
   sent twice.
3. Submit one ordinary conversation turn per chunk. The new user message says
   to read/retain that chunk and contains only that chunk's files plus a small
   fixed per-turn instruction. It must not include any prior file text.
4. Preserve the preceding conversation as normal chat history so the same
   conversation continues. The server may receive the preceding messages as
   protocol history, but `cache_prompt` must reuse their exact prefix; only
   the new user turn and its assistant reply count as new prefill. Record
   `prompt_tokens`, `cached_tokens`, `new_tokens`, occupied-slot position, and
   wall/prompt timing for every turn. Reject a measurement if cached history
   is unexpectedly recomputed or if the current user message contains an
   earlier file span.
5. Keep each added file chunk near 32K tokens. Add explicit smaller chunks only
   at requested boundary checkpoints (64K, 70K, 76K, 84K); they must also be
   fresh source spans. Continue with new chunks until the requested final
   context is reached, leaving the declared output/context reserve.
6. Each MTP-on and MTP-off comparison is a separate fresh server process and
   starts from an empty slot. Never reuse a filled slot across configurations.

## Invalid methods and evidence

- Invalid: each request's one user message is the entire accumulated repository
  corpus, even if the server reports cached prefix reuse. That is not the
  requested one-new-chunk-per-turn workload.
- Invalid: repeat or overlap repository spans, use synthetic filler, or infer
  new prefill from a cached-history request without checking token counters.
- Prior runs at
  `/srv/ai/paged-kv/results/release/aco-128k-hot-256k-fill-20261021` and
  `/srv/ai/paged-kv/results/release/aco-no-mtp-128k-hot-256k-fill-20261021`
  used cumulative single-user prompts and are invalid for this method. Their
  raw artifacts may be retained for audit, but their summaries must not be
  included in performance comparisons.
