# ACO tool-conversation boundaries and retention

## Observed failure

An OpenCode conversation became incoherent after exceeding the GPU hot window.
The live process requested 256 pages but admitted 218: 55,808 tokens at 256
tokens/page. Its selected page inventory omitted several pages in the initial
instruction region. The server repeatedly logged ambiguous rendered-user spans
and fell back to a query starting at token zero.

The native Qwen template wraps tool replies in synthetic `user` messages, while
the previous boundary check counted only input messages whose role was `user`.
The fallback therefore lost the historical/current-query separation. Historical
page eligibility requires a page before `query_start`; a zero query start
cannot identify older pages for retrieval. Ordinary speed prompts and MTP
acceptance do not prove this multi-turn tool behavior is correct.

## Implemented behavior

- ACO performs one extra CPU template render with temporary markers in user
  and tool content. Removing the markers must reproduce the original prompt
  byte for byte. Only then are source byte ranges accepted. The actual model
  prompt is unchanged. Assistant reasoning content is never marked or edited.
- Byte ranges are mapped to native token positions against the exact rendered
  prompt. Source ordinals distinguish adjacent parallel tool replies from
  separate tool rounds with an intervening empty assistant tool call. Role-like
  text in a file and repeated question text do not change the boundary.
- The latest source user message or consecutive tool-result group is the
  query. Exact mapped positions need no legacy delimiter-position correction.
  Unsupported ambiguous text-chat layouts fail explicitly instead of treating
  the entire conversation as a new query from zero.
- The initial instruction prefix includes system/developer text and tool
  schemas. These pages are protected during prefill, history selection and
  generation. A bounded recent allowance prioritizes the latest user goal and
  a contiguous recent resident conversation, including assistant invocations.
  The suffix is fitted newest-first to at most half of admitted H, while roughly
  one eighth of the available historical budget remains open to ranked older
  history. Already-cold optional recent messages remain
  normal ranking candidates; this does not invent a second H2D implementation.
- Retention uses admitted H and the real query/history/generation budgets.
  Shared instruction/query boundary pages are charged once. Large file/tool
  payloads do not pin the entire cache. An instruction prefix which cannot fit
  or is missing from the actual attention view causes an explicit error.
- Empty ranking results reconcile mandatory resident pages into frozen R and
  leave generation space available. Single/batch write eviction obeys the same
  retained set, and request cleanup releases the retention metadata.
- A partial page shared with the incoming query stays writable; it is not
  frozen as historical R merely because its old valid extent ends at the
  query start. Optional recent retention also excludes that shared page, so
  the rolling output window can eventually recycle it during a long answer.
  Live admission uses the page geometry for this distinction.
- Old cold catalogue entries inside an upcoming query are not current query
  writes. Early routing does not demand uploads of invalidated future entries.
  A failed early publication also discards its unsubmitted server batch before
  releasing the slot, so the next scheduler cycle cannot consume stale rows.
- Automatic generation sizing starts at H/4 (maximum 16K tokens) and grows
  toward a bounded request's output allowance, up to H/2. Short subsequent
  requests restore the ordinary allowance. This partitions the same pool; it
  does not add VRAM. The former one-page default only retained roughly 512
  generated tokens on the live geometry, unlike OpenCode's long output budget.
- Owner budgeting shares the rolling query/output allowance with Q. It charges
  only query pages exceeding G against historical R. Subtracting the whole Q
  after already reserving G unnecessarily dropped still-fitting history at the
  first selective transition. The physical target-size checks remain intact.
- A tool query includes the native assistant-response prefix, leaving the final
  prompt token after the commit seam to produce logits. Large tool payloads keep
  the existing fast early-routing capability; an experiment forcing their full
  all-layer query pass did not fix stale answers and was removed.

The new metadata is bounded by the hot-page capacity and lives on the host.
There is no second K/V pool, extra model evaluation, per-token readback or
generation-token synchronization. Existing page authentication, transfers,
publication, replay certificates and target/draft history checks remain active.
Long-context selection is still approximate: these fixes do not make every
historical fact guaranteed to be retrieved.

## Validation

Builds and runtime checks are performed by Luna High agents with CUDA 13.4,
Release, architecture 89 and `--parallel 16`. Raw artifacts remain outside Git
under `/srv/ai/paged-kv/results/aco-toolturn-resilience-20261111/`; final results
are recorded after the final source and runtime checks finish.
