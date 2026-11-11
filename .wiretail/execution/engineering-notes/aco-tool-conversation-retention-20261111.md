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
- Automatic generation sizing uses a stable H/4 (maximum 16K tokens), reduced
  when necessary to retain the instruction prefix. G is a rolling window, not
  a reservation for the entire max-output ceiling. The earlier attempt to grow
  G toward OpenCode's 24,576-token ceiling halved historical attention even for
  short replies and was removed. This partitions the same pool, without extra
  VRAM. The former one-page default only retained roughly 512 generated tokens.
- A retained instruction page shared with Q never blocks the generation FIFO.
  It stays resident but is excluded from recyclable G indexing; transient
  seal/event/pin fences still preserve the oldest-first recycling rule.
- Owner budgeting shares the rolling query/output allowance with Q. It charges
  only query pages exceeding G against historical R. Subtracting the whole Q
  after already reserving G unnecessarily dropped still-fitting history at the
  first selective transition. The physical target-size checks remain intact.
- A tool query includes the native assistant-response prefix, leaving the final
  prompt token after the commit seam to produce logits. Large tool payloads keep
  the existing fast early-routing capability; an experiment forcing their full
  all-layer query pass did not fix stale answers and was removed.
- Re-evaluating Q inside an otherwise identical cached prompt also invalidates
  speculative branch/carry state before the physical rewind. A carry from the
  old prompt frontier must not seed an earlier replay row. A later coordinated
  checkpoint restore can reinstall its own authenticated carry. Normal appended
  tool turns do not take this path.

The new metadata is bounded by the hot-page capacity and lives on the host.
There is no second K/V pool, extra model evaluation, per-token readback or
generation-token synchronization. Existing page authentication, transfers,
publication, replay certificates and target/draft history checks remain active.
Long-context selection is still approximate: these fixes do not make every
historical fact guaranteed to be retrieved.

## Validation

Final implementation commit: `7c993829047cf513035d3efeaa51480aec9e204a`.
Luna High performed the builds and live checks. CUDA 13.4/GNU 13, Release,
architecture `89-real`, incremental build with `--parallel 16`.

All six focused suites passed: chat-template, kv-pager, kv-policy, kv-prefetch,
server-prompt-cache and speculative-mtp-state. Coverage includes instruction
retention during eviction, single/batch output rollover past a protected
shared-Q prefix, bounded recent/history budgets and speculative carry transitions.
The model-backed query-replay executable was built, but its mere build is not
counted as a live replay proof.

The final native-tool conversation used L=262,144, admitted H=65,536, G=16,384,
B=1024/U=256, GPU Turbo4 target/draft KV, native GPU MTP n-max=2, low reasoning
and a 24,576-token output ceiling. It appended ten distinct repository-file
chunks, preserving actual assistant replies and normal server prefix caching.
There were no disk saves/restores, forced paging or CPU attention fallback.

| Result | Final candidate |
| --- | --- |
| Correct latest-file tool reports | 10/10 |
| Occupied tokens after last request | 83,656 |
| First completed request beyond hot capacity | 67,189 |
| Final cold pages | 103 |
| Instruction prefix resident | Every request |
| MTP accepted/drafted, cumulative for this smoke | 218/224 (97.3%) |
| HTTP, admission, nonfinite or capacity failures | None |

The ten short file summaries were also reviewed for coherent English and
obvious wrong-file content or repeated filler; none was found. This is a
multi-turn tool-coherence check, not the canonical throughput benchmark or a
guarantee of retrieval accuracy for every cold fact at 256K.

Two identical 2,522-token native requests were then sent to the managed final
candidate. Both produced the same coherent `report_file(chat.cpp)` call, with
MTP accepted/drafted counts of 56/86 and 52/90. The second request reused the
446-token prefix before Q and evaluated the 2,076-token query suffix again.
That is the intended new-request query capture, not an unsuccessful prefix
cache or a replay of the whole conversation.

Raw artifacts remain outside Git under
`/srv/ai/paged-kv/results/aco-toolturn-resilience-20261111/`:

- `ctest-final-7c9938290.log`
- `production-7c9938290-selected/summary.json` and its requests/responses/slots
- `managed-cached-repeat-7c9938290/summary.json` and its requests/responses

## Managed deployment

The existing `qwen38-aco` physical profile now uses the immutable server and
matching project DSOs in
`/srv/ai/paged-kv/builds/aco-toolturn-resilience-7c9938290/bin/`.
The loaded executable and `/proc/PID/maps` were verified. Existing ACO model
names and reasoning aliases are unchanged; L=262,144, H=65,536, B/U=1024/256,
Turbo4 target/draft and native GPU MTP n-max=2 remain configured.
Backend, gateway and long-memory health were checked; 8092 was preserved.
Test-owned locks and runtime service masks are removed after validation.

Subsequent inspection found that the live gateway still passed client messages
through an external history-rewriting proxy, which the direct backend smoke
above did not exercise. That layer has now been removed from the serving path;
8091 is decommissioned. See [native gateway and session storage](aco-native-gateway-20261111.md)
for the diagnosis, host changes, verification and future transparent KV resume.
