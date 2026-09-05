# 20-01 retrieval evidence

Local implementation verification passes. The routing API now receives an
explicit complete inventory, including authenticated cold host pages, and the
pager preserves summaries across clean GPU eviction. The summary index contains
one bounded table per runtime attention ordinal/KV head; the query coordinate
identity rejects mismatched projected/transformed Q/K domains.

The production graph callback records each post-positioned `Qcur` tensor. At
the existing scheduler fence the cache reads the current query, scores the
matching layer/head table over all logical pages, unions bounded results, and
passes selected cold pages to the existing H2D/residency transaction. Without
a fresh Qcur, the boundary fails closed and does not synthesize an LRU choice.

Verification:

- CPU focused build: `test-kv-routing-summary`, `test-kv-routing-retrieval`,
  `test-kv-pager` passed.
- CPU `llama-server` target build passed.
- `git diff --check` passed.
- CUDA focused build with `-j8` passed, and the three CUDA routing/pager
  binaries passed.

The named live checkpoint remains deferred. No Qwen35 Turbo4 model/corpus
fixture is available in this checkout, so no live answer, cold promotion trace,
selected-layer trace, or dense-reference attention-mass recall is claimed.
The RTX 4080 was detected, but no model-backed workload was started or
disturbed.
