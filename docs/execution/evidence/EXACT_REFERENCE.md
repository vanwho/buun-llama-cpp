# Exact page-wave reference evidence

Task 06-05 adds an internal exact route and a bounded page-wave executor. The
route is a correctness/quality oracle; it is not included in the selective
throughput gate.

## Numerical contract

Each resident or cold wave returns an unnormalized state `(m, l, o)`:

```text
m = max(logits)
l = sum(exp(logit - m))
o = sum(exp(logit - m) * value)
```

Two states are merged as:

```text
m = max(m_a, m_b)
l = l_a * exp(m_a - m) + l_b * exp(m_b - m)
o = o_a * exp(m_a - m) + o_b * exp(m_b - m)
output = o / l
```

Normalization is exposed as a separate operation and is rejected for an
invalid state. Empty/tail partitions are represented with `m=-inf`, `l=0`,
and a zero `o`; they merge as the identity. The deterministic CPU fixture
compares a highly unequal-max split against direct softmax with an absolute
output tolerance of `1e-5`.

## Page coverage and staging

`llama_kv_attention_exact_wave_plan` sorts input metadata by logical page,
rejects duplicate or invalid/tail pages, verifies resident IDs and physical
slots against the published residency epoch, requires canonical host backing
for cold pages, and emits all hot pages before cold pages. Cold pages are
partitioned into deterministic waves and charged to a separately bounded
staging ring. The executor records every completed logical page and rejects a
duplicate or missing visit.

The focused fixture covers a permuted four-page set with two resident and two
cold pages, a 17-token final tail, one-page waves, two-page double-buffered
waves, and serial-vs-double-buffered output equality. A 305-page plan covers
the 304-page-plus-tail shape without allocating K/V payloads. The four-page
fake ledger reports two cold pages, two H2D page units, and one peak staging
page; the large plan reports 304 cold pages and a 17-page peak wave.

The executor has no CPU Turbo4 fallback. It requires upload, wait, and compute
callbacks for cold waves and returns `not_configured` when that GPU boundary is
absent. Serial execution uses synchronous upload callbacks; double buffering
pre-submits the next cold wave to a second bounded staging slot.

## CUDA boundary

The existing direct Turbo4 page decoder now optionally writes bounded per-head
partial state in `[m, l, o[256]]` layout and permits the normalized output
pointer to be omitted in partial-only mode. The fixture exercises this path
with native-position causal filtering, a compact mask, a permuted physical
page table, and the existing Turbo4 K/V decoding. The RTX 4080 run and
memcheck output are preserved in [`exact-cuda-fixture.log`](exact-cuda-fixture.log).

This is a bounded kernel/partial-state check, not a claim that a live Qwen
graph streams a 256K host catalog. The full graph integration must bind the
executor callbacks to the host catalog, upload scheduler, CUDA streams, and
per-layer/query partial buffers.

## Deferred verification

The current server still does not expose a production `--kv-pager exact`
model route, full-context exact graph, page-transfer ledger, or expected
answer stream. Therefore 256K exact runtime, dense-vs-exact model logits, and
the frozen selective-vs-exact quality delta are deferred. The frozen
`pager-corpus-v1` and acceptance controls remain unchanged; no threshold was
retuned.
