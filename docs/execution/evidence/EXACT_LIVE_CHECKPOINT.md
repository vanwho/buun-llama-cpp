# Exact live page-wave checkpoint

Task 10-05 adds the production-side exact inventory boundary. The pager now
enumerates live canonical host pages and reconciles them with the sequence's
resident table. The exact context preflight converts that immutable inventory
into a bounded `llama_kv_attention_exact_wave_plan`, records its coverage
ledger, and refuses execution if the graph/backend binding is unavailable.
This prevents exact mode from silently falling through to dense attention.

## Deterministic evidence

The CPU exact fixture passed with four logical pages, two resident pages, two
cold pages, a 17-token tail, one-page serial waves, and two-page
double-buffered waves. The online-softmax `(m,l,o)` merge matched direct
softmax within the existing `1e-5` tolerance. The executor visited all four
pages exactly once; the projected telemetry reported four planned/visited
pages, two cold pages, 200 useful H2D bytes in the fixture, and one peak
staging page.

The pager fixture also passed live host-page enumeration, including removal of
invalidated pages from the returned inventory. Missing host backing and stale,
duplicate, or missing page coverage remain fail-closed plan errors.

## Live boundary

The current graph has no per-layer exact node that can stage a cold page,
compute a CUDA partial state, and merge it before the next transformer layer.
Therefore the exact context route records the validated all-page plan and
returns `not_configured` for the absent CUDA upload/wait/compute binding. No
dense fallback, CPU Turbo4 dequantization, full logical-context GPU cache, or
model parity result is claimed.

## Verification commands

```text
cmake --build build --target llama test-kv-pager test-kv-attention-exact test-kv-attention-execution -j2
ctest --test-dir build -R '^(test-kv-attention-exact|test-kv-attention-execution|test-kv-pager|test-kv-residency|test-kv-attention-view)$' --output-on-failure
git diff --check
```

All five selected tests passed, and `git diff --check` passed.

## Deferred verification

The following require a linked CUDA target, a Qwen GGUF, and the eventual
per-layer graph callback implementation:

- CUDA upload/wait/partial compute and stream/event fence checks, including
  memcheck/racecheck;
- dense vs selected-all vs exact per-layer and final-logit parity;
- odd-tail, permuted hot/cold, cancellation, and maximum-context exact runs;
- raw timing, transfer bytes, staging high-water, and graph memory evidence.

Intended model path: `/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf`.
Intended raw log root: `/srv/ai/paged-kv/results/10-05/`.
