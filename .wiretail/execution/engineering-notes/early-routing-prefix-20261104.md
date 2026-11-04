# Automatic early routing prefix for ACO

## Purpose and measured starting point

Keep GPU Turbo4 target hot pages, canonical Turbo4 host storage and GPU Turbo4
MTP. Remove the second **full-model** evaluation of a new user span when its
history selection changes. The all-hot reuse lifetime repair is already in
`f12d3ff14`; it must remain intact. The clean baseline is `3dfcd0248`, with its
binary and DSOs frozen outside Git at
`/srv/ai/paged-kv/builds/aco-before-prefix-3dfcd0248/bin`. Set that directory as
`LD_LIBRARY_PATH` when comparing: the executable alone does not identify the
loaded implementation.

The previous 256K fresh-content run had no replay through C=62,465, then one
full query replay per append beyond H=65,536. Backend versus net-new speeds
were 818/409 tok/s at C=78,849 and 756/378 at C=256,001. Thus reducing graph
construction alone cannot remove the roughly twofold work beyond H.

## Implementation decisions

1. Automatically use an internal routing-prefix graph for a supported hybrid
   model with `probe-rerank` and leading recurrent layers before its first full-attention layer.
   Start with Qwen3.5/Qwen3.8's `qwen35` implementation. Discover the boundary
   from `hparams.is_recr()`, not a hardcoded layer number. Do not replace a
   model's native sparse indexer with this approximation. Unsupported models,
   non-text/multiple-sequence batches, all-hot requests and unsafe boundaries
   retain the existing path. There is no separate user-facing mode.
   **Measured adjustment:** apply the early-layer proxy only to spans larger
   than one configured batch (B=1024 here). Short recall questions retain the
   stronger all-layer evidence and its bounded query replay. A matched small
   test doubled bulk-ingestion speed but lost a fact with the proxy on the
   short question; that is evidence against using it for every query.
2. Before evaluating the final new-user span normally, copy only the active
   leading recurrent state into a scratch owner, sharing the model weights.
   Run those leading blocks and the first attention layer's normalized,
   position-encoded Q projection. Stop before attention, target K/V writes,
   upper blocks, output projection and MTP. Canonical recurrent and draft state
   must not change. Do not restore or replay old prefix tokens.
3. Process the span in the configured micro-batch (U=256), retaining only the
   existing four query probes. Reuse the context's sequential scheduler and
   scratch. Never retain prompt-length hidden activations, make a second model
   instance, duplicate full-L K/V, or grow staging with context length.
4. Run the existing authenticated coarse shortlist and encoded-key mass rerank
   against **the same first full-attention layer only**. Its Q and K must have
   matching head dimensions, RoPE and Turbo4 coefficient transforms. Name this
   source explicitly as early-layer mass: it is not all-layer exact ranking.
   Publish common logical page bundles across all target attention layers via
   the existing validated transfer transaction; a proxy score is never a
   substitute for page/content/generation authentication.
5. Freeze the selected history once, before the real full-model prefill. The
   final-user completion boundary then acknowledges the already frozen route
   without restoring the checkpoint or incrementing full-query replay count.
   Generation and MTP verification use that same frozen history. Cancellation
   discards scratch queries and follows the existing turn cleanup.
6. Fail safely before publication when capability/state/probe checks cannot
   support the early graph. Clear its pending query artifacts before the
   ordinary all-layer path captures anything. Do not turn a partial/failed
   publication into silent success. Preserve existing publication failures as
   explicit request failures rather than generating with mixed identities.

## Concrete source boundaries

- `src/models/qwen35.cpp`: graph dispatch and leading recurrent/FFN loop; share
  the exact Q norm/RoPE projection with the production attention path.
- `src/llama-graph.h/.cpp`: internal routing graph type and recurrent-only input
  construction. No attention-mask or packed-K/V workspace for this graph.
- `src/llama-memory-recurrent.h/.cpp`: filtered scratch state owner and active
  row copying; preserve source row/rollback-plane semantics without mutating
  canonical cells. Keep scratch tensors alive across CUDA graph reuse. Override
  only the prefix graph's `cparams.n_rs_seq=0` to match this single-plane owner;
  the target and draft retain their original speculative rollback settings.
- `src/llama-memory-hybrid.h/.cpp`, `src/llama-context.h/.cpp`: scratch recurrent
  graph context plus read-only selector access, sequential bounded execution.
- `src/llama-kv-cache.h/.cpp`, `src/llama-kv-router-job.*`: explicit participating
  source layer and provenance, existing authentication and bundle publication.
- `tools/server/server-context.cpp`: exact final-user seam, before its first
  real decode; use the task's actual token IDs/positions. No endpoint/client
  opt-in, filename inference or synthetic expected-answer injection.

## Memory and timing contract

For this model the first full attention follows three recurrent blocks. Their
active R/S state is 9,805,824 bytes (9.35 MiB). Four Q probes of 256 dimensions
by 24 heads use 98,304 bytes; the existing separate query accumulator adds
another 98,304 bytes plus 88 bytes of count/validity sidebands per source layer.
A U=256 by 5,120 F32 hidden slab is 5 MiB and is
temporary scheduler storage, not a persistent new allocation. Retaining 32K
hidden rows would cost 640 MiB and is explicitly forbidden. Record actual
allocated/sampled peaks; estimates do not establish fit. No per-token/layer
CPU readback, broad tensor callback, telemetry fence, inventory dump or
diagnostic CUDA synchronization belongs in the normal path.

## Validation order and decisions

Build incrementally with `cmake --build build-cuda --parallel 16` and focused
targets only. Run the existing repository benchmark drivers in place and put
results outside Git. Do not copy a driver by itself to a temporary tree: its
chat template, tokenizer helper and `include/llama.h` paths are repository-relative.
Freeze binaries and DSOs for controls, rather than constructing an incomplete
copy of the benchmark repository. First compare immutable baseline and candidate at L=16,384,
H=8,192 (or H=4,096 to make cold history unambiguous), B=1024/U=256, Turbo4
target/draft, GPU native MTP n-max=2, reasoning off. Use disjoint repository
content, retained prior context, and one bounded recall sequence; record
answers independently from route execution. Include a normal all-hot request
to detect regressions and a request spanning H to prove the new route is used.

Only after small correctness works, repeat a representative L=262,144,
H=65,536 fresh-content prefill through C=78,848 using the frozen 32K source map.
Measure net-new/backend/wall tok/s separately, routing time, full replay count,
actual recurrent allocation, GPU-wide/process VRAM peak and host RSS/swap.
Compare against the frozen baseline on identical content and geometry. Use one
Qwen process, the lifecycle lock and managed service restoration. Preserve
Jellyfin and ports 8091/8092. Do not run the full reasoning/decode matrix as an
implementation gate. If the proxy degrades recall materially, repair selection
or retain the original route for that unsupported case; speed alone cannot
certify it. If scratch grows with L or prompt length, fix ownership before a
long test. Write compact actual results here or in a linked evidence document;
raw logs/binaries stay outside Git.

## Status

Implemented in `6068d4a003c6ad0e117a4808097f34f0056e0e85`. Incremental CUDA
builds and four focused assertion-enabled checks pass. The first beyond-hot
attempt caught and repaired the scratch/rollback graph mismatch. The initial
unrestricted proxy doubled ingestion speed but missed the short fact recall;
that failure caused the automatic bulk-versus-short-query policy above.

The final matched small test recovered the source figures, measured 1,537
versus 752 HTTP-wall ingestion tok/s for the beyond-hot append, and retained 93.17% Prompt-1 MTP
acceptance (400 output tokens). Full L=262,144/H=65,536 prefill completed at
C=261,121: 261,121 net-new tokens versus 262,102 reported backend prompt tokens,
768.04 net-new prompt tok/s and 764.22 HTTP-wall tok/s. The old run reported
459,657 backend tokens, 449.94 net-new prompt tok/s and 448.61 HTTP-wall tok/s.
All nine early-routed appends used one target pass and constant 9,805,824-byte
prefix state. A 993-token tail used the deliberate short-query fallback and
replayed 981 tokens; the other 14 of 15 appends had no full query replay.

GPU-wide sampled peak was 15,567 MiB versus baseline 15,589 MiB. Both runs ended
with the same 168.3167 MiB CUDA compute allocator; server swap remained zero.
Services were restored and Jellyfin/8091/8092 preserved. These are bounded
functional/performance results, not a full retrieval-quality or reasoning-mode
matrix certification. See [the compact validation report](../evidence/EARLY_ROUTING_PREFIX_20261104.md)
for exact curves, identities and raw pointers. The full benchmark suite remains
paused pending operator direction.

The subsequent user-requested low-reasoning decode check also completed:
one fresh fill to C261121, one 40-token warmup and three measured requests
capped at 400 tokens for each original prompt. Decode medians were
59.06/44.45/55.49 tok/s versus historical ACO 33.82/27.73/31.35 at C256001;
median MTP acceptance was 90.00%/56.12%/81.25%. All 12 requests passed,
sampled GPU peak was 15,585 MiB, and services were restored. The two builds
and exact occupied content differ, so this establishes no observed decode
regression in the comparison rather than isolating one change's effect.
See [the low-mode comparison](../evidence/EARLY_ROUTING_PREFIX_LOW_DECODE_20261104.md).

A subsequent natural growing-conversation test also passed: 32K of repository
content including `src/llama-kv-fixed-window.cpp`, an optimization/code-generation
question, 96K of different files, an unrelated question, then a paraphrase of
the first question without resending the file. The critical source page64 was
cold/host-backed before recall and was naturally authenticated, promoted,
published and used by target/MTP verification with its identity unchanged.
Both generated replacement functions compiled unchanged and passed ten
behavior checks; recall MTP acceptance was 91.29%.

For this growing conversation use the **native model template and constant low
reasoning/preserve-thinking settings**. The isolated-benchmark effort template
moves a steering message when the last user changes; that edit caused full old
prefix recomputation in the excluded diagnostic attempt. The corrected native
run preserved the file/history cache through all pressure appends.

The recall's first generated token arrived at 2.779s, versus 2.625s for an
identical request with that source page already hot. Both turns promoted other
pages, so this is not an all-hot control or an isolated promotion surcharge.
Measured key rerank jobs were about 775/786ms; page-transfer counter deltas were
5.1/5.2ms. Timers overlap and do not establish a single additive ACO-versus-off
cost. See [the source-promotion and latency evidence](../evidence/NATURAL_FILE_PROMOTION_20261104.md).

A matched all-hot 32K file-context comparison with pristine master now measures
the ordinary short-question/request path: median first-token latency was
117.62 ms on master and 434.42 ms on ACO, a 316.80 ms difference. Each of three
clean samples per build reused 32,832 cached tokens and evaluated the same 91
new tokens. The short query skips the bulk early-routing prefix; logs show
no query replay, H2D queueing, promotion or eviction. This does not isolate
cold-selection cost or one code change. See [matched 32K master control](../evidence/MATCHED_32K_MASTER_ACO_LATENCY_20261104.md).
