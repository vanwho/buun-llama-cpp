# YaRN and 1M context with ACO

Research date: 2026-11-11. No model build, inference experiment, profile change
or service restart was performed for this assessment. Source: current ACO
implementation `7c9938290` and current deployment configuration.

## Assessment

The positional encoding support is already present. Qwen explicitly recommends
YaRN beyond its native 262,144 tokens, with factor 4, original context 262,144,
RoPE theta 10,000,000 and the existing partial/interleaved MRoPE geometry.
Its examples configure 1,000,000 total tokens. Our binary convention makes the
4x allocation **1,048,576 tokens**; that exact capacity still needs validation.

ACO has no identified page-ID or absolute-position limit blocking this size.
The main RTX 4080 obstacle is real GPU memory, particularly full-context MTP
KV, followed by the allocation peak of the larger ranking catalogue. Reaching
1M while retaining 64K hot target KV and GPU MTP is plausible, but is not a
validated configuration obtained simply by increasing `-c`.

Primary sources:

- [Qwen3.8-27B model card, ultra-long context](https://huggingface.co/Qwen/Qwen3.8-27B#usage-guidelines).
- [Qwen3.8-27B config](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json).
- [YaRN paper](https://arxiv.org/abs/2309.00071).
- [Qwen's llama.cpp instructions for YaRN](https://github.com/QwenLM/Qwen3/blob/main/docs/source/run_locally/llama.cpp.md).

## Actual model and scaling configuration

The local `/srv/ai/models/text/current.gguf` points to Qwen3.8-27B UD-IQ4_XS.
Reading the GGUF metadata, without loading a model context, confirmed:

| Field | Value |
| --- | --- |
| Architecture | `qwen35` |
| Native context | 262,144 |
| RoPE base | 10,000,000 |
| Rotary dimensions | 64 of 256 per attention head |
| MRoPE sections | `[11, 11, 10, 0]` |
| Attention heads / KV heads | 24 / 4 |
| Stored blocks | 65, including native MTP |
| YaRN/scaling-factor overrides in this GGUF | None |

The required new context/position flags are:

```bash
-c 1048576 --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144
```

Retain the GGUF's frequency base, rotary dimensions and MRoPE sections. The
existing implementation resolves YaRN attention magnitude and beta defaults;
do not add an independently guessed attention multiplier. `--rope-scale 4`
means internal frequency scale 0.25; `--rope-freq-scale 4` would be wrong.
`-c` must also be set: scaling frequencies alone does not allocate 1M context.
No metadata override to pretend the native model was trained at 1M is needed.

The initial allocation candidate, **not yet admitted or tested**, is this
single-line command. It replaces the managed Qwen process through its normal
lifecycle; it must never be launched alongside another Qwen allocation.

```bash
LD_LIBRARY_PATH=/srv/ai/paged-kv/builds/aco-toolturn-resilience-7c9938290/bin /srv/ai/paged-kv/builds/aco-toolturn-resilience-7c9938290/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-aco-turbo4-mtp -ngl 999 --fit off -fa on -c 1048576 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager selective --kv-router probe-rerank --kv-page-size 256 --kv-hot-pages 256 --kv-pin-recent 0 --device CUDA0 --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4 --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144
```

Static YaRN affects short requests too. Qwen recommends using scaling when
long contexts are needed and factor 2 for a 524,288-token workload. Keep the
validated native-256K profile available alongside a future 1M profile.

## Code consistency: no new RoPE implementation appears necessary

- `common/arg.cpp:3410` parses scaling type, scale and original context.
- `src/llama-context.cpp:509` and `:554` resolve effective YaRN parameters;
  `src/llama-graph.cpp:2545` supplies them to model graphs.
- `src/models/qwen35.cpp:398` applies them to target Q/K through
  `ggml_rope_multi`; the native MTP path uses the same parameters around `:715`.
- The CUDA interleaved-MRoPE path actually calls the YaRN transform with those
  frequency/mixing parameters in `ggml/src/ggml-cuda/rope.cu:263–290`.
- `common/speculative.cpp:6389` copies target context parameters into native
  MTP, preserving the position encoding settings.
- Early routing uses the same Qwen attention builder and explicit absolute
  positions, with separate recurrent routing state. Its captured Q is already
  rotated. Cold-key reranking consumes that Q and the stored rotated K rows;
  `src/llama-kv-router-job.cpp:607` does not rotate keys a second time.
- `llama_init_from_model` adjusts shared `n_ctx_train` to original/scale at
  `src/llama-context.cpp:11431`. Target and native MTP repeat the same assignment
  idempotently. A warning before that adjustment is not proof scaling failed.

Start a fresh process and prefill for a different YaRN setting. Cached keys
were rotated under their original settings; existing unscaled KV must not be
reused as scaled KV. Resume identity already includes RoPE/YaRN fields in
`tools/server/server-context.cpp:341` and the resume-format contract.

## Memory: real payloads versus admission estimates

Read-only live snapshot: L=262,144, admitted H=65,536, occupied C=163,744.
Native MTP is observed as GPU Turbo4. The CUDA/driver free reading was about
139 MiB. These values describe the running system, not a 1M admission result.

| Owner | 256K | 1M estimate at the same H | Change |
| --- | ---:| ---:| ---:|
| Target hot KV, GPU | 1.03125 GiB | 1.03125 GiB | Fixed |
| Full-context native MTP KV, GPU | 264.125 MiB observed | About 1,056.5 MiB | About +792.4 MiB |
| Full canonical target KV payload, host RAM | 4.125 GiB capacity | 16.5 GiB capacity | +12.375 GiB |
| Logical 256-token pages | 1,024 | 4,096 | 4x |

MTP's small allocation differences from raw row payload include alignment and
auxiliary storage; the projection is an estimate, not a byte-exact reservation.
Host backing grows as pages are produced/sealed. It is not necessarily all
resident at server startup. Model weights, H-sized attention storage and fixed
recurrent state should not be multiplied by four.

The live snapshot also reports `routing_table_bytes=268,435,456` and
`catalogue_bytes=100,876,288`. These are **admission ledger calculations** copied
from `snapshot.admission`, not independently measured CUDA allocations. Their
formulas predict roughly another 1,056 MiB of reservation at 1M, but adding that
number to MTP as proven physical consumption would repeat past accounting errors.

There is real catalogue growth to handle: `src/llama-kv-cache.cpp:20063` builds
per-layer `[D,3,KV_heads,P]` F16 summary/bounds tensors plus page metadata and
membership. With this model, that is about 6 MiB per layer at 256K and 24 MiB
at 1M. Sixteen layers imply 96 versus 384 MiB for one catalogue set. Graph
liveness, input/output sharing and the active routing path determine how many
sets are simultaneously resident. The admission routing formula instead uses
four subblocks per page; it must not be mistaken for the measured three-vector
op layout. Inspect actual allocator high-water before changing its budget.

Current process RSS was about 5.5 GiB, host MemAvailable about 21.3 GiB (22.85 GB), and
llama's VmSwap about 5.6 MiB. The nearly full 8 GiB system swap was not all llama.
The 16.5 GiB canonical payload can plausibly fit this 64GB machine, provided
snapshots, duplicated buffers and other workloads do not consume that headroom.
Keep cold KV in physical RAM for predictable promotion latency; process-level
swap counters alone cannot identify which allocations were swapped.

## Best route to 1M with 64K hot and MTP

1. **Use actual allocation peaks.** Attribute MTP, selector catalogue tensors,
   graph buffers and target hot storage separately. Correct the routing-ledger
   layout/lifetime estimates; do not assume removing a conservative reservation
   creates real VRAM. MTP expansion alone exceeds today's free space.
2. **Bound catalogue staging by active layer plans.** Keep canonical summaries
   host-backed and upload/consume the layer plans needed for a turn with bounded
   staging, preserving normalized mass/authentication. The bulk early-routing
   path only uses its first attention source layer. The richer short-query path
   can stream layer plans and merge compact scores rather than require duplicate
   all-layer catalogues in VRAM simultaneously. Preserve CUDA batching and avoid
   per-token uploads or synchronizations.
3. **If full-L MTP still cannot fit, extend paging to its KV owner.** The efficient
   architectural extension is a separate Turbo4 draft page store coordinated
   with the target's selected logical pages. Reuse the same turn-level selection
   decision, host backing and publication transaction; the target and draft own
   different KV tensors. Preserve complete committed draft rows and carry through
   rejection/rollback. Do not independently rerank cold draft pages per token.
   A 64K draft payload is about 66 MiB, versus about 1,056 MiB full-L at 1M.
   That could save roughly 990 MiB. For equal row geometry, adding the one draft
   layer to a sixteen-layer target promotion bundle adds about 6.25% to payload
   traffic. These are design estimates requiring implementation and validation.
4. **Preserve the working hot-attention configuration.** Keep Turbo4 everywhere,
   MTP2, B/U=1024/256 and the 64K target pool as the design target. A real measured
   lower-H allocation can establish a temporary feasibility point, but must not
   be reported as the wanted 64K-hot result. A short MTP-off allocation control
   can distinguish draft memory from catalogue memory without becoming a new
   full benchmark campaign.

At four times the catalogue size, the existing bounded shortlist may also need
more candidate coverage. First retain the proven router. If a retrieval miss
is localized to an eligible page being ranked out, compare bounded larger
shortlists/batched key rereads or hierarchical page summaries. More GPU memory
or a larger original-context setting does not itself fix ranking recall.

## Launcher, client and measurement changes

- The physical launcher currently has no explicit YaRN profile arguments.
  Add validated `ROPE_SCALING`, `ROPE_SCALE`, `YARN_ORIG_CTX` handling to
  `/srv/ai/scripts/start-primary-llama-profile.sh`, appending the exact flags.
  CLI environment overrides also exist (`LLAMA_ARG_ROPE_SCALING_TYPE`,
  `LLAMA_ARG_ROPE_SCALE`, `LLAMA_ARG_YARN_ORIG_CTX`), but variables merely sourced
  from a profile file are not automatically exported. Verify effective argv.
- A new named profile also needs the two shell profile allowlists, gateway
  aliases and `PROFILE_META` capacity updated. Preserve current profile names.
- `tools/server/bench/run-occupancy-frontier.py:16` hardcodes
  `MAX_CONTEXT_TOKENS=262144`. Replace that test-driver ceiling with an explicit
  resolved runtime capacity and keep checks against that capacity.
- Extend the frozen repository file/chunk map to 1M of distinct real content.
  Keep mapped chunks around the already proven 8K–16K ingestion sizes, native
  conversation/prefix reuse, B/U=1024/256, no disk restores during the run and no
  synthetic repeated-file fill. Generated replies count toward occupancy.
- OpenCode must advertise 1,048,576 total tokens and reserve its output budget
  and framing space. Native gateway routing already preserves the history.
  With 24,576 allowed output tokens, a fill near 1,000,000 leaves additional
  protocol/tool headroom within the physical 1,048,576-token context.

## Minimal future validation

Begin with actual 1M startup allocation, then one short code/tool request to
check usable output, native MTP and routing. A startup success alone says
nothing about million-token recall. Once memory is workable, use **one**
continuous mapped-file fill under the final 4x YaRN configuration. Record the
curve and short sanity points around the native 262,144 boundary, 524,288,
786,432 and near 1M. Keep output budgets generous enough for normal tool calls;
completed semantic misses become findings rather than repeated full fills.

One older upstream [Windows/CUDA report](https://github.com/ggml-org/llama.cpp/issues/27090)
described a crash near 520K with 4x YaRN on build b10434. It is not evidence
that our newer Linux build has that failure. Cross that boundary deliberately
in the same fill and preserve any real failure; do not add a guessed fix.

At the large frontier, ask for an old source detail/code change whose original
pages are actually cold, and inspect the same selector→transfer→publication→
target/draft use receipt that already works at 163,744 occupied tokens. Verify
tool-call coherence and MTP behavior. Run the full canonical three-prompt speed
and reasoning matrix only after basic long-context operation works.
