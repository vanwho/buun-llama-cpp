# Request-time GPU memory growth: native-MTP scratch ownership

## Confirmed code defect

The 105-15 tiny request previously grew GPU use from 14,965 MiB to
15,943 MiB with allocated L=262,144 and target hot H=32,768. There was
one Qwen CUDA process. That was not proof that more GPU hardware was needed.

`llama_kv_cache::vbr_scratch_reserve()` in `src/llama-kv-cache.cpp`
overrode the submitted materialization width with `layer.k/v->ne[1]`
whenever `pager_plan_ == nullptr`. Native MTP satisfies that condition:
its encoded Turbo4 draft cache has full-L backing, independently of the
target pager's H. But `llama_kv_cache_context::get_k()/get_v()` expose
`attention_source_rows(false, n_kv, physical_rows) == n_kv`, the occupied,
padded prefix. The reservation was charging unused backing capacity, not
the attention tensor that would execute.

For this model each F16 side is 4 KV heads × 256 elements × 2 bytes:
2,048 bytes/row. The stale override reserved 262,144 × 2,048 =
536,870,912 bytes for K and the same for V: **1 GiB** in total. A tiny
256-row padded view requires only 524,288 bytes per side.

The allocation owner is CUDA's persistent `fattn_scratch.k/v`, not model
weights, an extra Qwen process, a host checkpoint, or speculative-token
accounting. See `ggml_backend_cuda_kv_dequant_scratch_reserve()` and
`kv_dequant_scratch_try()` in `ggml/src/ggml-cuda/fattn.cu` and the
backend context in `ggml/src/ggml-cuda/common.cuh`.

## Implemented correction

- Keep the submitted `materialized_k_rows/materialized_v_rows`; do not
  replace them with encoded cache capacity for non-paged/MTP contexts.
- Continue reserving at the decode boundary, before graph execution,
  with the existing overflow checks and recoverable allocation failure.
- Identify `LLAMA_CONTEXT_TYPE_MTP` as the draft role in
  `llama_context::prepare_kv_attention_graph()`. The previous architecture
  test only recognized DFlash and mislabeled native MTP as target.
- Defer native-MTP scratch admission until the actual graph has been
  allocated, but still before `graph_compute()`. Query the optional backend
  capability `ggml_backend_vbr_fused_turbo4_attn_v1` for each real
  `FLASH_ATTN_EXT` descriptor. CUDA shares the fused shape predicate with
  execution and checks the compiled feature, device and environment. A
  confirmed fused op requires **zero** K/V F16 scratch. Unknown/disabled
  capability reserves the actual submitted fallback view; F16 sides need
  no duplicate buffer. The legacy backend vtable ABI is unchanged.
- Do this on graph construction, not on every reused graph. Failed
  admission clears graph-reuse eligibility so a retry cannot skip reserve.
- Thread `graph_scratch_admission` explicitly from the native-MTP context
  through `llama_model::create_memory()` to the cache. Its legacy
  `prepare_with_slots()` watermark reserve must not run ahead of graph-aware
  admission. Preserve dynamic VBR mapping/transcode fences and non-MTP
  pregraph admission. The legacy request's default `role=target` label does
  not establish that the failing allocation belongs to target attention.
- Audit graph allocation separately from persistent scratch: the existing
  `ggml_cuda_get_best_fattn_kernel()` returns VEC for Turbo KV, and
  `ggml_cuda_flash_attn_ext_get_alloc_size()` already sizes only the output
  for matched Turbo4. It does **not** add graph-owned F16 K/V copies in
  this configuration. No speculative graph-allocator rewrite is retained.
- No codec, batch size, MTP placement, router, hot-page count, or admission
  margin was changed. No new always-on diagnostic instrumentation is needed.

## Meaningful validation, not a renamed geometry

Use old and corrected server **and loaded project DSO** identities. The
small launcher executable alone is insufficient: its absolute RPATH can
load newly rebuilt libraries even when copied beside old libraries. Bind
the old control's `LD_LIBRARY_PATH` to its preserved bundle and verify
`/proc/PID/maps` before issuing requests.

Compare fixed L=262,144/H=32,768 before and after, then fixed
L=262,144/H=65,536 after. `-c` is L; `--kv-hot-pages` is H divided by
256. Keep B/U=1024/256, Turbo4 target/draft and GPU MTP. A smaller L is
not an equivalent regression and must not be labeled as one.

Existing verbosity-5 reserve logs expose requested bytes and physical
projections. Pair those with startup/post-request GPU memory, exact
process argv, request artifacts and request-local MTP counters. A one-token
request diagnoses allocation only; it is not an MTP-acceptance benchmark.
Then cross H with ordinary cumulative context to check real growth.

## Controlled first-fix results

The corrected comparison on 2026-10-09 held **L=262,144** constant and
verified old/new loaded DSO paths, not just the launcher. Original candidate,
H=32,768: requested 512 MiB per side; free memory fell from 982 MiB to
4 MiB after the 16-token/max-one-token request. Occupied-prefix correction,
H=32,768: requested 524,288 bytes per side; free memory 982 → 998 MiB.
H=65,536: the same small reserve; free memory 454 → 470 MiB. Both corrected
80-token canonical prompt-1 probes drafted 65 tokens and accepted 46.

These measurements establish the initial full-L inflation defect. They
precede the graph-aware fused correction and do not prove its long-context
behavior. Raw comparison:
`/srv/ai/paged-kv/results/forward/vram-first-request-fix/raw/corrected-20261009T014737Z/`.
An earlier diagnostic accidentally changed L instead of H and is **not**
an equivalent comparison; its smaller-capacity measurements are excluded.

## Limits on conclusions

The first graph-aware live validation (2026-10-11) kept L=262,144,
H=65,536, B/U=1024/256, and Turbo4 GPU target/draft unchanged. The tiny
request and 80-token canonical prompt-1 probe passed, with 61 proposals and
48 accepted (78.69%). Nine ordinary cumulative requests reached 60,205
occupied tokens. Request 10 failed near 65,787 prompt tokens because the
**earlier legacy cache reservation** still requested 134,742,016 bytes
per K/V side, growing its physical workspace from 256 to 512 MiB. The
failure propagated from `llama_decode(ctx_dft)`; its legacy `role=target`
log label was not a reliable owner classification. This located the
remaining bypass; it did not establish that encoded Turbo4 itself needs
that workspace or that a larger card is necessary.

Raw:
`/srv/ai/paged-kv/results/forward/vram-first-request-fix/final-live/run-20261011T002757Z/`.
The managed service was restored with its exact original identity. The
legacy-reserve deferral and fused output-only allocation regression were
validated in the separate successful run below; do not relabel this failed
run as their proof.

Removing the unused full-L reservation does not prove a fully occupied
256K/64K-hot campaign. The target hot pool and the separate full-history
draft cache have different owners. F16 materialization, when actually
required by a CUDA fallback, grows with the occupied draft view. Eligible
fused Turbo4 attention instead consumes encoded KV directly. The second
correction now excludes its unused materialization from admission without
globally forcing another attention route or disabling OOM checks. Full
256K occupancy remains a separate measurement, not a claim from a tiny test.

Raw diagnostic artifacts and the old bundle are local-only under
`/srv/ai/paged-kv/results/forward/vram-first-request-fix/`.
Do not commit shared libraries, full logs, or raw request streams.

## Corrected build verification

Final source at `24669529e` builds with parallel 16. Focused CPU CTest
passed 3/3 and `test-mtp-attention-scratch --require-fused` passed on the
RTX 4080, including output-only graph allocation, invalid metadata,
disabled-fused and null-backend checks. A source edit raced the first CUDA
compile; the new null-backend test exposed its stale object. Refreshing
only `fattn.cu`'s build timestamp and recompiling resolved that mismatch.
No production allocation policy was changed to accommodate a test error.
Tests used the explicitly verified `build-cuda/bin` DSOs.

Compiled artifact SHA-256:

- server: `d3626be43428d7825c5f5d1269f58f2d6fda2e224700e3ebc0f8f2f23248e17a`
- libllama: `d411421d91f00a69d739d958a0637d2904ee1326fc7ac6dcba56261f9770a83b`
- libggml-cuda: `b0a79464d81a0511cd8d115e2520eaf441e209b201528ab82c40bff0a6a07404`

Logs: `/srv/ai/paged-kv/results/forward/vram-first-request-fix/final-build/`.

## Successful cross-hot-boundary validation

The final run at source `24669529e` used the exact same L=262,144,
H=65,536 (256 pages), B/U=1024/256, Turbo4 target K/V, full-L GPU
Turbo4 draft-MTP n-max=2 and automatic production routes. Actual loaded
project DSOs matched the compiled hashes above. One managed Qwen owner;
no CPU attention, codec substitution, smaller hot pool or route override.

Eleven ordinary cumulative source requests reached **C=71,218**, crossing
the 65,787-token point that failed previously. Their individual source
chunks were 5,304–8,293 tokens; these were not 70K fresh single-turn inputs.
An 80-token canonical prompt-1 follow-up completed at 71,249 prompt tokens
with 71,218 cached and 52 freshly processed tokens. MTP drafted 65 and
accepted 46 (70.77%); the initial clean canonical probe had the same counts.
These are short correctness/memory diagnostics, not the 400-token matrix
medians or a recall-accuracy certification.

GPU MiB: ready used/free **15,493/454**; after tiny approximately
**15,453/494**; after clean canonical **15,473/474**; sampled peak
**15,617/330**. The full journal contained zero backend
`vbr_scratch_reserve` lines and zero scratch-allocation/dequant failures;
fused native-MTP admission continued to request zero materialization.

Raw:
`/srv/ai/paged-kv/results/forward/vram-first-request-fix/final-live/run-20261011T004232Z/`.
The exact original qwen38-fast service was restored healthy on 8080,
matching argv, executable, loaded DSO set and configuration hashes; no
drop-ins remained. Port 8091 was preserved and 8092 was untouched.

This establishes 64K-hot operation past its occupied boundary on this
16 GiB card and resolves the observed scratch OOM. It does not establish
fully occupied 256K performance or guarantee arbitrary future workloads
will fit. Resume 105-15 preparation with a new frozen candidate cohort,
then collect the prescribed remaining curves and benchmark tables.
