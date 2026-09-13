# V9 upstream integration contract

Read for phase 33 only; later packets consume its compact receipt, not merge
history. Repository `CONTRIBUTING.md` remains authoritative for conventions.
No public issue/PR text is generated or published by these tasks.

## Pinned source and Git ownership

- Integration before planning: `4af26bd68f0875db852fe6973f30b98be7f0f158`.
- Upstream and fork master: `da458765dde0e0414fc530e2e577d9a8e2ac6795`.
- Merge base: `cb703be37e3628dadb71912f3b3b25b82090555b`.
- Main project branch: `plan/attention-aware-kv-paging` on fork `origin`.
- Read-only merge preview yielded five conflicted paths. It did not change
  the worktree/index. Recompute the preview after 33-01, since its generic
  harness changes may add a small, legitimate conflict.

33-01 fetches the pinned commit if necessary, records refs and a safety ref
request in its handoff (the outer Git owner owns branch creation), and does
not replace the pin with a moving master. Task 33-02 declares
`upstream_merge_commit` in state. Wiretail auto mode prepares
`git merge --no-ff --no-commit <pin>` on its task branch. The task resolves
content and validates; it does not abort/commit/rebase. The runner records the
real two-parent integration commit, then execution metadata separately.
Interrupted conflict resolution resumes on the same task branch. An unrelated
MERGE_HEAD or missing pin is a setup failure, not permission to reset files.

## Exact conflict resolutions

Use `git diff --name-only --diff-filter=U` and `rg -n '^(<<<<<<<|=======|>>>>>>>)'`
on the five paths. Never resolve a whole file with `--ours` or `--theirs`.

1. **`common/common.h`**: keep the fork declarations
   `common_speculative_draft_kv_offload`, `..._device_is_available`,
   `..._device_name`, **and** upstream `common_params_postprocess_vbr` /
   `common_params_resolve_vbr_codec_auto`. They are independent APIs.
2. **`src/llama-cparams.h`**: retain both `kv_attention_tokens` and upstream
   `vbr_codec = LLAMA_VBR_CODEC_TURBO`. Preserve all other pager fields.
3. **`common/speculative.cpp`**, target-only restore recovery near
   `llama_get_embeddings_nextn_ith`: both branches already implement the same
   recovery. Adopt upstream formatting once; retain the fork's independent
   draft residency, full resolved context, query-sized batch and rollback
   handling. Do not duplicate recovery, embeddings copy, or state transition.
4. **`src/llama-kv-cache.cpp`**, top declarations: retain the fork inverse-FWHT
   declaration and complete `pager_failure_reason` function, then upstream
   `llama_memory_vbr_budget_bytes_resolve` and `..._partition` as separate
   namespace-level functions. The shared closing brace in the conflict is
   easy to misplace; compile this translation unit first.
   At slot planning near `vbr_scratch_reserve(scratch_cells)`, retain upstream
   non-unified stream-span/watermark sizing for ordinary/draft contexts. Do
   not restore an unbounded H/L dequant reserve for the paged target. Guard
   that upstream reserve by the **actual paged-target ownership state**
   (`pager_plan_`, verify its type/meaning after merge), not by codec or model
   architecture; draft has no target pager. Phase 34 installs explicit bounded
   A scratch. Preserve upstream recoverable reserve failure and non-pager
   multi-slot behavior. Do not remove all Turbo4 scratch globally.
5. **`tests/test-llama-archs.cpp`**, target-only restore fixture: differences
   in the common recovery block are mostly comments/indentation. Keep **one**
   shared recovery block, its verified hidden-carry comparison and next-batch
   draft repopulation checks. Preserve the fork's other dynamic-batch/full-L
   tests outside that block. Compare conflict sides before copying; do not
   duplicate the full fixture under a different name.

## Semantic merge checks and improvements to retain

These often auto-merge; compilation alone does not verify semantics.

| Upstream commits / subsystem | Decision and verification |
| --- | --- |
| `f3d7c9b3`, `ggml/src/ggml-cuda/ggml-cuda.cu` | Preserve owned pinned upload ring (8 MiB per thread; <=1 MiB fast uploads) and event ordering. Mutable inputs must not reference caller storage after return. Reuse this facility, not another per-page synchronizing upload path. Test replay, pointer ownership and multiple contexts. |
| `c9c52d718`, `common/speculative.cpp` | Keep target-only restore recovery without claiming it restores full historical draft KV. Compare fresh-to-fresh or paired target+draft restore for speed. Independent `--spec-draft-kv-device gpu` and full-L Turbo4 draft remain required. |
| `eb858cef3`, `90c50c197`, GDN/RMSNorm kernels and graph fusion predicates | Retain upstream recurrent Q/K normalization and gated RMSNorm fusion. This hybrid model benefits even though only its full-attention layers use paged KV. Verify ordinary and selected graphs preserve applicable fusions. |
| `fe5ab7e95`, fused MROPE/set-rows | Preserve upstream support predicates. F16/F32 fused stores do not prove Turbo4 fused encoding exists. Use supported fusion plus separate Turbo4 store; do not reinterpret compressed bytes as F16. |
| `2479cbfdd` and consumer MMQ prefill changes | GA10x tuning does not imply Ada 8.9 tuning. Retain generic defaults; later compare U64/128/256, inspecting actual IQ4 dispatch. No copied 3090-only threshold table or RTX4080 hardcoding. |
| `d0f82fd41`, prompt restore / occupied unified KV | Preserve upstream fixes and run repeated request / slot reset fixtures; avoid restoring a target cache with stale draft state. |
| `a0bd2fe21`, `b4545104c` | Optional op trace localizes dispatch; its host times are not CUDA kernel times. Keep non-finite numerical validation failures. |

No model-format migration (EXL3/safetensors), MoE/Blackwell tuning, or dynamic
VBR ladder is part of this merge campaign. Existing upstream functionality
must remain available when the pager is off.

## Turbo4 host backing versus CPU TCQ fixtures

The target contract uses `GGML_TYPE_TURBO4_0` (Turbo4/PolarQuant 4-bit) for
canonical sealed host pages, GPU target pages, and native-MTP K/V. Host RAM is
backing storage: the pager transfers encoded Turbo4 bytes to CUDA, where the
Turbo4 attention path consumes them. It must not convert canonical pager pages
to F16, Q8_0, or standard q4_0 merely because a CPU-only fixture is present.

`GGML_TYPE_TURBO3_TCQ` is a separate 3-bit trellis-coded format. Accurate TCQ
quantize/dequantize is CUDA-only in this tree; the CPU reference routines are
stubs. A CPU `test-llama-archs` result that specifically reports unsupported
`turbo3_tcq`/missing TurboQuant CUDA backend is an expected fixture limitation
and is recorded as `expected_cpu_backend_limitation`. It is not a merge failure
and does not permit changing the pager representation. Any result involving
Turbo4, host capture bytes, CUDA Turbo4 attention/transfers, MTP Turbo4/Turbo4,
or memory/lifetime correctness remains a substantive failure and must be fixed.

The evidence must include a Turbo4 host-byte capture/round-trip check and the
CUDA Turbo4 page/attention check. F16/Q8_0 may appear only as ordinary CPU
controls or explicitly labeled fallback paths.

## Tests and receipt

First configure CPU and CUDA builds with tests enabled. Use an explicitly
selected build directory, Release/RelWithDebInfo and `CMAKE_CUDA_ARCHITECTURES=native`
locally, not a production architecture constant. Build focused targets before
requesting the complete suite. Use `ctest -N` and target help to resolve names
after merge. Current relevant targets: `llama-server`, `test-arg-parser`,
`test-llama-archs`, `test-kv-pager`, `test-kv-attention-execution`,
`test-server-mmproj-lifecycle`, `test-cuda-fattn-paged-turbo4`.

The merge's live check is a short **feature-off GPU Turbo4 native-MTP** request
and repeat on the actual Qwen model. Broken selected paging is repaired next,
not hidden by calling a help smoke live validation. Keep a passing candidate
loaded. Incremental build waits are progress-aware; do not kill CUDA compilation
at 240 seconds and restart from scratch.

`V9_INTEGRATION.json` contains pin, pre/post code SHAs, resolved conflict paths,
semantic decisions, exact build configuration/commands, executable+loaded-DSO
hashes, all-GPU q0 pp/tg/MTP results, repeated-request result, feature support,
remaining selected-path defects and the site recipe receipt path. Missing
selected proof is explicit and scheduled in phase 34, not deferred forever.

Primary upstream source: https://github.com/spiritbuun/buun-llama-cpp/commit/da458765dde0e0414fc530e2e577d9a8e2ac6795
Inspect individual commits with `git show <id>`; do not browse old issue #116.
