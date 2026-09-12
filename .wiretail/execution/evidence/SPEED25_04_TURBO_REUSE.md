# SPEED25_04_TURBO_REUSE

Result: `pass_with_deferred_live_reuse_measurement`.

The selected Turbo4 path now makes an explicit no-copy dense-view decision
when codec/domain, native causal order, valid tail, and physical row
contiguity all hold. Noncontiguous selections use a bounded per-layer compact
Turbo4 view: only selected K/V rows are copied as raw codec bytes, page
generation changes repack only changed pages, and current query pages receive
an ordered raw CUDA copy after the cache write. Dense Flash Attention consumes
the compact or no-copy typed view; no F16/F32 KV gather is introduced.

## Provenance and raw root

- Repository: `/srv/repos/vanwho/buun-llama-cpp`.
- Source identity: uncommitted source diff (`ggml`, `src`, and `tests`) SHA256
  `d25e5e15b590524997e964ff4b5131fa603a33ad8dbce372798002a60a5fc482` at
  receipt creation; no bundle or commit was created by this task.
- Raw CUDA fixture root:
  `.wiretail/execution/evidence/raw/SPEED25_04_TURBO_REUSE_cuda.txt`.
- Hardware fixture: NVIDIA GeForce RTX 4080, compute capability 8.9,
  15,945 MiB reported by the test.

## Local evidence

| Check | Result |
| --- | --- |
| `cmake --build build --target test-kv-attention-view test-kv-attention-execution -j2` | pass |
| `ctest --test-dir build -R 'kv-attention-(view\|execution)' --output-on-failure` | 2/2 pass |
| `cmake --build build-cuda --target test-cuda-fattn-paged-turbo4 llama -j2` | pass |
| `build-cuda/bin/test-cuda-fattn-paged-turbo4` | pass, exit 0 |
| `git diff --check` | pass |

The deterministic tests cover selected-page permutations, holes, partial tails,
native/physical contiguous eligibility, route selection, graph reuse keys,
changed-versus-unchanged pack accounting, and overflow accounting.

The CUDA fixture recorded the existing raw Turbo4 paged control at 7.703 ms
serial and 4.724 ms split-KV (16 query tokens, 530 selected rows, capacity
16). This is a kernel sanity result, not a claim that the new dense/packed
route wins end-to-end.

## Deferred verification

No model bundle or V6 live selected-all comparison was available for this
fresh implementation pass. Deferred measurements are the dense no-copy versus
packed compact route at Q=1/3/64/256, pack bytes/time per epoch versus token,
and one live 2–4K native-MTP request with output parity against the direct
oracle. The CUDA fixture above is the available hardware verification.

Next verification command when the V6 model fixture is available:
`cmake --build build-cuda --target llama-server -j2`, followed by the short
V6 selected-all/direct-oracle driver with its raw root recorded here.
