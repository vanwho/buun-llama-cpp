# 18-05 Turbo4 parity evidence

The generic selected-reference graph now restores Turbo4 K rows from the
stored forward-FWHT domain before ordinary MHA. K/V representation domains are
retained in operator metadata and graph content keys. Direct pager byte offsets
are resolved by the cache’s compact attention-layer ordinal mapped from the
model layer ID, covering hybrid/non-contiguous layer IDs.

## Local verification

- `make -C build/tests test-kv-pager-model test-kv-attention-view -j2`: passed.
- `make -C build/tests test-kv-attention-execution test-kv-attention-exact -j2`: passed.
- `cmake -S . -B build-cuda-test` followed by
  `make -C build-cuda-test/tests test-kv-pager-model -j2`: passed with CUDA.
- `LD_LIBRARY_PATH=build-cuda-test/bin build-cuda-test/bin/test-kv-pager-model`:
  synthetic F1/F2/F3 driver passed (exit 0).
- `ctest --test-dir build -R 'test-kv-(attention-view|attention-execution|attention-exact|pager|routing-retrieval)$' --output-on-failure`: 5/5 passed.
- `LD_LIBRARY_PATH=build-cuda-test/bin build-cuda-test/bin/test-cuda-fattn-paged-turbo4`: passed (exit 0; 529 selected rows).
- `git diff --check`: passed.

The driver’s fixed 300-token synthetic case uses page order 1,0, crosses
position 256, and has a 44-row tail. F1 reports wrong stored-domain dot
product `-6.32243`, corrected dot product `76.1587`, and inverse-WHT
restoration max error `2.98e-7`. F3’s dense/selected output max error is
`1.53e-5`, attributable to float accumulation order.

## Deferred verification

The named live target GGUF is available, but no safe model-backed run was
started because only 1128 MiB of RTX 4080 VRAM was free and the existing live
workload owns the device. This task does not claim dense-vs-selected live
logit parity, production direct write/read alias parity, or calibrated
non-identity InnerQ scale parity. Later setup: stop or isolate the existing
GPU workload through the site lifecycle owner, run the driver once per context
with MTP disabled and identical saved token IDs, then retain dense/reference/
direct JSON with route, layer, page, positions, domain/shape, max-abs/RMS, and
first-divergence fields. Production direct slab write binding remains the
19-02/I7 boundary.

## Next action

Proceed to 18-06; preserve the representation-domain and compact-layer
mapping contracts when testing native MTP and rollback.
