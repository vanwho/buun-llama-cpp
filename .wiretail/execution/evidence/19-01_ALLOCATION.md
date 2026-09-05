# 19-01 Allocation Receipt

Status: `pass_local_deferred_live`

The preallocation path is implemented and locally verified. Selective/exact
target caches receive a model-metadata plan before `create_memory()`, retain
full logical cell metadata, and allocate one bounded physical Turbo4 slab.
The pager/residency layer borrows that slab. Native MTP rows remain separately
reserved and are not target-pager victims.

Local checks:

- CUDA build targets `test-cache-budget`, `test-kv-pager`,
  `test-kv-residency`, and `test-kv-policy`: passed.
- All four focused executables: passed.
- `git diff --check`: passed.

Deferred: the Qwen3.8-27B, context-262144, native-MTP CUDA allocation-only
checkpoint was not run because the RTX 4080 had only 1128 MiB free. This is a
hardware-isolation requirement, not a claimed model result.
