# Task 15-04 — dynamic MTP admission evidence

## Raw local startup/teardown

These are process-level starts and clean exits of the deterministic production
boundary tests after the implementation build:

```text
$ ./build/bin/test-cache-budget
cache-budget checks passed

$ ./build/bin/test-kv-pager

$ ./build/bin/test-moe-cache-fit
MoE cache fit tests passed

$ ctest --test-dir build --output-on-failure -R 'test-arg-parser|test-kv-pager|test-moe-cache-fit|test-cache-budget'
100% tests passed, 0 tests failed out of 4
```

The production build also completed for `llama`, `server-context`, and
`llama-server`. The build emitted only existing signedness/deprecation warnings.

## Deterministic admission ladder

`test-cache-budget` exercises resolved contexts 256, 16,384, 32,768, 65,536,
65,537, 131,072, and 262,144. It verifies all-pages-fit, the 65,537 odd tail
(257 pages), an intermediate five-page hotset, exact context-sized Turbo4 MTP
bytes, and refusal when capacity is below the complete MTP reservation.

## Deferred verification

No compatible CUDA Qwen3.8 model-backed runtime is present in this environment,
so native MTP GPU residency, the live pager startup ledger, and GPU teardown
remain deferred. Later verification must launch the configured Qwen3.8 GGUF with
Turbo4 K/V, native MTP enabled, the resolved maximum target context, and pager
telemetry enabled; it must record the `KV pager startup` ledger (including
`mtp_rows`, `mtp_bytes`, `admitted_pages`, and refusal), then stop the server and
verify clean target/MTP context teardown. No hardware success is inferred from
the deterministic tests above.
