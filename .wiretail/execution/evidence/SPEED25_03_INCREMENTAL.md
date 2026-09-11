# SPEED25_03_INCREMENTAL

Result: `pass` for the V6 short native-MTP runtime checks and focused tests.

The same 4096-context, 2048-token, question-0, zero-warmup V6 case was used
with CUDA Turbo4 target KV and native GPU Turbo4 MTP on the RTX 4080
(driver 595.91.07).

| Metric | 25-02 selective before | 25-03 selective after |
| --- | ---: | ---: |
| Prefill us | 14,064,808 | 5,625,370 |
| Decode us | 14,119,361 | 504,824 |
| Summary build calls | 274,080 | 3,536 |
| Summary source bytes | 578,858,000 | 7,468,028 |
| Host seal D2H calls | 616 | 40 |
| Inventory copies | 17,738 | 221 |
| Immutable store copies | 273,472 | 128 |

The after run repeated with the same counters in
`/srv/ai/paged-kv/results/25-03-short-selective-live-20260912T053500Z`.
The OFF control completed 128 tokens at 1794.451 prefill tok/s and 98.631
decode tok/s in `/srv/ai/paged-kv/results/25-03-short-off-live-20260912T054000Z`.

The selective receipts have different generated-token counts (68 before, 22
after), so this receipt makes no output-quality equivalence claim. The
incremental counter reduction and focused deterministic tests are the direct
evidence for this task's asymptotic change.

Focused verification passed:

- `cmake --build build --target test-kv-pager test-kv-routing-summary -j2`
- `build/bin/test-kv-pager`
- `build/bin/test-kv-routing-summary`
- `cmake --build build-cuda --target llama-server -j2`
- `git diff --check`

CUDA event spans and 262K/pressure/quality runs are deferred.
