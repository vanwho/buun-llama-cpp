# Phase 23 checkpoint repair

Task 23-02 is locally complete with the fresh production probe deferred.

The compact KV state path now has an explicit pager format. Save resolves each
logical cell to its page identity and emits either canonical host bytes or the
checked resident physical row. The compact stream is page-major and includes
row ordinals, positions, row shapes, full `llama_kv_page_id` identity, and K/V
payloads. Restore rehydrates physical rows through pager write tickets, retains
the serialized identity, completes at the segment fence, and seals before
admitting another page. Invalid shapes, geometry, ordinals, offsets, capacity,
and unsupported transposed compact V fail explicitly.

The pager residency table accepts the intentional serialized-identity update
using the uniquely owned physical slot. The deterministic fake test exercises
noncontiguous positions, a partial tail, identity retention, cancellation, and
hot-set exhaustion. The original assertion journal remains the negative control.

## Verification

- `cmake --build build-cuda --target llama -- -j2` — passed.
- `cmake --build build-cuda --target test-kv-pager test-kv-residency test-state-restore-fragmented -- -j2` — passed.
- `ctest --test-dir build-cuda -R '^(test-kv-pager|test-kv-residency)$' --output-on-failure` — 2/2 passed.
- `git diff --check` — passed.

## Deferred verification

The three equal-work 6,401-token cycles, fresh compact save/restore with
selected-all logits/state comparison, MTP/recurrent continuation, and live
insufficient-buffer assertion-path check were not run. The existing successful
262,144-token CUDA candidate was already resident on the RTX 4080 and was
preserved; port 8092 was not touched. These checks require an authorized
benchmark transition to the new binary and are not claimed here.
