# V10 task 85-05 focused revalidation raw result

Revision: `hotpath-v10-20260914`  
Source commit: `d0472faa0e837c00a7e10983dcb32c2e4e146ba5`  
Candidate bundle: `/srv/ai/paged-kv/results/v10/85-05/candidate-bundle-wdJFg7`  
Candidate executable SHA-256: `83611b093c8031baaa3faefab89e75c93ae38765ab970cc5073e7af0b2063f89`  
Build receipt SHA-256: `a9bb445c21e5c21de730f7f0b8dc899dc7391091abe5bd3bab6701df449e001c`

## Fixed coordinate and identity

- Model/profile: `qwen38-fast-turbo4-mtp`, `qwen38-fast`; endpoint was the canonical `/v1/chat/completions` URL.
- Logical context `L=8192`, occupied/request prompt `C=6143`, target hot capacity `H=4096` (`16 * 256` token pages), attention allocation `A=4096` (`physical_write_capacity=4096`), `B=128`, `U=128`, page size 256.
- Target K/V: Turbo4/Turbo4. Native draft K/V: Turbo4/Turbo4, 8192 rows, 8,781,824 bytes, GPU/CUDA0. The CPU-main-KV control moved only the target KV placement to CPU; its native draft remained GPU/CUDA0.
- Every successful request used the immutable candidate bundle. The successful production service was left loaded with that bundle after the final cache pair.

## Focused rows

| Row | Status | Request-scoped result | First-divergence/state classification |
| --- | --- | --- | --- |
| Feature off | measured | `6143` prompt tokens, `256` completion tokens, `6399` total; no MTP denominator by design | Not applicable: feature off; pager counters are not applicable in this control |
| All-GPU native MTP | measured | draft delta `507`, accepted delta `0`, positive denominator `507`, acceptance `0%`; `6143/256` request completed | Native MTP state diagnostics reported restored target/draft state for the completed dense verification sequence; no selected pager path was enabled |
| CPU-main-KV native-MTP control | measured | draft delta `507`, accepted delta `0`, positive denominator `507`, acceptance `0%`; `6143/256` request completed | Same completed dense native-MTP control; target KV was CPU, draft KV remained GPU |
| Automatic selected native MTP | failed | Request reached exact `6143` processed prompt tokens and emitted `4` tokens, then stalled; canonical runner was interrupted and restored the profile | First observed selected path was `selected direct`, followed by `selected reference`; at the first verification diagnostic around position 6143, `target_state_restored_before_verification=false` and `draft_state_restored_before_verification=false`. The final graph state had 52 submissions versus 51 completions |
| Matched cached append/decode sample | measured | Cold seed: `6143` prompt / `64` output. Matched live continuation: `6139` cached rows of `6143`, `64` output; both canonical final-curve cases passed | Feature-off cache control; target GPU Turbo4/Turbo4, draft not present |

The two native rows have valid HTTP/request records and positive request-scoped
counter deltas. The adapter additionally reported its known large-mode
`missing_native_prompt_measurement` harness error (and a restore-identity
comparison error); these are retained as harness findings, not hidden as
successful adapter validation. No throughput ratio is claimed.

## Selected-row telemetry snapshot

The terminal selected-row snapshot was captured from
`selected-native/raw/progress-off-measured-1.jsonl` after the request reached
the fixed coordinate:

- Identity: target backend CUDA, target K/V Turbo4/Turbo4; native MTP backend GPU, draft K/V Turbo4/Turbo4.
- Automatic route counters: prefill dense/reference/direct/packed `0/49/1/0`; decode `0/0/0/0`; MTP verify `0/0/2/0`. No packed route was observed.
- GPU graph/cadence: capture `51`, replay `1`, rebuild `51`, submit `52`, complete `51`; waits `52`, wait time `86,214 us`; table epoch `12,482`, table epoch changes `51`.
- Catalogue/publication: natural-proof catalogue epoch `8421`, published epoch `8551`; table publication was observed. Summary builds `1472` / `198,967,296` bytes.
- Seal/host maintenance: seal calls `46`, pages scanned `47`, pages changed `23`; host-seal D2H calls `2208`, host-seal D2H bytes `99,483,648`.
- Transfers: H2D useful/aligned bytes `233,570,304/233,570,304`; D2H useful/aligned bytes `0/0`; transfer waits `0`.
- Fault/eviction deltas: faults `54`, evictions `54`; rejection histogram included identity mismatch `232`, missing host source `15`, admission rejected `44`, and transfer rejected `15`.
- GPU utilization samples: `not_run`; the canonical runner did not include a concurrent utilization sampler, so no utilization value is invented.

All off/native control rows report pager route, transfer, graph, seal,
catalogue, summary, wait, fault, eviction, and utilization fields as
`not_applicable` because paging was disabled, or `not_observed` where the
selected request did not complete. The selected snapshot above records every
counter that was available before the stall. No 256K occupancy, six-point
scaling, quality claim, or ratio was run.

## Raw artifacts

The complete canonical outputs are retained outside the repository under
`/srv/ai/paged-kv/results/v10/85-05/`:

- `feature-off-v2/`
- `all-gpu-native/`
- `cpu-main-kv-native/`
- `selected-native/`
- `cached-prefill-production-off/`
- `cached-continuation-production-off-v2/`

The selected failed row was restored to `qwen38-fast` after interruption. The
matched cache pair was run with the same immutable bundle, exact prompt
geometry, B/U128, and context 8192.
