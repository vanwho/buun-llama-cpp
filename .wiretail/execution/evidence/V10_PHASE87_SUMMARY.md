# Phase-87 compact summary

Revision: `hotpath-v10-20260914`; amendment: `repair86-20260921`.
Scope: phase-87 raw records and manifests only. Nulls remain explicit where a
measurement was not made or a claim was not established.

## Identity

The runtime candidate was
`/srv/ai/paged-kv/candidates/86-02-cuda/bin/llama-server`, SHA-256
`61fb0f5feb24fa74661e00fcb11a611cba593a618e1a6cd231c9227c96505643`, with
Qwen3.8-27B UD-IQ4_XS, SHA-256
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`, and
build receipt SHA-256
`402580238ef5008c49168770ca320f7e85a6dbc29481489fc6f4b72d33e53810`.
The controlled promotion fixture separately used the recorded CUDA test binary
(`7e0c5cea369a1008e7bfe107c029206902559b06074f2961d4f297e786cf11a0`) on an
RTX 4080, compute capability 8.9.

## Promotion origins

| Origin | Result | Evidence |
| --- | --- | --- |
| Controlled CUDA fixture (87-02) | Measured | Logical page 0 won by margin 297.858948, 1,081,344 useful H2D bytes, publication committed at epoch 6174, stable generation preserved, and completed target/reference parity. |
| Organic native Turbo4 (87-03) | Measured, with quality limitation | Cold candidate use was observed at selector rank 1 with 4,325,376 useful H2D bytes and target graph use. A-again returned `A-ARCHIVE-META`, not the expected `A-ARCHIVE-MARKER-914`. |
| Authenticated progress (87-01) | Progress only | C=27,008 of an L=65,536 request was observed; measured rows were 0 and no production cold-promotion success is claimed. |

The controlled and organic origins remain separate; the controlled fixture is
not relabeled as natural runtime promotion.

## MTP and attributed rates

The organic request-scoped MTP sum is 2 accepted of 12 drafted attempts
(16.6667%): A `0/2`, B `1/2`, A-again `1/6`, terminal `0/2`.
The controlled promotion fixture has `null` MTP counters because it does not
measure speculative drafting. Frontier occupancy exposes accepted target
tokens `0` with a speculative denominator of `0`; its acceptance rate is
therefore `null`, not zero.

The valid matched speed rows were fresh=58 tokens and cold=5469 tokens:

| Mode | Fresh prompt tok/s | Cold prompt tok/s | Fresh TTFT | Cold TTFT | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native GPU Turbo4 | 311.868 | 204.267 | 187.323 ms | 26783.172 ms | 33.234 |
| All-GPU MTP-off control | 822.182 | 1324.392 | 71.983 ms | 4140.827 ms | 45.367 |

Cached-append prompt tok/s is `null` in both matched rows because no valid
cached-append speed row was included. The native slowdown is retained as a
finding, not converted into a performance-success claim.

Organic attributed deltas include, for case B, 1,519,100 queue us, 63,540
wait us, 659 copy us, 4,325,370 useful H2D bytes, one direct and 30 reference
prefill routes, and target graph use. A-again added 101,470 queue us, 138,080
wait us, no H2D bytes, one direct and two reference prefill routes, and target
graph use. These are attribution counters, not universal rates.

## Capacity versus occupied frontier

| Quantity | Result |
| --- | ---: |
| Full-L allocation L | 262,144 tokens |
| Target allocated / physical pool | 69,206,016 / 69,206,016 bytes |
| H / A / B / U | 4096 / 2048 / 128 / 64 tokens |
| Page / pinned recent | 256 / 1024 tokens |
| Committed frontier sequence | 1200 → 5292 → 9384 → 13476 → 17568 → 21660 |
| Final committed C | 21,660 tokens |
| Final resident pages / host pages | 16 / 15 |
| Final target valid rows / host valid rows | 3996 / 3840 |
| Full capacity occupied | `false` |

The allocation is full-L, while the occupied frontier is only C=21,660. These
are intentionally separate claims; no full-context completion is asserted.

## Provenance and null policy

The structured source of truth is
`V10_PHASE87_SUMMARY.json`. Task receipts `V10_87-01.json` through
`V10_87-04.json` retain the raw paths, process identities, and hashes for each
phase-87 run. Missing fresh prompt, cached-append, controlled-MTP, completed
65536 frontier, and full-capacity occupancy values remain explicit `null` with
reasons in the JSON summary.
