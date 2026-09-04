# Final selective speed acceptance v2 — task 14-04

## Decision

**BLOCKED.** The required final speed run did not produce an accepted
model-maximum selective segment. Native GPU MTP failed closed before service
readiness at both 262,144 and 131,072 contexts. At the reduced 32,768 context,
native MTP reserved successfully but the first selective request failed while
rolling back speculative tokens. No speed ratio, 3x floor, 5x target, or 70%
all-GPU claim is made.

Machine-readable evidence is
[`PERFORMANCE_ACCEPTANCE_V2.json`](PERFORMANCE_ACCEPTANCE_V2.json). Raw output
is outside the repository at
`/srv/ai/paged-kv/results/14-04-speed-20260904T020000Z/`.

## Provenance

- Repository evaluation SHA: `8ae95b618581638c6d73d33168dfdade89fd6008`
- Release config hash: `2497f112026c71f904ac115cded4dd290a42bfd52a18ba202dc7337e5f88276b`
- Binary SHA256: `4cb1a90011d2d5b486b3f83b2ce3c9863eb65b55119b290fa52c8e70c1854626`
- Model SHA256: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`
- Corpus: `pager-corpus-v2`, hash `e8cd002b2a6215a5003db7b8c204602bba1879a5bdf91a2a9ef9b0a2e3ff0e35`
- GPU: NVIDIA GeForce RTX 4080, 16,376 MiB
- Sampling: temperature 0, seed 42

## Ladder and raw evidence

| Context | Mode | MTP | Result | Raw evidence |
| ---: | --- | --- | --- | --- |
| 262,144 | selective | native GPU | refused before listen: full native MTP context reservation | `server-probe2.log` |
| 131,072 | selective | native GPU | refused before listen: full native MTP context reservation | `server-ladder-131072.log` |
| 32,768 | selective | native GPU | startup passed; first request HTTP 500 during speculative rollback | `server-mtp-32768.log` |
| 32,768 | standalone control | disabled | not an accepted pager run; no pager telemetry | `server-standalone-32768.log` |

The successful 32,768 MTP startup recorded 32,768 Turbo4 rows, 34,734,080
bytes, CUDA0, and the `mtp_gpu_reserved` category. The independent selective
startup probe recorded 128 logical/admitted pages, 32,768 physical rows, a
553,648,128-byte target allocation, and a 40,374,656-byte runtime ledger
charge. These are startup observations, not throughput results.

## Trials, gates, and ratios

The final contract requires two warmups and at least ten measured trials for
warm focus, plus cold needles, focus shifts, churn, feature-off, observe, and
exact segments. Measured trials completed: **0 accepted trials**. Because the
required selective+MTP service/request boundary was not reached, the CPU-KV
denominator and fresh comparable all-GPU numerator were not paired. Ratios are
therefore `null` in the JSON artifact; historical controls were not substituted.

The 3x warm gate, 5x report target, 70% safe-all-GPU comparison, no-fault p95
tail gate, and observe-overhead gate are all **blocked without a result**.
Fault/churn segments were not averaged into any steady metric.

## Diagnostic paths

Three distinct runtime paths were tested and preserved:

1. Model-maximum selective context with native GPU MTP: startup reservation
   failed before listen.
2. The largest prior safe-ladder context with native GPU MTP: the same startup
   reservation failure reproduced.
3. A reduced selective context with native GPU MTP: startup succeeded, but the
   first request failed during speculative suffix rollback.

These are prerequisite/runtime diagnostics, not optimization wins. The speed
gate was never entered, so no repair or quality-regression claim is fabricated.

## Restore and deferred verification

The starting `qwen38-big` profile was restored. `llama-server.service` and
`ai-long-memory.service` are active; 8080 and 8091 return healthy responses.
The independent CPU service on 8092 was not touched. Restore evidence is
`teardown-restore.txt` in the raw root.

Nothing is deferred for lack of hardware or human setup. The final acceptance
is blocked by observed native-MTP allocation and speculative-rollback runtime
failures; a future acceptance must repair those boundaries and rerun the full
ten-trial contract with the same hashes.
