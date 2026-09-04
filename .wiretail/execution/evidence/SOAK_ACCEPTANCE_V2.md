# Final endurance, churn, concurrency, and teardown acceptance v2 — task 14-05

## Decision

**BLOCKED.** The required pager soak could not be entered because task 14-04
produced no accepted selective+native-MTP run. The ordinary restored service
did complete a bounded control soak: 12/12 requests returned HTTP 200, a large
generation was terminated at the timeout boundary, and the service plus its
8091 proxy recovered after restart. These controls do not establish pager
correctness, long-run resource boundedness, concurrency isolation, or
max-context behavior.

Machine-readable evidence is
[`SOAK_ACCEPTANCE_V2.json`](SOAK_ACCEPTANCE_V2.json). Raw output is outside
the repository at
`/srv/ai/paged-kv/results/14-05-soak-20260904T041000Z/`.

## Provenance and starting boundary

- Repository evaluation SHA: `8ae95b618581638c6d73d33168dfdade89fd6008`
- Release config hash: `2497f112026c71f904ac115cded4dd290a42bfd52a18ba202dc7337e5f88276b`
- Binary SHA256: `4cb1a90011d2d5b486b3f83b2ce3c9863eb65b55119b290fa52c8e70c1854626`
- Model SHA256: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`
- Corpus: `pager-corpus-v2`, hash `e8cd002b2a6215a5003db7b8c204602bba1879a5bdf91a2a9ef9b0a2e3ff0e35`
- GPU: NVIDIA GeForce RTX 4080, 16,376 MiB
- Starting/restored profile: `qwen38-big`

The live target was the ordinary managed `llama-server.service` boundary:
single slot (`-np 1`), context 131,072, Turbo4 K/V, `--fit off`, and no pager
flag. This was recorded as a control service because the accepted pager
boundary from 14-04 was unavailable.

## Verification results

The focused CUDA test selection ran 20 registered tests: 19 passed. The one
failure, `test-state-restore-fragmented`, was a configuration incompatibility
in its default tiny-model Turbo4 setup (block size 128 does not divide
`n_embd_head_k=48`), followed by device-fit abort. The explicit recovery
using `--cache-type f16 --fit off` passed, including fragmented sequence state
save, clear, restore, and decode. The q8_0 alternative was also rejected by
the model’s head dimension (`block size 32`), so neither failure is hidden.

The live control results were:

| Check | Result |
| --- | --- |
| Ordinary requests | 12/12 HTTP 200; payloads retained, not scored as pager correctness |
| Cancellation | `timeout_exit=124` during a bounded large generation |
| Resource sampling | 30 one-second samples; RSS 1,007,700–4,649,204 KiB; GPU used 15,481–15,523 MiB |
| Restart | 8080 healthy after restart; 8091 healthy with upstream 200 and DB path |
| Pager metrics | Not present on ordinary control service |
| Port 8092 | Untouched |

## Acceptance coverage

The following are intentionally marked blocked or not run rather than inferred
from the control soak: prolonged max-context pager conversation, focus shifts,
cache/checkpoint/clear/slot churn, pager cancellation boundaries, repeated
context creation, multi-slot isolation, pager telemetry, and long-run
monotonicity. Local deterministic tests provide supporting lifecycle coverage,
but do not replace the required model-backed pager run.

The 3x speed floor and phase-14 soak gates have no result. No correctness,
no-leak, no-corruption, no-deadlock, or concurrency pass is claimed for the
pager. The prerequisite failure is recorded in
[`PERFORMANCE_ACCEPTANCE_V2.json`](PERFORMANCE_ACCEPTANCE_V2.json).

## Restore and deferred verification

`qwen38-big` was restored. `llama-server.service` and
`ai-long-memory.service` are active; 8080 and 8091 return healthy responses.
The independent CPU service on 8092 was not touched. Final state and health
captures are in the raw root.

Deferred verification: **none due to unavailable hardware or human action**.
The required hardware and local harness were available. Pager endurance is
blocked by the observed native-MTP allocation and speculative-rollback
failures; a future acceptance must repair those boundaries and rerun this
contract with the pinned provenance.
