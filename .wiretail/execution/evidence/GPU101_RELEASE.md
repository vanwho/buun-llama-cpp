# GPU101 release decision — review 101-12h

Decision: **goal_miss**. Measured candidate source `a0086f056f20efa519100718f010bc15c2efb830`; binary SHA256 `300dc8cf420abbd0b9121ffafd9ed90e4caf299a797280186ac84426a10ade54`. This is an independent review of the unchanged 101-18 campaign. The original 101-18 release bytes are preserved under `provenance/101-12h/` and remain referenced by its completed receipt.

## Candidate and protocol

- Model SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`; L8192/H4096/P256/B1024/U256; target and draft Turbo4; native GPU MTP nmax2; frozen prefix SHA256 `8218b0f427cc931fa38d08f1c29f91ec7d82e12ac309ff7743eb7a60da678d20`.
- Three matched routes, three measured rows per prompt, one 40-token warmup, 400-token selected outputs, temperature 0, reasoning off, fresh slot per row. Every canonical row succeeded (HTTP 200) with zero cached input tokens. Model-backed query replay and natural old-file recall also passed.
- The raw-row recalculation is `.wiretail/execution/evidence/raw/101-12h/recalculation.json`.

## Independent selected medians and matched controls

| Prompt | Selected prefill tok/s | Selected decode tok/s | Selected MTP median | CPU prefill tok/s | CPU decode tok/s | Pager-off prefill tok/s | Pager-off decode tok/s | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| prompt_1 | 1452.08 | 60.05 | 48.56% | 660.72 | 25.50 | 1556.16 | 84.24 | MTP miss |
| prompt_2 | 1463.84 | 58.45 | 47.58% | 664.31 | 18.54 | 1556.18 | 62.89 | MTP pass |
| prompt_3 | 1451.19 | 65.41 | 59.79% | 660.74 | 21.09 | 1557.64 | 71.25 | MTP miss |

Selected prefill clears both the 500 tok/s minimum and the 750 tok/s preferred goal on all prompts. MTP floors remain 75% / 40% / 60%; prompt 1 misses by 26.44 percentage points and prompt 3 by 0.21 points. Selected decode beats ordinary CPU-main-KV decode by 2.36x / 3.15x / 3.10x. Selected prefill is 2.20x / 2.20x / 2.20x CPU, below the 3x and 5x comparisons; it is 0.93x / 0.94x / 0.93x pager-off GPU. Selected decode is 0.71x / 0.93x / 0.92x pager-off GPU.

## Functionality and remaining owner

Natural old-file recall passed: Python03 logical page 5, generation 7, content version 256 was cold and host-backed, naturally nominated at rank 4/5, transferred (34,603,008 useful/aligned bytes), mapped, then consumed by target and draft. Frozen-history Turbo4 MTP accepted 75 of 128 drafted tokens; query replay passed.

The canonical rows do not contain first-disagreement logits/state attribution. Schedule `101-12i` to capture the existing 101-09 first-disagreement evidence and repair only a proven state/mask/carry defect or measured selected-history distribution mismatch. Schedule `101-12j` for the exact changed-candidate canonical retake and final release review. `102-01` now depends on `101-12j`.

CPU summary optimization is not scheduled: the available 2112-build / 71,368,704-byte cumulative counters do not establish isolated summary time or a >=10% share of prefill. Sparse page-inventory counters are retained as diagnostics, but their counts disagree with adjacent counters and are not used for this decision. Driver CUDA event counts were not part of the canonical timing receipt; replay logs contain separate graph reuse evidence.

## Raw evidence and preserved history

Canonical raw roots: `/srv/ai/paged-kv/results/forward/101-18/attempt-01/`. The independent row script and results are under `.wiretail/execution/evidence/raw/101-12h/`. The immutable pre-review report copies are `.wiretail/execution/evidence/provenance/101-12h/GPU101_RELEASE.pre-review.json` and `.md`.

Review owner: **101-12h**. Ordered remediation successors: **101-12i**, then exact retake and review **101-12j**. Scaling remains gated until that chain passes.
