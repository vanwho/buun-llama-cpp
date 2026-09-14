# V10 phase-57 benchmark summary

- Result: **current_findings**
- Revision: `hotpath-v10-20260914`
- Matched geometry: L=8192, C=6144, H=8192, A=4096, B=128, U=64

## Identity and placement

- Requested model/mode: `qwen38-fast-turbo4-mtp` / `selective`.
- Observed model hash: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`; bundle manifest: `83d59a800698b1e3a5828de30ffb7b688b90ac94313e4118a40920e6d47616ba`.
- Target and full-L draft placement are Turbo4 on CUDA/GPU as recorded in JSON.

## Rates and request-scoped MTP

| workload | status | samples | median tok/s | range |
|---|---|---:|---:|---:|
| cold_prefill | measured | 3 | 180.8879673730395 | 180.70832191528703–180.9869497627772 |
| cached_append | not_run | 0 | None | None–None |
| committed_decode | measured | 3 | 25.949824392672742 | 24.98147857564977–28.17940772168608 |

- `q0-measured-1`: draft=29, accepted=16, acceptance=55.172413793103445%.

- `q1-measured-1`: draft=31, accepted=15, acceptance=48.38709677419355%.

- `q2-measured-1`: draft=32, accepted=14, acceptance=43.75%.

## Promotion, quality, and boundaries

- Controlled and organic physical promotion are explicit not-run findings; useful answer quality is also not run.
- The full-L allocation boundary was measured, but the first request failed before advancing occupancy; this is not a C262144 occupancy proof.
- 32K and 128K pilots are not run and no historical phase-55 rate is merged.

See the JSON for every raw row, reason, denominator, and checksum.
