# V10 phase-55 benchmark summary

- Result: **current_findings**
- Revision: `hotpath-v10-20260914`
- Matched proof geometry: L=8192, C=6144, H=4096, A=2048, B=128, U=128

## Identity and placement

- Candidate: `build-cuda/bin/llama-server` (`24333b2639ca140e53e1ddd53f60b2d2b954bee8f0d723351db0802b723032ce`)
- Model: `/srv/ai/models/text/current.gguf`; hash status `measured`.
- Target KV: Turbo4/CUDA. Full-L draft KV: Turbo4/GPU, 262144 rows.

## Measured rates

| workload | status | samples | median tok/s | range |
|---|---|---:|---:|---:|
| cold_prefill | measured | 9 | 186.03546464552537 | 185.35708587044113–186.61948044942247 |
| cached_append | measured | 2 | 456.823361276136 | 311.1917184103938–602.4550041418781 |
| committed_decode | measured | 9 | 13.103881635913153 | 13.03070604437444–13.150308436374903 |

## Boundaries and incomplete claims

- The 256K result proves allocation/startup plus occupied C1207; it does not prove occupied C262144.
- 32K and 128K pilots are explicitly not run in this phase-55-only summary; phase-53 rates are not merged.
- Selective rows are measured, but request-scoped native MTP deltas are not run because the required counters were absent.
- CPU-main-KV/GPU-draft and all-GPU/GPU-draft controls, physical promotion edges, and useful answer quality remain explicit not-run findings.

## Next bottlenecks

- **metrics exporter / benchmark observation** — `55-02 matched matrix with /metrics before and after each request`: prevents native MTP acceptance and CPU/all-GPU comparison.
- **run-final-curve.py profile identity** — `55-02 full-L and matrix campaigns`: campaign exports omit loaded executable, DSO, model, and bundle identity.
- **phase-56 benchmark review** — `V10_SUMMARY_55.json boundaries`: must schedule promotion/quality follow-up without treating allocation as C262144 occupancy.
