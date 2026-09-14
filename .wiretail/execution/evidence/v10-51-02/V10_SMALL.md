# V10 small live proof

- Matrix proof: PASS (three prompts × three modes × three measured trials).
- Paging/speed validation: PASS.
- Exact prompt occupancy: 6144 tokens in an 8192-token context; H=4096 and attended rows are reported from runtime telemetry.
- Append probes: 64 and 256 requested tokens, both with an exact 6144-token prefix; cache_n is reported without a hit-rate claim.

## Median rates by question

| question | mode | prefill tok/s | decode tok/s | MTP accepted/proposed | acceptance |
|---:|---|---:|---:|---:|---:|
| 0 | selective-v2 | 125.10 | 38.36 | 82/90 | 91.11% |
| 0 | cpu-main | 636.23 | 10.21 | 0/251 | 0.00% |
| 0 | all-gpu-v2 | 1542.69 | 36.99 | 0/251 | 0.00% |
| 1 | selective-v2 | 124.06 | 31.86 | 74/105 | 70.48% |
| 1 | cpu-main | 633.70 | 10.10 | 0/251 | 0.00% |
| 1 | all-gpu-v2 | 1547.35 | 37.09 | 0/251 | 0.00% |
| 2 | selective-v2 | 124.65 | 34.37 | 77/98 | 78.57% |
| 2 | cpu-main | 637.01 | 9.99 | 0/251 | 0.00% |
| 2 | all-gpu-v2 | 1548.89 | 37.09 | 0/251 | 0.00% |

## Validation

```json
[]
```
