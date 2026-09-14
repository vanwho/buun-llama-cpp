# V10 current benchmark summary

- Result: **current_findings**
- Revision: `hotpath-v10-20260914`
- Current geometry: L=8192, C=6144, H=4096, A=2048, B=128, U=128

## Source, placement, and proof boundaries

- Bundle manifest: `c6bbd6812ff5fbbef9a2a0c595b148a55797c64bcd6580773965a0bff624ce15`
- Model: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`
- Target and native draft are Turbo4; draft placement is GPU.
- Controlled promotion, organic promotion, and useful answer quality are separate claims.

## Original three-prompt comparison

| question | mode | samples | prefill/decode medians (tok/s) |
|---:|---|---:|---:|
| 0 | selective-v2 | 3 | 125.09874381436698 / 38.36009399421781 |
| 0 | cpu-main | 3 | 636.2347494975827 / 10.212424819000727 |
| 0 | all-gpu-v2 | 3 | 1542.6918114049217 / 36.989248554746254 |
| 1 | selective-v2 | 3 | 124.05919505491649 / 31.862695678646794 |
| 1 | cpu-main | 3 | 633.7031688562122 / 10.098448830923092 |
| 1 | all-gpu-v2 | 3 | 1547.3502005586479 / 37.08645344853313 |
| 2 | selective-v2 | 3 | 124.64723099036912 / 34.37206796861615 |
| 2 | cpu-main | 3 | 637.0141289816753 / 9.99225678163929 |
| 2 | all-gpu-v2 | 3 | 1548.8871281278055 / 37.09220310299462 |

## Capacity, failures, and next bottlenecks

- 32K and 128K states are retained as measured pilots; 256K startup and request failures are not full-occupancy proof.
- See `next_bottlenecks` in the JSON for the smallest reproducer and next experiment.

## Validation

```json
[]
```
