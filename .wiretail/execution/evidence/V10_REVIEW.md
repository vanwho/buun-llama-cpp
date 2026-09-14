# V10 benchmark review

Revision: `hotpath-v10-20260914`; reviewed summary: `V10_SUMMARY.json`.

## Verdict

`goal_met: false`. Identity and the required Turbo4/GPU placements are valid.
Controlled, organic, and file-roundtrip physical promotion all reached
completed H2D, mapping publication, and completed target-graph use. That is
separate from useful answer quality, which remains unproven/false.

The full 256K objective is unmet: all three first-request attempts at
`L262144` stopped before `C` advanced. The two concrete owners are
`ggml_cuda_fattn::kv_dequant_scratch` (VBR f16 scratch reservation missed a
512 MiB growth at `H30208/B128/U128`) and packed selected attention/
`launch_mul_mat_q` (buffer/shared-memory allocation and cleanup failure at
`H16384/B128/U64`). These are fixable implementation boundaries, so phase 53
contains two ordered repairs before remeasurement.

The matched original-three-prompt matrix contains 27 rows: three samples for
each of nine mode/question cells. Selective prefill medians are 124.059–125.099
tok/s, versus 633.703–637.014 CPU-main and 1542.692–1548.887 all-GPU. Selective
decode is 31.863–38.360 tok/s, versus 9.992–10.212 CPU-main and
36.989–37.092 all-GPU. Decode is competitive with all-GPU on two questions and
faster than CPU-main, but selective prefill is only roughly 8% of all-GPU and
20% of CPU-main; no practical end-to-end speed goal is claimed.

The measured 32K/128K pilots remain distinct from 256K allocation and full
`C262144` occupancy. Phase 53 therefore runs only the two repairs, a bounded
physical-chain/256K retry and matched matrix, and a compact summary. Phase 54
will repeat this algorithm from that summary only.

## Scheduled successors

`53-01 → 53-02 → 53-03 → 53-04 → 54-01`.

