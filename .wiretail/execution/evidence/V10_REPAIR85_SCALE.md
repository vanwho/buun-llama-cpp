# Repair85 scale findings

Task `85-17` measured startup allocation and bounded request behavior for the
immutable 85-15 CUDA candidate (`e020c2d4…`) with target and native MTP draft
on CUDA, Turbo4 K/V, page size 256, and `B=128`.

## Result

The full-L allocation boundary is viable at both `L=131072,H=8192` and
`L=262144,H=4096`: both services became healthy and authenticated `/metrics`
reported the expected target and MTP rows. This is allocation evidence only.

The first 1,200-token request at `L=32768,H=16384` failed with
`CUDA error: invalid argument` in `ggml_cuda_turbo_prefill_attend` at
`ggml/src/ggml-cuda/fattn.cu:3753`. The required recovery ladder reduced
`U=128` to `U=64`; it reproduced the same fault. A single resumable,
prefix-preserving frontier was then attempted at `L=262144,H=4096,U=64` and
stopped at `C=0` when the first response became invalid after the same fault.

Therefore this run establishes no successful occupied context beyond the
previously verified `C=6144` native-MTP profile, no warm decode speed, and no
full 256K occupancy claim. It does establish that shrinking hot capacity and
microbatch preserves startup allocation headroom but does not repair the
request-time prefill kernel failure. No repeated 200K prompt generation was
performed.

## Ledger highlights

| profile | target allocation | MTP draft | fixed recurrent state | reserved | headroom | request result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `L32768/H16384/U128` | 276,824,000 B | 34,734,080 B | unavailable before abort | 1,687,100,000 B | 201,327,000 B | CUDA invalid argument |
| `L32768/H16384/U64` | 276,824,000 B | 34,734,100 B | 470,680,000 B | 1,688,200,000 B | 201,327,000 B | same CUDA invalid argument |
| `L131072/H8192/U64` | 138,412,000 B | 138,543,104 B | 470,680,000 B | 1,762,150,000 B | 201,327,000 B | startup measured; request withheld |
| `L262144/H4096/U64` | 69,206,000 B | 276,955,136 B | 470,680,000 B | 1,990,850,000 B | 201,327,000 B | startup measured; frontier `C=0` |

The detailed ledger includes weights, recurrent state, MTP compute/graph,
Turbo4 scratch, routing/catalogue, packed workspace, page pool, and attention
window fields in `V10_REPAIR85_SCALE.json`. Raw authenticated metrics, command
lines, service journals, request payloads, and frontier state are under
`raw/85-17/`.

## Deferred verification

Full-L occupied 262K, cold promotion, warm decode throughput, batching
trade-off, and the 128K request path remain deferred because the immutable
candidate aborts in the CUDA prefill kernel before a request commits. The
final service was restored to the previously verified healthy `L=6144,
H=4096,U=128` native-MTP profile. A future repair/build must resolve the
`fattn.cu:3753` invalid-argument fault before these live measurements can be
repeated honestly.
