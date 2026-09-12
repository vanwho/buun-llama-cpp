# SPEED25-14 budget tuning

## Result

The existing admission contract passes the task’s budget coverage without a
production-code change. Native MTP is charged from resolved row geometry and
reserved before hot pages; H is admitted only from the remainder. The policy
keeps L=262144 as the logical target and does not treat a reduced context as a
successful fallback.

## Ledger and tests

The retained live ledger is from the matching Qwen/Turbo4/native-GPU-MTP
runtime bundle at L=4096, B=256. It records weights, fixed and recurrent state,
MTP compute, graph/compute scratch, routing, staging, allocator guard,
headroom, target storage, and exact MTP rows. Pager storage is charged once;
the fixed category excludes the storage tensor when the pager plan owns it.
This is a measured reference ledger, not a fabricated 262K extrapolation.

`tests/test-cache-budget.cpp` passes contexts 256 through 262144, including the
65537 partial-page case. It verifies actual Turbo4 row sizing, full-L MTP-first
admission, refusal below the MTP floor, zero/overflow budgets, headroom, all
late-startup categories, and a nonzero H remainder.

## Measured coordinate points

The retained native-MTP live pair used A=2304 rows, H=4096 rows, and
`mtp_n_max=2`:

| B | pp/s | committed tg/s | MTP accepted/proposed |
|---:|---:|---:|---:|
| 256 | 463.00 | 43.56 | 18/25 |
| 512 | 453.91 | 46.20 | 19/23 |

B=256 is retained for the prefill objective because it has the higher measured
prompt throughput and lower TTFT. B=512 remains the decode-oriented alternative.
No unmeasured H/A/n-max point is promoted to a production setting; H remains
automatic after fixed reservations and A remains independently derived.

## Deferred verification

A fresh live 262144 allocation and <=512 native-MTP smoke are deferred because
the available RTX 4080 is occupied by the resident `--spec-type none` server.
The old PHASE23 full-256K record is explicitly `not_run` and is not used as
evidence. A later run should start from the current source and matching native
MTP bundle, reserve full-L MTP first, then verify nonzero H and short
generation. The allocation-only result is intentionally separate from the
occupied-cache proof assigned to 26-01.

Raw roots and bundle identity are recorded in the adjacent JSON receipt.
