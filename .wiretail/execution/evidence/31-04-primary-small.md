# 31-04 primary-small evidence

Status: partial runtime-fault evidence. The repaired L8192/H4096 selected path
was measured with native full-L GPU Turbo4 MTP. Two matched no-warmup cold
single-request receipts completed; the requested warm-control comparison did
not complete, q0 repetition and q2 faulted, and natural cold recall faulted.
The result must not be read as a stable q0/q1/q2 or recall proof.

## Coordinate

L=8192, C=6144, H=16 pages/4096 tokens, A=256 attended rows, B=128,
U=64, page=256, CUDA query tile=64, selected/direct selective route,
runtime prefill policy, temperature 0, seed 42, thinking off. Target K/V and
native draft-MTP K/V were Turbo4; target was CUDA and MTP was GPU-resident,
with 8192 MTP rows (8,781,820 bytes).

## Matched completed receipts

| Question | Result | Prefill tok/s | Decode tok/s | TTFT | MTP proposed/accepted | H2D useful | Faults/evictions |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q0 | pass, 128 output | 205.203 | 36.466 | 29.949 s | 134/59 | 4,325,380 B | 1/1 |
| q1 | pass, 128 output | 204.932 | 36.382 | 29.990 s | 134/59 | 4,325,380 B | 1/1 |

These are matched to each other, but are no-warmup recovery measurements;
they are not a warm-vs-cold comparison. Full receipts and raw SSE are listed
in `31-04-primary-small.json`.

The completed q0/q1 snapshots agree on target allocation 69,206,000 B,
16 resident pages, 3,840 host-valid rows/64,880,600 B, 96 submitted and
event-completed transfers, 4,325,380 useful/aligned H2D bytes, 70 graph
rebuilds/captures, 95 graph replays, 164 table epoch changes, 70 table
rebuilds, 1,536 summary builds (207,618,000 B), zero summary reads, 2,304
host-seal D2H calls (103,809,000 B), 61 attention samples over 404 pages,
11,993,100 attention D2H bytes, and 1,328--1,352 us attention publication.
The successful HTTP receipts still report one pager fault and one eviction;
that is retained as a critical negative rather than normalized away.

## Natural retained continuation and recall

The authenticated incremental ladder completed six live-continuation turns to
occupied C=5000. Turns ended at 800, 1800, 2800, and 3800 without H2D; the
4800-token turn crossed H, recorded one fault/eviction, 4,325,380 useful H2D
bytes and 96 transfer completions, and ended with 16 resident/3,840 host-valid
rows. The final 5000-token turn completed with cached 4800 rows. The harness
receipt marks the continuation measurement valid but `natural_joint_proof` is
false.

The follow-up authenticated cold-topic recall reached only the text
`The retrieval topic`, then ended as `runtime_fault` with no validated answer;
the raw SSE, record, and summary are retained in the external result path.

## Negative runs

The warm-control run completed its warmup but its q0 measurement faulted after
prefill. A no-warmup q0x3 recovery attempt faulted on trial 1 and left trials 2
and 3 incomplete; q2 faulted after prefill. Each failure showed the same
`CUDA illegal memory access` from `ggml_backend_cuda_synchronize` through
`llama_context::decode`, with process status 6/ABRT and systemd recovery. The
partial raw SSE and harness receipts are retained in the paths in the JSON.

## Provenance and verification

The exact loaded bundle was `/tmp/buun-llama-cpp-31-03-final.WUe1Jf/bin/llama-server`,
model hash `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`,
and bundle-manifest hash
`8c346e1135b5623bad8a42d32633430bf418c0c93180b544045614f3ea1f94d7`.
The final live-service check returned `{"status":"ok"}` on port 8080.

The machine-verifiable evidence, raw paths, hashes, counters, and exact
configuration are in
`.wiretail/execution/evidence/31-04-primary-small.json`.
