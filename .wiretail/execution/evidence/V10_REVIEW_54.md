# V10 review 54

Revision: `hotpath-v10-20260914`. Reviewed input: `V10_SUMMARY_53.json/.md`.

## Decision

`goal_met` is false. Identity and the required Turbo4 placement are valid.
The controlled model edge and organic/file-roundtrip edges each reach queued
and completed H2D, publication, and target packed-FA use. That proves a
physical promotion chain, not useful answer quality; the summary keeps the
quality claim false.

The 8K proof is real at L8192/C6144/H4096/A2048/B128/U128. The 32K pilot
reached C24580 and the 128K pilot reached C4204 with H30208, but neither is a
full C262144 result. 256K startup/allocation and full-L GPU Turbo4 draft
reservation are reported separately. The three prior 256K request attempts
have max observed C=0, and the phase-53 retry did not start because an
unrelated CUDA process prevented the candidate from loading. Full occupancy,
full-context quality, and soak are therefore unproven.

## Measured rates and remaining owners

The original comparison has three samples per question/mode (27 rows total).
Selective prefill medians are 124.059–125.099 tok/s, versus 633.703–637.014
CPU-main and 1542.692–1548.887 all-GPU. Selective committed decode is
31.863–38.360 tok/s with positive draft and accepted deltas (70.476%–91.111%
acceptance); the CPU-main/all-GPU placement controls proposed drafts but
accepted zero and are not native-MTP acceptance evidence. Cached append
probes for 64 and 256 tokens pass functionally, but no phase-53 matched append
rate or revalidated q0/q1/q2 matrix exists.

The remaining full-L owners are `ggml_cuda_fattn::kv_dequant_scratch` /
`llama_kv_cache::vbr_scratch_reserve` at
L262144/H30208/B128/U128, and
`llama_kv_attention_packed_cache::find_or_create` /
`submit_graph` / `launch_mul_mat_q` at L262144/H16384/B128/U64. The selective
prefill path is also the measured speed gap: optional counters show substantial
summary-build, H2D, and wait activity, while the deferred matched run did not
isolate a narrower owner. This is a practical finding, not a 3x/5x gate.

## Ordered next work

Phase 55 appends one full-L resource-boundary repair, the bounded full-L and
matched revalidation benchmark, and a phase summary. Phase 56 then reviews
only that summary. The repair packets include exact symbols, hypotheses,
minimal fixtures, invariants, numerical before/after discriminators, and stop
conditions; this is not an evidence-only loop.

## Deferred verification

The live full-L retry and matched q0/q1/q2/append/MTP matrix require the
authorized GPU endpoint to load without the unrelated CUDA process. Until
then, no C262144, full-context quality, or practical end-to-end speed claim is
made.
