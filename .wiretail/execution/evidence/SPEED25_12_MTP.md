# SPEED25_12_MTP

## Outcome

Native MTP draft catch-up now has a guarded device-owned hidden-state handoff.
For compatible GPU/backend pairs, contiguous target `h_nextn` rows are copied
into the draft `mtp_h_input` after ordinary token/boundary input staging. CPU,
mismatched-device, unavailable-graph, non-F32, and host-buffer cases retain the
portable host path.

The pending boundary row, token shift, per-sequence positions, verification
rows, acceptance/rejection truncation, and MTP history lifecycle are unchanged.
The request is single-use and cleared before graph execution; the backend copy
is submitted with the source and destination backends, preserving device
ownership without exposing a host pointer as a device input.

## Verification

Build and focused deterministic tests passed. No live CUDA/model timing was
available, so before/after copy timings, proposal acceptance, output parity on
Qwen3.8, and n-max 1/2/4 selection are deferred.

See the JSON receipt for the exact command results and deferred reason.
