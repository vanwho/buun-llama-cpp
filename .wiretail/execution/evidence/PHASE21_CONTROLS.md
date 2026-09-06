# Phase 21 matched controls — task 21-08

## Result

The receipt is **incomplete with zero release-matched pairs**. Task 21-07
completed no contextual speed trial, so there is no target denominator for a
CPU-KV/GPU-MTP or dense all-GPU Turbo4 control. All per-question ratios,
variability, memory comparisons, and component decompositions are therefore
`null`.

The machine-readable receipt is [`PHASE21_CONTROLS.json`](PHASE21_CONTROLS.json).

## Matching decision

The current-release sentinel was retained as an unpaired diagnostic only. It
uses the 20-07 binary/model but the older corpus hash, a fixed 22,016-token
context, and only one complete measured prompt record. It is selective Turbo4
with native GPU MTP, not either requested control denominator. The historical
20-06 CPU-KV and all-GPU timings likewise use a different release, corpus,
context, and protocol. Neither artifact is promoted to a ratio.

No shorter all-GPU run was called comparable. Observe and MTP-off remain
separate, unmeasured ablations.

## Measured blockers

1. The first 20K target warmup rendered 3,072 of 19,600 prompt tokens in
   134.30 seconds before cancellation and produced no response. This is the
   first observed blocker in long-context prefill/context execution.
2. Automatic native-MTP startup failed at 175K during recurrent-state
   allocation and at 262,144 during CUDA compute-buffer allocation.
3. With no completed target/control pair, the available evidence cannot
   separate model compute, paged attention, draft/verify, transfers, routing,
   graph launch, and synchronization time.

The unpaired current-binary sentinel did record 59.094432 seconds for a
30-token prompt plus 313 generated tokens, with 1.254850 seconds of reported
wait and zero reported transfer bytes. Those values are diagnostic only and
are not representative long-context decomposition or a target/control ratio.

## Deferred verification

- Complete the six-context target curve under the frozen 20-07 bundle/model/
  corpus identity.
- Run release-matched CPU-KV/GPU-MTP and safely fitting dense all-GPU Turbo4
  controls, then calculate per-question wall/decode ratios and dispersion.
- Run observe and MTP-off as separate ablations and collect short/middle/long
  component counters.
- Repair or document the 175K/262K native-MTP allocation failures before
  claiming those capacities.

Raw artifacts and the exact null reasons are indexed in the JSON receipt. The
healthy fixed-four-page 262K candidate remains loaded on port 8080; port 8092
was not touched.
