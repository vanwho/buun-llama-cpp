# Live attention telemetry checkpoint

Task 11-03 wires the existing CUDA logical-page softmax reduction into the
selected direct graph. The optional F32 output is bounded by the resolved
logical page count and sampled query heads. It is copied only after the
scheduler fence; full K/V data and attention matrices never cross to the
controller.

The graph captures the complete page identity for the selected sequence. The
telemetry accumulator validates table epoch and all page/session/sequence and
representation generations before changing EMA, recent peak, frequency, or
normalization. Omitted pages remain explicitly unobserved. A non-due cadence
skips the reduction and D2H copy.

Configuration is available through `llama_kv_pager_config` and the common
options `--kv-telemetry-interval`, `--kv-telemetry-layer`,
`--kv-telemetry-head-begin`, and `--kv-telemetry-head-count` (zero means all
heads). Counters cover samples, skipped/stale/invalid observations, sampled
tokens/pages/layers/heads, GPU reduction time when supplied by the backend,
D2H bytes/time, publish time, and total observe overhead.

The machine-readable contract and raw-result location are in
[`LIVE_ATTENTION_TELEMETRY_CHECKPOINT.json`](LIVE_ATTENTION_TELEMETRY_CHECKPOINT.json).
The deterministic telemetry fixture covers known bins, EMA/peak updates,
unknown cold pages, cadence, stale generation rejection, and table rebind.

## Deferred verification

The linked Qwen3.8 model-backed CUDA calibration run is unavailable in this
workspace. Direct graph capture on a real model, dense/reference/direct output
parity, CUDA event timings, five observe and five selected trials, captured
mass consistency, and live graph cancellation/rebuild sanitizer checks remain
deferred. Later setup must run the frozen calibration harness and write raw
traces under `/srv/ai/paged-kv/results/11-03/` before phase-14 acceptance.
