# Repair85 review

## Decision

**Goal unmet.** The small selected path has a measured controlled promotion
chain, positive native-MTP counters, and descriptive short-prompt speed ratios:
selected-native is 1.72x fresh-prompt pp and 1.64x committed tg versus the
matched all-GPU-native rows. This is not a practical-speed signoff because the
context hot/cold control lacks valid stage timing and long-context requests do
not complete.

## Capability boundary

- Build identity and measured Turbo4 target/native full-L draft placement: true.
- Controlled model promotion: true; organic cold promotion and stable natural
  target consumption: false.
- 32K: CUDA invalid argument in `ggml_cuda_turbo_prefill_attend` before the
  first committed token. 128K: allocation-only.
- 256K: allocation measured, occupied frontier C=0; full occupancy: false.
- Practical speed goal: false/not established.

The measured context row was L8192/C6144/H4096/A2048/B128/U128 with full-L
draft capacity 8192, fresh pp 195.23 tok/s, committed tg 21.36 tok/s, TTFT
31.48 s, and native MTP 12/53. Host seal D2H was 103,809,000 bytes and
summary work 51,904,500 bytes; transfer/wait phase times were not exported.

## Ordered successors

1. 86-01 repairs the first long-context prefill launch invariant.
2. 86-02 measures post-repair 32K/128K/256K capability and occupied C.
3. 86-03 rebuilds matched speed and natural cold-promotion findings.
4. 86-04 creates the phase86 compact summary.
5. 86-05 reviews that summary and stops or schedules only evidence-directed work.

See the JSON for exact reproducers, capability booleans, and successor rationale.
