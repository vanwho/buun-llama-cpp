# V9 review - phase 43

`goal_met` is `false`. The selected-packed identity is valid: CUDA target and
draft K/V are Turbo4, the GPU draft uses native MTP, and
`L8192/H4096/A2048/B128/U64` satisfies
`selected <= A_by_layer <= H <= L`. Full-L GPU native-MTP capability is
present at this bounded point, but the active up-to-128K campaign and separate
256K goal remain unmet.

The natural C>H recall/append fixture completed coherently with 8 committed
tokens. Current-Q publication produced 36 attention and attention-mass
samples with 565 us publication time and zero recorded drop counters. The
required downstream proof did not occur: candidate and promotion publication,
submitted transfer, useful H2D bytes, and selected cold-page use were all
zero. This is the missing real Q-to-cold-to-target-use branch, not a runtime
crash or non-finite-output failure.

The current source owners for the next repair are
`llama_kv_attention_telemetry::publish_completed`,
`llama_kv_cache::capture_kv_routing_query`,
`llama_kv_cache::apply_pager_live_policy`, and
`llama_kv_prefetch_mailbox::poll`/`take_ready`. The repair must preserve the
table epoch, query generation, page generation, sequence identity, and page
identity across the candidate and promotion boundary. Forced transfer is not
natural proof.

Matched original q0/q1/q2 controls each have three samples and zero errors.
Selected decode is 34.1881/34.1358/33.8752 tok/s versus all-GPU
36.6931/36.8183/36.8335, with descriptive ratios 0.931/0.927/0.920.
Selected/CPU-main-KV decode ratios are 3.423/3.389/3.365, but
`measured_cpu_speedup` remains `null` because equivalent throughput is not
defined for that route.

Scale was stopped before 32K because natural-cold eligibility failed. No 32K,
128K, or 256K result is inferred from the L8192 run.

Actual successors were appended in execution order:

1. `45-01` repairs the current-Q candidate and promotion bridge.
2. `45-02` runs focused checks, bounded natural cold proof, and matched
   original q0/q1/q2 controls, stopping before scale when ineligible.
3. `45-03` writes the compact phase-45 summary.
4. `46-01` repeats this review algorithm against only that summary.

The new revision is `hotpath-v9-20260914-r3`. The receipt records all measured
values and leaves unmeasured speed and scale fields explicit.
