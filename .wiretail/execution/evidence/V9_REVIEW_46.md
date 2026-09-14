# V9 review - phase 45

`goal_met` is `false`. The selected-packed identity is valid: CUDA target and
draft K/V are Turbo4, the GPU draft uses native MTP, and
`L8192/H4096/A2048/B128/U64` satisfies
`selected <= A_by_layer <= H <= L`. Full-L GPU native-MTP capability is present
at this bounded point, but the active up-to-128K campaign and separate 256K
goal remain unmet.

The natural C>H recall/append fixture completed coherently with 8 committed
tokens. Current-Q publication produced 26 attention and 26 attention-mass
samples, but no non-sentinel candidate or promotion was published. Submitted
transfer, useful H2D bytes, and selected cold-page use were all zero. This is
the missing real Q-to-cold-to-target-use branch, not a completed capability.
The first natural run had a recoverable pager batch-write reservation failure
at C=2807; fresh managed recovery completed the bounded run.

Matched original q0/q1/q2 controls each have three samples and zero errors.
Selected decode is 34.7836/34.4398/34.1688 tok/s versus all-GPU
36.5296/36.6687/36.7821, with descriptive ratios 0.952/0.939/0.929.
Selected/CPU-main-KV decode ratios are 3.437/3.415/3.365, but
`measured_cpu_speedup` remains `null` because equivalent throughput is not
defined for that route.

Scale was stopped before 32K because natural-cold eligibility failed. No 32K,
128K, or 256K result is inferred. The next measured chain is:

1. `47-01` instruments and repairs the first missing candidate-readiness or
   promotion edge, preserving exact table/query/page generations and identity.
2. `47-02` runs focused checks, bounded natural cold proof, and matched
   original q0/q1/q2 controls using the existing B1-B5 boundary.
3. `47-03` writes the compact phase-47 summary.
4. `48-01` repeats this review algorithm against only that summary.

The chain stops at L8192 until candidate, promotion, useful H2D, selected
cold-page use, and coherent recall/append all pass. It does not claim CPU
speedup or 128K/256K achievement.
