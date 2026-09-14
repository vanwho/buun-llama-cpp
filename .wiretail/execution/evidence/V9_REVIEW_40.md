# V9 review — phase 39

`goal_met` is `false`. The phase-39 identity is valid: CUDA selected-packed
Turbo4 target/draft K/V, GPU draft, native MTP, and `L8192/H4096/A2048` satisfy
`selected <= A_by_layer <= H <= L`. Full-L GPU native MTP is present, but the
active up-to-128K campaign and the separate overall 256K goal are both unmet.

The deterministic decision reaches the missing natural-proof branch. The
incremental C>H run was coherent, but it recorded zero current-Q attention
samples, no candidate or promotion, zero submitted transfers, zero useful H2D
bytes, and zero selected cold-page use. This is a measurement-boundary
failure, not a runtime failure; selection quality is unmeasured. The first
owners are `llm_graph_context::build_attn_inp_kv`,
`llama_context::publish_kv_attention_telemetry`, and
`llama_kv_attention_telemetry::publish_completed`.

The selected route measured 35.2779/34.3649/33.9990 tok/s decode for q0/q1/q2,
versus 36.4845/36.6288/36.7347 tok/s for the all-GPU control. The ratios are
0.9669/0.9387/0.9251 (median 0.9387). The CPU-main-KV control has zero current
samples because the site launcher did not expose the required explicit
`--no-kv-offload` boundary; no CPU speedup is claimed.

Actual successors were appended in execution order:

1. `41-01` repairs or isolates the live current-Q publication boundary.
2. `41-02` exposes the explicit CPU-main-KV/GPU-native-MTP control boundary.
3. `41-03` runs focused checks, bounded natural proof, and matched q0/q1/q2
   controls, stopping before scale if cold eligibility is absent.
4. `41-04` writes the compact phase-41 summary.
5. `42-01` repeats this review algorithm against only that summary.

No 32K, 128K, or 256K process/allocation was launched. The full 256K goal is
not inferred from the L8192 result.
