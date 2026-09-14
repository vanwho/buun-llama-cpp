# V9 review — phase 41

`goal_met` is `false`. Identity is valid: CUDA selected-packed Turbo4
target/draft K/V, GPU draft, native MTP, and
`L8192/H4096/A2048/B128/U64` satisfy
`selected <= A_by_layer <= H <= L`. Full-L GPU native MTP is present, but the
active up-to-128K campaign and the separate overall 256K goal remain unmet.

The deterministic review reaches the missing real Q-to-cold-to-target-use
branch. The incremental C>H run was coherent and completed 20 output tokens,
but it recorded zero attention samples, no candidate or promotion, zero
submitted transfers, zero useful H2D bytes, and zero selected cold-page use.
This is a production producer/publication measurement-boundary failure, not a
runtime failure. The selected route measured 33.7603/33.6107/32.9828 tok/s
decode for q0/q1/q2, versus 36.7151/36.8280/36.8460 for all-GPU; ratios are
0.9198/0.9121/0.8952. The explicit CPU-main-KV ratios are 3.4012/3.3882/3.3029,
but no CPU speedup is claimed because the receipt does not define equivalent
throughput. Each profile has three matched samples.

The current owners are `llm_graph_context::build_attn_inp_kv`,
`llm_graph_input_attn_kv::refresh_direct_telemetry`,
`llama_context::publish_kv_attention_telemetry`, and
`llama_kv_attention_telemetry::publish_completed`. Four successors were
appended in execution order:

1. `43-01` repairs the real selected-packed producer/publication edge.
2. `43-02` runs focused checks, bounded natural-cold proof, and matched
   original q0/q1/q2 controls, stopping before scale when ineligible.
3. `43-03` writes the compact phase-43 summary.
4. `44-01` repeats this review algorithm against only that summary.

No 32K, 128K, or 256K process/allocation was launched. The full 256K goal is
not inferred from the L8192 result.
