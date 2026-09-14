# V9 review — phase 38

`goal_met` is `false`. The selected identity is valid and the three matched
small prompts show 3.1971–3.5298× committed-decode speedup over the real
CPU-main-KV control (median 3.4164×), while selected/all-GPU decode is only
0.9875–1.0840× (median 1.0504×). Selected prefill is 0.1492× the all-GPU
control and TTFT is 6.6901× it, so this is not an overall speed win.

The deterministic decision reaches the missing-proof branch. The natural cold
edge did not publish a usable current-Q sample or catalogue candidate:
`attention_samples=0`, `transfer_submitted=0`, `h2d_bytes=0`, and
`selected_cold_pages_used=0`. The 32K and 128K measurements correctly remain
`not_run`; 256K is an estimate, not a demonstration. This is a measurement
boundary failure, not evidence of a hardware or runtime crash. Selection
quality is unmeasured because the diagnostic selected all 16 pages.

The first repair owner is the Q/sample publication chain:
`llm_graph_context::build_attn_inp_kv` →
`llama_context::publish_kv_attention_telemetry` →
`llama_kv_attention_telemetry::publish_completed`. A second bounded task checks
packed Q transform/grouping in `build_attn_mha`, `build_attn`, and
`llama_kv_attention_execution::prepare`. Both retain the mature Turbo4 packed
route and full-L GPU native-MTP contract.

Actual successors were appended in execution order:

1. `39-01` repairs or isolates current-Q sample/candidate publication.
2. `39-02` checks the selected packed Q/layout boundary.
3. `39-03` runs focused tests, the bounded natural proof, and the matched
   original-three-prompt controls, stopping before scale if the cold edge is
   still absent.
4. `39-04` writes the compact phase-39 summary.
5. `40-01` repeats this review contract against only that new summary.

The complete 256K goal remains separate from the active up-to-128K campaign.

