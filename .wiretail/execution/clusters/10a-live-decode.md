# Cluster 10a — production selected Turbo4 decode

Tasks: `10-01`, `10-02`, `10-03`.

Purpose: make live Qwen decode consume the compact selected target cache and establish correctness and
performance before dynamic selection is allowed.

Carry forward:

- first wire selected-all-pages through a real graph as the reference route; do not begin with dynamic
  selection;
- direct CUDA qualification is causal, Turbo4/Turbo4, validated Qwen head geometry, supported batch/query
  shape, native logical positions/masks, and an immutable table epoch;
- page table entries reference physical slots but carry logical position runs; arbitrary physical order,
  gaps, tails, and repeated valid logical positions must behave correctly;
- the kernel dequantizes Turbo4 tiles locally and emits output plus optional bounded page-mass/partial
  state. Never gather the selected cache to a full F16 tensor;
- graph capture/reuse keys include route, table content/epoch, representation, shape, and device ownership;
  pages remain pinned until the completion event;
- unsupported shapes use the explicit selected reference route or fail before submission. They never
  reinterpret compact memory as dense logical storage.

Read: plan sections 8.5, 9.4–9.5, 22; handoffs 03-01–03-04, 06-02, 09-02–09-05;
Qwen graph builders, `src/llama-kv-attention-*`, `ggml/src/ggml-cuda/fattn*`, graph allocator/capture,
and direct CUDA tests.

Exit artifacts: live selected-all-pages parity, live direct-paged parity, graph epoch/fence tests, real
allocation evidence, and a decode checkpoint comparing dense/all-pages reference/direct routes at
contexts that fit all pages. Record kernel and end-to-end timing without claiming final speed.
