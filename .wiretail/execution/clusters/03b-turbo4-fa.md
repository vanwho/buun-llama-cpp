# Cluster 03b — direct Turbo4 paged flash attention

Tasks: `03-03`, `03-04`.

Purpose: implement and integrate the performance data path after metadata semantics are proven.

Carry forward:

- first supported geometry is Qwen3.8 causal decode, T4 K/T4 V, GQA=4, Dk=Dv=256, batch 1;
- kernels consume compressed pages directly and dequantize tiles locally; full-cache F16 gather is
  forbidden;
- preserve codebook, rotation/unrotation, scale, mean subtraction, native masks, and tail rows;
- physical page permutation and logical gaps must match the gather reference;
- prefill/decode transitions and CUDA graph reuse invalidate on table epoch changes;
- optional telemetry reduction must not alter attention output.

Read: plan sections 8.5, 8.7, 9.4–9.5, 12.2, 17; cluster 03a handoffs;
`ggml/src/ggml-cuda/fattn-mma-turbo.cuh`, `fattn-common.cuh`, `fattn.cu`, relevant
`template-instances/`, and upstream/community sparse operator references. Avoid router/policy work.

Exit artifacts: bounded direct-Turbo4 paged FA and graph integration with numerical, sanitizer, scratch,
and feature-off evidence.
