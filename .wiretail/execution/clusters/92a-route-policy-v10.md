# repair92a — automatic fast-route policy closure

Revision: `hotpath-v10-20260914`. Amendment: `repair92-20260922`.

This cluster has one implementation task. It is intentionally separate from
frontier measurement: the task changes route admission and its diagnostics,
then proves that unsupported automatic shapes cannot silently enter the slow
selected-reference consumer. The reference route remains available only as an
explicit correctness oracle until the final fast-path acceptance review.

Load only the task packet, the active V10 plan/testing/receipt contracts,
`src/llama-kv-attention-execution.{h,cpp}`, the selected-view construction in
`src/llama-context.cpp`, the CUDA dispatch code, the focused execution tests,
and the repository contribution instructions. Do not load old phase diaries
or treat historical selected-reference timings as performance evidence.
