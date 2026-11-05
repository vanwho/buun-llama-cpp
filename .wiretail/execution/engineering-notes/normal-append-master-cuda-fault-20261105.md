# Master control: CUDA fault on an ordinary appended question

The R8 master GPU/Turbo4, MTP-off control failed after a successful 6,144-token
repository fill, on the first original prompt-1/off warmup. The request appended
33 tokens with LCP/reusable prefix 6,144 and rewind zero. No benchmark KV-file
restore, prefix rewind, or context overflow preceded it. This is a runtime
failure, not an acceptable property of normal prompting.

## Evidence and exact observation point

- Frozen source: `d2407c33f9595d1badccfd9a82a33c4c5ceda490`, clean worktree
  `/srv/ai/paged-kv/worktrees/master106-d2407c33f959`.
- Server SHA-256: `527c34de411d4cad6290e828deb17a71816ae2b9620a3f1166b833e1267e0f0f`.
- Raw attempt: `/srv/ai/paged-kv/results/comparison-completion-20261104-r8/master-gpu-mtp-off/attempt-001-L122880`.
- `tools/server/server-context.cpp:22665–22668`: `llama_process` returned zero;
  the queue-yield lambda then called target `llama_synchronize` because output
  was requested. The asynchronous CUDA error surfaced there, before output.
- `ggml/src/ggml-cuda/ggml-cuda.cu:4328`: failing CUDA check wraps
  `cudaStreamSynchronize` (illegal memory access).
- Checkpoint 3 was captured before decode (`server-context.cpp:21748–21751`;
  creation logging at `22368–22372`), not restored by this request.

Those line numbers refer to the frozen master worktree, not the evolving fork.
The host backtrace cannot identify the accessing device kernel. A fault in the
33-token target graph or an earlier still-asynchronous operation are hypotheses,
not established causes. The successful fill and ordinary append do not justify
blaming the prompt, memory capacity, or a benchmark restore.

## Next diagnostic and comparison treatment

After the independent primary measurements, reproduce only the first warmup
under Compute Sanitizer memcheck, with the same frozen binary and server argv.
The prepared plan is
`/srv/ai/paged-kv/results/post-queue-fault-diagnostic-r8-20261105/plan.json`.
Retain its slow instrumented timings as diagnostics only. Identify the first
invalid device access before proposing an inference fix or changing flags.

Record the failed control cells as unavailable with this runtime reason; do
not invent throughput, silently substitute the fork, or repeat the full matrix.
Keep unrelated completed measurements and continue healthy configurations.

## Follow-up frozen-build diagnostics

Compute Sanitizer memcheck did not complete the measured request: its
900-second limit expired after 322 output tokens, and no invalid access was
reported before that timeout. This is inconclusive instrumentation, not a
passing 400-token benchmark or proof that the failure disappeared.

CUDA-GDB subsequently reproduced `CUDA_EXCEPTION_5` (warp out-of-range address)
in the actual Flash Attention device kernel. External captures are under
`/srv/ai/paged-kv/results/cuda-gdb-master-gpuoff-r9-20261105/`.
Run 003 used the same frozen L122880/B1024/U256 Turbo4/MTP-off command and normal
append schedule. The 40-token warmup completed; the first measured generation
failed before producing output. The exception is in a
`flash_attn_ext_f16<256,256,1,8,...>` specialization with Turbo4 K/V template
type 44, not evidence that the requested Turbo4 storage was replaced with Q4_0.
The run's startup output also confirms Turbo4 K/V initialization.

Run 003 recorded grid dimensions (152,1,1), block dimensions (32,4,1), exception
block (57,0,0), thread (0,2,0), and device 0. The device backtrace contains only
the kernel frame in this optimized build; the host stack again ends in CUDA
stream synchronization. This locates the accessing kernel, but does not yet
identify its invalid pointer or the source instruction that produced it.

A separately identified graph-disabled diagnostic can distinguish a captured
pointer/reuse issue from a fault that also occurs with direct launches. It
cannot replace an ordinary frozen-configuration throughput measurement.

The follow-up at
`/srv/ai/paged-kv/results/graphs-disabled-master-gpuoff-r9-20261105/`
also completed the 40-token warmup and failed on the first measured request.
Its only diagnostic environment change was `GGML_CUDA_DISABLE_GRAPHS=1`;
the server argv and binary were unchanged. CUDA reported a misaligned address
at stream synchronization in this run. Thus graph capture is not required to
reproduce a memory-access failure. The changed error label does not establish
graph capture as the cause or identify the offending pointer.

Compact CUDA-GDB evidence: `cuda-gdb-summary.json` in the CUDA-GDB root,
SHA-256 `ff136a8123da86855b076385ff35e5e2c191436ad43bc61fc6c9cf8c9c2c05b2`.
Combined graph-disabled evidence:
`/srv/ai/paged-kv/results/graphs-disabled-diagnostics-20261105/graphs-disabled-summary.json`,
SHA-256 `41c9a00aee4755c4b118454dc6c7e27c066ec196a5fbf1e409a08de20e90471b`.
The diagnostic owner restored server/gateway health and stopped all owned
clients; no changed graph flags were retained in the managed service.

## Separate benchmark serialization defect

An offline comparison of R8 fork CPU/MTP-on C12,288 requests proves that client
message history is preserved exactly, but `qwen38-shared-prefix.jinja` moves its
reasoning-effort system instruction from the previous last user to the new last
user. This changes the rendered historical prefix and can require reprocessing.
A repair is being prepared outside active files to retain each turn's steering
as durable chat history. Apply it only after the primary comparison matrix and
its diagnostics, with reasoning-mode transition checks. It is not proof of the
CUDA fault's cause: the crashing off-mode request had rewind zero.
