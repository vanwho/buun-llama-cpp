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

## Separate benchmark serialization defect

An offline comparison of R8 fork CPU/MTP-on C12,288 requests proves that client
message history is preserved exactly, but `qwen38-shared-prefix.jinja` moves its
reasoning-effort system instruction from the previous last user to the new last
user. This changes the rendered historical prefix and can require reprocessing.
A repair is being prepared outside active files to retain each turn's steering
as durable chat history. Apply it only after the primary comparison matrix and
its diagnostics, with reasoning-mode transition checks. It is not proof of the
CUDA fault's cause: the crashing off-mode request had rewind zero.
