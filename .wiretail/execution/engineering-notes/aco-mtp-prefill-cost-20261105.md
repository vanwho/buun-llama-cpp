# ACO prefill: native MTP on versus off

## Matched measurements

The completed November 4 ACO curves use the same binary, loaded DSOs, model,
template, frozen repository map, target Turbo4 K/V, L=262,144, H=65,536,
B=1024/U=256 and thread settings. Normalized server argv differs only in MTP
switches and per-run artifact paths. Both ingest 261,121 net-new tokens and
process 262,072 backend prompt tokens across the same 15 append groups.

| Aggregate measurement | MTP on | MTP off | On relative to off |
|---|---:|---:|---:|
| Backend prompt tok/s | 784.72 | 941.85 | -16.68% |
| Net-new tokens / backend second | 781.88 | 938.44 | -16.68% |
| Net-new tokens / HTTP wall second | 777.90 | 932.60 | -16.59% |
| Backend prompt seconds | 333.97 | 278.25 | +20.02% |

The extra reported prompt time is 55.72 seconds. This is not a difference in
how many tokens the two runs count as new. Backend seconds are the server's
reported prompt-phase duration, not an isolated target-model CUDA kernel sum.

The four paired full chunks each add 32,783 net-new tokens:

| Actual occupied tokens at append end | MTP on tok/s | MTP off tok/s | Difference |
|---:|---:|---:|---:|
| 61,456 | 671.38 | 748.98 | -10.36% |
| 162,832 | 815.90 | 988.28 | -17.44% |
| 195,615 | 775.82 | 970.67 | -20.07% |
| 229,391 | 742.79 | 965.31 | -23.05% |

These are append endpoints, not the later nominal checkpoint labels. The
difference persists on large matched chunks; tiny checkpoint tails alone do
not explain it. Runs were sequential, not randomized paired trials.

## Source facts

Native MTP is not restricted to proposing tokens after prefill. It also builds
the draft context from prompt tokens and target hidden states. In the frozen
fork source (`8a79008b5473676bb99a54cc1505d11a4e107357`):

- `tools/server/server-context.cpp:25020` invokes
  `common_speculative_process` after successful target processing, before
  `metrics_post_decode` at line 25202 advances prompt timing.
- `common/speculative.cpp:3085` attempts the GPU hidden-state handoff for a
  contiguous batch; line 3160 runs `llama_decode(ctx_dft, batch)` for the MTP
  layer(s). This API call processes a prompt batch too; its name does not
  mean it generates one output token here.
- `common/speculative.cpp:3171` records a completion event, falling back to
  draft synchronization if that event cannot be established.
- `tools/server/server-common.h:539` computes prompt milliseconds from the
  prompt-phase timestamps. These timings can include draft preparation and
  blocking work within the prefill lifecycle.

The ACO target has bounded hot GPU storage; the native MTP draft context is
configured for the full L. Maintaining that draft history adds work during
ingestion, so MTP-on and MTP-off need not have identical prefill throughput.
The widening gap is consistent with that extra work, but is not a measured
attribution of all 55.72 seconds to draft attention.

The pinned GGUF identifies architecture `qwen35`, 65 total blocks, one next-token
block and full-attention interval four. The main trunk has 64 layers: 16 full
attention and 48 recurrent/GDN layers (`src/llama-hparams.cpp:348` and
`src/models/qwen35.cpp:17`). Its one extra MTP block uses dense attention
(`src/models/qwen35.cpp:609`). Do not estimate this as one extra attention layer
on top of 64 full-attention layers. At a large total/hot ratio, that extra
full-context attention is a meaningful workload.

There is also a specific synchronization cost worth profiling. After the bulk
device handoff, `common/speculative.cpp:3204–3221` retrieves the final target
hidden row into host carry state. If not already materialized, the device-row
path in `src/llama-context.cpp:3961–3987` performs D2H and scheduler
synchronization. It preserves state correctness, but can prevent asynchronous
submission across prompt batches. A future optimization should keep the carry
row under stable device ownership with event ordering and materialize its host
shadow only when required by an actual consumer. It must preserve rejection,
checkpoint and query-replay identity; simply skipping the row is incorrect.
The present data does not quantify how much time this fence costs.

The server's MTP-verification attention flag excludes ordinary prompt batches
(`tools/server/server-context.cpp:24959`). Source inspection therefore does
not support a claim that enabling MTP switches normal target prefill to a slow
verification/reference route. Graph scratch reservation occurs at context
setup and should not be described as repeated per-token allocation overhead.

Existing MTP-on logs confirm that the device handoff actually ran: 58,595
direct handoff/transfer records, no bulk-handoff-unavailable warning and no
completion-event-unavailable warning. There are 131 no-handoff records; a
bounded trace check identifies the one-token shape, which is ineligible for
the multi-row handoff. These log totals include generation and verification,
not just repository fills. Do not use them as per-prefill cost measurements.
Direct handoff still reports a single hidden-row-sized target D2H/draft H2D
transfer (20,480 bytes each); do not describe this path as zero host traffic.

## Memory confound and limits of the conclusion

The MTP-on server had zero VmSwap during the first full-chunk fill. During
the later three it had approximately 465–467, 462–465 and 438–441 MiB swapped,
respectively. MTP-off had zero throughout those intervals. The 530.4 MiB
MTP-on peak occurred during an earlier measured decode at C=100,352.

Fill windows were inferred from request/SSE file modification times and
approximately five-second process samples. VmSwap measures swapped-out
process memory, not active swap I/O, and does not identify which allocations
were swapped. It is a real comparison caveat, not proof that swap caused the
slowdown or that canonical context pages were swapped. Do not claim the whole
17% throughput difference is inherent, optimized or explained without phase
timing that separates target work, draft work, waits, checkpoints and I/O.

Keep the active comparison matrix frozen. Any follow-up profiling must be a
separate diagnostic at an idle boundary, with matched chunks and existing
binaries. Do not shorten prompts, lower batch sizes or weaken MTP state to
make the measured cost disappear.

## Evidence

- `.wiretail/execution/evidence/ACO_FULL_CURVES_20261104.json` and `.md`.
- `/srv/ai/paged-kv/results/candidate-full256k-aco-mtp-on-20261104T105341378940038Z`.
- `/srv/ai/paged-kv/results/candidate-full256k-aco-mtp-off-20261104T115523300080953Z`.
- Each root's `cell/C261120/campaign/run-config.json` binds runtime identity;
  the compact curve JSON binds per-append request/raw hashes and resource logs.
