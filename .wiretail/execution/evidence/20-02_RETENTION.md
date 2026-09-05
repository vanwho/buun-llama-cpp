# 20-02 retention evidence

Result: local implementation/build/fixture checks passed; the named live
Qwen35 Turbo4 focus-shift and multi-hop checkpoint is deferred because its
model and corpus fixture are not present in this checkout.

The implementation keeps attention retention separate from query retrieval.
Retention records carry layer identity, EMA, recent peak, reuse/frequency,
sample count, and last observation. Clean host residency preserves
authenticated history, while an unknown cold page is not represented as a
measured zero. Policy pins are established first and remaining pages use
capacity-relative recent/history/transient choices with deterministic fallback
and bounded exploration.

The CUDA paged-attention fixture passed serial-vs-split global mass checks for
partition capacities 1, 2, and 3, including valid-tail/mask cases. Observed
fixture timings and hashes are recorded in `20-02_RETENTION.json`; timings are
findings, not acceptance gates.

Local verification:

- CPU focused CTest: 5/5 routing, pager, policy, and telemetry tests passed.
- CUDA focused CTest: 2/2 policy and telemetry tests passed.
- CUDA paged-attention mass fixture passed.
- `llama-server` and all changed targets built successfully.
- `git diff --check` passed.

Deferred live verification must use the runtime-supported Qwen35 Turbo4 fixture
and compare recency-only, query-only, and query-plus-retention at equal hot
bytes, recording quality, selected rows, transfer rate, and elapsed time.
