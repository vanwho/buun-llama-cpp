# Optional work after primary architecture acceptance

Revision: `hotpath-v10-20260914`. Amendment: `forward-turn-retrieval-20260927`.

These implement source-plan tasks I/J. They are not active dependencies of
phases94–102. The final review schedules them only from measured benefit, using
next unused phases (suggested103/104 if no primary remediation occupies them).
Each retains Luna High, the forward TESTING/receipt contract and small tests.

## Follower MTP — optional task103-01

Prerequisite: full256K target backing, query replay/freeze/ring, stable canonical
MTP acceptance and a measured byte ledger. First compute whether reclaiming
part of this model's approximately264MiB full-L draft meaningfully increases H
or reserve. If it does not, record a skip decision without code churn.

Owners: `common/speculative.cpp` native draft-mtp and carry lifecycle;
`src/llama-kv-cache.cpp/.h` draft/target mapping; `src/llama-kv-pager.cpp/.h`
slot identities; `src/llama-context.cpp` draft context admission;
`tests/test-speculative-mtp-state.cpp`, `tests/test-cuda-kv-promotion.cpp`.

Implement a follower interface keyed by target logical block identity and
generation, not pointer aliasing of target K/V bytes. Draft and target codecs,
layer geometry and weights are separate. Follow frozen historical selection;
rebind draft slots only at safe turn publication. Guard current draft writes,
target generation slot reuse, pinned proposals and rollback independently.
Keep full-L GPU draft as the selectable reference/default until measurements
prove the follower. Rejected draft pages never become target canonical data.

Tests: deterministic follower rebind/version/refusal and accepted-frontier
rollback; small CUDA target-equivalent greedy comparison; one canonical suite
native-MTP acceptance before/after on the same target history. Record VRAM
saved, admitted-H change, prepare/copy cost and speed. Adopt only with preserved
correctness and useful measured capacity/speed benefit. No draft-to-CPU fallback.

## First-full-attention Q one-pass retrieval — optional task104-01

Prerequisite: query replay is the validated reference and its TTFT cost is
material in current results. Owners: `src/models/qwen35.cpp` or the actual
GGUF-resolved Qwen model builder, `src/llama-graph.cpp::build_attn_mha` /
`Qcur_routing`, `src/llama-memory-hybrid.cpp`, `src/llama-kv-cache.cpp`,
`tools/server/server-context.cpp` query state machine, CUDA selector tests.

Derive the first full-attention layer from model layout; this model has three
GDN layers before it. Capture that layer's user-span Q after those recurrent
layers but before any full-attention operation. Retrieval/promotion then allows
the full query to traverse attention once. Do not advance preceding recurrent
states twice, recalculate the old history, or use a Q already contaminated by
the provisional historical selection as a claimed one-pass proof.

Use the replay implementation as the correctness/reference path. Add a guarded
internal split/resume seam only if the graph/scheduler can preserve causal
state safely; no CPU tensor extraction or per-layer synchronization. Compare
required-page recall against all-layer whole-span Q on the fixed small recall
set, then one small live conversation. Measure total TTFT including split,
selector and publication. Adopt only if recall/answer quality is acceptable
and TTFT improves materially; otherwise retain replay and remove experimental
hot-path hooks. Capability remains generic for supported model layouts.

VBR/mixed Turbo codecs and YaRN beyond256K are separate later experiments.
Neither is required to accept the fixed-Turbo4 primary implementation.
