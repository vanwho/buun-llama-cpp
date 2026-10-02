# Cluster forward102-repo-baseline

Revision: `hotpath-v10-20260914`. Amendment: `repo-context-scale-20261002`.

Tasks in this context area: `102-02`.

Create and run the 8K/4K baseline using actual tracked Buun source/document
contents and the deterministic A→B→A prompts under
`tools/server/bench/fixtures/repo-context-v1/`. Compare GPU-resident target KV,
host-resident target KV, and selected target KV; all model compute and full-L
Turbo4 MTP stay on GPU. Do not turn this into a CPU-only model benchmark.

Use exact file hashes and full rendered Qwen token counts. Keep the same slot,
conversation and candidate through A, B, A2; A2 must not resend A files.
Reserve context for the final query/output/replay and verify no context shift
or truncation. The selected row must commit C>H=4096. Capture route/selector,
host-backed precondition, transfer completion/publication and target use as
separate facts; a semantically correct answer alone is not proof of promotion.

This is the only stage with a CPU target-KV baseline. Use an already-supported
ordinary host/offload codec and label its actual type if CPU Turbo4 attention
is not supported; do not add a CPU TurboQuant implementation. The two
measured repository prompts and the selected retrieval chain use the semantic
rubric in `prompts.md`, not brittle exact strings or constrained short answers.
