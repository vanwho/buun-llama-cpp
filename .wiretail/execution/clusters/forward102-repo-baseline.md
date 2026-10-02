# Cluster forward102-repo-baseline

Revision: `hotpath-v10-20260914`. Amendment: `repo-context-scale-20261002-16k`.

Tasks in this context area: `102-02`.

Create and run the 16K/8K baseline using actual tracked Buun source/document
contents and the deterministic A→B→A prompts under
`tools/server/bench/fixtures/repo-context-v1/`. At L=16,384/H=8,192, compare
full-GPU target KV, full-host/CPU target KV, and selected target KV in that
order; these are all required baseline rows. All model compute and full-L
Turbo4 MTP stay on GPU. Do not turn this into a CPU-only model benchmark.

Use exact file hashes and full rendered Qwen token counts. Keep the same slot,
conversation and candidate through A, B, A2; A2 must not resend A files.
Reserve context for the final query/output/replay and verify no context shift
or truncation. The prior 8K run proved the complete mandatory A/B/A history
plus answer/replay reserve requires 10,829 tokens, so use L=16,384. The
selected B request must be rendered to at least H+2,048 before sending; a
runtime failure after request start is recorded as a partial result and does
not require committed occupancy. Host-valid rows/bytes, page
inventory, route/selector, transfer and promotion fields are best-effort
diagnostics only: capture them when exposed, but their absence or incompleteness
must not fail a request, cause a retry, or block the benchmark. Do not claim
promotion unless its full transition was actually observed.
After the comparisons, leave the selected candidate loaded and verify its
identity/health so 102-03 starts from the tested server rather than restoring
an unrelated profile.

If a live request in any row returns a runtime error, capture that failed
sample once and stop that row; do not investigate/fix eviction or promotion
here. Continue to the next independent row/task. Mark the measurement artifact
partial and keep its speed null for the failed request; do not mislabel it as
a passing selected-speed result. Only pre-request setup errors are repaired
and repeated in this baseline.

This is the only stage with a CPU target-KV baseline. Use an already-supported
ordinary host/offload codec and label its actual type if CPU Turbo4 attention
is not supported; do not add a CPU TurboQuant implementation. The two
measured repository prompts and the selected retrieval chain use the semantic
rubric in `prompts.md`, not brittle exact strings or constrained short answers.
