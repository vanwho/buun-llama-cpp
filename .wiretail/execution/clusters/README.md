# Active cluster contract — V10

New `*-v10` clusters are context areas: group consecutive tasks that share
stable source/design context and benefit from the same Codex prompt-cache
prefix. There is no fixed task-count cap. Keep a cohesive sequence together
while its context remains useful and within provider/runner guardrails; split
at a subsystem, risk, provenance, toolchain, model, or context-budget boundary.
One-task clusters are appropriate for isolated live tests/reviews, and larger
clusters are valid when their bounded context remains productive. Each starts
fresh; only tasks within that cluster may reuse context. Each packet's
context_files supersedes recursive dependency-handoff reading. Do not load
completed old cluster files or old session dumps.

Repository instructions still apply. Inspect implementation source as needed,
but keep historical plans and benchmark diaries out of the reading set.
Future remediation revisions must use new cluster IDs and the same bounded
context contract, with explicit source/test directions in every task.
