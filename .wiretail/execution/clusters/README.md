# Active cluster contract — V9

New `*-v9` clusters contain at most three adjacent tasks, normally one or two
in a single subsystem. Each starts fresh; only tasks within that cluster may
reuse context. Each packet's context_files supersedes recursive dependency
handoff reading. Do not load completed old cluster files or old session dumps.

Repository instructions still apply. Inspect implementation source as needed,
but keep historical plans and benchmark diaries out of the reading set.
Future remediation revisions must use new cluster IDs and the same bounded
context contract, with explicit source/test directions in every task.
