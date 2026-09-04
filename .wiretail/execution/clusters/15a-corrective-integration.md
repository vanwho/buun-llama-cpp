# Cluster 15a — corrective integration and immutable evidence

Tasks: `15-01`.

Purpose: establish a clean implementation base after the fork sync, preserve V2 as historical evidence,
and make the corrective work reviewable. Read `CONTRIBUTING.md`, `AGENTS.md` if present, the task README,
the plan sections 13, 19, 20, 25–27, and the V2 acceptance manifests. Never edit `/srv/ai/paged-kv/repos`.

The current local checkout may be behind the synced fork. Create or prepare a fresh integration branch
from the fork's current master, record upstream ancestry and range-diff, and keep execution metadata in
separate commits from production code. Do not rewrite V2 manifests or claim their blocked gates passed.

Exit artifact: source/provenance handoff, exact branch relations, conflict forecast, and a corrective
baseline build record.
