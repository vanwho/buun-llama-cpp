# 51c-summary-v10

Revision: `hotpath-v10-20260914`.
Tasks: `51-05`.

Generate compact current findings from V10 JSON/raw records. Preserve invalid/not_run reasons and null unmatched ratios. Do not reread historical acceptance ledgers or manually copy benchmark numbers.

Read each task's explicit context_files and packet, then named source by symbol.
Do not load whole state/log or earlier plan/phase diaries. Cluster size follows
subsystem/context overlap, not a fixed number of tasks. New cluster IDs start
fresh context; reuse cohesive context and current candidate within a cluster.
Every task records named proof receipt plus compact handoff with current build,
smallest fixture, findings and next action. Required implementation CUDA/live
checks cannot be replaced by generic CPU passes or deferred notes.

Use sudo -n for authorized named service lifecycle; passwordless sudo is available.
Keep successful candidate loaded except explicit controls/reverts or unsafe failure.
Full-L native MTP K/V and main/canonical K/V remain Turbo4; draft remains GPU.
Server-specific raw data stays /srv/ai; metadata stays .wiretail, not code commits.
