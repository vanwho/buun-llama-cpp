# Cluster forward102-driver

Revision: `hotpath-v10-20260914`. Amendment: `repo-context-scale-20261002`.

Tasks in this context area: `102-01`.

This is a small offline validation change only. Raise the occupancy-frontier
CLI's accepted logical-context ceiling from 32,768 to 262,144 while preserving
the default and all geometry/resume checks. Add the smallest regression to
`tools/server/bench/test_occupancy_frontier.py`; do not start Qwen, contact a
service, compile, or benchmark. Do not load the former 32K task handoff: its
attempts are superseded-scope provenance, not this task's proof.
