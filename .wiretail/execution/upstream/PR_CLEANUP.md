# Upstream PR cleanup and execution-path denylist

The `.wiretail/` tree is fork-local execution metadata. It is not production
Buun/llama.cpp code and must never be included in an upstream pull request.
Do not create an upstream PR from the plan branch merely by deleting these
files in a final commit: the execution commits would still be present in the
PR history.

## Current execution paths

The current layout is:

- `.wiretail/execution/` — plan, task packets, state, clusters, handoffs,
  evidence, and work logs;
- `.wiretail/build/` — runner logs, Codex JSONL/final-message artifacts,
  assessments, and retry records;
- `.wiretail/` runtime files — lock, stop, pause, session, and retry-control
  files;
- `/srv/wiretail/wiretail.sh` and
  `/srv/wiretail/task_state.py` — external execution programs, never upstream
  source files.

## Historical paths to exclude

Earlier commits used these paths. They remain a denylist when auditing history:

- `docs/execution/`;
- `tool/codex/`;
- `build/codex-autonomous/`;
- `.codex-runner/`.

The current `.wiretail/` paths are also denylisted. Runtime paths are normally
ignored locally, but tracked execution metadata must be excluded explicitly
from production commits.

The existing `.codex-runner/` session file and lock were migrated to
`.wiretail/`, and the existing `build/codex-autonomous/` artifacts were
migrated to `.wiretail/build/`; no prior session or attempt artifact was
discarded during this layout change.

## Audit before preparing an upstream branch

Run from the repository with `UPSTREAM_BASE` set to the reviewed upstream tip:

```bash
UPSTREAM_BASE=upstream/master
git diff --name-only "$UPSTREAM_BASE...HEAD" -- \
  '.wiretail/**' 'docs/execution/**' 'tool/codex/**' \
  'build/codex-autonomous/**' '.codex-runner/**'
git log --format='%H %s' "$UPSTREAM_BASE..HEAD" -- \
  '.wiretail/**' 'docs/execution/**' 'tool/codex/**' \
  'build/codex-autonomous/**' '.codex-runner/**'
```

Both commands are review gates. The first must be empty for the proposed
upstream range. The second identifies every historical commit that must be
omitted or manually split; deleting the files later is insufficient.

Prepare the upstream branch from `UPSTREAM_BASE`, then cherry-pick only
production commits after reviewing each candidate. If a commit mixes source
changes with execution metadata, split it manually and retain only the source
change. Before opening the PR, verify both the final path denylist and the
commit list again, and run the repository's normal upstream tests.

This file itself is fork-local execution metadata and is excluded along with
the rest of `.wiretail/execution/`.
