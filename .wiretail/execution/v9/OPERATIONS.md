# V9 execution and handoff operations

## Authorized environment and boundaries

Project root `/srv/repos/vanwho/buun-llama-cpp`, fork branch
`plan/attention-aware-kv-paging`; shared runner `/srv/wiretail/wiretail.sh`.
All server-specific paths and commands in this file are execution metadata,
not source defaults. Generic runtime code must not embed them.

Site inputs to inspect in 33-01 and persist in a **redacted recipe receipt**:

- canonical launcher `/srv/ai/benchmarks/run-profile-benchmark.sh`;
- model symlink `/srv/ai/models/text/current.gguf` (resolve and confirm
  Qwen3.8-27B UD-IQ4_XS against the existing service configuration);
- active profile `/srv/ai/config/llama/active-profile`;
- API key file `/srv/ai/config/llama/api-keys` (read privately to authenticate,
  never print its values or place them in a receipt/argv);
- target `llama-server.service`, 8080; companion `ai-long-memory.service`,
  8091; do not stop or reconfigure unrelated 8092 service;
- current build trees may be reused after checking CMakeCache and loaded DSOs;
  new candidate binaries must be explicitly passed via `BENCH_SERVER_BIN`.

Use `sudo -n` for exact authorized privileged operations. Test `sudo -n true`
and `sudo -n systemctl show llama-server.service -p MainPID --value`.
Bare `systemctl stop` denied by polkit is not proof sudo is unavailable.
`sudo -n -v` is not the capability test. If the exact command genuinely fails,
retain stderr/uid/effective environment and report setup failure; never claim
an interactive requirement without testing the command. Do not alter sudoers
or credentials as part of model tasks.

The user authorizes stopping/replacing the active Qwen service for candidate,
control and debugging work. Keep a **successful tested** candidate loaded for
the next task; this does not mean preserving an old binary that prevents live
validation. Capture entry identity, switch sequentially, confirm loaded
candidate and health. Restore only for explicit control/revert, failed-start
recovery, teardown or final cleanup. A failed candidate cannot be declared
restored by finding a different healthy process on another port.

Raw outputs `/srv/ai/paged-kv/results/v9/<task>/<UTC-run-id>/`. Persistent build
and site recipe outputs `/srv/ai/paged-kv/v9/`. Do not put essential evidence
only in `/tmp`. Do not kill a process by broad pattern (`python /dev/fd/3`,
`llama-server`): verify PID, executable and ownership of the exact child/service.
Wiretail's output follower is not an orphaned benchmark client.

## Git and contribution discipline

Read `CONTRIBUTING.md`. Agents edit code/tests/docs; outer Wiretail auto mode
owns task branches, implementation commits, separate `.wiretail/execution`
commits, fork pushes and integration. The sole declared upstream-merge task
uses the runner's opt-in merge hook. No task does its own rebase/cherry-pick or
push; fetch/read-only Git inspection is fine. Preserve unrelated dirty edits.

Public upstream issues/PR prose is a human action under this repository's
rules. Do not open an AI-authored issue/PR. Future upstream patches must be
generic, focused, tested, AI assistance disclosed as required, and based on a
clean code-only branch. Do not PR the whole plan branch: its history contains
execution metadata. Do not squash that metadata into a code patch or delete
unrelated upstream repository docs.

Historical cleanup inventory (never PR these, including historical commits):
`docs/execution/`, `tool/codex/`, `build/codex-autonomous/`, `.codex-runner/`,
`.codex_runner/`, `.wiretail/`, old planning branches and site-specific results.
Keep raw benchmarks/config/secrets out of code commits. The canonical current
instructions are only `.wiretail/execution/v9/` and current packets/clusters.

## Per-task workflow and bounded context

1. Read repository instructions; current task entry with
   `jq --arg id '<id>' '.tasks[]|select(.id==$id)|del(.token_usage.events,.token_usage.turns)' .wiretail/execution/WORK_STATE.json`;
   current packet/cluster and its `context_files`. Do not read the full state
   merely to find a field. Completed dependencies are scheduling facts, not
   a recursive history-reading requirement.
2. Verify branch/current candidate provenance. Inspect specified symbols and
   existing tests. Modify only the task scope; no generic performance audit.
3. Build touched targets incrementally, run deterministic regression, then
   the named short live probe if required. Capture raw output once.
4. On failure identify first divergent operation, not just final error. Fix
   within scope; if a missing prerequisite is discovered, the assessment may
   add one precise predecessor with source/test/invariant/expected result.
   Do not re-run unrelated tests/whole corpus to manufacture progress.
5. Update a compact current handoff (target <=120 lines) and evidence receipt,
   validate task graph and `git diff --check`, then mark done only under the
   packet's acceptance. Append only one short WORK_LOG entry per material
   outcome; preserve token accounting via Wiretail, never reset usage.

Handoff sections: `Result`, `Changed symbols`, `Validation / raw receipt`,
`Measured findings`, `Unresolved next action`, `Loaded candidate / resume`.
Replace stale attempt narratives with the latest bounded diagnosis; archive
the old handoff content to its raw task result root if necessary. Do not put
multi-megabyte logs, full `--help`, process lists, historical manifests or all
previous attempts in the active handoff. Evidence says pass/fail/not_run with
reason; a machine/model present here cannot be declared unavailable merely
because another profile is running.

## Retry policy for these tasks

Initial task-recommended Luna Medium or Luna High; retry1 same family High with
the artifact-aware prefix; retry2 after Luna High assessment returns to the
task's original model/reasoning; retry3 after Terra High assessment returns to
the task's original model/reasoning. Task fields opt in to retry1 High; shared
tool defaults unchanged. Assessments also receive the bounded context list. Three
substantive retries then block is correct if distinct repairs are exhausted.
Provider/network transport backoff is not implementation progress. Never
reset the budget by appending another diary-only task with the same failing
command. If remediation tasks are added, they must change an identified
implementation/configuration defect, not simply repeat the last audit.

Validation command:

```bash
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp python3 /srv/wiretail/task_state.py validate
python3 .wiretail/execution/v9/validate_plan.py
git diff --check
```

Run command (operator, not within a task):

```bash
PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp \
PROJECT_BRANCH=plan/attention-aware-kv-paging /srv/wiretail/wiretail.sh
```
