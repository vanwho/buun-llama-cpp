# Task packet contract

Each packet is the complete task-specific context for one bounded task. Its shared session context is
`docs/execution/clusters/<task.cluster>.md`; read that file once per fresh cluster session and reuse it
for consecutive tasks in the cluster.

Global rules for every packet:

1. Work only in `/srv/repos/vanwho/buun-llama-cpp`, except for explicitly read-only references and
   benchmark output paths named by the packet.
2. Re-read repository `CONTRIBUTING.md` and any applicable `AGENTS.md` before editing. The current
   Buun tree has no `AGENTS.md`; do not assume that remains true after an upstream sync.
3. Preserve unrelated dirty work. Never alter or clean `/srv/ai/paged-kv/repos/buun-llama-cpp`.
4. Do not create GitHub issues/PRs, write their descriptions/replies, push, merge, or invent human
   approval. The user permits iterative commits in `vanwho/*`, but the clustered runner uses manual Git
   mode and leaves reviewable commits/pushes to a human or an explicitly authorized outer agent.
5. Prefer internal APIs and existing source/test files. Do not add a new test file without maintainer
   approval recorded in the task handoff.
6. Never claim a CUDA, benchmark, quality, or hardware result without raw output. Missing reference
   hardware blocks a hardware acceptance task; it does not block pure/fake-backend work.
7. Keep feature-disabled behavior unchanged and make unsupported configurations fail closed.
8. Keep implementation portable: never hard-code `/srv/ai`, `/srv/repos`, `/home/ninja`, local service
   units/PIDs/ports, absolute model/profile filenames, or RTX 4080 assumptions. Qwen3.8 geometry belongs
   in runtime capability checks and test fixtures; server profiles, lifecycle commands, and benchmark
   artifacts stay in execution metadata or the external `/srv/ai` repository.
9. Read existing handoffs for every `depends_on` task before implementation. Create/update
   `docs/execution/handoffs/<task-id>.md` with scope, changed files, commands/results,
   invariants checked, raw artifact paths, risks, and deferred checks.
10. Update only this task in `WORK_STATE.json`. Use `tool/codex/task_state.py`; mark complete only after
   every locally executable acceptance item passes. Mark blocked with one concrete reason and needed
   input only after distinct recovery paths have been tried. The runner automatically reopens a blocked
   task twice, using a fresh session each time and escalating a Luna task to Terra/high on the final
   approach; do not treat the first failed command as a sufficient blocker.

Suggested handoff skeleton:

```markdown
# <task-id> handoff

## Outcome
## Changed files
## Commands and results
## Invariants/evidence
## Raw artifacts
## Deferred verification
## Remaining risks
```
