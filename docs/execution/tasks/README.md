# Task packet contract

Each packet is the complete local context for one bounded task. Follow it exactly.

Global rules for every packet:

1. Work only in `/srv/repos/vanwho/buun-llama-cpp`, except for explicitly read-only references and
   benchmark output paths named by the packet.
2. Re-read repository `CONTRIBUTING.md` and any applicable `AGENTS.md` before editing. The current
   Buun tree has no `AGENTS.md`; do not assume that remains true after an upstream sync.
3. Preserve unrelated dirty work. Never alter or clean `/srv/ai/paged-kv/repos/buun-llama-cpp`.
4. Do not create GitHub issues/PRs, write their descriptions/replies, push, merge, or invent human
   approval. The user permits iterative commits in `vanwho/*`, but the clustered runner uses local Git
   mode and leaves reviewable commits/pushes to a human or an explicitly authorized outer agent.
5. Prefer internal APIs and existing source/test files. Do not add a new test file without maintainer
   approval recorded in the task handoff.
6. Never claim a CUDA, benchmark, quality, or hardware result without raw output. Missing reference
   hardware blocks a hardware acceptance task; it does not block pure/fake-backend work.
7. Keep feature-disabled behavior unchanged and make unsupported configurations fail closed.
8. Create/update `docs/execution/handoffs/<task-id>.md` with scope, changed files, commands/results,
   invariants checked, raw artifact paths, risks, and deferred checks.
9. Update only this task in `WORK_STATE.json`. Use `tool/codex/task_state.py`; mark complete only after
   every locally executable acceptance item passes. Mark blocked with one concrete reason and needed
   input when progress cannot continue.

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
