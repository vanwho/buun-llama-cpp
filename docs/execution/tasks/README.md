# Task packet contract

Each packet is the complete task-specific context for one bounded task. Its shared session context is
`docs/execution/clusters/<task.cluster>.md`; read that file once per fresh cluster session and reuse it
for consecutive tasks in the cluster.

Global rules for every packet:

1. Work primarily in `/srv/repos/vanwho/buun-llama-cpp`. A phase 08–14 packet may also explicitly
   authorize server-specific benchmark corpus/config/script edits under `/srv/ai/benchmarks` and raw
   outputs under `/srv/ai/paged-kv/results`; inspect that repository's instructions and dirty state
   first, preserve unrelated work, and never put those machine paths in production source.
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
11. For phases 08–15, native MTP rows must equal the resolved target context and remain Turbo4/GPU;
    the trained model context is not an allocation floor. No production default may encode a fixed hot
    token/page count. Historical benchmark artifacts may retain their measured configuration names.
12. Required model-backed phase 14 gates cannot be deferred. A failing result starts diagnosis and
    repair through the owning implementation task; it is not converted to a documentation success.
13. Live benchmark tasks may stop/restart the active Qwen service using passwordless sudo and the
    established profile scripts. Capture the starting profile, restore it, verify ports 8080 and 8091,
    and never stop or reconfigure the unrelated service on port 8092.
14. Performance tasks preserve raw before/after results for every attempted optimization. They test at
    least three evidence-driven repair hypotheses before blocking on the final 3x speed floor.

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
