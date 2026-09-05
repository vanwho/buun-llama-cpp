# Task packet contract

Each packet is the complete task-specific context for one bounded task. Its shared session context is
`.wiretail/execution/clusters/<task.cluster>.md`; read that file once per fresh cluster session and reuse it
for consecutive tasks in the cluster.

Global rules for every packet:

1. Work primarily in `/srv/repos/vanwho/buun-llama-cpp`. A live implementation/benchmark packet may also explicitly
   authorize server-specific benchmark corpus/config/script edits under `/srv/ai/benchmarks` and raw
   outputs under `/srv/ai/paged-kv/results`; inspect that repository's instructions and dirty state
   first, preserve unrelated work, and never put those machine paths in production source.
2. Re-read repository `CONTRIBUTING.md` and any applicable `AGENTS.md` before editing. The current
   Buun tree has no `AGENTS.md`; do not assume that remains true after an upstream sync.
3. Preserve unrelated dirty work. Never alter or clean `/srv/ai/paged-kv/repos/buun-llama-cpp`.
4. Do not create GitHub issues/PRs, write their descriptions/replies, push, merge, or invent human
   approval. The user permits iterative commits in `vanwho/*`; manual Git mode applies only when
   explicitly configured. The current outer runner uses auto mode and owns fork
   commits/pushes/merges. Task agents must not race that Git owner or publish upstream prose.
5. Prefer internal APIs and existing source/test files. Extend existing regression tests first;
   add a focused new test only when justified by the actual contribution rules and explain why.
6. Never claim a CUDA, benchmark, quality, or hardware result without raw output. Missing reference
   hardware blocks a hardware acceptance task; it does not block pure/fake-backend work.
7. Keep feature-disabled behavior unchanged and make unsupported configurations fail closed.
8. Keep implementation portable: never hard-code `/srv/ai`, `/srv/repos`, `/home/ninja`, local service
   units/PIDs/ports, absolute model/profile filenames, or RTX 4080 assumptions. Qwen3.8 geometry belongs
   in runtime capability checks and test fixtures; server profiles, lifecycle commands, and benchmark
   artifacts stay in execution metadata or the external `/srv/ai` repository.
9. Read existing handoffs for every `depends_on` task before implementation. Create/update
   `.wiretail/execution/handoffs/<task-id>.md` with scope, changed files, commands/results,
   invariants checked, raw artifact paths, risks, and deferred checks.
10. Update only this task in `WORK_STATE.json`. Use `/srv/wiretail/task_state.py`; mark complete only after
   every locally executable acceptance item passes. Mark blocked with one concrete reason and needed
   input only after distinct recovery paths have been tried. The runner automatically reopens a blocked
   task three times after its initial approach, using fresh sessions as needed; the transient recovery
   budget resets when a new runner invocation resumes the project. Retry 1 only adds the direct
   artifact-aware prefix to the original task model/reasoning. Before retry 2, the task's own model family
   runs a High-reasoning assessment; retry 2 uses the original task model/reasoning with no retry prefix.
   Before retry 3, the next-tier/high assessment uses `luna -> terra -> sol` (Sol remains the ceiling); retry
   3 also uses the original task model/reasoning with no retry prefix. Do not treat the first failed command
   as a sufficient blocker.
11. For all live phases, native MTP rows must equal the resolved target context and remain Turbo4/GPU;
    the trained model context is not an allocation floor. No production default may encode a fixed hot
    token/page count. Historical benchmark artifacts may retain their measured configuration names.
12. After 17-15, use POST17_IMPLEMENTATION_STRATEGY.md, the packet's TECHNICAL_CHANGE_SPEC.md sections,
    and BENCHMARK_PROTOCOL_V5.md instead of historical acceptance ledgers. 17-16 bridges the evidence;
    phases 18–20 implement repairs; phase 21 measures; 22-01 (Sol High) reviews only that compact summary
    and creates another measured remediation/review chain if needed. Implementation must demonstrate
    its live behavior, not merely report an unsupported callback. Evidence tasks may honestly finish
    with negative findings, but cannot promote them as overall-goal success.
13. Live benchmark tasks may stop/restart the active Qwen service using passwordless sudo and the
    established profile scripts. Capture the starting profile and verify ports 8080 and 8091. A
    successful benchmark keeps its tested server/profile loaded for the next task or retry by default;
    restore only when the packet explicitly requests a control/revert benchmark, teardown, failed-start
    recovery, or final cleanup, and record the lifecycle decision. Never stop or reconfigure the unrelated
    service on port 8092.
14. Performance tasks preserve raw before/after results for every attempted optimization. The historical
    3x/5x/70% speed numbers are findings, not gates. Investigate measured bottlenecks with distinct
    hypotheses; do not block solely on a throughput threshold or silently change the three prompts.
    The six 20K/40K/60K/100K/175K/256K coordinates are for the final results campaign only, after
    functionality works. Other tests choose the smallest appropriate context/occupancy/hot budget
    for their own purpose; no task must replay that entire ladder as a routine gate.

15. If Codex reports `codex_core::tools::router` or `apply_patch verification failed`, treat it as a
    patch-anchor/context failure. Re-read the current file and exact surrounding lines with `rg`/`sed`,
    check whether the intended change already exists, and apply one small patch against the current
    content. Never repeat the same stale patch, blindly append another handoff note, or spend the turn
    retrying an anchor that is no longer present. The runner records this signal, rotates the session
    before the next substantive attempt, and injects the same recovery direction into the retry and
    High-reasoning assessment prompts.

16. Every post-17 packet has a concrete recipe. Read its named sections of
    `EXECUTION_COOKBOOK.md` and `IMPLEMENTATION_CONTRACTS.md` once per cluster.
    Use the prescribed minimal fixture/checkpoint before a live campaign;
    export actual new symbols, layout/domain/ownership contracts and tested
    commands in the handoff. A proposed CLI/API is work to implement, not an
    already supported option. The shared model-parity driver is owned by 18-05;
    later parity tasks extend it rather than invent another diagnostic loop.
    Implementation receipts must pass their named live checks; truthful negative
    benchmark receipts are allowed only where the evidence packet says so.

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
