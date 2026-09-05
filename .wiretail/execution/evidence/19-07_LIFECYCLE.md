# 19-07 lifecycle evidence

Classification: local implementation and deterministic regression verification
passed; the named model-backed pressure/native-MTP checkpoint is deferred because
the shared RTX 4080 had 14,815 MiB used and 1,128 MiB free.

The lifecycle boundary now rejects incomplete page identities and frontier
regressions, validates policy versions, rolls back companion publication, and
invalidates the old lifecycle on a rejected destructive transition. VBR
terminal commit is reported through the memory boundary and runs before pager
host sealing and live table policy publication; a failed commit suppresses
dependent publication. Clear/remove/copy/keep/add state paths reserve their VBR
operation before pager mutation, while unsupported cross-stream copies and
invalid keep IDs fail closed with diagnostics.

## Verification

| Command | Result |
|---|---|
| `cmake --build build --target test-kv-policy -j2` | pass |
| `./build/bin/test-kv-policy` | pass; lifecycle trace: 4 submitted, 1 completed, 3 cancelled, 32 useful/40 aligned bytes, overlap observed |
| `cmake --build build --target test-kv-pager test-kv-residency test-kv-attention-execution test-kv-attention-exact test-server-prompt-cache test-recurrent-state-rollback -j2` | pass |
| `ctest --test-dir build --output-on-failure -R 'test-(kv-(policy\|pager\|residency\|attention-(execution\|exact))\|server-(prompt-cache\|recurrent-expansion)\|recurrent-state-rollback\|mtp-vocab-trim)$'` | 10/10 pass |
| `ctest --test-dir build --output-on-failure -R '^test-server-prompt-cache-(clone-fault\|accounting-fault\|restore-checkpoint-fault)$'` | 3/3 pass |
| `git diff --check` | pass |

The first adjacent run used stale binaries and exposed ABI/fixture failures;
all affected targets were rebuilt and the rerun passed. The attempted broad
CUDA build was interrupted during unrelated template instantiation before
changed host sources were reached; the changed C++ sources compile in the
CPU production build.

## Deferred verification

After GPU isolation, run the runtime-supported single-slot Qwen forced-pressure
sequence covering first/middle/last speculative rejection, remove/clear/reuse,
queued cancellation, checkpoint save/restore, and full restart with native MTP
off/on. Record exact continuation/logit comparisons, committed page
generations/checksums, nonzero cold movement, and host/device/pinned/event/
descriptor high-water plus post-cycle plateaus. Do not disturb the existing
workload; retain its healthy tested state.

## Raw pointers

- Lifecycle identity/frontier and transition guards: `src/llama-kv-live-lifecycle.cpp:17`, `:150`, `:354`, `:422`.
- Commit fence before host/table publication: `src/llama-context.cpp:1883`.
- State-operation preflight and unsupported diagnostics: `src/llama-kv-cache.cpp:2934`, `:3054`, `:3223`, `:3377`, `:3475`.
- Deterministic lifecycle trace: `build/bin/test-kv-policy`.
