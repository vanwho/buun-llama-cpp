# 18-03 Resume Evidence

The long-experiment boundary is resumable and progress-aware. Shared case
identity/state code is used by the quality and soak runners; the site adapter
forwards the same resume and deadline contract to the canonical shell runner.

## Verification

- Focused Python regression suite: 45 tests passed.
- `py_compile`, `bash -n`, and `git diff --check`: passed.
- Local fake HTTP coverage includes 501 capability refusal, 400 sizing error,
  delayed progress, client disconnect, and mixed-token SSE timing.
- Quality live manifest `/srv/ai/paged-kv/results/18-03-live-resume-20260905T0720Z/`:
  three unique completed cases, one interrupted attempt, zero duplicate
  completed rows, and completed progress with ETA 0.
- Canonical live manifest `/srv/ai/paged-kv/results/18-03-site-resume-final-20260905T071106Z/`:
  one completed case; resume kept records at 1, produced no duplicate request,
  and passed lifecycle validation with `resume_usable=true`.
- `llama-server.service` ended active/running, health HTTP 200, Result=success,
  NRestarts=0. The unrelated service on port 8092 was not touched.

## Deferred verification

The full 22016-token quality matrix and final speed curves remain deferred;
this task’s live gate is durable interruption/resume behavior, not semantic
quality or throughput acceptance. No human upstream action is required.

## Next action

Proceed to task 18-04 using the shared case manifest/state and evidence
validator contracts.
