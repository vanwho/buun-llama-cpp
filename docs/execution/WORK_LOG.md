# Work log

Append-only task transitions are maintained by `tool/codex/task_state.py`.


## 2026-09-03T02:10:52+00:00 — 00-01 — todo

- Branch: `plan/attention-aware-kv-paging`
- Commit at update: `cb703be37`
- Summary: User authorized experimental iteration and commits in vanwho forks; upstream actions remain deferred.

## 2026-09-03T02:10:52+00:00 — 00-01 — in_progress

- Branch: `plan/attention-aware-kv-paging`
- Commit at update: `cb703be37`
- Summary: Task started

## 2026-09-03T02:10:52+00:00 — 00-01 — done

- Branch: `plan/attention-aware-kv-paging`
- Commit at update: `cb703be37`
- Summary: Recorded experimental-fork-only direction and existing upstream references.

## 2026-09-03T04:08:35+00:00 — 00-03 — blocked

- Branch: `codex/task-00-03`
- Commit at update: `1bebf618f`
- Summary: Controlled profile restore failed because required health endpoint 127.0.0.1:8091 was unavailable; Qwen 8080 is healthy but the harness restore gate cannot be certified.

## 2026-09-03T04:40:01+00:00 — 00-03 — todo

- Branch: `codex/task-00-03`
- Commit at update: `07cf19764`
- Summary: Automatic blocker-recovery attempt 1/2; previous blocker: Controlled profile restore failed because required health endpoint 127.0.0.1:8091 was unavailable; Qwen 8080 is healthy but the harness restore gate cannot be certified.

## 2026-09-03T04:52:25+00:00 — 00-03 — done

- Branch: `codex/task-00-03`
- Commit at update: `ad8ebd1fe`
- Summary: Captured Fast default/large, full-context ordinary CPU-KV, and Big spec-off controls; manifests and summaries verified, final qwen38-big and 8080/8091 health restored.

## 2026-09-03T04:54:38+00:00 — 00-04 — done

- Branch: `codex/task-00-04`
- Commit at update: `9e4e10100`
- Summary: Added and validated the prototype/community salvage matrix with all required rows, explicit VBR-page boundary, and no reference worktree changes.

## 2026-09-03T05:57:57+00:00 — 01-01 — done

- Branch: `codex/task-01-01`
- Commit at update: `175ec305f`
- Summary: Froze selective-attention semantics, retrieval/retention boundary, lifecycle ownership, fail-closed scope, and telemetry contract in design/SEMANTICS.md; validation and diff checks passed.

## 2026-09-03T06:01:45+00:00 — 01-02 — in_progress

- Branch: `codex/task-01-02`
- Commit at update: `65180956d`
- Summary: Task started

## 2026-09-03T06:02:01+00:00 — 01-02 — done

- Branch: `codex/task-01-02`
- Commit at update: `65180956d`
- Summary: Added and tested CPU-only llama_kv page identity, checked forward/reverse residency uniqueness, immutable snapshots, and generation-checked publish/rollback.
