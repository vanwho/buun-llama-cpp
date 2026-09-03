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

## 2026-09-03T06:26:17+00:00 — 02-01 — done

- Branch: `codex/task-02-01`
- Commit at update: `0ffec87e3`
- Summary: Added bounded 16-layer Turbo4 selected-page capture descriptors with exact preflight quotes, pinned-ring segmented transfers, generation/representation rechecks, and fake-provider fault coverage.

## 2026-09-03T06:43:51+00:00 — 02-02 — done

- Branch: `codex/task-02-02`
- Commit at update: `a687880e2`
- Summary: Added authenticated canonical Turbo4 selected-page host catalog with pageable/pinned accounting, budget admission, aliasing, invalidation, and deterministic tests.

## 2026-09-03T07:10:58+00:00 — 02-03 — done

- Branch: `codex/task-02-03`
- Commit at update: `53d1f37c1`
- Summary: Added bounded KV residency transfer plans and separate backend pool with coalesced D2H/H2D runs, pre-submit slot/event/catalog admission, async cancellation-safe VBR seams, generation rechecks, clean eviction, fixed counters, and fake-backend failure tests.

## 2026-09-03T07:28:11+00:00 — 02-04 — done

- Branch: `codex/task-02-04`
- Commit at update: `b50a11691`
- Summary: Added failure-atomic KV residency transactions with explicit snapshot/plan/reserve/pin/reseal/drop/load/fence/recheck/publish ordering, reversible victim rollback, generation and epoch rejection, dirty/all-pinned/source failure handling, and deterministic phase fault tests.

## 2026-09-03T07:38:32+00:00 — 02-05 — done

- Branch: `codex/task-02-05`
- Commit at update: `a4f869753`
- Summary: Added derived fixed/manual 304-page residency window with exact Turbo4 geometry, scalar host/device/staging ledger, partial-tail seal/reseal, checksum validation, mutation/promotion/clean-eviction proof, and deterministic tests.

## 2026-09-03T07:42:58+00:00 — 03-01 — done

- Summary: Added immutable compact selected-cache attention view with bounded physical row dimensions, native logical positions/masks, source-slot mapping, graph-fence snapshot lifetime, tail support, and deterministic permutation/gap/rejection tests.

## 2026-09-03T07:50:02+00:00 — 03-02 — done

- Summary: Added backend-neutral selected-page operator metadata with Turbo4/GQA/causal/batch contracts, typed CPU/unsupported-backend capability results, table-epoch graph reuse fencing, and deterministic validation tests.

## 2026-09-03T08:56:50+00:00 — 03-03 — done

- Summary: Added direct selected-page CUDA Turbo4 decode with validated physical-page addressing, native masks/causal tails, bounded online logical-page mass reduction, fail-closed geometry/type checks, deterministic CUDA coverage, timing/resource capture, and compute-sanitizer verification.

## 2026-09-03T09:19:01+00:00 — 03-04 — done

- Summary: Added the selected-attention execution boundary for multi-chunk prefill/tail admission, decode transition, reference/direct route selection, observe/off/refusal logging, resident-plus-transfer/router scratch sizing, selected-content graph keys, and scheduler-completion page fences; focused lifecycle, graph, residency, and prior CUDA primitive checks passed.
