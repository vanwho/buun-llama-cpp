# Phase 06 selective acceptance

This is the immutable foundation-series disposition. Phases 08–15 now own
production completion and must write `FINAL_ACCEPTANCE_V2.md/json`; this record
must not be mistaken for the final project decision.

## Decision

Selective capacity, quality, and speed acceptance is deferred. The RTX 4080,
Qwen3.8 model, canonical controls, and both profile health endpoints are
available, but the current server build does not expose a live 262,144-token
selective pager runtime or the required pager telemetry. No selective result or
release gate is claimed.

The machine-readable record is [`FINAL_ACCEPTANCE.json`](FINAL_ACCEPTANCE.json).

## Frozen comparison and raw controls

All runs use greedy sampling (`T=0`, seed 42) and the existing canonical result
contract. The valid controls are retained outside this repository:

- `/srv/ai/paged-kv/results/profile-benchmark-default-fast-20260903-055158/`:
  77,824 all-GPU Turbo4 target plus Turbo4 MTP.
- `/srv/ai/paged-kv/results/profile-benchmark-large-fast-20260903-055411/`:
  approximately 44K large scratch run with five measured trials.
- `/srv/ai/paged-kv/results/ordinary-cpu-kv-full-20260903/`:
  262,144 ordinary CPU-KV control with explicit `--no-kv-offload`.
- `/srv/ai/paged-kv/results/profile-benchmark-default-big-20260903-065012/`:
  canonical feature/spec-off rollback control.

The adapter's frozen corpus identity is `pager-corpus-v1`; expected needle
answers are not configured yet. The exact required-run and gate status is in
the JSON record.

## Gate disposition

| Gate | Status | Reason |
| --- | --- | --- |
| 262,144 logical target and near-304-page hot set | Deferred | No live pager allocation/ledger |
| Turbo4 host backing for cold pages | Deferred | Deterministic backing tests pass; model-backed run absent |
| Full-context Turbo4 MTP in VRAM | Deferred | Only the 77,824 Fast control has placement evidence |
| Bounded target GPU allocation | Deferred | No 262,144 pager allocation was run |
| Warm-focus speed and 3x stretch | Deferred | No selective trials |
| Needle/conversation quality | Deferred | Expected answers and selective outputs absent |
| Churn degradation disclosure | Deferred | No churn trials |
| Feature-off paired regression | Deferred | Existing controls are not a paired pager-off run |

## Deferred verification

The established profile workflow originally lacked its required 8091 health
service; `ai-long-memory.service` was restarted through systemd and now both
8080 and 8091 return HTTP 200. A live benchmark was not started because the
current server still lacks pager mode and telemetry, so running the canonical
profile harness would only produce another non-pager control.

06-05 must reuse `pager-corpus-v1`, its exact prompts, and the raw control paths
above when the model-backed selective runtime and expected answers are
available. Thresholds must not be retuned after selective results exist.
