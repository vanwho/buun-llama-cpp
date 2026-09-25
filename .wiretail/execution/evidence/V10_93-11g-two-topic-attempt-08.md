# 93-11g attempt 08 — natural two-topic recall

## Result and current state

The exact three-request candidate-bound sequence completed under the managed Qwen candidate with the required 16,384 context, 4,096 hot-token capacity, 256-token pages, B=128/U=64, one slot, selective Turbo4 target KV, and GPU Turbo4 native MTP. All three candidate-rendered cumulative preflights fit with a 128-token completion reserve (5,317 / 10,650 / 10,712 tokens). Ten selected fixture hashes, no-BOS tokenizer counts, Python AST checks, and Bash syntax checks passed.

Filename scoring: request 1 **fail** (`merge_sorted_lists_05.py`); request 2 **fail** (verbatim answer `ะПРЯ БЫ\n\n`); request 3 **fail** (explanatory prose without a filename). These outcomes did not stop later turns.

The final `RETRIEVAL_KEY` fact spans fixture bytes `[3852,3931)` and candidate-tokenized prompt tokens `[5245,5261)`, overlapping logical page 20. It was resident after request 1. Before request 3, the same sequence/page/generation (sequence 0/generation 1, logical page 20/generation 22) was cold and host-backed at content version 22. Request 3 produced no natural selector nomination: selector proof remained the invalid-page sentinel, with no H2D, publication, target use, or draft use. The answer-bearing page was therefore **not promoted**; the full fixture was not hot after request 3. Request-level GPU Turbo4 MTP placement was verified, but it does not prove page-specific use.

Attempt 07's resident IDs 8–15 and attempt 08's boundary snapshots are recorded. The candidate snapshot exposes `pin_recent_tokens=0` and residency/host backing, but does not expose per-page current/pin counts, routing-summary readiness/version, or live-policy eligibility/decision reason. Therefore the reason those early pages remained resident cannot be determined from these traces; no attention-selection or pinning claim is inferred from residency.

93-11g remains `in_progress`; 93-12 remains gated. The attempt-08 receipt is diagnostic and intentionally does not satisfy the natural-promotion gate. Historical attempt receipts are unchanged.

## Decisions and invariants

- Preserve the Python fixture order `01, 02, 04, 05, 03`, then Bash `01..05`, then the exact repeated Python question with actual prior replies.
- Keep request answers separate from page movement. No forced eviction, selection, route, or promotion was used.
- Page correlation across snapshots uses sequence/page/generation plus overlap with the original token span; content version and bounds are refreshed from each snapshot.
- No attention/KV/MTP source was changed. Current/page policy telemetry is absent from the loaded candidate snapshot.

## Changed files and relevant symbols

- `tools/server/bench/pager_promotion.py`: fixed the CLI's live target description/options and refreshed logical-page identity across partial-to-complete bounds.
- `tools/server/bench/run-pager-promotion.py`: final-fact prefix tokenization, byte offsets, request raw-response reference, preflight totals, request-1 residency check, and page identity correlation across expanded page bounds.
- `.wiretail/execution/v10/validate.py` and `test_validate.py`: enforce the corrected fixture order/content, final answer-bearing fact, request metadata, and cold/host-backed precondition.
- `tools/server/bench/fixtures/pager-promotion/README.md`: document the Python fixture order.

## Validation commands/results and raw artifact paths

- `python3 -m py_compile tools/server/bench/pager_promotion.py tools/server/bench/run-pager-promotion.py tools/server/bench/prompt_sizing.py .wiretail/execution/v10/validate.py .wiretail/execution/v10/test_validate.py` — pass.
- `python3 tools/server/bench/test_pager_promotion.py` — pass, 6 tests, including partial-tail page-bound refresh.
- `python3 .wiretail/execution/v10/test_validate.py` — pass, 34 tests.
- `git diff --check` — pass.
- `python3 .wiretail/execution/v10/validate.py --task 93-11g --receipt .wiretail/execution/evidence/V10_93-11g-two-topic-attempt-08.json` — exit 1 as required for incomplete natural-promotion proof; no artifact/hash/geometry/content errors. It reports missing proof keys plus no answer-bearing promotion chain.
- Focused logs: `.wiretail/build/93-11g-attempt-1-20260925/`.
- Candidate-bound raw root: `/srv/ai/paged-kv/results/v10/93-11g/attempt-08/`; exact requests, responses, rendered prompts/token IDs, per-request snapshots and placement, and lifecycle identity are under `campaign/`.
- Page reconciliation: `/srv/ai/paged-kv/results/v10/93-11g/attempt-08/campaign/cases/PY_MERGE_03/reconciled-page-analysis.json`.
- Lifecycle and fixture integrity: `/srv/ai/paged-kv/results/v10/93-11g/attempt-08/managed-lifecycle.json` and `fixture-integrity.json`.

## Next concrete action

Keep this task in progress and 93-12 gated. The next task-local recovery must add the missing bounded per-page diagnostic fields at the existing once-per-request snapshot seam (current/pin state, routing-summary readiness/content version, and live-policy eligibility/reason) before making a retention or selector-defect claim. Do not repeat this conversation or add another live request sequence within this attempt.

## Deferred verification

Natural physical promotion remains unproven because request 3 produced no selector nomination or later transition. Page-specific current/pin and live-policy retention diagnostics are unavailable in the candidate snapshot, so attempt 07's early-page retention cause remains undetermined. Answer correctness also failed on all three turns. These are recorded as incomplete evidence, not a pass.
