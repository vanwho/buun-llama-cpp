# 93-11g two-topic live diagnostic

## Result

All three candidate-bound requests completed under the required geometry. The campaign is diagnostic only because the answer-bearing Python pages were not naturally evicted and promoted. Filename selection and physical promotion are recorded independently.

## Answers

- Request 1 (compare_python): `merge_sorted_lists.py`; intended filename matched: **false**.
- Request 2 (compare_bash): ```` / { /     " /     filename": /     " /     " / ``` / ````; intended filename matched: **false**.
- Request 3 (repeat_python): ```` / merge_sorted_lists.py / ````; intended filename matched: **false**.

## Page finding

The complete `PY_MERGE_03` body spans rendered tokens [2131, 3156]. Its answer-bearing source line spans [2131, 2352] and overlaps logical pages 8, 9. Both pages remained resident and host-backed after requests 1 and 2 and immediately before request 3. Therefore the first missing transition is **cold before request 3**; natural selector nomination, H2D, publication, target use, and draft use did not occur. The whole tracked fixture was resident after request 3, but it was not promoted.

## Runtime and artifacts

Candidate SHA-256: `36aea000698c168a3017317a789ae36eb56becb8b5fa5910f243b5dd28fa0bfb`. Model SHA-256: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`. Full requests, responses, rendered prompts, tokenizer outputs, slot snapshots, page inventories, and raw hashes are under `/srv/ai/paged-kv/results/v10/93-11g/attempt-04/campaign/`. The campaign driver log is `/srv/ai/paged-kv/results/v10/93-11g/attempt-04/campaign-driver.log`.

Focused checks passed: planner 6 tests, receipt validator 34 tests, Python
compilation, fixture manifest verification, and `git diff --check`. The
versioned receipt validation command returned exit 1. It reported the incomplete
natural-promotion proof and promotion acceptance, plus stale packet/cluster
revision markers in the broader task plan. No receipt artifact hash errors were
reported. The full output is in
`.wiretail/build/93-11g-attempt-1-20260924/validate-two-topic.log`.

## Acceptance

The required cold → selector → H2D → publication → target → draft chain is incomplete. Keep 93-11g in progress and 93-12 gated. Do not treat the answer quality or whole-fixture hot residency as promotion proof.
