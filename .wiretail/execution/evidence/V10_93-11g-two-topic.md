# V10 93-11g two-topic natural recall and promotion — attempt 07

## Result

The pinned candidate completed all three same-slot HTTP requests after the exact 16,384-context / 4,096-hot-token preflight. All placeholder cumulative prompts and the actual final cumulative prompt fit with the 128-token completion reserve. Actual request-3 preflight: 10,791 tokens, leaving 5,465 completion tokens. The request sequence and raw responses, prompts, token IDs, slot snapshots, page events, and lifecycle identity are retained under `/srv/ai/paged-kv/results/v10/93-11g/attempt-07/campaign/`.

Filename scoring is separate from page movement:

- Request 1: `merge_sorted_lists.py` — **incorrect**; expected `merge_sorted_lists_03.py`.
- Request 2: malformed explanatory answer, no filename — **incorrect**; expected `watch_directory_new_files_01.sh`.
- Request 3: `merge_sorted_lists_01.py` — **incorrect**; expected `merge_sorted_lists_03.py`.

## Page and MTP findings

Candidate-rendered request 1 maps the full `PY_MERGE_03` body to tokens `[2131, 3156)` and its answer-bearing source line to `[2131, 2352)`. The fixture body byte span is `[0, 3932)`. The mapped pages are logical pages 8–12, generations 10–14, content version 256. Answer-bearing pages are 8 and 9.

Immediately before request 3, all five mapped pages remained present, resident, and host-backed. The answer-bearing pages were therefore **not cold** before the repeat question. The selector nominated no tracked page: `natural_proof.logical_page` was the invalid sentinel `4294967295`, and selector/H2D/publication/target-page-use/draft-page-use flags were all false. The answer-bearing page was not promoted; the first missing transition is cold-before-request-3 because it remained resident and host-backed. Since the fixture was already hot before request 3, `whole_fixture_hot_after_request_3=true` does not mean it returned from host backing.

All three responses report GPU Turbo4 target K/V and GPU Turbo4 MTP placement. Request-level MTP counters were 4/4 accepted draft tokens (request 1), 60/60 (request 2), and 6/6 (request 3). Request 3 completed with HTTP 200 and natural finish. These are request-level MTP observations; no tracked Python page was selected or consumed as a promotion.

## Runtime identity and geometry

Managed lifecycle reload used `sudo -n` and `/srv/ai/scripts/activate-ai-profile.sh qwen38-fast`. Verified PID/start ticks are `50225/187483`; candidate SHA-256 is `36aea000698c168a3017317a789ae36eb56becb8b5fa5910f243b5dd28fa0bfb`; model SHA-256 is `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`. Runtime admitted context 16,384, 16 hot pages/4,096 tokens, 256-token pages, B128/U64, one slot, selective Turbo4 target K/V, and GPU Turbo4 MTP.

## Validation and raw artifacts

- Selected ten fixture hashes, 1,024 no-BOS token counts (trailing newline excluded as in the generator), and Python/Bash syntax: pass. Log: `/srv/ai/paged-kv/results/v10/93-11g/attempt-07/fixture-integrity.log`.
- `python3 tools/server/bench/test_pager_promotion.py`: pass, 6 tests.
- `python3 .wiretail/execution/v10/test_validate.py`: pass, 34 tests.
- Receipt validation command exits 1 because physical promotion was not proven and existing packet/cluster revision markers are stale; artifact hashes and raw references validate. Log: `.wiretail/build/93-11g-attempt-2-20260925/validate-final.log`.
- `python3 -m py_compile tools/server/bench/pager_promotion.py tools/server/bench/run-pager-promotion.py tools/server/bench/prompt_sizing.py`: pass.
- Full campaign driver output: `/srv/ai/paged-kv/results/v10/93-11g/attempt-07/campaign-driver.log`.
- Per-request records, responses, rendered prompts, and token counts are in the corresponding `request-01`, `request-02`, and `request-03` directories. Page snapshots and correlated identities are recorded in `cases/PY_MERGE_03/*tracked-pages.json` and the JSON receipt.

## Disposition

Execution completed, but physical promotion acceptance is incomplete: the winning Python pages were never cold before request 3 and no natural selector/H2D/publication/use chain occurred. Keep 93-11g `in_progress` and keep 93-12 gated. Do not infer promotion from runtime placement, MTP token counters, or whole-fixture residency after request 3.
