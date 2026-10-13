# Successful natural promotion reference for 107-02

Revision: natural-aba-promotion-timing-20261013.

The 2026-10-13 manual run reproduced natural promotion with the current
build-aco-latest-155fc9816 candidate at L8192/H4096/B1024/U256, Turbo4 target
and GPU Turbo4 MTP2. Source A: PY_MERGE_01 and PY_MERGE_03; B: BASH_WATCH_02,
03 and 04; A again asks the existing PY_MERGE_03 write-cursor question.
Actual earlier assistant replies are kept in the cumulative conversation.
After the final A request, repeat its exact HTTP body without adding its answer.

The answer-bearing page was logical page5 at native positions1280..1536,
generation9/content256: cold and host-backed before A2, resident after A2 and
selected in its target graph. These coordinates are observations, not injected
selection IDs: remap the source anchor using the current server tokenizer.
The singleton natural_proof named page4, another promoted page in the batch;
its full event chain must not be falsely attributed to page5. A2 added
34,603,008 useful H2D bytes and answered the source question correctly.

Final manual reference: /srv/ai/paged-kv/results/forward/107-cold-promotion-cost/manual-8k-20261013T212237Z/RESULT.md
and RESULT.json. Read these compact files only if needed; do not ingest raw
snapshots, full source text, SSE or prior Codex logs.

| Request | HTTP wall ms | Server prompt ms | Output | Decode tok/s | MTP |
|---|---:|---:|---:|---:|---:|
| Cold A2 |2458.5|656.0|131|72.44|73/116 =62.93%|
| Target-hot A4 |2356.5|555.0|131|72.48|73/116 =62.93%|

The body was identical; usage cached counts5501 vs5503 meant68 vs66 new
input tokens. The repeat also moved4,325,376 bytes for another page, so this
was a target-hot mixed comparison, not a pure promotion-cost oracle. A3/A4
alternated secondary pages17/13. Bound repeats at two and report this behavior;
do not run an endless settling loop or repair production ranking in this task.

The manual helper scripts/run_manual_8k_promotion.py preserves lifecycle,
timing and exact-body repetition. Generalize the measurements in a new helper;
its fixed baseline L262144/H65536 is not a universal restore assumption.
Its old run_case classification is not authoritative: it wrongly calls a
transition missing if the one stored witness names another promoted page.
The real source page transition plus actual selected IDs/H2D evidence are
available. Keep physical transition, semantic answer and full witness-chain
attribution as distinct fields.

Every comparison must exclude all107-01 artifacts, as registered in
evidence/RESULT_EXCLUSIONS.json. result_eligibility.py checks this without
reading the historical229MB report. Usage and raw files remain preserved.
