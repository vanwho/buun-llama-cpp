# 107-02 natural A/B/A measurement

Candidate: `6e438ffbdb919b42e61e057f027b0bd10473f3443e6789c23453e7aba02fd881`  
Model: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`  
Source commit: `d4ab69d3ffc2423d4d788a0b01c9a39235d4a728`

| Geometry | Status | C before A2 | Fill turns | Target outcome | Cold/H1/H2 evidence |
|---|---|---:|---:|---|---|
| 8K-4K-hot | completed | 5501 | 0 | natural_target_promotion | cold 2447.4 ms; H1/H2 target_hot_mixed_control, target_hot_mixed_control |
| 16K-8K-hot | resident_control | 14865 | 3 | target source pages remained resident or lacked valid host-backed cold page | no cold A2/repeat timing (resident control) |
| 32K-16K-hot | resident_control | 31234 | 8 | target source pages remained resident or lacked valid host-backed cold page | no cold A2/repeat timing (resident control) |
| 64K-32K-hot | resident_control | 64008 | 16 | target source pages remained resident or lacked valid host-backed cold page | no cold A2/repeat timing (resident control) |
| original-profile | resident_control | 76280 | 19 | target source pages remained resident or lacked valid host-backed cold page | no cold A2/repeat timing (resident control) |
| 128K-64K-hot | resident_control | 129533 | 33 | target source pages remained resident or lacked valid host-backed cold page | no cold A2/repeat timing (resident control) |
| 192K-64K-hot | resident_control | 195092 | 50 | target source pages remained resident or lacked valid host-backed cold page | no cold A2/repeat timing (resident control) |
| 256K-64K-hot | resident_control | 260602 | 68 | target source pages remained resident or lacked valid host-backed cold page | no cold A2/repeat timing (resident control) |

## 8K cold and repeat result

The source page transitioned cold→resident and was selected; request-local useful H2D was 34603008 bytes. Semantic recall: `pass`. The singleton natural witness identifies logical page `4` and the target page is `4`; witness evidence and the observed transition are recorded independently.
Cold request: HTTP wall 2447.379 ms, prompt 638.308 ms, decode 1801.59 ms. H1 and H2 resend the exact A2 bytes and are both mixed controls because secondary pages moved: 2281.403 ms and 2275.135 ms.

Seven larger geometries remained resident after one extra natural pressure request. They are recorded as `resident_control`; they have no cold A2/H1/H2 timing, semantic recall outcome, or promotion-cost estimate. No promotion was inferred from another page’s witness.

## Raw artifacts

- Report JSON: `/srv/ai/paged-kv/results/forward/107-natural-aba/107-02/attempt-01/ACO107_02_NATURAL_ABA.json` (SHA-256 `881694c0ef718478c251520364ecd61ff359cb4be5a50832978ba52e820df6e7`).
- Report Markdown: `/srv/ai/paged-kv/results/forward/107-natural-aba/107-02/attempt-01/ACO107_02_NATURAL_ABA.md` (SHA-256 `aa9e519b14794741a5c37733ae158d008eb2aad4195a089cc4f11feca69b8dda`).
- Report validator output: `/srv/ai/paged-kv/results/forward/107-natural-aba/107-02/attempt-01/report-validator.stdout` (SHA-256 `4a6c7ef9cf49a4dad2cde94ae95ccd75500fdbf534a173aba031640486349caa`).
- Frozen inputs: `/srv/ai/paged-kv/results/forward/107-natural-aba/107-02/attempt-01/FROZEN_INPUTS.json` (SHA-256 `a83b361b564eda8d48f5b482f3ede2fb80a582b7405ac6161fc2e20aabc2a26f`).
- Managed service restored to the original 256K profile; health passed; ports 8091 and 8092 were preserved.
