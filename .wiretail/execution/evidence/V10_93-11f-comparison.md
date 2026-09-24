# 93-11f native MTP geometry comparison

Candidate SHA-256: `34380fb3a12c7fe3172eb9933179093ea3f03c0d9153a7bee33a30461f54763e`  
Model SHA-256: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`  
Context 8192; target hot limit 4096; selected pager; target/draft GPU Turbo4; reasoning off. Each cell uses three measured 400-token requests after one 40-token warmup. Prompt token counts are from the candidate server exact-rendered input-token preflight.

| Prompt | Exact input tokens | Geometry | Prompt tok/s min / median / max | Decode tok/s min / median / max | TTFT proxy min / median / max (s) | MTP acceptance min / median / max | Gate |
|---|---:|---|---:|---:|---:|---:|---|
| prompt_1 | 30 | primary_1024_256 | 66.90 / 83.67 / 102.67 | 75.84 / 76.12 / 76.49 | 0.039 / 0.048 / 0.060 | 79.41% / 79.41% / 79.41% | pass
| prompt_2 | 27 | primary_1024_256 | 88.60 / 104.97 / 105.78 | 59.00 / 59.12 / 59.12 | 0.038 / 0.038 / 0.045 | 54.03% / 54.03% / 54.03% | pass
| prompt_3 | 28 | primary_1024_256 | 72.95 / 82.88 / 104.62 | 63.30 / 63.50 / 63.68 | 0.038 / 0.048 / 0.055 | 62.94% / 62.94% / 62.94% | PASS (reassessed at 60%)
| prompt_1 | 30 | secondary_512_128 | 66.97 / 83.13 / 101.08 | 76.27 / 76.32 / 76.54 | 0.040 / 0.048 / 0.060 | 79.41% / 79.41% / 79.41% | pass
| prompt_2 | 27 | secondary_512_128 | 88.31 / 104.92 / 104.93 | 59.09 / 59.31 / 59.31 | 0.038 / 0.038 / 0.045 | 54.03% / 54.03% / 54.03% | pass
| prompt_3 | 28 | secondary_512_128 | 74.65 / 82.35 / 105.09 | 63.32 / 63.47 / 63.62 | 0.038 / 0.049 / 0.054 | 62.94% / 62.94% / 62.94% | PASS (reassessed at 60%)

## Secondary-to-primary speed ratio (512/128 ÷ 1024/256)

| Prompt | Prompt tok/s | Decode tok/s |
|---|---:|---:|
| prompt_1 | 0.994× | 1.003× |
| prompt_2 | 1.000× | 1.003× |
| prompt_3 | 0.994× | 1.000× |

## Acceptance diagnosis

Current required median acceptance floors: prompt 1 75%, prompt 2 40%, prompt 3 60%. All three prompts meet the current floor in both geometries; prompt 3 selected medians are 62.94% in both geometries.

| Dense all-GPU control | Prompt-3 median acceptance | Result |
|---|---:|---|
| 1024/256 | 63.75% | meets 60% |
| 512/128 | 63.75% | meets 60% |

Both same-candidate dense controls also meet the current prompt-3 floor (63.75%). The former 70% policy would have triggered investigation, but the revised 60% floor is satisfied in selected and dense modes. No MTP source defect is established by this acceptance result; the original request data and hashes remain unchanged. The original benchmark command returned exit 1 under the superseded 70% threshold. The current-policy receipt validator now passes; the original request data and hashes remain unchanged.

## Runtime and transfer evidence

The live service command used the same immutable candidate and model in both rows. Both request sets recorded route `selected dense`, target/draft GPU Turbo4, and zero H2D/D2H useful bytes; the short prompts remained resident, so no host transfer occurred. Global GPU free memory was at least 1822 MiB in the primary post-run sample and across all secondary in-run samples; maximum pager scratch high-water was 8650752 bytes. The pager allocator headroom value is a separate reservation metric (192 MiB on primary snapshots), not device-wide free memory.

TTFT is represented by the server-reported `prompt_ms` prefill duration because the canonical runner did not preserve first-stream-chunk timestamps. Per-request source pointers and hashes are in the receipt and raw artifact manifest.

## Raw roots and hashes

- Primary: `/srv/ai/paged-kv/results/v10/93-11f/attempt-01/primary-1024-256-retry-01`
- Secondary: `/srv/ai/paged-kv/results/v10/93-11f/attempt-01/secondary-512-128`
- Dense controls: `/srv/ai/paged-kv/results/v10/93-11f/attempt-01/dense-control-prompt3-1024-256-correct` and `.../dense-control-prompt3-512-128`
- Complete file SHA-256 inventory: `/srv/ai/paged-kv/results/v10/93-11f/attempt-01/raw-artifact-manifest.json`
- Wrong case-index control attempt is preserved and explicitly excluded in receipt diagnosis.


## Supplementary attempt-04 diagnosis

A new immutable candidate (SHA-256 `9418466c46e6139df39487700abac870c75a26a68ab9c56fd23869ad6cacff04`) was used for one 40-token warmup and one 400-token measured dense prompt-3 request at each geometry. Both requests recorded 71.52% acceptance. The bounded diagnostic observed the first raw-logit greedy mismatch at step 7, draft position 1 (proposal token 23791; target top-1 token 7650) in both runs. This confirms a proposal/target difference but does not identify a source defect; it is supplemental and does not replace or alter the complete attempt-01 matrix. Trace receipt: `.wiretail/execution/evidence/V10_93-11f-attempt-04-trace.json` (SHA-256 `2cde387ab075d8b9f5d740f6e7fd6dfa5ce0ef96ee4fd055a999ee053c1a5d33`).
