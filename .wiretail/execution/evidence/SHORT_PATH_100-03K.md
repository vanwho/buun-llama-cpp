# 100-03k selected prompt-2 retake

Decision: **goal miss** against the unchanged 500 tok/s floor.

- Fresh prompt: 4326 tokens, 0 cached; one selected measured row: 212.122 tok/s, 20.394 s.
- Matched 100-03j row: 210.614 tok/s; single-row delta +0.716%.
- Direct attribution totals: prompt-batch pre-decode 0.185 s (render 0.000002 s); three checkpoint captures 0.161 s; prompt-marked llama_decode API wall 20.098 s over 9 calls. Separately exposed explicit wait totaled 4 us. API wall includes internal backend work and possible internal waits.
- MTP accepted/drafted: 68/126 (53.97%); adapter validation passed, canonical exit 0.
- Candidate binary SHA256 `87878b33bbd6fb27fa4cc00baf611d97e5a9ffa189cd024ea113c892209e2675`; source `938df7a8f6990bc08dc4187935fce70de573dd9b`; model SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`; frozen prefix SHA256 `ad528d87807f615a95b0694c0b0dd097e3facd24be14b71914a5ecf9694f4067`.

The prompt-batch direct assembly eliminated the redundant render copy pass, but the selected speed is effectively flat and remains below target. A bounded follow-up 100-03l is scheduled to inspect the `llama_context` prefill/decode owner and reduce call-path latency before 100-04.

Request payload records are reconstructed from canonical templates because the runner did not retain wire bodies; see `/srv/ai/paged-kv/results/forward/100-03k/attempt-01/selected-p2-retake-final/requests/manifest.json`. Raw timing detail: `/srv/ai/paged-kv/results/forward/100-03k/attempt-01/measured-stage-timing-summary.json`. Full selected retake: `/srv/ai/paged-kv/results/forward/100-03k/attempt-01/selected-p2-retake-final`.
