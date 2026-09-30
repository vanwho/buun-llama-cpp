# 100-03j selected prompt-2 retake

Decision: **goal miss**. The unchanged prefill floor is 500 tok/s.

- Fresh prompt: 4326 tokens, 0 cached.
- One measured row: 212.158 tok/s (20.390 s request-local prefill); decode 29.398 tok/s.
- MTP: 105/329 accepted/drafted (31.91%).
- Matched predecessor row: 210.614 tok/s; the single-row change is +0.733%. There is no median or confidence interval.
- Live-rewind captures decreased from 5 to 3: [1130, 4066, 4299, 4319, 4322] -> [1130, 4299, 4322]. The final-user seam at 4299 and newest rollback frontier at 4322 remain.
- Pager counters report 0 faults, 0 evictions, and 0 H2D / 0 D2H bytes.

This confirms fewer snapshot materializations but not a meaningful prefill speed pass. Checkpoint wall time and TTFT remain unmeasured. A further bounded source-owner task is scheduled before 100-04.

Candidate source: `2a5b594ed5b398bc7326ad25be646ffe474f96db`; binary SHA256 `d0edda5f1d725988d9c2208187a888639184c64664640c3e3ead5bac896eca34`; model SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`; frozen-prefix SHA256 `ad528d87807f615a95b0694c0b0dd097e3facd24be14b71914a5ecf9694f4067`.

Raw row artifacts: `/srv/ai/paged-kv/results/forward/100-03j/attempt-01/selected-p2-retake-final`. Request bodies are documented reconstructions because the canonical runner does not persist the outbound HTTP bodies.
