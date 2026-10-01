# GPU101 final speed decision

**Decision: `goal_miss`; 102-01 remains gated.** The canonical selected H4096 campaign stopped at the first measured request after its warmup and slot reset. Reservation failed with `no_victim` for sequence 0, positions 4085–4105, 21 rows. A stand-alone fresh request succeeds, which narrows the failure to the warmup/reset path; it does not establish a canonical speed row.

## Geometry and identity

- Candidate: Qwen3.8-27B UD-IQ4_XS, binary `build-cuda/bin/llama-server`; full hashes and active settings are in the JSON receipt.
- L8192/H4096 (16 pages)/P256; B1024/U256; automatic selective pager; target and native nmax2 MTP Turbo4 on GPU.
- Frozen system prefix SHA-256 `8218b0f427cc931fa38d08f1c29f91ec7d82e12ac309ff7743eb7a60da678d20`; server-tokenized prompt sizes 4115/4112/4113, all C>H. Fresh-row `cached_tokens=0`.
- Canonical protocol: Python, mmap/read, Bash prompts; 40-token warmup then three 400-token requests per prompt.

## Selected rows

| Prompt | Warmup | Measured rows | Selected fresh prefill / decode / MTP median |
|---|---|---:|---|
| Python sorted merge | Passed; 4115 fresh tokens | 0; first request failed `no_victim` | Not established |
| mmap versus read | Not run after first failure | 0 | Not established |
| Bash directory watcher | Not run after first failure | 0 | Not established |

A separate one-request diagnostic at the same H4096 geometry completed at 267.17 prefill tok/s, 31.62 decode tok/s, 38.77% MTP acceptance. Another fresh same-process request also completed. These are not canonical paired medians and do not clear the slot-reset failure.

## Completed controls

| Prompt | Pager-off all-GPU prefill / decode / MTP median | CPU main-KV + GPU MTP prefill / decode / MTP median |
|---|---:|---:|
| 1 | 1632.24 / 84.34 / 78.20% | 674.29 / 24.56 / 80.37% |
| 2 | 1631.99 / 62.76 / 51.61% | 675.11 / 18.36 / 50.00% |
| 3 | 1632.94 / 71.25 / 63.32% | 673.79 / 21.29 / 65.03% |

CPU control uses ordinary CPU main-KV and GPU Turbo4 MTP. Raw summaries and all rows remain under `/srv/ai/paged-kv/results/forward/101-12/attempt-01/`; every chosen file is hashed in `GPU101_RELEASE.json`.

## Scheduled successors

1. **101-12a** owns the source-directed H4096 batch-write reservation/slot-reset defect. Reproduce warmup→erase→C>H admission, fix the eligibility/reservation state transition, run focused deterministic regression and the smallest live recheck at unchanged L8192/H4096/P256/B1024/U256.
2. **101-12b** depends on 101-12a and owns the repeated canonical speed decision. Run selected, matched pager-off and CPU-KV controls if the source/build identity changes; otherwise reuse this task’s exact completed controls. Release 102-01 only if every required gate passes; otherwise schedule another measured repair chain.

The 102-01 dependency is changed to 101-12b. No occupancy campaign was run.
