# Pristine master CUDA 13.4 smoke results

**Scope:** phase 1 GPU MTP-off and CPU-KV MTP-on startup/decode smokes only. These results are not the completed comparison matrix. Raw artifacts are under `/srv/ai/paged-kv/results/pristine-master-958b444c677a-cuda134-validation-20261105/phase1-startup-smokes-corrected-source/`.

## Identity

- Candidate source: detached pristine master worktree `/srv/ai/paged-kv/worktrees/master-pristine-958b444c677a`, commit `958b444c677ad9b02058d5fbf8fd0e13b7d24286`; source remained clean.
- Candidate executable: `/srv/ai/paged-kv/builds/master-pristine-958b444c677a-cuda134/bin/llama-server`, SHA256 `ab32d497d19389e7bc1c12310fe0e7cd8b1254cf5fffa492edf37bb963baf675`.
- CUDA project DSO: `libggml-cuda.so` SHA256 `875111574883716d6777436e0151b3b7d18a6532b16a82fb7d867b5eacbd0ead`; it links CUDA 13 SONAMEs (`libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`) from CUDA 13.4.
- Toolchain: CUDA 13.4.92, GCC/G++ 13.4.0, host compiler G++ 13. Driver: NVIDIA open kernel module 615.71.09; smoke-time GPU use peaked at 15,903 MiB for CPU-KV MTP-on and 15,397 MiB for GPU-KV MTP-off.
- Model: `/srv/ai/models/text/current.gguf`, SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
- Mapped input: fixture worktree `/srv/ai/paged-kv/worktrees/benchmark-source-a5a1110` at commit `a5a1110ca620db4f1c04669d0a2f569852ec1ec7`; map `/srv/ai/paged-kv/fixtures/repo-prefill-a5a1110-32k-v1/map.json`, SHA256 `5ca53650661d610ff2632eae262cb794bceb5b72efaf47364b8bcd43247dcf4f`.

Both smokes used B1024/U256 and 16 threads. The prefix stayed resident and the conversation accumulated normally; the CPU-KV profile continued from C6144 to C12288 without reloading or repeating mapped bytes.

## Results

| Smoke | Selected decode work | Decode repetitions (tok/s) | Outcome |
|---|---|---|---|
| GPU KV Turbo4, MTP off; context 122,880; nominal C6144 | Reasoning off, prompt 0; 1 warmup + 3 measured repetitions (all 4 requests passed) | Raw: 41.4297, 41.4105, 41.2505; median 41.4105; max graph value 41.4297 | Complete. Actual context was 6,145 before decode and 7,430 after the cumulative turn history. MTP acceptance is disabled/not applicable. |
| CPU KV Q4_0, MTP on with Turbo4 draft K/V and n-max 2; context 200,704; nominal C6144 then C12288 | C6144 was prefill-only. At C12288, low prompts 0–2 and medium prompt 0; 4 warmups + 12 measured repetitions (all 16 selected requests passed, 3 measured repeats per cell). | Low p0: 25.5062, 26.6377, 26.3089 (median 26.3089, max 26.6377); p1: 19.4721, 25.4535, 25.5664 (median 25.4535, max 25.5664); p2: 22.7281, 25.0135, 24.9792 (median 24.9792, max 25.0135); medium p0: 25.1887, 24.8573, 24.6096 (median 24.8573, max 25.1887). | Accepted as a successful smoke. At C12288 actual context was 12,289 before the selected requests and 15,790 after their accumulated history. MTP acceptance medians were 99.00%, 97.44%, 100.00%, and 100.00%, respectively. |

### CPU-KV optional compact-head fallback

The historical supervisor marked the CPU-KV series `capacity_limit` because model startup logged a CUDA allocation failure for the optional 220.25 MiB compact MTP draft head. The adjacent messages state that compact-head allocation failed and the server kept the full head. The server then loaded successfully and every selected request completed; 12 measured requests reported real accepted/draft-token counter deltas. This is a handled optional allocation fallback, not a fatal model/server OOM. The adjudication suppresses only that exact matched diagnostic sequence; unrelated OOM, device, and scratch-allocation errors remain fatal.

Adjudication: `.../smoke-cpu-mtp-on-C6144-C12288/attempt-001-L200704-smoke-C6144-C12288/adjudication.json`, status `accepted_as_successful_smoke_with_handled_optional_allocation_fallback`. It records the narrow classifier helper SHA256 `5759118b5a7f4c3b7a95dcb92fedbb2ae4619ec9bfaa34ef3f1577691b3f7178` (original helper `b1b288c59cb7e0ce5633ec8c1ca04fa33f974e1dd349c10d0abd0a8dd75077c1`).

## Raw result files

- GPU row: `/srv/ai/paged-kv/results/pristine-master-958b444c677a-cuda134-validation-20261105/phase1-startup-smokes-corrected-source/smoke-gpu-mtp-off-C6144/attempt-001-L122880-smoke-C6144/cell/C6144/campaign/row.json`
- CPU row: `/srv/ai/paged-kv/results/pristine-master-958b444c677a-cuda134-validation-20261105/phase1-startup-smokes-corrected-source/smoke-cpu-mtp-on-C6144-C12288/attempt-001-L200704-smoke-C6144-C12288/cell/C12288/campaign/row.json`
- CPU startup diagnostics and adjudication are in the same CPU attempt directory.

## Phase 2 GPU MTP-off missing-cell pass

The cumulative mapped-prefix run used the same candidate and continued the resident history across targets. Its supervisor ended the profile with `resource_interference` (exit 1), so treat this as partial data rather than a completed series. The pooled summary below is descriptive across different reasoning modes and prompts; its maximum must not be used as a graph point. Per-context, per-mode, per-prompt graph values are listed separately, and exist only for complete three-repeat groups. Raw per-request repetitions remain in each row's adjacent `records.jsonl`.

| Nominal C | Row | Measured passes | Invalid measured cases | Pooled median tok/s (descriptive) | Pooled max tok/s (descriptive only) |
|---:|---|---:|---:|---:|---:|
| 6,144 | complete | 36/36 | 0 | 40.5824 | 41.2934 |
| 12,288 | incomplete | 22/30 | 8 | 39.7569 | 40.0771 |
| 28,672 | incomplete | 27/30 | 3 | 38.7389 | 39.2038 |
| 62,464 | complete | 3/3 | 0 | 35.9855 | 35.9924 |
| 78,848 | fill-only complete | 0 | 0 | — | — |
| 88,064 | incomplete | 10/18 | 8 | 34.0309 | 34.1722 |
| 100,352 | fill-only complete | 0 | 0 | — | — |
| 120,832 | not reached | 0 | — | — | — |

The incomplete decode rows failed their output-validity checks (short, typically one-token completions); these are not fatal OOMs. The C120832 stage stopped at the pre-request capacity guard: actual context was 116,601 against a 116,600 ceiling with 6,280 tokens reserved, so no decode request ran. The phase2 supervisor's `resource_interference` label and restoration record are preserved in `series-result.json`; protected Jellyfin, port 8091, and port 8092 were recorded restored/unchanged. No phase2 GPU rows are represented as complete where the row itself is incomplete.

Phase 2 artifacts: `/srv/ai/paged-kv/results/pristine-master-958b444c677a-cuda134-validation-20261105/phase2-primary-missing-cells/master-gpu-mtp-off-missing-gaps/attempt-001-L122880-selected-missing-cells/` (`series-result.json`, per-target `cell/C*/campaign/row.json` and `records.jsonl`).

Per-cell graph values (tok/s) use only complete sets of three valid measured repetitions. Incomplete sets are shown with their valid repetitions and explicitly have no graph point; their median and raw values are descriptive only.

| C | Reasoning | Prompt | Valid / 3 | Median (tok/s) | Graph value: max (tok/s) | Valid measured repetitions (tok/s) | Invalid repetitions |
|---:|---|---:|---:|---:|---:|---|---|
| 6,144 | low | 0 | 3/3 | 40.7984 | 41.0263 | 41.0263, 40.7984, 40.7588 | — |
| 6,144 | low | 1 | 3/3 | 40.8317 | 40.8339 | 40.7912, 40.8339, 40.8317 | — |
| 6,144 | low | 2 | 3/3 | 40.5395 | 40.7758 | 40.7758, 40.4590, 40.5395 | — |
| 6,144 | medium | 0 | 3/3 | 40.3974 | 40.6445 | 40.3974, 40.3810, 40.6445 | — |
| 6,144 | medium | 1 | 3/3 | 40.5080 | 40.5277 | 40.5069, 40.5080, 40.5277 | — |
| 6,144 | medium | 2 | 3/3 | 40.4851 | 40.6094 | 40.6094, 40.4851, 40.3327 | — |
| 6,144 | off | 0 | 3/3 | 41.2862 | 41.2934 | 41.2934, 41.2862, 41.1551 | — |
| 6,144 | off | 1 | 3/3 | 41.0913 | 41.1447 | 41.0913, 41.1447, 41.0845 | — |
| 6,144 | off | 2 | 3/3 | 41.0682 | 41.0787 | 41.0787, 41.0682, 41.0569 | — |
| 6,144 | xhigh | 0 | 3/3 | 40.3899 | 40.5554 | 40.5554, 40.1001, 40.3899 | — |
| 6,144 | xhigh | 1 | 3/3 | 40.2770 | 40.2875 | 40.2377, 40.2770, 40.2875 | — |
| 6,144 | xhigh | 2 | 3/3 | 40.3023 | 40.3498 | 40.3023, 40.3498, 40.2245 | — |
| 12,288 | low | 0 | 3/3 | 40.0684 | 40.0771 | 40.0684, 40.0171, 40.0771 | — |
| 12,288 | low | 1 | 3/3 | 40.0465 | 40.0619 | 40.0465, 40.0619, 39.9886 | — |
| 12,288 | low | 2 | 3/3 | 39.7273 | 39.7600 | 39.7600, 39.7273, 39.7047 | — |
| 12,288 | medium | 1 | 3/3 | 39.7631 | 39.7941 | 39.7941, 39.7631, 39.7478 | — |
| 12,288 | medium | 2 | 3/3 | 39.7541 | 39.7597 | 39.6768, 39.7597, 39.7541 | — |
| 12,288 | off | 0 | 1/3 | 39.9324* | — | 39.9324 | rep2, rep3: output_invalid, 1 token each |
| 12,288 | off | 1 | 0/3 | — | — | — | rep1: 2-token output_invalid; rep2, rep3: 1-token output_invalid |
| 12,288 | off | 2 | 0/3 | — | — | — | rep1, rep2, rep3: 1-token output_invalid |
| 12,288 | xhigh | 1 | 3/3 | 39.5434 | 39.6043 | 39.6043, 39.5434, 39.4951 | — |
| 12,288 | xhigh | 2 | 3/3 | 39.5064 | 39.5209 | 39.4672, 39.5209, 39.5064 | — |
| 28,672 | low | 0 | 3/3 | 39.1871 | 39.2038 | 38.9841, 39.2038, 39.1871 | — |
| 28,672 | low | 1 | 3/3 | 38.8164 | 38.8509 | 38.8164, 38.8509, 38.7955 | — |
| 28,672 | low | 2 | 3/3 | 38.8106 | 38.8108 | 38.8108, 38.7389, 38.8106 | — |
| 28,672 | medium | 0 | 3/3 | 38.7720 | 38.8451 | 38.6441, 38.8451, 38.7720 | — |
| 28,672 | medium | 1 | 3/3 | 38.7664 | 38.8124 | 38.7149, 38.8124, 38.7664 | — |
| 28,672 | medium | 2 | 3/3 | 38.5852 | 38.8377 | 38.8377, 38.5852, 38.5321 | — |
| 28,672 | off | 2 | 0/3 | — | — | — | rep1, rep2, rep3: 1-token output_invalid |
| 28,672 | xhigh | 0 | 3/3 | 38.5171 | 38.5288 | 38.5288, 38.3105, 38.5171 | — |
| 28,672 | xhigh | 1 | 3/3 | 38.5765 | 38.5916 | 38.5409, 38.5916, 38.5765 | — |
| 28,672 | xhigh | 2 | 3/3 | 38.5172 | 38.5461 | 38.5172, 38.5461, 38.4909 | — |
| 62,464 | low | 2 | 3/3 | 35.9855 | 35.9924 | 35.9924, 35.9855, 35.9689 | — |
| 88,064 | low | 0 | 3/3 | 34.0094 | 34.0346 | 33.9456, 34.0094, 34.0346 | — |
| 88,064 | low | 1 | 3/3 | 34.0410 | 34.0463 | 34.0410, 34.0463, 34.0273 | — |
| 88,064 | low | 2 | 3/3 | 33.9306 | 34.0382 | 34.0382, 33.8633, 33.9306 | — |
| 88,064 | off | 0 | 1/3 | 34.1722* | — | 34.1722 | rep2, rep3: output_invalid, 1 token each |
| 88,064 | off | 1 | 0/3 | — | — | — | rep1, rep2, rep3: 1-token output_invalid |
| 88,064 | off | 2 | 0/3 | — | — | — | rep1, rep2, rep3: 1-token output_invalid |

*The listed median is partial (one valid repetition); it is not a graph point. Fill-only targets have no selected decode cells.

## Phase 2 CPU-KV MTP-on missing-cell pass

The CPU-KV Q4_0, MTP-on run is still marked `running` at step 2 in its series result; the four rows below have been written, and later planned contexts are not included here. It used the same pristine master executable, mapped fixture, B1024/U256, 16 threads, Turbo4 draft K/V, and n-max 2. Values in the summary table are aggregated across the selected reasoning/prompt cases; pooled medians and maxima are descriptive only, and the pooled maximum is not a graph point. The per-context, per-mode, per-prompt table below carries graph values, each calculated only from a full set of three valid measured repetitions. Repetition lists follow the campaign record order and omit only the explicitly identified invalid request.

| Nominal C | Row / valid measured repeats | Actual prefix occupancy → final context | Pooled median tok/s (descriptive) | Pooled max tok/s (descriptive only) | Invalid case |
|---:|---|---:|---:|---:|---|
| 6,144 | complete, 9/9 | 6,145 → 10,043 | 27.5776 | 28.7961 | None |
| 12,288 | complete, 27/27 | 12,286 → 18,843 | 24.9509 | 27.2596 | None; initial prefix was 2 tokens below nominal C |
| 28,672 | incomplete, 26/27 | 28,673 → 35,208 | 20.8134 | 21.6917 | medium, prompt 0, repetition 3; 8-token non-substantive output |
| 61,440 | incomplete, 26/27 | 61,441 → 67,472 | 15.2084 | 15.7032 | medium, prompt 0, repetition 3; 8-token non-substantive output |

All valid measured decode repetitions (tok/s):

- C6144: 26.8237, 28.7961, 28.3004, 22.5438, 27.9062, 27.5776, 24.9075, 27.5247, 27.6672.
- C12288: 27.2596, 27.0314, 26.2813, 26.2388, 25.9220, 26.6815, 23.5564, 25.5593, 25.3686, 25.2204, 24.5543, 25.7307, 25.1548, 24.8481, 24.8113, 24.0605, 23.4170, 23.4687, 24.2036, 24.7085, 24.6944, 24.5899, 25.1950, 24.9509, 23.7487, 24.4143, 25.0642.
- C28672: 21.4644, 21.6917, 21.6203, 21.3926, 21.1053, 21.1202, 20.5923, 20.8569, 20.5705, 20.7700, 17.6162, 21.3924, 21.1156, 21.0644, 19.7158, 20.9876, 20.9939, 20.9335, 20.7475, 20.5633, 20.5347, 20.1967, 20.4158, 20.5299, 20.0361, 20.0591.
- C61440: 15.4856, 15.3815, 15.4043, 15.5766, 15.7032, 15.2991, 15.3458, 15.1500, 15.2527, 15.0725, 12.6282, 15.4301, 15.2899, 15.1153, 14.7634, 14.9570, 14.9851, 15.3043, 15.5560, 15.2250, 15.0321, 15.1887, 15.1918, 15.0388, 15.0548, 14.9599.

Per-cell graph values (tok/s) use only complete sets of three valid measured repetitions. Incomplete sets have no graph point; any partial median is descriptive only.

| C | Reasoning | Prompt | Valid / 3 | Median (tok/s) | Graph value: max (tok/s) | Valid measured repetitions (tok/s) | Invalid repetitions |
|---:|---|---:|---:|---:|---:|---|---|
| 6,144 | medium | 0 | 3/3 | 28.3004 | 28.7961 | 26.8237, 28.7961, 28.3004 | — |
| 6,144 | medium | 1 | 3/3 | 27.5776 | 27.9062 | 22.5438, 27.9062, 27.5776 | — |
| 6,144 | medium | 2 | 3/3 | 27.5247 | 27.6672 | 24.9075, 27.5247, 27.6672 | — |
| 12,288 | low | 0 | 3/3 | 27.0314 | 27.2596 | 27.2596, 27.0314, 26.2813 | — |
| 12,288 | low | 1 | 3/3 | 26.2388 | 26.6815 | 26.2388, 25.9220, 26.6815 | — |
| 12,288 | low | 2 | 3/3 | 25.3686 | 25.5593 | 23.5564, 25.5593, 25.3686 | — |
| 12,288 | medium | 0 | 3/3 | 25.2204 | 25.7307 | 25.2204, 24.5543, 25.7307 | — |
| 12,288 | medium | 1 | 3/3 | 24.8481 | 25.1548 | 25.1548, 24.8481, 24.8113 | — |
| 12,288 | medium | 2 | 3/3 | 23.4687 | 24.0605 | 24.0605, 23.4170, 23.4687 | — |
| 12,288 | xhigh | 0 | 3/3 | 24.6944 | 24.7085 | 24.2036, 24.7085, 24.6944 | — |
| 12,288 | xhigh | 1 | 3/3 | 24.9509 | 25.1950 | 24.5899, 25.1950, 24.9509 | — |
| 12,288 | xhigh | 2 | 3/3 | 24.4143 | 25.0642 | 23.7487, 24.4143, 25.0642 | — |
| 28,672 | low | 0 | 3/3 | 21.6203 | 21.6917 | 21.4644, 21.6917, 21.6203 | — |
| 28,672 | low | 1 | 3/3 | 21.1202 | 21.3926 | 21.3926, 21.1053, 21.1202 | — |
| 28,672 | low | 2 | 3/3 | 20.5923 | 20.8569 | 20.5923, 20.8569, 20.5705 | — |
| 28,672 | medium | 0 | 2/3 | 19.1931* | — | 20.7700, 17.6162 | rep3: output_invalid, 8 tokens |
| 28,672 | medium | 1 | 3/3 | 21.1156 | 21.3924 | 21.3924, 21.1156, 21.0644 | — |
| 28,672 | medium | 2 | 3/3 | 20.9876 | 20.9939 | 19.7158, 20.9876, 20.9939 | — |
| 28,672 | xhigh | 0 | 3/3 | 20.7475 | 20.9335 | 20.9335, 20.7475, 20.5633 | — |
| 28,672 | xhigh | 1 | 3/3 | 20.4158 | 20.5347 | 20.5347, 20.1967, 20.4158 | — |
| 28,672 | xhigh | 2 | 3/3 | 20.0591 | 20.5299 | 20.5299, 20.0361, 20.0591 | — |
| 61,440 | low | 0 | 3/3 | 15.4043 | 15.4856 | 15.4856, 15.3815, 15.4043 | — |
| 61,440 | low | 1 | 3/3 | 15.5766 | 15.7032 | 15.5766, 15.7032, 15.2991 | — |
| 61,440 | low | 2 | 3/3 | 15.2527 | 15.3458 | 15.3458, 15.1500, 15.2527 | — |
| 61,440 | medium | 0 | 2/3 | 13.8504* | — | 15.0725, 12.6282 | rep3: output_invalid, 8 tokens |
| 61,440 | medium | 1 | 3/3 | 15.2899 | 15.4301 | 15.4301, 15.2899, 15.1153 | — |
| 61,440 | medium | 2 | 3/3 | 14.9570 | 14.9851 | 14.7634, 14.9570, 14.9851 | — |
| 61,440 | xhigh | 0 | 3/3 | 15.3043 | 15.5560 | 15.3043, 15.5560, 15.2250 | — |
| 61,440 | xhigh | 1 | 3/3 | 15.1887 | 15.1918 | 15.0321, 15.1887, 15.1918 | — |
| 61,440 | xhigh | 2 | 3/3 | 15.0388 | 15.0548 | 15.0388, 15.0548, 14.9599 | — |

*Partial median across two valid repeats; no graph point is emitted for the incomplete triplet.

The C12288 base occupied 12,286 tokens for a 12,288-token target, a 2-token prefix shortfall; other listed starts were within one token of nominal. Actual final context includes the accumulated selected request history and is not the nominal fill target. The two invalid repetitions are output-quality failures, not OOMs; each affected reasoning/prompt triplet has two valid measured repeats and one failed repetition. Raw records, row summaries, and the running series result are under `/srv/ai/paged-kv/results/pristine-master-958b444c677a-cuda134-validation-20261105/phase2-primary-missing-cells/master-cpu-mtp-on-L200704-missing-gaps/attempt-001-L200704-selected-missing-cells/`.

Provenance is the pristine detached source commit `958b444c677ad9b02058d5fbf8fd0e13b7d24286`; executable SHA256 `ab32d497d19389e7bc1c12310fe0e7cd8b1254cf5fffa492edf37bb963baf675`; mapped fixture worktree commit `a5a1110ca620db4f1c04669d0a2f569852ec1ec7` and map SHA256 `5ca53650661d610ff2632eae262cb794bceb5b72efaf47364b8bcd43247dcf4f`; NVIDIA driver `615.71.09` (boot ID `f79550f5-bc13-4100-a561-0467a4201951`). Full compiler, CUDA, DSO hashes, and build/link provenance are recorded above and in `/srv/ai/paged-kv/results/master-pristine-958b444c677a/build-provenance-cuda134-matched-20261105.md`.

### C126976 (127K) completed row

The CPU-KV MTP-on row is complete: 27/27 measured requests passed, with three valid measured repetitions in each of the nine selected reasoning/prompt cells. The fill base held 126,977 tokens for nominal C126976 (+1); final accumulated conversation context was 133,995 tokens. The +1 token is a one-token occupancy offset, not a prefix shortfall. Each maximum below is the graph value for that specific cell.

| Reasoning | Prompt | Valid / 3 | Median tok/s | Graph value: max tok/s | All measured repetitions tok/s |
|---|---:|---:|---:|---:|---|
| low | 0 | 3/3 | 10.2333 | 10.2457 | 10.2457, 10.2217, 10.2333 |
| low | 1 | 3/3 | 10.2737 | 10.3246 | 10.3246, 10.2737, 10.2087 |
| low | 2 | 3/3 | 10.1393 | 10.1541 | 10.1393, 10.0303, 10.1541 |
| medium | 0 | 3/3 | 9.8338 | 10.0744 | 9.8338, 10.0744, 9.5737 |
| medium | 1 | 3/3 | 10.0929 | 10.1078 | 10.1078, 10.0662, 10.0929 |
| medium | 2 | 3/3 | 9.8082 | 9.9555 | 9.8082, 9.6344, 9.9555 |
| xhigh | 0 | 3/3 | 10.0732 | 10.1369 | 10.0732, 10.1369, 10.0534 |
| xhigh | 1 | 3/3 | 9.9875 | 10.0465 | 9.9594, 10.0465, 9.9875 |
| xhigh | 2 | 3/3 | 9.9599 | 9.9887 | 9.9599, 9.9420, 9.9887 |

Raw row and repetition records: `/srv/ai/paged-kv/results/pristine-master-958b444c677a-cuda134-validation-20261105/phase2-primary-missing-cells/master-cpu-mtp-on-L200704-missing-gaps/attempt-001-L200704-selected-missing-cells/cell/C126976/campaign/row.json` and `records.jsonl`. The overall CPU series remained running at step 2 when checked; no later context row had completed yet.

## Phase 6 CPU-KV MTP-off, ~192K row

The CPU-Q4 MTP-off series uses context 262,144, `-ngl 999 --device CUDA0`, `-b 1024 -ub 256`, 16 threads, target K/V Q4_0, and `--no-kv-offload`. The C195664 row is incomplete: 45/48 selected requests passed, including 33/36 measured repetitions. Eleven mode/prompt groups have complete three-repetition sets; medium/prompt index 1 had three invalid measured outputs (6 tokens each) and has no graph value. Maxima below are per C × reasoning × prompt and only appear for complete triplets.

The mapped base fill occupied exactly 195,664 tokens for nominal C195664 (zero-token fill delta); the row uses `resident_cumulative_conversation`, with one base fill and append-only history across selected requests. Final context reached 203,496 tokens, 7,832 above the nominal base because of that appended history. This is not a base-prefix shortfall. The phase 6 supervisor later ended with `runtime_or_driver_failure`; the C195664 row's recorded invalids are output-validity failures, not OOMs.

| C | Reasoning | Prompt | Valid / 3 | Median tok/s | Graph value: max tok/s | All valid measured repetitions tok/s | Invalid repetitions |
|---:|---|---:|---:|---:|---:|---|---|
| 195,664 | low | 0 | 3/3 | 3.5792 | 3.5911 | 3.5911, 3.5457, 3.5792 | — |
| 195,664 | low | 1 | 3/3 | 3.5754 | 3.5764 | 3.5764, 3.5733, 3.5754 | — |
| 195,664 | low | 2 | 3/3 | 3.5532 | 3.5623 | 3.5623, 3.5532, 3.5296 | — |
| 195,664 | medium | 0 | 3/3 | 3.5489 | 3.5533 | 3.5489, 3.5533, 3.5433 | — |
| 195,664 | medium | 1 | 0/3 | — | — | — | rep1, rep2, rep3: output_invalid, 6 tokens each |
| 195,664 | medium | 2 | 3/3 | 3.5293 | 3.5346 | 3.5293, 3.5237, 3.5346 | — |
| 195,664 | off | 0 | 3/3 | 3.5821 | 3.6134 | 3.6134, 3.5488, 3.5821 | — |
| 195,664 | off | 1 | 3/3 | 3.5970 | 3.6048 | 3.5783, 3.5970, 3.6048 | — |
| 195,664 | off | 2 | 3/3 | 3.6014 | 3.6026 | 3.6026, 3.6014, 3.5953 | — |
| 195,664 | xhigh | 0 | 3/3 | 3.5435 | 3.5479 | 3.5479, 3.5435, 3.5271 | — |
| 195,664 | xhigh | 1 | 3/3 | 3.5366 | 3.5415 | 3.5415, 3.5241, 3.5366 | — |
| 195,664 | xhigh | 2 | 3/3 | 3.5299 | 3.5357 | 3.5259, 3.5357, 3.5299 | — |

Raw campaign artifacts: `/srv/ai/paged-kv/results/pristine-master-958b444c677a-cuda134-validation-20261105/phase6-cpu-mtp-off-L262144-192k-256k/master-cpu-mtp-off-L262144-192k-256k-content/attempt-002-L262144-C195664-C236544-full-matrix/cell/C195664/campaign/` (`row.json`, `records.jsonl`, `conversation-frontier.json`).

### C255488 (~250K) incomplete row

The second phase6 target row is incomplete: 5/12 selected requests passed and only 2/9 measured requests were valid. The base occupancy was 255,489 for nominal C255488 (+1 token), with final accumulated context at 255,949. The prefix was not short; the final delta reflects appended request history. The only valid measured repetitions were off/prompt index 0, reps 1–2 (2.8980 and 2.9044 tok/s; two-repeat median 2.9012 is descriptive only). Rep 3 there returned a non-substantive 5-token fragment. Off/prompt index 1 had three non-substantive 2-token outputs; off/prompt index 2 had three one-token EOS/empty outputs. No reasoning/prompt cell reached three valid measured repetitions, so this row contributes no graph point.

The overall phase6 series ended `runtime_or_driver_failure` after its two targets. Preserve C195664 as recorded above (11 full triplets plus the invalid medium/prompt-1 triplet); neither row's output-invalid cases are evidence of OOM. Raw C255488 artifacts are in `/srv/ai/paged-kv/results/pristine-master-958b444c677a-cuda134-validation-20261105/phase6-cpu-mtp-off-L262144-192k-256k/master-cpu-mtp-off-L262144-192k-256k-content/attempt-002-L262144-C195664-C236544-full-matrix/cell/C255488/campaign/` (`row.json`, `records.jsonl`, `conversation-frontier.json`).
