# Pristine master CUDA 13.4 smoke results

**Scope:** consolidated pristine-master GPU MTP-off and CPU-KV results, including phases 1, 2, 6, 7, 8, and 10, independent-base retries, historical partial rows, and clean replacement cells where identified. A separate fork build diagnostic is recorded at the end. This is not the completed comparison matrix. Each section links its raw artifacts under `/srv/ai/paged-kv/results/`.

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
| 12,288 | incomplete (phase 2 historical; off cells replaced by phase 10) | 22/30 | 8 | 39.7569 | 40.0771 |
| 28,672 | incomplete (phase 2 historical; off/prompt 2 replaced by phase 10) | 27/30 | 3 | 38.7389 | 39.2038 |
| 62,464 | complete | 3/3 | 0 | 35.9855 | 35.9924 |
| 78,848 | fill-only complete | 0 | 0 | — | — |
| 88,064 | incomplete (phase 2 historical; replaced by clean phase 8 cells) | 10/18 | 8 | 34.0309 | 34.1722 |
| 100,352 | fill-only complete | 0 | 0 | — | — |
| 120,832 | not reached | 0 | — | — | — |

The incomplete decode rows failed their output-validity checks (short, typically one-token completions); these are not fatal OOMs. The C120832 stage stopped at the pre-request capacity guard: actual context was 116,601 against a 116,600 ceiling with 6,280 tokens reserved, so no decode request ran. The phase2 supervisor's `resource_interference` label and restoration record are preserved in `series-result.json`; protected Jellyfin, port 8091, and port 8092 were recorded restored/unchanged. No phase2 GPU rows are represented as complete where the row itself is incomplete. Phase 2 C88064 results below are historical: the clean phase 8 run at the same nominal context supplies the current graph cells. Phase 2 C12288 off prompts 0–2 and C28672 off prompt 2 are likewise historical and are replaced by the clean phase 10 retries. All phase2 raw repetitions remain here for audit.

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
| 12,288 | off | 0 | 1/3 | 39.9324* | historical; phase10 replaces | 39.9324 | rep2, rep3: output_invalid, 1 token each |
| 12,288 | off | 1 | 0/3 | — | historical; phase10 replaces | — | rep1: 2-token output_invalid; rep2, rep3: 1-token output_invalid |
| 12,288 | off | 2 | 0/3 | — | historical; phase10 replaces | — | rep1, rep2, rep3: 1-token output_invalid |
| 12,288 | xhigh | 1 | 3/3 | 39.5434 | 39.6043 | 39.6043, 39.5434, 39.4951 | — |
| 12,288 | xhigh | 2 | 3/3 | 39.5064 | 39.5209 | 39.4672, 39.5209, 39.5064 | — |
| 28,672 | low | 0 | 3/3 | 39.1871 | 39.2038 | 38.9841, 39.2038, 39.1871 | — |
| 28,672 | low | 1 | 3/3 | 38.8164 | 38.8509 | 38.8164, 38.8509, 38.7955 | — |
| 28,672 | low | 2 | 3/3 | 38.8106 | 38.8108 | 38.8108, 38.7389, 38.8106 | — |
| 28,672 | medium | 0 | 3/3 | 38.7720 | 38.8451 | 38.6441, 38.8451, 38.7720 | — |
| 28,672 | medium | 1 | 3/3 | 38.7664 | 38.8124 | 38.7149, 38.8124, 38.7664 | — |
| 28,672 | medium | 2 | 3/3 | 38.5852 | 38.8377 | 38.8377, 38.5852, 38.5321 | — |
| 28,672 | off | 2 | 0/3 | — | historical; phase10 replaces | — | rep1, rep2, rep3: 1-token output_invalid |
| 28,672 | xhigh | 0 | 3/3 | 38.5171 | 38.5288 | 38.5288, 38.3105, 38.5171 | — |
| 28,672 | xhigh | 1 | 3/3 | 38.5765 | 38.5916 | 38.5409, 38.5916, 38.5765 | — |
| 28,672 | xhigh | 2 | 3/3 | 38.5172 | 38.5461 | 38.5172, 38.5461, 38.4909 | — |
| 62,464 | low | 2 | 3/3 | 35.9855 | 35.9924 | 35.9924, 35.9855, 35.9689 | — |
| 88,064 | low | 0 | 3/3 | 34.0094 | historical; superseded | 33.9456, 34.0094, 34.0346 | — |
| 88,064 | low | 1 | 3/3 | 34.0410 | historical; superseded | 34.0410, 34.0463, 34.0273 | — |
| 88,064 | low | 2 | 3/3 | 33.9306 | historical; superseded | 34.0382, 33.8633, 33.9306 | — |
| 88,064 | off | 0 | 1/3 | 34.1722* | — | 34.1722 | rep2, rep3: output_invalid, 1 token each |
| 88,064 | off | 1 | 0/3 | — | — | — | rep1, rep2, rep3: 1-token output_invalid |
| 88,064 | off | 2 | 0/3 | — | — | — | rep1, rep2, rep3: 1-token output_invalid |

*The listed median is partial (one valid repetition); it is not a graph point. The phase2 C88064 cells are retained as historical raw data and are not graph points because phase 8 provides clean replacement cells at the same nominal C. Phase 2 C12288 off prompts 0–2 and C28672 off prompt 2 are also historical and superseded by phase 10. Fill-only targets have no selected decode cells.

## Phase 8 clean GPU MTP-off series

Phase 8 repeated C88064 under a clean timer-managed lifecycle and continued the same resident cumulative conversation to C116000. Both targets used the same pristine master binary, model, B1024/U256, 16 threads, Turbo4 KV, GPU placement (`-ngl 999`, `CUDA0`), and MTP off. One server handled the full series. The runner recorded the checkpoint service inactive before the run and active after restoration; the checkpoint timer remained active, with no restoration errors. Jellyfin and ports 8091/8092 matched their protected prior state. The supervisor and both rows completed successfully, without a resource-interference label.

At nominal C88064, the fill base occupied 88,065 tokens (one above target); the row completed 48/48 requests, including 12 warmups and 36 valid measured repetitions. Final cumulative context was 98,937. At nominal C116000, the fill base occupied 116,001 tokens (one above target); the selected xhigh-only row completed 12/12 requests, including 3 warmups and 9 valid measured repetitions. Final cumulative context was 119,263. The prefix remained resident and accumulated across requests/targets; the records show cached prompt lengths of 88,136–96,336 tokens at C88064 and 115,996 tokens for each selected prompt at C116000. The C88064 fill curve used two 32,768-token groups followed by a 22,489-token checkpoint tail; C116000 continued with two checkpoint-tail groups. The graph value in each row below is the maximum of that cell's three valid measured repetitions; medians and all repetitions are retained.

| Nominal C (actual base → final) | Reasoning | Prompt | Valid / 3 | Median (tok/s) | Graph value: max (tok/s) | Valid measured repetitions (tok/s) |
|---:|---|---:|---:|---:|---:|---|
| 88,064 (88,065 → 98,937) | off | 0 | 3/3 | 34.2009 | 34.2521 | 34.2521, 34.1954, 34.2009 |
| 88,064 (88,065 → 98,937) | off | 1 | 3/3 | 34.1608 | 34.1737 | 34.1737, 34.1608, 34.0621 |
| 88,064 (88,065 → 98,937) | off | 2 | 3/3 | 34.0520 | 34.0614 | 34.0520, 34.0614, 34.0236 |
| 88,064 (88,065 → 98,937) | low | 0 | 3/3 | 34.0072 | 34.0270 | 34.0072, 34.0065, 34.0270 |
| 88,064 (88,065 → 98,937) | low | 1 | 3/3 | 33.8665 | 33.9403 | 33.9403, 33.8665, 33.8479 |
| 88,064 (88,065 → 98,937) | low | 2 | 3/3 | 33.8163 | 33.8165 | 33.8165, 33.8163, 33.7289 |
| 88,064 (88,065 → 98,937) | medium | 0 | 3/3 | 33.7686 | 33.7740 | 33.7740, 33.6267, 33.7686 |
| 88,064 (88,065 → 98,937) | medium | 1 | 3/3 | 33.7028 | 33.7302 | 33.7302, 33.7028, 33.6474 |
| 88,064 (88,065 → 98,937) | medium | 2 | 3/3 | 33.7156 | 33.7191 | 33.7191, 33.5804, 33.7156 |
| 88,064 (88,065 → 98,937) | xhigh | 0 | 3/3 | 33.7278 | 33.7352 | 33.5739, 33.7352, 33.7278 |
| 88,064 (88,065 → 98,937) | xhigh | 1 | 3/3 | 33.6061 | 33.6574 | 33.6574, 33.6061, 33.5816 |
| 88,064 (88,065 → 98,937) | xhigh | 2 | 3/3 | 33.4433 | 33.4643 | 33.4643, 33.4433, 33.4227 |
| 116,000 (116,001 → 119,263) | xhigh | 0 | 3/3 | 32.3338 | 32.3545 | 32.3545, 32.3338, 32.2733 |
| 116,000 (116,001 → 119,263) | xhigh | 1 | 3/3 | 32.2510 | 32.2870 | 32.2510, 32.2870, 32.1853 |
| 116,000 (116,001 → 119,263) | xhigh | 2 | 3/3 | 32.1579 | 32.1956 | 32.1956, 32.1579, 32.1344 |

Raw evidence: `/srv/ai/paged-kv/results/pristine-master-958b444c677a-cuda134-validation-20261105/phase8-gpu-mtp-off-clean-88064/phase8-gpu-mtp-off-clean-88064/attempt-002-L122880-C88064-C116000-clean/` (`series-result.json`, `prior-state.json`, `cell/C*/campaign/row.json`, `records.jsonl`, and `prefill-curve.json`).

## Phase 10 clean GPU MTP-off retries

Phase 10 reran only the invalid phase2 cells, using the same pristine master source/build/model/fixture identity documented above and one GPU server for the cumulative C12288 → C28672 series. The server used B1024/U256, 16 threads, Turbo4 KV, `-ngl 999`, CUDA0, and MTP off. C12288 completed 12/12 requests (3 warmups plus 9 valid measured repeats) from a base occupancy of 12,289 tokens; final cumulative context was 15,648. C28672 completed 4/4 requests (1 warmup plus 3 valid measured repeats) from a base occupancy of exactly 28,672; final context was 29,961. The resident cumulative prefix continued across the two targets. Cached prompt lengths were 12,360–15,240 at C12288 and 28,741–29,554 at C28672. The checkpoint service was inactive before and after; the checkpoint timer remained active. The run completed without restoration errors or resource-interference status; Jellyfin and ports 8091/8092 matched their protected prior state.

Each graph point below is the maximum of all three valid measured repetitions in the cell. Output lengths are included because the C12288 off/prompt 1 responses hit the 167-token cap, while off/prompt 2 produced 400, 400, and 377 tokens; all passed the campaign validity checks.

| Nominal C (actual base → final) | Reasoning | Prompt | Valid / 3 | Median (tok/s) | Graph value: max (tok/s) | Valid measured repetitions (tok/s) | Output tokens per repetition |
|---:|---|---:|---:|---:|---:|---|---|
| 12,288 (12,289 → 15,648) | off | 0 | 3/3 | 40.5835 | 40.5955 | 40.5955, 40.5835, 40.5770 | 400, 400, 400 |
| 12,288 (12,289 → 15,648) | off | 1 | 3/3 | 40.5439 | 40.5702 | 40.5317, 40.5439, 40.5702 | 167, 167, 167 |
| 12,288 (12,289 → 15,648) | off | 2 | 3/3 | 40.3181 | 40.4242 | 40.4242, 40.3181, 40.3164 | 400, 400, 377 |
| 28,672 (28,672 → 29,961) | off | 2 | 3/3 | 38.9368 | 39.1373 | 39.1373, 38.9368, 38.8685 | 377, 377, 377 |

Raw evidence: `/srv/ai/paged-kv/results/pristine-master-958b444c677a-cuda134-validation-20261105/phase10-gpu-mtp-off-clean-12k-28k/phase10-gpu-mtp-off-clean-12k-28k/attempt-001-L122880-C12288-C28672-off-gaps/` (`series-result.json`, `prior-state.json`, `cell/server-argv.json`, and each target's `campaign/row.json` and `records.jsonl`). The executable SHA256 is `ab32d497d19389e7bc1c12310fe0e7cd8b1254cf5fffa492edf37bb963baf675`; source commit is `958b444c677ad9b02058d5fbf8fd0e13b7d24286`; model SHA256 is `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`; fixture commit is `a5a1110ca620db4f1c04669d0a2f569852ec1ec7` and map SHA256 is `5ca53650661d610ff2632eae262cb794bceb5b72efaf47364b8bcd43247dcf4f`. Driver: NVIDIA open module 615.71.09; CUDA 13.4.92, GCC/G++ 13.4.0. Phase10 plan SHA256: `652aad79e78a9356ceaa62490d1226b6d5b1b5c1820232c24f920915cb080199`.

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

### Independent-base retry: C195664 CPU-Q4 MTP-off, medium prompt index 1

This is a separate independent-base cohort and must not be merged into the phase6 cumulative-conversation curve. It retries the same C × reasoning × prompt cell that was invalid in phase6: all three cumulative phase6 measured requests returned 6-token outputs and had no graph value. The phase attempt used the independent-base-one-query protocol with one base at actual C195664; each request starts from that base rather than inheriting prior query history. Target KV was CPU Q4_0 with MTP off, context 262,144, `-ngl 999 --device CUDA0`, B1024/U256, and 16 threads. The row reports 4/4 passing requests (one warmup plus three measured), with 400 output tokens in each measured repetition. Each measured request began at actual C195664 and ended at actual C196091; the cached prompt count was 195,687.

| Nominal C (independent base) | Reasoning | Prompt index | Valid / 3 | Median tok/s | Graph value: max tok/s | Measured repetitions tok/s | Output tokens |
|---:|---|---:|---:|---:|---:|---|---|
| 195,664 (base actual 195,664) | medium | 1 (P2) | 3/3 | 3.627703 | 3.630899 | 3.630899, 3.627703, 3.605923 | 400, 400, 400 |

Attempt: `/srv/ai/paged-kv/results/independent-base-192k-medium-p2-20261105/attempt-003-L262144-C195664-medium-P2/`. Row and repetitions: `run-L262144-C195664-medium-P2/C195664/campaign/row.json` and `records.jsonl`; independent protocol, lifecycle, and server configuration are recorded in `lifecycle.json` and `run-L262144-C195664-medium-P2/C195664/campaign/run-config.json`. Provenance: pristine source commit `958b444c677ad9b02058d5fbf8fd0e13b7d24286`; executable SHA256 `ab32d497d19389e7bc1c12310fe0e7cd8b1254cf5fffa492edf37bb963baf675`; model SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`; fixture commit `a5a1110ca620db4f1c04669d0a2f569852ec1ec7` and map SHA256 `5ca53650661d610ff2632eae262cb794bceb5b72efaf47364b8bcd432dcf4f`; chat-template SHA256 `0103d81679a0cc43ce416da910e7db86a04c0e8548d7df3f41564c98a203deb7`; NVIDIA open driver 615.71.09. The lifecycle file records all loaded project DSO SHA256 values, benchmark runner SHA256 `5759118b5a7f4c3b7a95dcb92fedbb2ae4619ec9bfaa34ef3f1577691b3f7178`, and matrix driver SHA256 `c8293454dbb3fd2fb716931c985a4b5337facc4c77c729fb83a6f29136dad7c0`; services and protected processes were restored, with no lifecycle errors.

## Phase 7 separate 192K CPU-KV MTP-on recovery cohort

This is a separate targeted cohort for the two cells missing from the earlier broad C195664 CPU-KV MTP-on series; keep it separate rather than pooling it into that profile. The completed C195664 row passed all eight requests (2 warmups + 6 measured), with no startup error. It ran at context 200,704 using the pristine candidate and CPU Q4_0 target KV with native MTP and Turbo4 draft K/V.

The mapped base occupancy was exactly 195,664 tokens for nominal C195664 (zero-token fill delta). The prefix remained resident and the two selected cells appended to it; final actual context was 198,223 (+2,559 accumulated tokens), not a refill or a prefix shortfall.

| C | Reasoning | Prompt | Valid / 3 | Median tok/s | Graph value: max tok/s | All measured repetitions tok/s |
|---:|---|---:|---:|---:|---:|---|
| 195,664 | off | 2 | 3/3 | 7.5103 | 7.5591 | 6.5931, 7.5591, 7.5103 |
| 195,664 | xhigh | 1 | 3/3 | 7.4307 | 7.4754 | 5.5410, 7.4307, 7.4754 |

Phase 7 artifacts: `/srv/ai/paged-kv/results/pristine-master-958b444c677a-cuda134-validation-20261105/phase7-cpu-mtp-on-L200704-near192k/master-cpu-mtp-on-L200704-C195664-missing-upper-cells/attempt-001-L200704-C195664-offP2-xhighP1/cell/C195664/campaign/` (`row.json`, `records.jsonl`, `run-config.json`).

## Independent-base near-256K CPU-Q4 MTP-off cohort (complete)

This is a separate independent-base cohort from the earlier cumulative C255488 phase6 series; do not merge the two profiles. It completed all 48/48 requests: 12 warmups and 36 valid measured repetitions, with no invalid requests. The run used the pristine master binary (`ab32d497d19389e7bc1c12310fe0e7cd8b1254cf5fffa492edf37bb963baf675`), source commit `958b444c677ad9b02058d5fbf8fd0e13b7d24286`, model SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`, mapped fixture commit `a5a1110ca620db4f1c04669d0a2f569852ec1ec7` / map SHA256 `5ca53650661d610ff2632eae262cb794bceb5b72efaf47364b8bcd43247dcf4f`, and NVIDIA open driver `615.71.09` (boot ID `f79550f5-bc13-4100-a561-0467a4201951`). Loaded project DSO hashes match the build provenance above. Configuration: `-ngl 999 --device CUDA0`, context 262,144, 16 threads, B1024/U256, target CPU Q4_0 KV (`--no-kv-offload -ctk q4_0 -ctv q4_0`), MTP off.

The independent base occupied 255,489 tokens for nominal C255488 (+1), and every request began from actual context 255,489 with the same frozen base plus exactly one query. `actual_C_after` varied from 255,670 to 255,961 within individual requests; final row context reset to 255,489. Cached prompt counts ranged from 255,488 to 255,517. The row's `filled_prefix_method` field says `resident_cumulative_conversation`, but that inherited label is stale for this cohort: the 48 request payloads and `actual_C_before` records show independent base-plus-one-query requests, without accumulated conversation history between cells.

Each graph value below is the maximum of that cell's full three measured repetitions. Medians and all three raw repetitions are retained.

| Reasoning | Prompt index | Valid measured / 3 | Median tok/s | Graph value: max tok/s | All measured repetitions tok/s |
|---|---:|---:|---:|---:|---|
| low | 0 | 3/3 | 2.8962 | 2.8982 | 2.8982, 2.8962, 2.8943 |
| low | 1 | 3/3 | 2.9035 | 2.9054 | 2.9054, 2.9035, 2.8893 |
| low | 2 | 3/3 | 2.8991 | 2.8996 | 2.8996, 2.8988, 2.8991 |
| medium | 0 | 3/3 | 2.8973 | 2.8977 | 2.8973, 2.8773, 2.8977 |
| medium | 1 | 3/3 | 2.9063 | 2.9092 | 2.9063, 2.9004, 2.9092 |
| medium | 2 | 3/3 | 2.9040 | 2.9053 | 2.9053, 2.8974, 2.9040 |
| off | 0 | 3/3 | 2.8939 | 2.8947 | 2.8939, 2.8908, 2.8947 |
| off | 1 | 3/3 | 2.9056 | 2.9080 | 2.9080, 2.9024, 2.9056 |
| off | 2 | 3/3 | 2.9031 | 2.9051 | 2.9031, 2.9025, 2.9051 |
| xhigh | 0 | 3/3 | 2.9001 | 2.9034 | 2.9001, 2.8902, 2.9034 |
| xhigh | 1 | 3/3 | 2.9061 | 2.9062 | 2.9062, 2.9061, 2.9061 |
| xhigh | 2 | 3/3 | 2.9038 | 2.9048 | 2.9038, 2.9048, 2.9023 |

The supervisor logged this wall-time prefill curve for this independent base fill. It belongs to this cohort and is separate from the cumulative phase6 fill curve:

| Actual C after fill | New tokens | Wall prefill tok/s |
|---:|---:|---:|
| 32,780 | 32,779 | 1,112.16 |
| 65,563 | 32,782 | 791.70 |
| 98,345 | 32,781 | 619.79 |
| 131,128 | 32,782 | 511.28 |
| 163,911 | 32,782 | 433.95 |
| 196,694 | 32,782 | 376.96 |
| 229,477 | 32,782 | 333.25 |
| 255,489 | 26,011 | 301.35 |

Completed row and raw repetitions: `/srv/ai/paged-kv/results/independent-base-queries-256k-prototype-20261105/attempt-003-L262144-C255488/run-L262144-C255488/C255488/campaign/row.json` and `records.jsonl`; fill curve and lifecycle provenance are in `/srv/ai/paged-kv/results/independent-base-queries-256k-prototype-20261105/attempt-003-L262144-C255488/supervisor.log` and `lifecycle.json`.

## Separate fork CPU-Q4 native MTP-on driver-615 diagnostic

This is a fork-build diagnostic, separate from all pristine-master results above. With NVIDIA open driver 615.71.09 loaded, the fork candidate started successfully and reached nominal C12288 with actual occupancy 12,289. The configuration was context 262,144, CPU Q4_0 target KV (`-ctk q4_0 -ctv q4_0 --no-kv-offload`), `-ngl 999 --device CUDA0`, B1024/U256, 16 threads, native MTP (`--spec-type draft-mtp --spec-draft-n-max 2`), Turbo4 draft K/V, and draft KV on GPU. It used the same current model file and shared-prefix chat template identity recorded in the attempt.

The medium/P1 warmup and measured repetition 1 were normal: the warmup generated 40 tokens at 20.8411 tok/s with 22/31 MTP accepted (70.97%); repetition 1 generated 367 tokens at 24.9512 tok/s with 231/270 accepted (85.56%). The quality gate then stopped the run after measured repetitions 2 and 3 each emitted 400 slash-only reasoning tokens. Both had zero accepted of 795 draft tokens (0.00% acceptance); every nonspace character was `/`, with no alphanumeric characters. The gate classifies output with at least 64 nonspace characters, zero alphanumeric characters, and at least 80% dominant punctuation as low quality. These responses passed the transport/campaign status but failed output quality, so the run was intentionally aborted before later contexts. This is a quality failure, not an OOM or a driver crash. The 615.71.09 check therefore shows that updating the NVIDIA driver alone did not cure this fork's native-MTP CPU-Q4 output defect. The 33 other planned old-fork cells remain unfilled; no performance conclusions should be drawn for them.

| Fork diagnostic cell | Valid behavior | Output / MTP details |
|---|---|---|
| C12288, medium/P1 warmup | completed | 40 tokens; 20.8411 tok/s; 22/31 accepted (70.97%) |
| C12288, medium/P1 measured rep1 | completed | 367 tokens; 24.9512 tok/s; 231/270 accepted (85.56%) |
| C12288, medium/P1 measured rep2 | quality gate failed | 400 slash-only tokens; 0/795 accepted; 9.1429 tok/s |
| C12288, medium/P1 measured rep3 | quality gate failed; run stopped | 400 slash-only tokens; 0/795 accepted; 8.9777 tok/s |

Provenance: fork source commit `8a79008b5473676bb99a54cc1505d11a4e107357`; executable `/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server`, SHA256 `90426c19b1d3d2e6f322907a35447adcb9a1278341a587f09137d242c8ffab8c`; model `/srv/ai/models/text/current.gguf`, SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`; chat-template SHA256 `0103d81679a0cc43ce416da910e7db86a04c0e8548d7df3f41564c98a203deb7`; mapped fixture commit `a5a1110ca620db4f1c04669d0a2f569852ec1ec7`, map SHA256 `5ca53650661d610ff2632eae262cb794bceb5b72efaf47364b8bcd432dcf4f`; NVIDIA open driver 615.71.09. Lifecycle records driver/checker SHA256 `ee4e73093f0186d8fbc25a49d7d06b31e69dc0eae1d1db2fc8e0e8b285846ecd`, matrix SHA256 `d71b10f51e18df5b9e5f424283deec2a3a201620b90d5f0fd2b21c0ce1210d48`, and the executable plus all loaded project DSO hashes. The quality record preserves base SHA256 `f50e31be5cca945ac2d58e10f2f03daffbc3bf165609e03b5a540c9d6aee8d8c` and records SHA256 `a536b700812ec1aa73b6c837c3c86dd2bfc721eb2536c7fc54997bfd09112c0e`.

Raw artifacts: `/srv/ai/paged-kv/results/fork-cpu-mtp-on-nvidia615-20261105/attempt-002-L262144-C12288-to-196608/quality-stop.json`, `lifecycle.json`, `cell/C12288/campaign/records.jsonl`, `cell/server-argv.json`, and `cell/server-startup-20261105T195721Z.log`. The quality record hashes the slash-only raw streams: rep2 `fa3b286f0d5c5220dd12c63da543d714d3eb0985a0199ec93d029bf28ebded3b`; rep3 `9f265fde7bd5e62336560b85cd2d40442236886e37c32f84fe5817fcd4546f6a`. Contexts 28,672, 61,440, 126,976, and 196,608 were skipped.

## Independent-base master GPU MTP-off C120832 xhigh retry

This retry fills the exact master GPU MTP-off C120832 xhigh/P1–P3 cells. It completed all 12 requests (3 warmups + 9 measured) with no failed cases. Context was 122,880; the mapped base occupied 120,833 tokens (+1 from nominal C120832). Every measured query began at that same base occupancy. Cached prompt tokens were 120,901 for P1 and 120,832 for P2/P3, with no per-prompt or per-repetition refills. Although the inherited `row.json` label says `resident_cumulative_conversation`, the lifecycle protocol is `independent-base-one-query` and each record shows `actual_C_before=120833`; treat this as an independent frozen-base cohort.

The graph value is the maximum of the three valid repetitions in each cell. The table retains each repetition and its median.

| C | Reasoning | Prompt | Valid / 3 | Median tok/s | Graph value: max tok/s | All measured repetitions tok/s |
|---:|---|---:|---:|---:|---:|---|
| 120832 | xhigh | 1 | 3/3 | 32.084035 | 32.095688 | 32.084035, 32.076703, 32.095688 |
| 120832 | xhigh | 2 | 3/3 | 32.075815 | 32.081824 | 32.081824, 32.075815, 32.055659 |
| 120832 | xhigh | 3 | 3/3 | 32.058503 | 32.059907 | 32.058503, 32.059907, 32.048491 |

P1 and P2 stopped normally at 355 and 293 tokens. All three P3 repetitions reached the 400-token cap with a `length` finish. P3 emitted coherent, task-relevant xhigh reasoning about the requested directory-watcher script, rather than punctuation-only or slash-only filler; it did not reach a final script answer before the cap. Under the previously specified policy to retain normal reasoning-only length-stopped output, P3 is graph-eligible.

Configuration: pristine master source commit `958b444c677ad9b02058d5fbf8fd0e13b7d24286`; executable `/srv/ai/paged-kv/builds/master-pristine-958b444c677a-cuda134/bin/llama-server`, SHA256 `ab32d497d19389e7bc1c12310fe0e7cd8b1254cf5fffa492edf37bb963baf675`; model SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`; context 122,880, GPU Turbo4 KV, `-ngl 999 --device CUDA0`, B1024/U256, 16 threads, MTP off. Map SHA256 `5ca53650661d610ff2632eae262cb794bceb5b72efaf47364b8bcd432dcf4f`; NVIDIA open driver 615.71.09. Loaded project DSO hashes and runner/matrix hashes are recorded in the attempt `lifecycle.json`; the run used the same pristine binary/DSO identity as the provenance above.

Artifacts: `/srv/ai/paged-kv/results/independent-base-gpu-mtp-off-120k-xhigh-20261105/attempt-001-L122880-C120832-xhigh-P123/`; row and raw repetitions are under `run-L122880-C120832-xhigh-P123/C120832/campaign/`. Lifecycle confirms services and protected processes were restored with no errors.
