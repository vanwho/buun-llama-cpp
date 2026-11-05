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

The cumulative mapped-prefix run used the same candidate and continued the resident history across targets. Its supervisor ended the profile with `resource_interference` (exit 1), so treat this as partial data rather than a completed series. The valid completed rows and their maximum measured repetition (the graph value) are below; medians are retained for reference. Raw per-request repetitions remain in each row's adjacent `records.jsonl`.

| Nominal C | Row | Measured passes | Invalid measured cases | Median tok/s | Max graph value tok/s |
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
