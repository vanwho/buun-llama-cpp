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

| Smoke | Selected decode work | Outcome |
|---|---|---|
| GPU KV Turbo4, MTP off; context 122,880; nominal C6144 | Reasoning off, prompt 0; 1 warmup + 3 measured repetitions (all 4 requests passed) | Complete. Median decode 41.41 tok/s. Actual context was 6,145 before decode and 7,430 after the cumulative turn history. MTP acceptance is disabled/not applicable. |
| CPU KV Q4_0, MTP on with Turbo4 draft K/V and n-max 2; context 200,704; nominal C6144 then C12288 | C6144 was prefill-only. At C12288, low prompts 0–2 and medium prompt 0; 4 warmups + 12 measured repetitions (all 16 selected requests passed, 3 measured repeats per cell). | Accepted as a successful smoke. At C12288 actual context was 12,289 before the selected requests and 15,790 after their accumulated history. Median decode rates were 26.31, 25.45, 24.98 tok/s for low prompts 0–2 and 24.86 tok/s for medium prompt 0. MTP acceptance medians were 99.00%, 97.44%, 100.00%, and 100.00%, respectively. |

### CPU-KV optional compact-head fallback

The historical supervisor marked the CPU-KV series `capacity_limit` because model startup logged a CUDA allocation failure for the optional 220.25 MiB compact MTP draft head. The adjacent messages state that compact-head allocation failed and the server kept the full head. The server then loaded successfully and every selected request completed; 12 measured requests reported real accepted/draft-token counter deltas. This is a handled optional allocation fallback, not a fatal model/server OOM. The adjudication suppresses only that exact matched diagnostic sequence; unrelated OOM, device, and scratch-allocation errors remain fatal.

Adjudication: `.../smoke-cpu-mtp-on-C6144-C12288/attempt-001-L200704-smoke-C6144-C12288/adjudication.json`, status `accepted_as_successful_smoke_with_handled_optional_allocation_fallback`. It records the narrow classifier helper SHA256 `5759118b5a7f4c3b7a95dcb92fedbb2ae4619ec9bfaa34ef3f1577691b3f7178` (original helper `b1b288c59cb7e0ce5633ec8c1ca04fa33f974e1dd349c10d0abd0a8dd75077c1`).

## Raw result files

- GPU row: `/srv/ai/paged-kv/results/pristine-master-958b444c677a-cuda134-validation-20261105/phase1-startup-smokes-corrected-source/smoke-gpu-mtp-off-C6144/attempt-001-L122880-smoke-C6144/cell/C6144/campaign/row.json`
- CPU row: `/srv/ai/paged-kv/results/pristine-master-958b444c677a-cuda134-validation-20261105/phase1-startup-smokes-corrected-source/smoke-cpu-mtp-on-C6144-C12288/attempt-001-L200704-smoke-C6144-C12288/cell/C12288/campaign/row.json`
- CPU startup diagnostics and adjudication are in the same CPU attempt directory.
