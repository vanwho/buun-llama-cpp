# Upstream-master all-GPU Turbo4, no-MTP 128K prefill attempt

Date: 2026-10-20

## Setup

- Binary: `/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server`
- Upstream master: `d2407c33f9595d1badccfd9a82a33c4c5ceda490`
- Binary SHA-256: `527c34de411d4cad6290e828deb17a71816ae2b9620a3f1166b833e1267e0f0f`
- Model SHA-256: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`
- Configuration: `L=131072`, target K/V Turbo4 on CUDA0, `-ngl 999`, Flash Attention on, MTP off (`--spec-type none`), `B/U=1024/256`, 16 threads, no context shift.
- The managed 8080 server was stopped while this isolated candidate ran on 18080, then restarted. The test sent one fresh repository-source prompt once; `n_prompt_tokens_cache=0`. It did not replay prior context.
- Prompt prefix contained 126,900 source tokens, leaving roughly 4K context reserve. The server's chat template brought the actual prompt to approximately 127K tokens.
- Exact argv and raw artifacts are under `/srv/ai/paged-kv/results/release/master-turbo4-allgpu-no-mtp-128k-20261020/one-shot/`.

## Result

The 128K fill did **not** complete. The server processed 95,274 prompt tokens, then aborted with GPU out-of-memory errors. The log first reports failure to reserve two 188.5 MiB F16 dequant scratch buffers, followed by CUDA OOM in `launch_mul_mat_q` while setting maximum dynamic shared memory. The server retried the batch at smaller sizes, down to 64 rows, but still failed. This is a real capacity/scratch failure for this exact all-GPU configuration, not a client timeout or prompt replay issue.

| Observed frontier | Elapsed | Interval rate |
| ---: | ---: | ---: |
| 9,258 | 5.003 s | 1,850 tok/s from start |
| 16,426 | 9.006 s | 1,790 tok/s |
| 32,810 | 19.016 s | 1,636 tok/s |
| 65,578 | 46.050 s | 1,212 tok/s |
| 95,274 | 75.140 s | 1,021 tok/s to final server progress record |

These are single-pass progress rates, not a completed 128K benchmark. No 128K result should be reported as successful. MTP was disabled throughout. The server's response connection ended because the process aborted; there is no completion timing/response record.

The most likely constraint is insufficient VRAM headroom for growing all-GPU Turbo4 KV plus attention/dequant scratch at large prefill shapes. The log shows the scratch reservation failure directly; it does not by itself establish whether the dominant pressure is KV allocation, a separate CUDA allocation, or another driver allocation. Do not claim that 128K all-GPU fits on this 16-GiB card based on the partial fill.

## Comparison with the actual qwen38-big profile

The relevant profile is `/srv/ai/config/profiles/qwen38-big.env`. It specifies
`CTX=131072`, `BATCH=1024`, `UBATCH=128`, `CACHE_TYPE_K=turbo4`,
`CACHE_TYPE_V=turbo4`, `N_GPU_LAYERS=999`, Flash Attention on, and `MTP=off`.
The failed one-shot above used `B/U=1024/256`, so it was **not an exact
qwen38-big reproduction**. Its OOM cannot be used to conclude that the
qwen38-big configuration fails at 131K. The higher ubatch may increase
prefill scratch requirements; that is a plausible difference, not yet an
established cause.

The exact-geometry retry and a separate idle-headroom measurement are recorded
below. The earlier ACO selective-pager run is a different configuration and
is not the relevant baseline for this question.

## Exact-qwen38-big-geometry retry

The retry used `B/U=1024/128` with the same master binary, model, Turbo4 K/V,
MTP-off, `L=131072`, one-shot prompt, and fresh server. It advanced beyond the
first `U=256` failure, reaching `108544` prompt tokens at 106.21 seconds
(cumulative 1,021.99 tok/s); the next batch failed. The server reported failure
to reserve `212.5 + 212.5 MiB` of F16 dequant scratch, retried progressively
smaller batches down to one row, and returned HTTP 500 “Context size has been
exceeded.” The slot had committed 109,568 tokens when it was released. Thus
the exact-qwen38-big ubatch improves the frontier by about 14.3K tokens versus
the `U=256` attempt, but this particular master binary/run still does not
complete the requested ~127K prompt. It is a scratch/capacity failure, not a
client timeout.

The GPU had 15,487 MiB used / 456 MiB free after candidate model load; before
the candidate started it had 150 MiB used / 15,793 MiB free. During prefill,
usage approached the device limit. The managed llama service was restarted
afterward and health checks on ports 8080 and 8091 returned HTTP 200. Raw retry
artifacts are in
`/srv/ai/paged-kv/results/release/master-turbo4-allgpu-no-mtp-128k-20261020/ubatch-128/`.
The captured candidate command and hashes are in `argv.json` and
`binary.sha256`; server log SHA-256 is
`efef2f24c67b2b1b0ce73149b090a1a70c443a42380294e11c162e358d09da4d`.

## Idle VRAM headroom for `B/U=1024/256`

A fresh master server was started with `L=131072`, Turbo4 K/V, MTP off,
`B/U=1024/256`, and no prompt submitted. Across six five-second idle samples
after model load, `nvidia-smi` consistently reported 15,555 MiB used and 388
MiB free out of 16,376 MiB. This 388 MiB is the **idle free-VRAM upper bound**
available to later allocations, not a scratch reservation: as context is
ingested, additional KV/runtime allocations consume headroom. At the observed
95,274-token frontier, the 1024/256 run could not reserve 188.5 MiB each for
K and V scratch, so less than the required 377 MiB was available then. Raw
idle samples and startup log are in
`/srv/ai/paged-kv/results/release/master-turbo4-allgpu-no-mtp-128k-20261020/idle-1024-256/`.

## Raw evidence

- Result root: `/srv/ai/paged-kv/results/release/master-turbo4-allgpu-no-mtp-128k-20261020/one-shot/`
- Server log SHA-256: `28517b4f697fec645790b33e22e407ed93aedc5ed797cc108756ef1a590262f7`
- Partial frontiers: `frontiers.jsonl`
- Failure classification: `failure.json`
- Config: `argv.json`; prompt identity/count: `prompt-info.json`, `prompt.txt`, and `request.json`
- Managed service was restored active on port 8080; no other service port was intentionally accessed.
