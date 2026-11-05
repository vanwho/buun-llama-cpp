# Exploratory master MTP startup: `--no-op-offload`

**Result:** Exploratory startup check only; this does not establish serving or benchmark performance. Adding `--no-op-offload` did not prevent the 1,730 MiB CUDA graph-compute allocation failure during native MTP context initialization at `-c 262144`, `-b 1024`, `-ub 256`.

## Frozen build

- Worktree commit: `958b444c677ad9b02058d5fbf8fd0e13b7d24286`
- Binary: `/srv/ai/paged-kv/builds/master-pristine-958b444c677a-cuda134/bin/llama-server`
- Version: `0.6.0-dev (build 13383, commit 958b444c6)`, GNU 13.4.0, Linux x86_64
- Binary SHA-256: `ab32d497d19389e7bc1c12310fe0e7cd8b1254cf5fffa492edf37bb963baf675`
- Loaded DSO SHA-256:
  - `libggml-base.so.0.26.0`: `c5cfc04fb7f3a271ffd4056b3c0c54d1f316336a6ab3a15fa281c8d74e61db39`
  - `libggml-cpu.so.0.26.0`: `a796741e585709c577452789c5ee837ada963ce2c46c161068a90f4f7f87b837`
  - `libggml-cuda.so.0.26.0`: `875111574883716d6777436e0151b3b7d18a6532b16a82fb7d867b5eacbd0ead`
  - `libggml.so.0.26.0`: `bb8df8427f33ea69b36e28e255e617acbe036650583a0c9b3c35b7d4a76a0ad9`
  - `libllama-common.so.0.6.0`: `52466c3e64500834cdebd3818a51b10ea6fe08c18fd9b9c26e426aa8f69d5e6d`
  - `libllama-server-impl.so`: `a3d14a6ebeccfcc8b743e8550c40e9ed79093f68948e0497c82bb2ee64f50ebe`
  - `libllama.so.0.6.0`: `d92d985ce279465a20ef000dfd0a2ec7f5091783b8768b78ba2480c8f844d386`
  - `libmtmd.so.0.6.0`: `70c8d29000a07664c5d0bb73167630db9958f19ea74d1836aac25479df795bbf`

## Test and outcome

The launch argv matched the existing failed L262144 CPU-Q4/native-MTP startup argv, with only `--no-op-offload` appended:

```text
/srv/ai/paged-kv/builds/master-pristine-958b444c677a-cuda134/bin/llama-server
-m /srv/ai/models/text/current.gguf
--alias qwen38-fast-turbo4-mtp
--jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja
-ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -t 16 -tb 16
-ctk q4_0 -ctv q4_0 --no-kv-offload -b 1024 -ub 256 --poll 0
--host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys
--metrics --reasoning-preserve --no-context-shift
--slot-save-path /srv/ai/paged-kv/results/pristine-master-958b444c677a-cuda134-validation-20261105/phase3-long-context-startup/master-cpu-mtp-on-L262144-startup-fill-only/attempt-001-L262144-allocation-startup/slot-state
--cache-ram 12288 --ctx-checkpoints 32 --checkpoint-min-step 8192
--cache-idle-slots --kv-unified --spec-type draft-mtp --spec-draft-n-max 2
--spec-draft-type-k turbo4 --spec-draft-type-v turbo4 --no-op-offload
```

The candidate reached 14,820 MiB allocated on the RTX 4080 before failing. Startup logged CPU-bound movable KV sides pinned at `q8_0`, then repeated `allocating 1730.00 MiB on device 0: cudaMalloc failed: out of memory` (1,814,041,344 bytes). The final reserve diagnostic was `needed 256.0 MiB ... 1158.2 MiB free after 8.0s — giving up`, followed by `failed to allocate compute buffers` and `failed to create MTP draft context`. Candidate exited 1 before readiness; no prompt/generation was sent and no acceptance counters exist. Immediately after candidate exit, GPU memory was 44 MiB used / 15,984 MiB free. No GPU performance result was measured.

## Restoration and raw evidence

After cleanup, production `llama-server.service` and `ai-profile-gateway.service` were active; `http://127.0.0.1:8080/health` returned `{"status":"ok"}`. The checkpoint timer was restored active and its oneshot remained inactive. Port 8091 remained on its existing owner; protected CPU-only port 8092 remained on PID 5947. Jellyfin remained running; no ffmpeg process was present. The test held `/tmp/ai-pager-benchmark.lock` and stopped/restarted production in server-then-gateway order.

Raw external evidence directory: `/srv/ai/paged-kv/results/pristine-master-958b444c677a-cuda134-validation-20261105/phase11b-no-op-offload-startup/` (`argv.json`, `state.json`, `candidate.log`). Original failed baseline argv: `/srv/ai/paged-kv/results/pristine-master-958b444c677a-cuda134-validation-20261105/phase3-long-context-startup/master-cpu-mtp-on-L262144-startup-fill-only/attempt-001-L262144-allocation-startup/cell/server-argv.json`.
