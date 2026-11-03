# Frozen remaining benchmark commands

Ten variants; nine have missing measurements. All commands use the already-built immutable binaries recorded in remaining-matrix.json. No C++ rebuild or gateway is involved.

One clean server per variant/configuration; append disjoint repo source chunks through the required checkpoints. At each checkpoint all selected prompts for off, then low, then medium, then xhigh use the same loaded base. Each selected prompt has one 40-token warmup and three measured 400-token-budget requests. Valid early EOS is allowed. Existing complete measurements are retained.

K = 1024 tokens. L is server allocation; H is GPU hot capacity. Final C=256000 in L=262144 leaves output reserve and counts as near-full 256K. The 128K GPU-off endpoint is C=130048 in L=131072. B/U is always 1024/256; target CPU KV uses Q4_0, GPU/ACO KV uses Turbo4. MTP uses n-max=2 and Turbo4 draft type requests. Master uses stock draft placement; the fork CPU-MTP and ACO-MTP variants explicitly pin draft KV to GPU.

Ordinary controls retain the known-working 8-checkpoint/12288-MiB prompt-cache profile. The zero-checkpoint profile previously forced stock Qwen hybrid recurrent state to re-prefill the entire context on the second independent query. ACO retains its measured zero-checkpoint, no-idle-cache profile. Do not remove checkpoints from the ordinary controls.

Gateway: bypassed. Direct endpoint: http://127.0.0.1:8080/v1/chat/completions. The suite temporarily pauses the profile gateway and managed server and restores services afterward. 8091 and 8092 are not changed. Any unrelated GPU compute process is a setup block, never a capacity finding.

Reasoning is a request setting; changing it does not restart the server or refill the repository base:

- off: `{"chat_template_kwargs":{"enable_thinking":false,"preserve_thinking":true},"reasoning_effort":"none"}`
- low: `{"chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true},"reasoning_effort":"low"}`
- medium: `{"chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true},"reasoning_effort":"medium"}`
- xhigh: `{"chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true},"reasoning_effort":"xhigh"}`

## master-cpu-mtp-on

- C=6,144: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=12,288: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=28,672: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=62,464: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=78,848: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=88,064: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=100,352: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=130,048: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=196,608: off: prompts 3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=256,000: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.

Primary server command:

```bash
env LD_LIBRARY_PATH=/srv/ai/paged-kv/builds/master106-d2407c33f959/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -t 16 -tb 16 -ctk q4_0 -ctv q4_0 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/final-remaining-20261103/master-cpu-mtp-on/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4
```

Predeclared startup-OOM fallback (never changes B/U):

```bash
env LD_LIBRARY_PATH=/srv/ai/paged-kv/builds/master106-d2407c33f959/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 200704 -np 1 -t 16 -tb 16 -ctk q4_0 -ctv q4_0 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/final-remaining-20261103/master-cpu-mtp-on/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4
```

Predeclared startup-OOM fallback (never changes B/U):

```bash
env LD_LIBRARY_PATH=/srv/ai/paged-kv/builds/master106-d2407c33f959/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 133120 -np 1 -t 16 -tb 16 -ctk q4_0 -ctv q4_0 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/final-remaining-20261103/master-cpu-mtp-on/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4
```

Run this variant:

```bash
python3 /srv/repos/vanwho/buun-llama-cpp/tools/server/bench/run-benchmark-gaps.py --run --variant master-cpu-mtp-on
```

## master-cpu-mtp-off

- C=6,144: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=12,288: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=28,672: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=62,464: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=78,848: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=88,064: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=100,352: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=130,048: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=196,608: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=256,000: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.

Primary server command:

```bash
env LD_LIBRARY_PATH=/srv/ai/paged-kv/builds/master106-d2407c33f959/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -t 16 -tb 16 -ctk q4_0 -ctv q4_0 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/final-remaining-20261103/master-cpu-mtp-off/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --spec-type none
```

Run this variant:

```bash
python3 /srv/repos/vanwho/buun-llama-cpp/tools/server/bench/run-benchmark-gaps.py --run --variant master-cpu-mtp-off
```

## master-gpu-mtp-on

- C=28,672: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=62,464: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.

Primary server command:

```bash
env LD_LIBRARY_PATH=/srv/ai/paged-kv/builds/master106-d2407c33f959/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 65536 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/final-remaining-20261103/master-gpu-mtp-on/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4
```

Run this variant:

```bash
python3 /srv/repos/vanwho/buun-llama-cpp/tools/server/bench/run-benchmark-gaps.py --run --variant master-gpu-mtp-on
```

## master-gpu-mtp-off

- C=6,144: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=12,288: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=28,672: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=62,464: low: prompts 1,2,3.
- C=88,064: off: prompts 1,2,3; low: prompts 1,2,3.
- C=130,048: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.

Primary server command:

```bash
env LD_LIBRARY_PATH=/srv/ai/paged-kv/builds/master106-d2407c33f959/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 131072 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/final-remaining-20261103/master-gpu-mtp-off/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --spec-type none
```

Predeclared startup-OOM fallback (never changes B/U):

```bash
env LD_LIBRARY_PATH=/srv/ai/paged-kv/builds/master106-d2407c33f959/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 122880 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/final-remaining-20261103/master-gpu-mtp-off/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --spec-type none
```

Run this variant:

```bash
python3 /srv/repos/vanwho/buun-llama-cpp/tools/server/bench/run-benchmark-gaps.py --run --variant master-gpu-mtp-off
```

## fork-cpu-mtp-on

- C=12,288: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=28,672: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=62,464: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=78,848: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=88,064: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=100,352: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=130,048: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=196,608: off: prompts 3.

Primary server command:

```bash
env LD_LIBRARY_PATH=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -t 16 -tb 16 -ctk q4_0 -ctv q4_0 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/final-remaining-20261103/fork-cpu-mtp-on/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --kv-pager off --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4 --spec-draft-kv-device gpu
```

Run this variant:

```bash
python3 /srv/repos/vanwho/buun-llama-cpp/tools/server/bench/run-benchmark-gaps.py --run --variant fork-cpu-mtp-on
```

## fork-cpu-mtp-off

- C=78,848: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=88,064: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=100,352: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=196,608: off: prompts 3.

Primary server command:

```bash
env LD_LIBRARY_PATH=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -t 16 -tb 16 -ctk q4_0 -ctv q4_0 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/final-remaining-20261103/fork-cpu-mtp-off/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --kv-pager off --spec-type none
```

Run this variant:

```bash
python3 /srv/repos/vanwho/buun-llama-cpp/tools/server/bench/run-benchmark-gaps.py --run --variant fork-cpu-mtp-off
```

## fork-gpu-mtp-on

- C=6,144: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=28,672: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=62,464: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.

Primary server command:

```bash
env LD_LIBRARY_PATH=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 65536 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/final-remaining-20261103/fork-gpu-mtp-on/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --kv-pager off --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4
```

Run this variant:

```bash
python3 /srv/repos/vanwho/buun-llama-cpp/tools/server/bench/run-benchmark-gaps.py --run --variant fork-gpu-mtp-on
```

## fork-gpu-mtp-off

- C=130,048: off: prompts 1,2,3; low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.

Primary server command:

```bash
env LD_LIBRARY_PATH=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 131072 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/final-remaining-20261103/fork-gpu-mtp-off/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --kv-pager off --spec-type none
```

Predeclared startup-OOM fallback (never changes B/U):

```bash
env LD_LIBRARY_PATH=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 122880 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/final-remaining-20261103/fork-gpu-mtp-off/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --kv-pager off --spec-type none
```

Run this variant:

```bash
python3 /srv/repos/vanwho/buun-llama-cpp/tools/server/bench/run-benchmark-gaps.py --run --variant fork-gpu-mtp-off
```

## aco-mtp-on

Complete; excluded from execution queue. Retain existing 40 complete mode/context cells.

Primary server command:

```bash
env LD_LIBRARY_PATH=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/final-remaining-20261103/aco-mtp-on/slot-state --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --kv-pager selective --kv-page-size 256 --kv-hot-pages 256 --kv-router probe-rerank --kv-pin-recent 0 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4 --spec-draft-kv-device gpu
```

Run this variant:

```bash
python3 /srv/repos/vanwho/buun-llama-cpp/tools/server/bench/run-benchmark-gaps.py --run --variant aco-mtp-on
```

## aco-mtp-off

- C=6,144: medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=12,288: medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=28,672: medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=62,464: medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=78,848: medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=88,064: medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=100,352: medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=130,048: medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=196,608: low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.
- C=256,000: low: prompts 1,2,3; medium: prompts 1,2,3; xhigh: prompts 1,2,3.

Primary server command:

```bash
env LD_LIBRARY_PATH=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/final-remaining-20261103/aco-mtp-off/slot-state --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --kv-pager selective --kv-page-size 256 --kv-hot-pages 256 --kv-router probe-rerank --kv-pin-recent 0 --spec-type none
```

Run this variant:

```bash
python3 /srv/repos/vanwho/buun-llama-cpp/tools/server/bench/run-benchmark-gaps.py --run --variant aco-mtp-off
```

## Execution and failure rules

The /tmp/run_remaining_benchmark_matrix.py scheduler is superseded. It searched only its new output root and wrongly queued existing results plus unrequested small-hot variants. Do not execute it. The checked-in runner uses only the explicit remaining selections in this manifest.

Validation/listing (no server or benchmark requests):

```bash
python3 /srv/repos/vanwho/buun-llama-cpp/tools/server/bench/run-benchmark-gaps.py --validate
```

Execute all remaining variants serially:

```bash
python3 /srv/repos/vanwho/buun-llama-cpp/tools/server/bench/run-benchmark-gaps.py --run
```

Raw results and resumable state stay outside Git under the manifest output_root. No binary, raw SSE or KV snapshot is committed. Each attempt has a new directory. Completed mode/prompt pairs are skipped on restart; an incomplete pair is rerun with its warmup. There are at most two completed clean attempts per allocation across restarts; interrupted attempts recover their finished pairs. The runner records genuine OOM/runtime/answer failures and proceeds to other variants; it never writes a successful speed for a failed point. Stock master failures do not authorize patching the master control binary.

Master CPU+MTP startup fallbacks are L262144 -> L200704 -> L133120, allowing smaller contexts to be measured even if the largest stock MTP workspace cannot fit. GPU-off fallback is L131072 -> L122880; a failed 128K setup remains a recorded capacity limit, not a claimed 128K benchmark. ACO remains fixed H65536/L262144.

## Shared-prefix reasoning template

The original GGUF template prepends low/xhigh effort instructions before the entire repository history. Those modes therefore cannot share an identical filled KV prefix under the unmodified template. Every frozen server command now explicitly loads `qwen38-shared-prefix.jinja`. It preserves the exact effort instruction words and their system role but moves them immediately before the final user query. Off/medium generation suffixes are unchanged. `preserve_thinking=true` is explicit for all modes and fill requests. The model, CUDA kernels, KV policy and benchmark questions are unchanged.

The template and original-template SHA-256s are recorded in `qwen38-shared-prefix.provenance.json`. This is benchmark-only prompt plumbing. Existing earlier-template results remain individually valid and are retained; new late-effort results must be identified by protocol/template hash. Do not claim that early-effort and late-effort prompt streams are byte-identical. This manifest fills the previously audited gaps; it does not silently rerun all previously completed low/xhigh rows.

Server environment is explicit per command: candidate-local LD_LIBRARY_PATH, CUDA_VISIBLE_DEVICES=0, OMP_NUM_THREADS=16. The Python launcher also removes inherited GGML_* and LLAMA_* exports so old diagnostic routes or launcher geometry cannot override the benchmark. Each actual environment/argv is recorded with the result.

## Focused hot-capacity boundary investigation

The checkpoint ladder is unchanged. Before the missing matrix, repeat C=62464 and C=78848 with H=65536/L=262144 for ACO MTP off and on. Use reasoning off and the original three prompts, one 40-token warmup plus three 400-token-budget measurements per prompt. Both points use one incremental repository fill per MTP configuration. Only the already collected before/after snapshots are now persisted; no CUDA hot-path diagnostic flags, forced routes or changed batch settings are enabled. Use request epoch timestamps to compare GPU resources during requests rather than including idle periods.

Run only these investigations (48 canonical requests total):

```bash
python3 /srv/repos/vanwho/buun-llama-cpp/tools/server/bench/run-benchmark-gaps.py --run --investigation aco-hot-boundary-mtp-off --investigation aco-hot-boundary-mtp-on
```

aco-hot-boundary-mtp-off exact server command:

```bash
env LD_LIBRARY_PATH=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/final-remaining-20261103/aco-hot-boundary-mtp-off/slot-state --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --kv-pager selective --kv-page-size 256 --kv-hot-pages 256 --kv-router probe-rerank --kv-pin-recent 0 --spec-type none
```

aco-hot-boundary-mtp-on exact server command:

```bash
env LD_LIBRARY_PATH=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/final-remaining-20261103/aco-hot-boundary-mtp-on/slot-state --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --kv-pager selective --kv-page-size 256 --kv-hot-pages 256 --kv-router probe-rerank --kv-pin-recent 0 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4 --spec-draft-kv-device gpu
```
