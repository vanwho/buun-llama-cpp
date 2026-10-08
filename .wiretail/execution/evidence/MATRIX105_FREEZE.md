# Matrix 105 frozen configuration

Revision: `hotpath-v10-20260914`

Candidate: `6b5b225b4791c1c5a9402304207c74e1a119e5c7f51a5300a891909dbdae34c7`; receipt `10612b68cd18a9f24543549193a4f6d0b6a0950e090c7f8a0e1188bcf3c05759`.
Model: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
Harness: `fe59102934baff94ba8d81adb1c50bbefdc4cf60e8a2bc7e1ab25053fd09544e`; self-test passed.
8K source seed: 5344 tokens, H=4096, free=2848 (minimum reserve 1536): PASS.

## Frozen launch commands

### 105-06 — original fast-profile settings, GPU-only, empty slot, L=77,824

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 77824 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

### 105-07 — CPU-F16 target-KV, empty slot, L=77,824

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 77824 -np 1 -ctk f16 -ctv f16 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

### 105-08 — ACO, empty slot, L=77,824, H=65,536 (256 pages)

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 77824 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager selective --kv-router probe-rerank --kv-page-size 256 --kv-hot-pages 256 --kv-pin-recent 0 --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

### 105-09 — ACO, L=8,192, H=4,096 (16 pages)

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 8192 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager selective --kv-router probe-rerank --kv-page-size 256 --kv-hot-pages 16 --kv-pin-recent 0 --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

### 105-10 — GPU-only control, L=8,192

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 8192 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

### 105-11 — CPU-F16 target-KV control, L=8,192

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 8192 -np 1 -ctk f16 -ctv f16 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

### 105-12 — ACO, L=262,144, H=16,384 (64 pages)

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager selective --kv-router probe-rerank --kv-page-size 256 --kv-hot-pages 64 --kv-pin-recent 0 --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

### 105-13 — GPU-only control, L=32,768

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 32768 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

### 105-14 — CPU-F16 target-KV control, L=32,768

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 32768 -np 1 -ctk f16 -ctv f16 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

### 105-15 — ACO, L=262,144, H=32,768 (128 pages)

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager selective --kv-router probe-rerank --kv-page-size 256 --kv-hot-pages 128 --kv-pin-recent 0 --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

### 105-16 — GPU-only control, L=65,536

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 65536 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

### 105-17 — CPU-F16 target-KV control, L=65,536

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 65536 -np 1 -ctk f16 -ctv f16 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

### 105-18 — ACO, L=262,144, H=65,536 (256 pages), fill C=120,000

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager selective --kv-router probe-rerank --kv-page-size 256 --kv-hot-pages 256 --kv-pin-recent 0 --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

### 105-19 — ACO, L=262,144, H=65,536 (256 pages), fill C=184,000

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager selective --kv-router probe-rerank --kv-page-size 256 --kv-hot-pages 256 --kv-pin-recent 0 --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

### 105-20 — ACO, L=262,144, H=65,536 (256 pages), fill C=250,000

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager selective --kv-router probe-rerank --kv-page-size 256 --kv-hot-pages 256 --kv-pin-recent 0 --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

## 105-06 empty-context baseline

Runtime preflight: exact frozen argv active; service health 200; process resolves to frozen candidate; slot snapshot empty. No generation startup probe was sent.

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 77824 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

The 48 request rows use the three canonical prompts, request-level modes off/low/medium/xhigh, one 40-token warmup and three measured requests per prompt/mode. Each request is independent; the baseline sends no repository seed, source context, or A/B/A turns.
