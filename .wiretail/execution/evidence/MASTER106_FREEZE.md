# Frozen master build and runtime identity — 106-01

- Master commit: `d2407c33f9595d1badccfd9a82a33c4c5ceda490`
- Executable SHA-256: `527c34de411d4cad6290e828deb17a71816ae2b9620a3f1166b833e1267e0f0f`
- Model SHA-256: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`
- GPU/driver: NVIDIA GeForce RTX 4080, GPU-022c1422-d7bf-7443-27f1-0dab1254f05e, 595.99.02
- Configure: `cmake -S /srv/ai/paged-kv/worktrees/master106-d2407c33f959 -B /srv/ai/paged-kv/builds/master106-d2407c33f959 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89-real -DGGML_CUDA_FA=ON -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=ON`
- Build: `cmake --build /srv/ai/paged-kv/builds/master106-d2407c33f959 --parallel 16 --target llama-server`
- Build retry environment: CPATH shim `/tmp/master106-gcc12/shim`; source unchanged, GGML AMX disabled.
- The fork master does not accept `--spec-draft-kv-device`; the managed service supplied the recorded site checkpoint/cache options.
- Filled-row argv arrays below are the observed `/proc/<pid>/cmdline`; the empty baseline retains its original startup argv.

## Rows

| Row | PID | L/C | Cases | Snapshot | Identity capture |
|---|---:|---:|---:|---:|---|
| gpu-empty-77824 | 2543646 | 77,824/empty | 48 | baseline | `/srv/ai/paged-kv/results/forward/106-master/106-01/raw/startup-identity.json` |
| gpu-8192-6144-r2 | 2556944 | 8,192/6,144 | 48 | 6,144/6,144 | `/srv/ai/paged-kv/results/forward/106-master/106-01/raw/runtime-identities/gpu-8192-r2.json` |
| gpu-16384-14336 | 2563260 | 16,384/14,336 | 48 | 14,336/14,336 | `/srv/ai/paged-kv/results/forward/106-master/106-01/raw/runtime-identities/gpu-16384.json` |
| gpu-32768-30000-r2 | 2570683 | 32,768/30,000 | 48 | 30,000/30,000 | `/srv/ai/paged-kv/results/forward/106-master/106-01/raw/runtime-identities/gpu-32768-r2.json` |
| gpu-65536-61440-r4 | 2587190 | 65,536/61,440 | 48 | 61,440/61,440 | `/srv/ai/paged-kv/results/forward/106-master/106-01/raw/runtime-identities/gpu-65536-r4.json` |

## Loaded project DSOs

```json
{
  "/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/libggml-base.so.0.26.0": "9f883e824e6cbf40d5fbedafe1a185c70e09fabc7574b6b0d8cf2e84ec1e2166",
  "/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/libggml-cpu.so.0.26.0": "9a1bbce31a0387341cf6057d688b582cf9de0183a109ca59b62692c6c297d6ea",
  "/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/libggml-cuda.so.0.26.0": "e35eeb2413d026fc1651599d67ff85474f88099343bac05887b3a1fb7dbd42d0",
  "/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/libggml.so.0.26.0": "60685d4e302c602021f74b3ae6226f37f62ef7d84bf352fc79803560dc3d5b23",
  "/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/libllama-common.so.0.6.0": "e34553856be80a2cc373ea8559b9ecfb7c5ab5c6611e3903b79d87ae33843952",
  "/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/libllama-server-impl.so": "96cb874ff3d286784378c3a5949d4f68300266b9e3ee40b7f8f126dff1e08f1b",
  "/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/libllama.so.0.6.0": "011722e51f00e142778ce181d363a9f5774b3340e41c030b6bb51a8190f769c9",
  "/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/libmtmd.so.0.6.0": "0c285981b7cb11fbd8d61cb24ed3451e7731b75782119cfa078a69071a6c79fc"
}
```

## Effective argv — gpu-empty-77824

```json
[
  "/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server",
  "-m",
  "/srv/ai/models/text/current.gguf",
  "--alias",
  "qwen38-fast-turbo4-mtp",
  "-ngl",
  "999",
  "--fit",
  "off",
  "-fa",
  "on",
  "-c",
  "77824",
  "-np",
  "1",
  "-ctk",
  "turbo4",
  "-ctv",
  "turbo4",
  "-b",
  "1024",
  "-ub",
  "256",
  "--poll",
  "0",
  "--host",
  "0.0.0.0",
  "--port",
  "8080",
  "--api-key-file",
  "/srv/ai/config/llama/api-keys",
  "--metrics",
  "--reasoning-preserve",
  "--no-context-shift",
  "--device",
  "CUDA0",
  "--ctx-checkpoints",
  "0",
  "--cache-ram",
  "0",
  "--no-cache-idle-slots",
  "--spec-type",
  "draft-mtp",
  "--spec-draft-n-max",
  "2",
  "--spec-draft-type-k",
  "turbo4",
  "--spec-draft-type-v",
  "turbo4"
]
```

## Effective argv — gpu-8192-6144-r2

```json
[
  "/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server",
  "-m",
  "/srv/ai/models/text/current.gguf",
  "--alias",
  "qwen38-fast-turbo4-mtp",
  "-ngl",
  "999",
  "--fit",
  "off",
  "-fa",
  "on",
  "-c",
  "8192",
  "-np",
  "1",
  "-ctk",
  "turbo4",
  "-ctv",
  "turbo4",
  "-b",
  "1024",
  "-ub",
  "256",
  "--poll",
  "0",
  "--host",
  "0.0.0.0",
  "--port",
  "8080",
  "--api-key-file",
  "/srv/ai/config/llama/api-keys",
  "--metrics",
  "--reasoning-preserve",
  "--no-context-shift",
  "--device",
  "CUDA0",
  "--cache-ram",
  "12288",
  "--ctx-checkpoints",
  "8",
  "--checkpoint-min-step",
  "8192",
  "--slot-save-path",
  "/srv/ai/paged-kv/results/forward/106-master/106-01/raw/slot-state",
  "--cache-idle-slots",
  "--kv-unified",
  "--spec-type",
  "draft-mtp",
  "--spec-draft-n-max",
  "2",
  "--spec-draft-type-k",
  "turbo4",
  "--spec-draft-type-v",
  "turbo4"
]
```

## Effective argv — gpu-16384-14336

```json
[
  "/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server",
  "-m",
  "/srv/ai/models/text/current.gguf",
  "--alias",
  "qwen38-fast-turbo4-mtp",
  "-ngl",
  "999",
  "--fit",
  "off",
  "-fa",
  "on",
  "-c",
  "16384",
  "-np",
  "1",
  "-ctk",
  "turbo4",
  "-ctv",
  "turbo4",
  "-b",
  "1024",
  "-ub",
  "256",
  "--poll",
  "0",
  "--host",
  "0.0.0.0",
  "--port",
  "8080",
  "--api-key-file",
  "/srv/ai/config/llama/api-keys",
  "--metrics",
  "--reasoning-preserve",
  "--no-context-shift",
  "--device",
  "CUDA0",
  "--cache-ram",
  "12288",
  "--ctx-checkpoints",
  "8",
  "--checkpoint-min-step",
  "8192",
  "--slot-save-path",
  "/srv/ai/paged-kv/results/forward/106-master/106-01/raw/slot-state",
  "--cache-idle-slots",
  "--kv-unified",
  "--spec-type",
  "draft-mtp",
  "--spec-draft-n-max",
  "2",
  "--spec-draft-type-k",
  "turbo4",
  "--spec-draft-type-v",
  "turbo4"
]
```

## Effective argv — gpu-32768-30000-r2

```json
[
  "/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server",
  "-m",
  "/srv/ai/models/text/current.gguf",
  "--alias",
  "qwen38-fast-turbo4-mtp",
  "-ngl",
  "999",
  "--fit",
  "off",
  "-fa",
  "on",
  "-c",
  "32768",
  "-np",
  "1",
  "-ctk",
  "turbo4",
  "-ctv",
  "turbo4",
  "-b",
  "1024",
  "-ub",
  "256",
  "--poll",
  "0",
  "--host",
  "0.0.0.0",
  "--port",
  "8080",
  "--api-key-file",
  "/srv/ai/config/llama/api-keys",
  "--metrics",
  "--reasoning-preserve",
  "--no-context-shift",
  "--device",
  "CUDA0",
  "--cache-ram",
  "12288",
  "--ctx-checkpoints",
  "8",
  "--checkpoint-min-step",
  "8192",
  "--slot-save-path",
  "/srv/ai/paged-kv/results/forward/106-master/106-01/raw/slot-state",
  "--cache-idle-slots",
  "--kv-unified",
  "--spec-type",
  "draft-mtp",
  "--spec-draft-n-max",
  "2",
  "--spec-draft-type-k",
  "turbo4",
  "--spec-draft-type-v",
  "turbo4"
]
```

## Effective argv — gpu-65536-61440-r4

```json
[
  "/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server",
  "-m",
  "/srv/ai/models/text/current.gguf",
  "--alias",
  "qwen38-fast-turbo4-mtp",
  "-ngl",
  "999",
  "--fit",
  "off",
  "-fa",
  "on",
  "-c",
  "<L>",
  "-np",
  "1",
  "-ctk",
  "turbo4",
  "-ctv",
  "turbo4",
  "-b",
  "1024",
  "-ub",
  "256",
  "--poll",
  "0",
  "--host",
  "0.0.0.0",
  "--port",
  "8080",
  "--api-key-file",
  "/srv/ai/config/llama/api-keys",
  "--metrics",
  "--reasoning-preserve",
  "--no-context-shift",
  "--device",
  "CUDA0",
  "--cache-ram",
  "12288",
  "--ctx-checkpoints",
  "8",
  "--checkpoint-min-step",
  "8192",
  "--slot-save-path",
  "/srv/ai/paged-kv/results/forward/106-master/106-01/raw/slot-state",
  "--cache-idle-slots",
  "--kv-unified",
  "--spec-type",
  "draft-mtp",
  "--spec-draft-n-max",
  "2",
  "--spec-draft-type-k",
  "turbo4",
  "--spec-draft-type-v",
  "turbo4"
]
```
