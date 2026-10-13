# GPU-only master context baselines — 106-01

- Frozen upstream master: `d2407c33f9595d1badccfd9a82a33c4c5ceda490`
- Build executable SHA-256: `527c34de411d4cad6290e828deb17a71816ae2b9620a3f1166b833e1267e0f0f`
- Model SHA-256: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`
- GPU/driver: NVIDIA GeForce RTX 4080, GPU-022c1422-d7bf-7443-27f1-0dab1254f05e, 595.99.02
- Batch/ubatch: 1024/256; target and embedded MTP draft K/V: Turbo4; n-max=2.
- All rows: 48 unique requests (12 warmups + 36 measured), T=0, seed=42; medians use only the three measured repetitions.
- Empty baseline was first. Filled rows are separate fresh server starts at exact L/C with GPU-only placement.

## Row summary

| L | C | Cases | Decode median range (tok/s) | MTP median range (%) | Source fill median (tok/s) | Snapshot save/restore |
|---:|---:|---:|---:|---:|---:|---:|
| 77,824 | empty | 48/48 | 80.5–107.3 | 56.4–93.1 | — | — |
| 8,192 | 6,144 | 48/48 | 74.2–105.9 | 50.8–94.9 | 1510.57 | 6,144/6,144 |
| 16,384 | 14,336 | 48/48 | 75.1–101.6 | 54.5–92.1 | 1455.59 | 14,336/14,336 |
| 32,768 | 30,000 | 48/48 | 71.2–96.0 | 55.1–90.1 | 1352.18 | 30,000/30,000 |
| 65,536 | 61,440 | 48/48 | 65.7–91.9 | 57.2–95.5 | 1110.22 | 61,440/61,440 |

Rates below are per prompt and reasoning mode. Full per-row values, actual argv, identities, raw paths and hashes are in [`MASTER106_GPU.json`](MASTER106_GPU.json).

## L=77,824, C=empty

Cases: 48 unique; 12 warmups; 36 measured. Runtime faults in preserved attempts: 0.
Runtime identity: PID `2543646`, capture `/srv/ai/paged-kv/results/forward/106-master/106-01/raw/startup-identity.json`, model SHA `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
Snapshot: `baseline`; save/restore `n/a`.

#### Decode median tok/s

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 107.34 | 88.86 | 104.84 |
| low | 106.26 | 83.04 | 96.90 |
| medium | 103.53 | 85.28 | 97.69 |
| xhigh | 102.80 | 80.53 | 97.41 |

#### MTP acceptance median %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 93.10 | 67.39 | 88.27 |
| low | 91.49 | 60.38 | 78.71 |
| medium | 87.59 | 63.53 | 79.80 |
| xhigh | 87.10 | 56.36 | 80.33 |

#### Fresh-prefill median tok/s

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 234.16 | 219.19 | 215.47 |
| low | 239.33 | 240.02 | 242.47 |
| medium | 222.83 | 197.86 | 206.33 |
| xhigh | 308.97 | 290.29 | 297.98 |

Effective argv:

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

Raw row directory: `/srv/ai/paged-kv/results/forward/106-master/106-01/gpu-empty-77824`.

## L=8,192, C=6,144

Cases: 48 unique; 12 warmups; 36 measured. Runtime faults in preserved attempts: 2.
Runtime identity: PID `2556944`, capture `/srv/ai/paged-kv/results/forward/106-master/106-01/raw/runtime-identities/gpu-8192-r2.json`, model SHA `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
Snapshot: `aec1319634b0cbdea1a637d02f2c3d7e526ead08606e7ff7fd526eea79478f39`; save/restore `6144/6144`.

#### Decode median tok/s

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 105.92 | 74.16 | 95.01 |
| low | 104.15 | 79.36 | 96.66 |
| medium | 95.31 | 78.88 | 97.08 |
| xhigh | 100.06 | 78.54 | 90.72 |

#### MTP acceptance median %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 94.91 | 50.81 | 79.55 |
| low | 92.15 | 58.74 | 83.56 |
| medium | 83.56 | 57.57 | 82.67 |
| xhigh | 87.95 | 58.82 | 74.14 |

#### Fresh-prefill median tok/s

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 115.00 | 103.92 | 109.75 |
| low | 1726.96 | 1722.26 | 1702.91 |
| medium | 101.51 | 93.42 | 97.04 |
| xhigh | 1695.59 | 1712.69 | 1726.51 |

Effective argv:

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

Fill curve points at the first append reaching each 8,192-token boundary:

| Boundary C | Observed C | New frontier tokens | Evaluated prompt tokens | Elapsed (s) | Fresh tok/s | Request SHA-256 | Response SHA-256 |
|---:|---:|---:|---:|---:|---:|---|---|
| 6,144 | 6,144 | 2254 | 2254 | 1.53 | 1469.99 | `f30a5e8a88098bf4693a137108cca4450acbfc092c0b45b8fdcd2405ac8d5cc2` | `4735b7a4e82fca00692d0e8a1c61dbd81363df310524ad9c2c5e29fc057bf139` |

Incomplete runtime attempts (preserved; targeted retry passed):
- Case 37: `incomplete SSE after CUDA illegal-memory-access server restart; see raw CUDA fault log`; raw `/srv/ai/paged-kv/results/forward/106-master/106-01/gpu-8192-6144-r2/raw/case-37-00.sse`.
- Case 43: `incomplete SSE after CUDA illegal-memory-access server restart; see raw CUDA fault log`; raw `/srv/ai/paged-kv/results/forward/106-master/106-01/gpu-8192-6144-r2/raw/case-43-00.sse`.

Raw row directory: `/srv/ai/paged-kv/results/forward/106-master/106-01/gpu-8192-6144-r2`.

## L=16,384, C=14,336

Cases: 48 unique; 12 warmups; 36 measured. Runtime faults in preserved attempts: 0.
Runtime identity: PID `2563260`, capture `/srv/ai/paged-kv/results/forward/106-master/106-01/raw/runtime-identities/gpu-16384.json`, model SHA `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
Snapshot: `9177fa8c9f61bb18d10d0d32b2c894a35cf8fee93f1df20bfe27a31b49ed414d`; save/restore `14336/14336`.

#### Decode median tok/s

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 100.07 | 75.08 | 91.02 |
| low | 101.57 | 82.11 | 89.31 |
| medium | 97.30 | 80.15 | 89.21 |
| xhigh | 98.83 | 78.42 | 84.28 |

#### MTP acceptance median %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 87.88 | 54.49 | 76.04 |
| low | 92.13 | 65.29 | 74.69 |
| medium | 84.73 | 61.17 | 73.83 |
| xhigh | 89.02 | 59.50 | 67.75 |

#### Fresh-prefill median tok/s

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 116.48 | 106.55 | 111.05 |
| low | 1733.11 | 1735.22 | 1735.06 |
| medium | 112.33 | 101.62 | 104.23 |
| xhigh | 1740.05 | 1740.70 | 1739.86 |

Effective argv:

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

Fill curve points at the first append reaching each 8,192-token boundary:

| Boundary C | Observed C | New frontier tokens | Evaluated prompt tokens | Elapsed (s) | Fresh tok/s | Request SHA-256 | Response SHA-256 |
|---:|---:|---:|---:|---:|---:|---|---|
| 8,192 | 9,257 | 1272 | 1272 | 0.96 | 1350.77 | `8c25fde53035cc303ce6241de51fb3afaa3bc6d096898ee75d734fc0ebd35334` | `6aaeec427424bae6f4fa7a172d563aa0f13a19569629dd04872f85d7acd614a6` |
| 14,336 | 14,336 | 2604 | 2604 | 1.80 | 1462.43 | `667ff9228fc2704189efd069b3fb00b1cb0aed90fdcd0d373c381f388c7b760c` | `c1cf4b1bc1c5eadc5410189d3f2639821c0023e42f2c6a6aee1b68b43b895c0e` |

Raw row directory: `/srv/ai/paged-kv/results/forward/106-master/106-01/gpu-16384-14336`.

## L=32,768, C=30,000

Cases: 48 unique; 12 warmups; 36 measured. Runtime faults in preserved attempts: 1.
Runtime identity: PID `2570683`, capture `/srv/ai/paged-kv/results/forward/106-master/106-01/raw/runtime-identities/gpu-32768-r2.json`, model SHA `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
Snapshot: `12ba28eb6221840382600bff908b07cde9322f06badf45ddf45456652de4927e`; save/restore `30000/30000`.

#### Decode median tok/s

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 96.00 | 76.48 | 90.10 |
| low | 93.74 | 72.86 | 85.81 |
| medium | 92.16 | 74.27 | 86.81 |
| xhigh | 94.45 | 71.15 | 72.41 |

#### MTP acceptance median %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 89.39 | 60.00 | 80.39 |
| low | 88.66 | 57.31 | 76.75 |
| medium | 83.46 | 57.50 | 76.19 |
| xhigh | 90.12 | 55.10 | 56.99 |

#### Fresh-prefill median tok/s

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 115.31 | 105.57 | 108.72 |
| low | 1611.32 | 1609.93 | 1613.33 |
| medium | 156.16 | 141.77 | 146.21 |
| xhigh | 1613.99 | 1612.81 | 1613.51 |

Effective argv:

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

Fill curve points at the first append reaching each 8,192-token boundary:

| Boundary C | Observed C | New frontier tokens | Evaluated prompt tokens | Elapsed (s) | Fresh tok/s | Request SHA-256 | Response SHA-256 |
|---:|---:|---:|---:|---:|---:|---|---|
| 8,192 | 9,257 | 1272 | 1272 | 0.96 | 1352.18 | `8c25fde53035cc303ce6241de51fb3afaa3bc6d096898ee75d734fc0ebd35334` | `ed6f95780df33850d0baa284cb8e7830ab74ab73131d78f8b3d201ec83cdeb63` |
| 16,384 | 18,071 | 3224 | 3224 | 2.23 | 1460.77 | `a1ad793ecb8504eab92eaef0511fa1338c601203d1c6d1c08985bc94dfab1c25` | `85be961b13bd46daf332e9a8fc29ef42026c36d4998e5753eeaa43f28632dc07` |
| 24,576 | 26,653 | 4096 | 4096 | 3.06 | 1353.39 | `ab20d5b1c76ec3f29e61c3ac4db2ea13c4036e7fb4ac04b9894ec5fca7dfa69e` | `92f707e0f0bbb3c2d56b0c4e14bd243c7c97ad1e61b3faed289ea04b1125c866` |
| 30,000 | 30,000 | 698 | 698 | 0.80 | 914.71 | `c502d0d12d98a3e8dceb70619e4bf34ceb1c75d07754289a1ed99e7f79f69474` | `2c0a398e4e152f6fdbcb85d79c2954489d785f0e687ae1da9c10bdef0779a52a` |

Incomplete runtime attempts (preserved; targeted retry passed):
- Case 13: `SSE error [500]: got exception: DFlash target argmax row lookup failed`; raw `/srv/ai/paged-kv/results/forward/106-master/106-01/gpu-32768-30000-r2/raw/case-13-00.sse`.

Raw row directory: `/srv/ai/paged-kv/results/forward/106-master/106-01/gpu-32768-30000-r2`.

## L=65,536, C=61,440

Cases: 48 unique; 12 warmups; 36 measured. Runtime faults in preserved attempts: 0.
Runtime identity: PID `2587190`, capture `/srv/ai/paged-kv/results/forward/106-master/106-01/raw/runtime-identities/gpu-65536-r4.json`, model SHA `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
Snapshot: `2a22a1334a1cd4d861e0dbcff432d0d7823830577cf9f6ef3f884f95202b66c4`; save/restore `61440/61440`.

#### Decode median tok/s

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 91.91 | 74.48 | 82.24 |
| low | 86.24 | 65.70 | 78.19 |
| medium | 86.51 | 69.12 | 81.08 |
| xhigh | 89.18 | 71.06 | 77.79 |

#### MTP acceptance median %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 94.26 | 67.57 | 79.72 |
| low | 90.37 | 57.20 | 77.32 |
| medium | 86.67 | 59.02 | 78.14 |
| xhigh | 95.51 | 65.89 | 76.75 |

#### Fresh-prefill median tok/s

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 104.50 | 96.38 | 98.29 |
| low | 1365.53 | 1365.01 | 1364.98 |
| medium | 96.58 | 87.76 | 90.06 |
| xhigh | 1364.87 | 1364.77 | 1364.62 |

Effective argv:

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

Fill curve points at the first append reaching each 8,192-token boundary:

| Boundary C | Observed C | New frontier tokens | Evaluated prompt tokens | Elapsed (s) | Fresh tok/s | Request SHA-256 | Response SHA-256 |
|---:|---:|---:|---:|---:|---:|---|---|
| 8,192 | 9,257 | 1272 | 1272 | 0.94 | 1378.53 | `8c25fde53035cc303ce6241de51fb3afaa3bc6d096898ee75d734fc0ebd35334` | `385df07244cd60b8b1b94df018a65adf356973e9e48bc27a49b9aafc22d07e77` |
| 16,384 | 18,071 | 3224 | 3224 | 2.20 | 1481.58 | `a1ad793ecb8504eab92eaef0511fa1338c601203d1c6d1c08985bc94dfab1c25` | `d74d91834c3132ddcd4abc975400d2f3db53d4a716c94892ccd70a34bfb0f997` |
| 24,576 | 26,653 | 4096 | 4096 | 3.02 | 1373.52 | `ab20d5b1c76ec3f29e61c3ac4db2ea13c4036e7fb4ac04b9894ec5fca7dfa69e` | `d8f517a2046f87ec197db9cf0f2156a09038de831383751be3df693e0652fb85` |
| 32,768 | 35,080 | 4096 | 4096 | 3.29 | 1260.54 | `b379d760dae0d06c139166db6beca7af56b2e96d9caed890fac8741692e0bd75` | `a830699db0639128eee3c5f9522a3210f9c9c57a9f7ed7b1a035572e9ab0dd81` |
| 40,960 | 41,953 | 2777 | 2777 | 2.55 | 1110.22 | `e1e45ac9dc0dfa5a77056d01a5242c1a5ebc0a43dd9fa66896428a903ed2f507` | `7dcee1b8af7913b3a63f914e03b3fa3a2c8e24b6c8f495f820b04c5f8a35ce51` |
| 49,152 | 49,339 | 4096 | 4096 | 3.77 | 1102.62 | `e4190931875623e0c7f6f014bed262e1c8c74be691f2cac8aa5f5c60c60abd7e` | `a7c2df01580d63402dfe8ea656fb770549599ad5afcf23744ea8d500a9679ba1` |
| 57,344 | 59,757 | 4096 | 4096 | 4.10 | 1014.83 | `36289fea6e8e6d0890f4f760acdb48c0e3303d5d05351b0e5f0ebc810073dce6` | `71f85495abae3609a3ed463aaf8c4b4733c5e646bb81da90fdabbb586383f0eb` |
| 61,440 | 61,440 | 2 | 2 | 0.20 | 14.51 | `93db09cae1d22f0294711c3aeab3d922416875441128d2eecac529fa6fbe8ed5` | `25921bf663b6ab6cc6a4823967574f09b3d26e10a1d1f41c9942520ca9edc56d` |

Tail adjustment: deterministic assistant closure reached exact C after the source tokenizer skipped the two-token remainder. See the raw fill schedule/request/SSE in the JSON report.

Raw row directory: `/srv/ai/paged-kv/results/forward/106-master/106-01/gpu-65536-61440-r4`.

## Deferred verification

None. The required GPU rows completed on the RTX 4080; the prior managed qwen38-aco profile was restored and verified.
