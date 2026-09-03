# Phase 00 baseline provenance

Captured 2026-09-03 on `/srv/repos/vanwho/buun-llama-cpp`, before implementation
changes. The machine-readable source of record is [reference.json](reference.json).

## Source and remotes

- Buun source: `6de47d6ffdaa2bf6bff76024067bf68c861b9353`, remote
  `git@github.com:vanwho/buun-llama-cpp.git` (origin).
- Compared upstream source: `cb703be37e3628dadb71912f3b3b25b82090555b`, remote
  `https://github.com/spiritbuun/buun-llama-cpp.git` (upstream).
- After separate read-only `git fetch --prune origin` and `git fetch --prune upstream`,
  the working source is 13 commits ahead and 0 behind both `origin/HEAD` and
  `upstream/HEAD`; both remote tips are the pinned base commit. No relevant upstream
  movement was observed.

## Model and geometry

The deployed `current.gguf` resolves to
`/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf`. Its SHA256 is recorded in
`reference.json`. A read-only GGUF dump confirmed architecture `qwen35`, 65 blocks,
262144 native context, 4 KV heads, 256 K/V width, `full_attention_interval=4`, and
one native `nextn_predict_layers` MTP layer. There is no separate MTP sidecar file;
MTP identity is the embedded native layer in this same GGUF.

Using the observed geometry and Turbo4's measured 4.125 bits/value:

- target: 16 full-attention layers × 4 KV heads × (256 K + 256 V) = 16,896 bytes/token;
  256-token page = 4,325,376 bytes (4.125 MiB); 77,824 tokens = 1.224609375 GiB.
- native MTP: 1 layer with the same heads and widths = 1,056 bytes/token;
  262,144 tokens = 276,824,064 bytes (264 MiB).
- combined target plus MTP payload at 262,144 tokens = 4.3828125 GiB.

These are encoded payloads only; allocator/VMM, tables, masks, staging, graph reserves,
recurrent state, and scratch are not included.

## Build and benchmark contract

The captured server is `/srv/ai/paged-kv/build/buun/bin/llama-server`, build version
`1 (7d30a72)`, SHA256 in `reference.json`. It is a Release, shared-library, CUDA,
CUDA-FA, CUDA-graphs, native-optimized build using `/usr/bin/c++` and `/usr/bin/nvcc`.
The exact fast profile and benchmark command are preserved in the manifest and the
profile remains unmodified. Benchmark API keys and tokens were not read or recorded.

No model, profile, service, or API-key state was mutated. No benchmark was rerun in this
task; historical harness identity is retained only as provenance, not new acceptance
evidence. Hardware-specific pager verification is deferred until the later benchmark
and GPU correctness tasks.
