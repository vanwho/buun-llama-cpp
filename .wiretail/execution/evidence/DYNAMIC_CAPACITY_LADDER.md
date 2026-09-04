# Dynamic capacity ladder - task 14-02

## Verdict

The release server was exercised against the real Qwen 3.8 GGUF on the RTX
4080. The bounded KV pager target is derived from the resolved context and the
admitted page count: target bytes and physical rows change with `H`, while the
page byte cost remains constant because the model geometry and 256-token page
size are constant. Auto-fit, explicit hot-page limits, headroom, too-small
refusal, GPU MTP residency, odd-context rounding, and service restoration were
observed.

The raw run also records two boundaries that are material to interpreting the
result. Native MTP reserves its extra KV independently after pager creation,
so the pager startup line has `mtp_rows=0`; the authoritative MTP residency
line reports the exact resolved context, Turbo4 types, GPU backend, and bytes.
Also, the ordinary attention-cache category still grows with the requested
context. The pager target itself follows `H`, but this run does not claim that
the ordinary cache has been replaced by the bounded target allocation.

## Provenance

- Raw root: `/srv/ai/paged-kv/results/14-02-ladder-20260904T004547Z`
- Model: `/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf`
- Model SHA256: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`
- Model size: `14252845984` bytes
- GGUF maximum context: `262144`
- Architecture and geometry: Qwen35, 65 layers, KV heads 4, K/V head length
  256, native M-RoPE (`n_pos=4`)
- Release binary: `build-14-01-cuda/bin/llama-server`
- Release binary SHA256: `4cb1a90011d2d5b486b3f83b2ce3c9863eb65b55119b290fa52c8e70c1854626`
- Release version: `0.3.0-dev (build 11993, commit 8a550d30c)`
- GPU: NVIDIA GeForce RTX 4080, 16376 MiB, compute capability 8.9
- Raw request: `{"prompt":"hello","n_predict":1,"cache_prompt":false}`

The ladder ran the release binary directly on port 18080 with the primary
service stopped. The unrelated CPU server on port 8092 was not changed.

## Auto-fit ladder

Each successful row reached health, listened on port 18080, and completed the
one-token request. `H` is `physical_rows`, and `target_bytes = H *
target_page_bytes`.

| requested | resolved | result | logical pages | admitted pages | H rows | page bytes | target bytes | usable | charged/reserved | headroom | GPU serving used | selected rows |
| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 256 | 256 | pass | 1 | 1 | 256 | 4325376 | 4325376 | 2761687040 | 32051584 / 32051584 | 0 | 13345 MiB | prefill 2, decode 1 |
| 16384 | 16384 | pass | 64 | 64 | 16384 | 4325376 | 276824064 | 2484862976 | 36180352 / 36180352 | 0 | 13867 MiB | prefill 2, decode 1 |
| 32768 | 32768 | pass | 128 | 128 | 32768 | 4325376 | 553648128 | 2203844608 | 40374656 / 40374656 | 0 | 14399 MiB | prefill 2, decode 1 |
| 65536 | 65536 | pass | 256 | 256 | 65536 | 4325376 | 1107296256 | 1641807872 | 48763264 / 48763264 | 0 | 15463 MiB | prefill 2, decode 1 |
| 131072 | 131072 | pass | 512 | 104 | 26624 | 4325376 | 449839104 | 517734400 | 65540480 / 65540480 | 0 | 15909 MiB | prefill 2, decode 1 |
| 262144 | 262144 | safe refusal | allocation failed before listen | allocation failed before listen | - | - | - | - | - | - | 148 MiB | - |

The maximum-context refusal was before partial serving. The server log reports
an 8448.1 MiB unmet VRAM demand and then fails context/model initialization;
the raw result is `ready=0` and has no completion response.

The odd-tail run is `raw/odd-65537`. The requested 65537 tokens resolve to
65792 tokens, or 257 pages, with `H=65792` and target bytes `1111621632`.
It listened and completed the request. The rounding is the explicit 256-token
page contract, not a fixed context floor.

## Fixed-cap and refusal checks

At resolved context 16384:

- `raw/fixed-h2-pass`: `--kv-vram-budget 512MiB`,
  `--kv-host-budget 1GiB`, `--kv-safety-headroom 128MiB`, and
  `--kv-hot-pages 2` passed. The pager line reports logical pages 64,
  admitted pages 2, `H=512`, target bytes 8650752, and the requested 128 MiB
  headroom.
- `raw/fixed-h2`: the same context with a 64 MiB VRAM budget and 128 MiB
  headroom refused with `KV pager initialization failed: admission` before
  listening.
- `raw/too-small`: a 1 MiB VRAM budget refused with the same admission error
  before listening.

These runs show the hot-page cap and headroom are applied to admission rather
than silently replaced with a trained-context or fixed-capacity floor.

## MTP GPU residency

The MTP runs used draft MTP, `--spec-draft-n-max 2`, Turbo4 K/V, and
`--spec-draft-kv-device gpu`.

| raw run | resolved context | runtime MTP rows | type K/V | bytes | bytes/token | backend | category | result |
| :--- | ---: | ---: | :--- | ---: | ---: | :--- | :--- | :--- |
| `raw/mtp-gpu-256` | 256 | 256 | turbo4 / turbo4 | 401408 | 1568 | CUDA0 / gpu | mtp_gpu_reserved | pass |
| `raw/mtp-gpu-16384` | 16384 | 16384 | turbo4 / turbo4 | 17432576 | 1064 | CUDA0 / gpu | mtp_gpu_reserved | pass |
| `raw/mtp-gpu-131072` | 131072 | no reservation; recurrent-state allocation failed first | - | - | - | - | - | safe refusal |

For the first two rows, the raw log contains both
`common_speculative_mtp_log_residency` and `MTP KV reservation committed`.
The pager target is constructed before the native MTP context, so its startup
snapshot reports `mtp_rows=0 mtp_bytes=0`; the separate MTP reservation lines
are the authoritative runtime rows and bytes. The 131072 run fails before
native MTP reservation because it cannot allocate the recurrent-state cache.

## Host catalog, sealing, and ledgers

The production pager binds a host catalog and a 64 MiB maximum pinned transfer
ring. The `llama_context::get_kv_pager_metrics()` snapshot contains
`host_pageable_bytes`, `host_metadata_bytes`, and `host_pinned_bytes`, but the
server metrics scrape captured in these one-token runs exported only the
generic counters. Consequently those three per-run values are recorded as
`not emitted` in `DYNAMIC_CAPACITY_LADDER.json`, rather than inferred from RSS.
The one-token request also leaves the current partial page as the write
frontier; it is not evidence that every full target page has been sealed.

The admission ledger values above are copied from the `KV pager startup` line.
The invariant checked from every successful line is:

```text
pages = ceil(resolved_context / 256)
H = admitted_pages * 256
target_bytes = H * target_page_bytes
charged/reserved/headroom = the startup admission ledger terms
```

The fixed page byte value is mathematically justified by the unchanged Qwen35
layer/head/type geometry and the fixed 256-token page size. `H`, target bytes,
admitted pages, and the selected execution rows are not constant across the
ladder.

The CUDA memory breakdown in the raw logs also shows the ordinary context
category growing from 153 MiB at 256 tokens to 2261 MiB at 131072 tokens. This
is listed explicitly as a separate allocation observation; the bounded pager
target values in the table are not substituted for that category.

## Teardown and restore

Every isolated run captured `gpu-before.csv`, `gpu-serving.csv`, and
`gpu-after-teardown.csv`. Successful run teardown returned the isolated
process to the 148 MiB baseline snapshot. The final restore record is
`teardown-restore.txt` and shows:

- the fast and big profile `SERVER_BIN` values restored to
  `/srv/ai/paged-kv/build/buun/bin/llama-server`;
- all temporary pager, MTP-device, benchmark, and debug manager environment
  variables unset;
- `llama-server.service` active and healthy;
- `ai-long-memory.service` active with upstream HTTP 200 and healthy DB;
- the primary service back on the big profile at context 131072;
- the independent 8092 CPU server still present; and
- final GPU state 15481 MiB used, 462 MiB free, matching the restored primary
  service rather than a leaked isolated process.

## Source fixes exercised by the ladder

- Production host capture now binds representation identity to `LLAMA_COMMIT`,
  so the live model does not reject its own Turbo4 identity as unavailable.
- Qwen35 M-RoPE selected-reference admission accepts `n_pos=4` and uses the
  first coordinate of each token as the causal KV row position.
- Full sequence reset releases the partial write-frontier pin before pager
  removal and invalidates a removed current-page index, allowing repeated
  short startup/request/teardown cycles.

## Deferred verification

No required hardware, credential, or human-upstream verification is deferred:
the real GPU/model runs, refusals, and service restore were executed locally.
The non-emitted host metrics and the separate native-MTP/pager construction
boundary are recorded as observed runtime limitations in this evidence rather
than asserted as passes.

