# Matched 32K file-question latency: master versus ACO

**Result:** at the identical 32K file context and short A1 optimization question, ACO added **316.80 ms to median first-token latency** versus the clean master control. Server-reported question processing differed by **316.55 ms**. This is the all-hot short-query cost, not cold-page selection/promotion time.

Each build ran three independent fresh-server samples, interleaved M1/A1/A2/M2/M3/A3. Each session loaded the frozen 32K repository payload once, then asked the exact A1 question from the natural file-promotion test. No source, binary, or performance setting changed during the six samples.

## Matched inputs and settings

- Body 32,768 tokens; framed initial prefix 32,832; question 91 new tokens; total rendered A1 input 32,923. All six rendered prompt token hashes match, and every question reused all 32,832 cached tokens.
- Both servers L=65,536 to leave answer room. ACO H=65,536 (256 pages). GPU Turbo4 target/draft KV, native MTP2, B=1024/U=256, 16 threads, 1 slot, CUDA0, GPU layers 999, flash attention on, fit/context-shift off.
- Native GGUF template, constant low reasoning and preserved thinking, T=0/seed=42, SSE, 4096-token answer cap with normal EOS. No benchmark template override, diagnostic polling or opt-in timing/selector flags.
- Existing built-in selective-mode hotpath logging remained enabled by the runtime. Neither binary was rebuilt for this comparison.
- Clean master source `d2407c33f9595d1badccfd9a82a33c4c5ceda490`; ACO source-equivalent `6068d4a003c6ad0e117a4808097f34f0056e0e85`. Binary/loaded-library hashes and commands are in the JSON companion.

## Results

| Metric | Master median | ACO median | ACO minus master |
|---|---:|---:|---:|
| First generated token | 117.62 ms | 434.42 ms | +316.80 ms |
| Question processing | 90.11 ms | 406.66 ms | +316.55 ms |
| First visible answer | 16.534 s | 18.352 s | +1.818 s |
| Complete response | 22.590 s | 25.153 s | +2.563 s |
| Decode throughput | 91.58 tok/s | 87.27 tok/s | -4.32 tok/s |
| MTP acceptance | 86.54 % | 87.04 % | +0.50 percentage points |
| Generated tokens | 2059 tokens | 2158 tokens | +99 tokens |
| Initial file-fill backend throughput | 1631.85 tok/s | 1039.52 tok/s | -592.33 tok/s |
| Initial file-fill HTTP wall time | 20.144 s | 31.609 s | +11.464 s |

TTFT ranges: **master 115.45–119.15 ms; ACO 433.67–435.55 ms**. The prompt/TTFT difference is stable across clean sessions. First token means the first generated reasoning token; first visible answer includes low-mode model reasoning.

ACO produced 99 additional tokens in every sample (2158 versus 2059). Its 2.563-second longer complete response therefore includes different generation length/rate; it is not an isolated fixed ACO surcharge. The pre-answer difference is approximately 0.317 seconds.

## What this measures

The A1 query is only 91 tokens, below B=1024, so the new bulk early-routing prefix is skipped. This measures the ordinary ACO short-question/request path against the pristine master build. It does not time the approximate bulk-routing prefix.

The ACO3 server log reports `query replay: unchanged history; replay_count=0`, and `h2d_queued=0 promotion_pages=0 eviction_pages=0` at the question freeze. Generated-span deltas also show no promotions/transfers/evictions. Thus this is an all-hot comparison; no cold promotion was required. A one-shot post-response inventory capture missed the sample server exit and returned connection refused; it was not a request failure, and no extra model run was made for it.

The earlier 133K natural recall test remains the evidence for an unchanged cold source page becoming resident and used. It measured approximately 775 ms key-rerank time and 5.1 ms page-transfer time, with overlapping component scopes. Do not subtract this 32K all-hot master timing from that 133K cold turn and call the remainder a matched promotion surcharge. See [natural promotion timings](NATURAL_FILE_PROMOTION_20261104.md).

## Artifacts and lifecycle

- Raw records, SSE, launch argv, per-process loaded DSO hashes and ordinary server logs: `/srv/ai/paged-kv/results/matched32k-latency-20261104T024052153560202Z`.
- External reproducible runner and frozen requests: `/srv/ai/paged-kv/tests/matched32k-latency-20261104/`.
- No raw payload, binary or log is committed. The JSON companion retains compact metrics and SHA256 references.
- Managed gateway/server restored successfully; Jellyfin PID 1000914 and port 8091 PID 76881 preserved. Port 8092 untouched. Full benchmark suite remains paused.
- Two setup-only attempts (a missing script import and an absent slot-state directory) were corrected before measurements. They sent no generation requests and are excluded.
- The similarly named `master106-draftkv-20261013` build was excluded: it has six source modifications and is not a pristine-master control.

## Exact completed launch commands

master:

```bash
/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja -ngl 999 --device CUDA0 --fit off -fa on -c 65536 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 127.0.0.1 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/matched32k-latency-20261104T024052153560202Z/master-1/slot-state --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4 --slots
```

aco:

```bash
/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja -ngl 999 --device CUDA0 --fit off -fa on -c 65536 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 127.0.0.1 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/matched32k-latency-20261104T024052153560202Z/aco-1/slot-state --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4 --slots --kv-pager selective --kv-page-size 256 --kv-hot-pages 256 --kv-router probe-rerank --kv-pin-recent 0 --spec-draft-kv-device gpu
```
