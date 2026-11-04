# Natural file recall, promotion and turn latency

Runtime source: `6068d4a00` (same tested build/DSOs; no runtime changes). Raw run: `/srv/ai/paged-kv/results/natural-aba-native-20261104-20261104T020522828038941Z`.

## Outcome

**Pass:** an unchanged original source page was naturally selected, promoted from host to GPU, published and used by the target graph. Native-model chat template and constant low reasoning preserved the old prefix across the growing conversation. No forced eviction, page nomination, route override or source resubmission was used.

The file was `src/llama-kv-fixed-window.cpp`, SHA256 `51dfe4f48cefd6ec1387278c2e85f9672943c894203d740f471fee76d1dc7f62`. It occupied token span [15507, 19992]; its duplicate/capacity-check span [16515, 16581] is on page 64. That page was resident after the first question, cold/host-backed/summary-ready before the last question, then resident and selected afterward. Sequence generation **1**, page generation **66**, content version **256** stayed unchanged across the cold-to-resident transition. This distinguishes transfer from recomputation. Original file pages promoted during A2: **[64]**; this is page-wise retrieval, not a claim that the whole file must return to GPU.

A2's terminal proof confirms cold-candidate authentication, H2D completion, mapping publication and target-graph use. Both A1 and A2 generated replacement functions compiled **unchanged** and passed all ten differential behavior checks against the original implementation. No generated code was applied to the repository.

## Exact scenario

L=262,144; H=65,536 (256 pages x 256); B=1024/U=256; 16 threads; GPU Turbo4 target and draft; native MTP n-max=2; greedy seed42; low reasoning throughout. Initial context: **32,832 tokens** including framing. Three subsequent chunks each contained 32,768 new repository tokens from other files, then an unrelated palindrome question. Pre-A2 occupied frontier: **133,777**.

A1: Using src/llama-kv-fixed-window.cpp loaded above, provide a drop-in implementation of llama_kv_fixed_window_select that avoids repeatedly scanning the output to detect duplicate explicit page IDs. Keep extra storage proportional to resident_capacity, not logical_page_count. Preserve the existing output order and every failure/clear behavior. Include any required include. Give the code and a brief explanation.

B: Write a short Python function that checks whether a string is a palindrome, ignoring spaces and letter case. Include a docstring and two examples.

A2: Please revisit src/llama-kv-fixed-window.cpp from earlier. Show a complete replacement for llama_kv_fixed_window_select using efficient duplicate membership tracking rather than repeatedly scanning the output. Keep its original signature, output ordering, and error handling; extra memory must scale with the resident capacity, not the logical address space. Explain what your version does if duplicate page IDs arrive after the resident capacity has been filled. Do not ask me to resend the file.

The original file was not sent again. The first generated answer's pages were tracked separately because retrieving a prior answer alone would not establish source-page promotion. Generous 4,096-token answer caps allowed normal completion. File hashes, payloads, source/answer spans, snapshots, SSE and code checks remain in the raw run directory.

## Request results

| Request | Cached prefix | Backend prompt evaluations | Output tokens | Decode tok/s | MTP acceptance |
|---|---:|---:|---:|---:|---:|
| A0-load32K | 0 | 32,832 | 1 | not a decode test | not measured (ingest ACK) |
| A1 | 32,832 | 91 | 2,056 | 87.14 | 88.92% |
| B1-load32K | 34,979 | 32,778 | 1 | not a decode test | not measured (ingest ACK) |
| B2-load32K | 67,757 | 32,783 | 1 | not a decode test | not measured (ingest ACK) |
| B3-load32K | 100,540 | 32,783 | 1 | not a decode test | not measured (ingest ACK) |
| B-question | 133,323 | 75 | 411 | 65.55 | 87.58% |
| A2 | 133,777 | 195 | 2,060 | 67.23 | 91.29% |
| A2-repeat | 133,779 | 193 | 1,636 | 68.77 | 94.52% |

All pressure requests reused the existing file/history KV. A2 and its identical-input repeat each performed one **bounded question replay**, not a replay of the old repository context. Answer variation is possible because the selected map changes; this is not a byte-identical generation comparison.

## Turn-lifecycle timings

TTFT means the first generated reasoning token. First visible answer includes model reasoning time, so it is not an ACO-only cost.

| Measurement | Cold source-page turn | Identical request, source page already hot |
|---|---:|---:|
| TTFT (seconds) | 2.779 | 2.625 |
| First visible content (seconds) | 25.339 | 18.269 |
| Whole response wall time (seconds) | 33.412 | 26.409 |
| Key rerank job (ms) | 774.696 | 786.252 |
| Query checkpoint capture (ms) | 77.148 | 34.890 |
| Query checkpoint restore (ms) | 243.462 | 236.506 |
| Bounded question replay (ms) | 275.246 | 274.798 |
| Live-policy counter delta (ms) | 102.506 | 102.869 |
| Page-transfer counter delta (ms) | 5.129 | 5.165 |

Each turn transferred **34,603,008 useful H2D bytes (33 MiB / eight complete page bundles)**. The repeat had page64 hot but fetched other pages, so it is **not an all-hot control**. Its 154 ms shorter TTFT does not isolate promotion cost. Its shorter total reply also had fewer generated tokens, so the wall difference cannot be labeled ACO overhead.

The rerank job took about **0.78 seconds**, including **658/670 ms** associated with **1,024 cold-key reader graphs/events** in each turn. The 5 ms page-transfer counter is therefore only one component. It excludes the key-rerank pipeline, checkpoint work and bounded question replay. Component scopes overlap and are not an additive total; an exact ACO-versus-disabled surcharge remains unisolated. These diagnostic timings are attribution data, not a clean production latency benchmark.

Idle slot snapshots clear query timers; the table uses maxima from active 250 ms samples. CUDA graph reuse counters refer to efficient kernel-graph execution, not repeated old-context input. Existing opt-in diagnostics were used; no runtime or performance configuration was changed during the sequence.

## Reproduction and artifacts

- External runner: `/srv/ai/paged-kv/tests/natural-aba-20261104/run-native.sh` and `run-native.py`.
- Source selection: `/srv/ai/paged-kv/results/natural-aba-native-20261104-20261104T020522828038941Z/source-manifest.json` and `source-span.json`.
- Requests/answers: `/srv/ai/paged-kv/results/natural-aba-native-20261104-20261104T020522828038941Z/A1/`, `B-question/`, `A2/`, `A2-repeat/`.
- Page/trace evidence: `A2/slots-before.json`, `A2/slots-after.json`, `A2/page-trace.jsonl`.
- Exact generated-code contract checks: `A1/comparison.log` and `A2/comparison.log` (external CPU-only).
- Timing ledger: `/srv/ai/paged-kv/results/natural-aba-native-20261104-20261104T020522828038941Z/server.log`; SSE first-token/wall times: `/srv/ai/paged-kv/results/natural-aba-native-20261104-20261104T020522828038941Z/records.json`.
- Runtime loaded-library hashes: `/srv/ai/paged-kv/results/natural-aba-native-20261104-20261104T020522828038941Z/runtime-dependencies.json`.
- Services restored: `/srv/ai/paged-kv/results/natural-aba-native-20261104-20261104T020522828038941Z/restoration.txt`; Jellyfin and port8091 preserved, port8092 untouched.

## Excluded diagnostic attempt

The earlier run at `/srv/ai/paged-kv/results/natural-aba-20261104-20261104T015228960462919Z` is **invalid for promotion and performance claims**. An isolated-benchmark template moved effort instructions inside growing history, causing old-prefix recomputation (cache_n=0) and a source-page generation change. Its transient residency was not promotion. For growing conversations use native template and constant reasoning/preserve-thinking settings, as in the valid run above. Keep benchmark-only effort relocation confined to isolated benchmark branches.

## Actual server launch (audit)

The native model template was used; no benchmark chat-template override was present. The wrapper generated a fresh output directory and applied the recorded observation-only environment. The exact command for this completed run was:

```bash
/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/natural-aba-native-20261104-20261104T020522828038941Z/cell/slot-state --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --kv-pager selective --kv-page-size 256 --kv-hot-pages 256 --kv-router probe-rerank --kv-pin-recent 0 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4 --slots --spec-draft-kv-device gpu
```
