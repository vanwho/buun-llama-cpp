# Cluster forward105-matrix — fixed original-profile, CPU-KV, and ACO context benchmark matrix

Revision: `hotpath-v10-20260914`.

This cluster replaces the earlier narrow 105-06 CPU comparison. Its purpose is
to produce comparable old-GPU, ordinary CPU-KV, and ACO results at the exact
context/placement geometries requested by the operator. It is a measurement
campaign, not a code-tuning phase.

## Immutable campaign rule

### Operator-directed MTP scratch repair: new preparation boundary at 105-15

The request-time allocation blocker was a source defect, not proof that the
RTX 4080 cannot admit 64K hot. Read only the compact engineering note
`engineering-notes/request-time-mtp-scratch.md` for the correction and its
controlled validation. Do not restore the obsolete full-L F16 reservation
or globally force native/reference attention. No batch, codec or geometry
change is part of this repair.

105-06 through 105-14 retain their original candidate cohort and
`MATRIX105_FREEZE.json/.md`; do not overwrite those manifests or historical
results. Before 105-15's scored campaign, make one new preparation revision
`MATRIX105_SCRATCH_FREEZE.json/.md` for the already-built corrected candidate
and the remaining 105-15 through 105-21 runs. Reuse the original model,
prompts, reasoning modes, request schedule and exact commands below.
Refresh the build receipt only after verifying compiled source-file hashes;
record executable and **actual loaded project DSO** hashes, source identity,
helper hashes and the prior manifest pointer. The executable is a small
launcher; its hash alone does not identify libllama/libggml-cuda.

Verify one exact managed owner and requested L/H before generation. The
scratch-fix diagnostics are not the 48-row benchmark and do not complete
105-15. Freeze once before scored rows, then do not edit code/configuration
or rebuild within the new cohort. Subsequent tasks use this new manifest.
105-22 labels both cohorts explicitly; never present measurements across
the repair boundary as same-binary comparisons. Preserve previous findings
without rerunning unrelated completed configurations.

### Original preparation boundary (historical cohort)

Task 105-06 is the only preparation boundary. It must validate or repair the
benchmark helper, request construction, tokenizer/preflight, row parser, and
launch configuration before any measured row. Before its first measured
request it writes a frozen matrix manifest with source SHA, the single frozen
candidate binary SHA-256, model SHA-256, helper SHA-256, prompt bytes, exact expanded server
`ExecStart` argv for all 15 configurations, effective profile values, and the
expected output schedule. From the first measured request through the final
task 105-20, do not edit source code, benchmark code, launch/profile settings,
sampling, batch sizes, KV types, MTP flags, or any other performance setting;
do not rebuild either binary. Each task changes only the explicitly named
configuration axis (binary family, target placement/pager geometry, or L/H).
If an argv, binary, model, helper, or configuration hash differs from the
frozen manifest, stop before sending a request. Do not repair it midway through
the matrix: record the setup defect, repair in a new preparation revision,
then restart the entire affected comparison set under a newly frozen manifest.
Ordinary low speed, a semantic miss, or low MTP acceptance is a finding, not a
setup error and never causes an unchanged rerun.

Reasoning modes are request-level values `off`, `low`, `medium`, and `xhigh`;
they are not separate qwen38-fast server profiles. The host has one installed
`qwen38-fast` profile. Keep server argv constant across the four modes; encode
the requested mode in each JSON request and retain the exact request body.
`off` must actually disable model thinking; the preparation task verifies this
in the emitted payload. All requests use SSE, temperature 0, seed 42, the exact
three prompts below, one 40-token warmup and three measured requests with a
400-token maximum for each prompt/mode. Thus each configuration has 48
canonical requests: 12 warmups and 36 measured. Warmups are not included in
medians. Native EOS is preserved; 400 is a ceiling, not a forced length.

Canonical prompts (verbatim):

1. `write a python function that merges two sorted lists into one sorted list, with docstring.`
2. `explain the difference between mmap and read for loading large files, one paragraph.`
3. `write a bash script that watches a directory and prints new files as they appear.`

## Frozen server settings and launch definitions

Common to every configuration: model `/srv/ai/models/text/current.gguf`,
alias `qwen38-fast-turbo4-mtp`, GPU model layers `999`, Flash Attention on,
fit off, parallel `1`, batch/micro-batch `1024/256`, target and draft KV
Turbo4 except the explicitly labeled ordinary CPU target-KV control, native
GPU draft-MTP with `n-max=2`, metrics enabled, context shift disabled, the
managed API-key file `/srv/ai/config/llama/api-keys`, port 8080, and the one
managed Qwen owner. Preserve 8091; never inspect, stop, or touch 8092. Acquire
`/tmp/ai-pager-benchmark.lock`; use the managed service lifecycle, and never
start a second Qwen model. Resolve the current service owner and ensure its
process is gone before replacing it. Restore the original healthy service at
the end of each task and verify its identity/health.

All 15 measured configurations use the same frozen ACO candidate binary
`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server`. The two
non-paged controls deliberately use that binary with `--kv-pager off`; this
holds source/build identity constant so the ACO-vs-control comparison measures
placement/paging, not unrelated binary changes. “Original fast-profile GPU”
means the original fast-profile settings with pager off, not the older
executable at `/srv/ai/paged-kv/build/buun/bin/llama-server` (that binary does
not expose the explicit native-MTP KV-device flag required by the managed
benchmark lifecycle). Ordinary CPU offload means F16 target K/V in CPU RAM via
`--no-kv-offload`; model weights and Turbo4 draft-MTP K/V remain on CUDA0. Label
it `CPU-F16 KV`, not codec-matched Turbo4. All GPU-target and ACO target K/V
are Turbo4. Every command below is a complete one-line `ExecStart`; tasks use
the named command verbatim, never reconstruct it from prose. The task owns the
managed service lifecycle and must not launch a second model.

### Exact `llama-server` commands

105-06 — original fast-profile settings, GPU-only, empty slot, L=77,824:

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 77824 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

105-07 — CPU-F16 target-KV, empty slot, L=77,824:

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 77824 -np 1 -ctk f16 -ctv f16 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

105-08 — ACO, empty slot, L=77,824, H=65,536 (256 pages):

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 77824 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager selective --kv-router probe-rerank --kv-page-size 256 --kv-hot-pages 256 --kv-pin-recent 0 --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

105-09 — ACO, L=8,192, H=4,096 (16 pages):

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 8192 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager selective --kv-router probe-rerank --kv-page-size 256 --kv-hot-pages 16 --kv-pin-recent 0 --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

105-10 — GPU-only control, L=8,192:

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 8192 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

105-11 — CPU-F16 target-KV control, L=8,192:

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 8192 -np 1 -ctk f16 -ctv f16 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

105-12 — ACO, L=262,144, H=16,384 (64 pages):

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager selective --kv-router probe-rerank --kv-page-size 256 --kv-hot-pages 64 --kv-pin-recent 0 --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

105-13 — GPU-only control, L=32,768:

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 32768 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

105-14 — CPU-F16 target-KV control, L=32,768:

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 32768 -np 1 -ctk f16 -ctv f16 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

105-15 — ACO, L=262,144, H=32,768 (128 pages):

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager selective --kv-router probe-rerank --kv-page-size 256 --kv-hot-pages 128 --kv-pin-recent 0 --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

105-16 — GPU-only control, L=65,536:

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 65536 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

105-17 — CPU-F16 target-KV control, L=65,536:

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 65536 -np 1 -ctk f16 -ctv f16 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager off --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

105-18 — ACO, L=262,144, H=65,536 (256 pages), fill C=120,000:

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager selective --kv-router probe-rerank --kv-page-size 256 --kv-hot-pages 256 --kv-pin-recent 0 --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

105-19 — ACO, L=262,144, H=65,536 (256 pages), fill C=184,000:

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager selective --kv-router probe-rerank --kv-page-size 256 --kv-hot-pages 256 --kv-pin-recent 0 --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

105-20 — ACO, L=262,144, H=65,536 (256 pages), fill C=250,000:

`/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --kv-pager selective --kv-router probe-rerank --kv-page-size 256 --kv-hot-pages 256 --kv-pin-recent 0 --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4`

Before freezing the manifest, 105-06 checks each exact option against the
candidate binary's `--help`/reported devices. The old August binary is not used
by this matrix: its CLI lacks explicit draft-KV device selection and the
managed lifecycle refuses it for native MTP. If any command above is rejected
by the candidate binary, fix this preparation/configuration before any scored
row and freeze the corrected command set. Thereafter, an effective argv
mismatch invalidates setup; never probe by launching a second model or sending
one-token requests.

## Which configurations are filled

The first three configurations—105-06 original GPU-only, 105-07 original
CPU-F16-KV, and 105-08 ACO—are the 77,824-token *empty-context baseline
group*. Set server `-c 77824`, clear the slot, send no A/B/A files, and do not
ingest repository content. Run only the 48 canonical prompt requests from a
clean isolated slot. The context setting is capacity, not preloaded occupancy.
This preserves the original fast-profile benchmark as an unfilled baseline.

Every subsequent configuration, beginning with 105-09 at 8K/4K-hot, is a
repo-filled context test. These tests use the A/B/A workload below, then append
tracked repository source to the task's occupied-context target before running
the 48 canonical requests. Thus all GPU-only/CPU-KV controls at 8K, 32K, and
64K receive the same repo-derived filled workload as their paired ACO case;
the 128K, 192K, and 250K ACO tests do too.

## A/B/A source-context workload for filled-context tasks

Every filled-context configuration first runs the same deterministic A/B/A
context build: load tracked repository source files in chronological user
turns; A1 contains
the chosen Python merge implementations, B contains the Bash directory-watch
implementations, and A2 asks about the best Python implementation again using
its filename/identifier. Use the canonical existing fixtures and prompt text
from `tools/server/bench/fixtures/repo-context-v1/`; the preparation task
selects a frozen, tokenizer-counted prefix and records every file/hash/span.
Do not use synthetic filler, gold answers in prompts, forced paging, artificial
eviction, or hidden settings. After this natural A/B/A seed, append frozen
tracked repository chunks in bounded turns until the target occupied-context
count for that test is reached. Keep at least 1,536 tokens for the next user
prompt, response, and normal runtime margin. If source text runs out or the
count cannot fit, record actual C and stop the fill; never exceed L or compact.

Use C targets: 6,144 at L=8,192; 30,000 for 32K; 60,000 for 64K; 120,000,
184,000, and 250,000 for the three large ACO rows. The 77,824 baseline group
is not filled. For ACO H is respectively 4,096, 16,384, 32,768, and
65,536 tokens as listed in the task packet. The 32K and 64K ACO tasks allocate
L=262,144 while filling only to their named C; their GPU/CPU controls allocate
L=32,768 or 65,536. The 8K ACO allocation is L=8,192. The baseline ACO
allocation is L=77,824 and H=65,536. The 128K/192K/256K rows use L=262,144,
H=65,536. Never reduce H, batch size, or other settings during the campaign.

The 8K A/B/A fixture must be sized in 105-06 to fit and still reach C>H. For
larger contexts, use that same frozen A/B/A seed then the fixed repository
source chunks. For each canonical probe, fork/restore the same frozen filled
base snapshot so all 48 requests start at the same occupied C; do not let
earlier prompts, reasoning modes, warmups, or generated answers contaminate
later prompt rows. If snapshot isolation cannot be validated in preparation,
do not begin the matrix.

## Metrics and output contract

Each configuration writes a compact JSON and Markdown result plus append-only
raw request/response, SSE, lifecycle, process-identity, and memory artifacts
under `/srv/ai/paged-kv/results/forward/105-matrix/<task-id>/attempt-01/`.
Report 3 tables per configuration, with rows `off`, `low`, `medium`, `xhigh`
and columns Prompt 1, Prompt 2, Prompt 3:

```text
DECODE MEDIAN TOK/S
| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
| off | median | median | median |
| low | median | median | median |
| medium | median | median | median |
| xhigh | median | median | median |

MTP ACCEPTANCE MEDIAN %
| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
| off | median % (accepted/drafted retained in JSON) | ... | ... |
| low | ... | ... | ... |
| medium | ... | ... | ... |
| xhigh | ... | ... | ... |

PREFILL MEDIAN TOK/S
| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
| off | median | median | median |
| low | median | median | median |
| medium | median | median | median |
| xhigh | median | median | median |
```

* `DECODE MEDIAN TOK/S`: median of the three measured response decode rates.
* `MTP ACCEPTANCE MEDIAN %`: per-response accepted/drafted percentage, then
  median of the three measured percentages. Record numerator/denominator and
  `not observed` distinctly; do not invent zero when no proposal was made.
* `PREFILL MEDIAN TOK/S`: executed fresh input tokens divided by the server's
  prompt-evaluation duration for that request. Do not count cached historical
  tokens as newly processed prefill. For filled-context configurations also
  save each append chunk's fresh ingestion tok/s against occupied C to produce
  the curve. The unfilled 77,824 baseline group has no append curve.

Do not impose an MTP acceptance or speed pass/fail threshold in these
measurement tasks. A completed request with low MTP, slow prefill, poor decode,
or semantic error is a valid measured result. Missing telemetry is `not
observed`, not zero, and only transport/setup failure may be repaired before
resuming a row. Medians require three valid measured runs; retain successful
raw rows and resume only missing identical rows. Each task's receipt must hash
the compact report and raw manifest. After the last matrix row, task 105-21
performs one bounded two-file cold-recall/promotion measurement at the 105-20
near-full context using the same frozen 105-20 server argv. It is
measurement-only: wrong answers or unobserved promotion are recorded, never
repaired or retried. Task 105-22 aggregates the 15 configuration reports and
the recall result without rerunning them, producing the cross-configuration
tables, context-fill prefill curve, and conclusions.
