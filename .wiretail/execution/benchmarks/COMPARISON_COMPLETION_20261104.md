# Numbered comparison-completion run plan — 2026-11-04

Execute the 7 remaining steps below in order, one model process at a time. The completed ACO-enabled MTP-on/off curves and R4 fork CPU/MTP-off curve are retained. Historical complete decode cells remain valid under their recorded build/template, with one explicitly declared MTP outlier recheck. R4 MTP-on target-file-restored series are quarantined from the normal resident-MTP comparison. Only an assessed unexplained anomaly permits a full rerun. The plan pins both binaries, all eight project DSOs, map, template, driver files, and exact server argv.

Committed queue command (single line): `python3 -u .wiretail/execution/benchmarks/run-comparison-series.py --queue`. The Luna run owner starts it detached with stdout/stderr redirected to the external queue log; closed terminals must not break benchmark output. After an assessed setup interruption, `--queue --resume-queue` preserves completed steps and starts a fresh attempt at the unfinished step.

## Fixed execution contract

- Server settings: B=1024/U=256, t=16/tb=16, CUDA0, GPU model layers=999, flash attention on, fit off, one slot, no context shift. GPU target KV is Turbo4; ordinary CPU target KV is Q4_0. MTP-on requests native draft-mtp n-max=2 with Turbo4 draft KV. The fork explicitly requests GPU draft KV; pristine master uses its stock route and actual placement must be recorded, not inferred.
- One fresh server and one persistent repository fill per series. Use frozen source `a5a1110ca620db4f1c04669d0a2f569852ec1ec7`, the exact 32K map below, and only the next new mapped source slice as new user content. Normal groups are 32,768 content tokens; checkpoint tails are measured and labeled separately. Do not resend old files as a new turn or restart/refill between prompts or modes.
- At each selected decode checkpoint: off → low → medium → xhigh; each mode runs all three original prompts, with one 40-token warmup and three measured requests capped at 400 tokens. T=0, seed=42, SSE. Normal early EOS is valid; two-token acknowledgments are not decode measurements. MTP-on controls and ACO retain the resident prefix and paired target/draft checkpoint state. Target-only slot files must never be substituted for normal native-MTP state. MTP-off controls may restore a complete target file outside timing; no repository re-prefill occurs.
- Even where decode selection is empty, fill and record that checkpoint. `checkpoint_selection: {}` means prefill-only, not skip the ingest point. Complete the missing reasoning-mode curves where needed, but never remeasure an already-complete mode curve merely because other modes are absent. Historical 192K two-token Bash outputs cause targeted cell validation first, not preemptive full-family reruns. Only a newly observed anomaly triggers a full-family escalation.
- All chunk groups contribute to the prefill curve: backend tok/s, net-new tok/s, and HTTP-wall tok/s separately; never use canonical question prompt timing or only a short final tail as the bulk-ingest curve. Report min/median/max of three decode repeats and MTP acceptance, plus raw references, candidate/model/DSO hashes, actual C, GPU/RAM peaks and swap.
- Keep B/U, quants, threads and MTP fixed. Use only the listed L fallbacks after a verified clean-start OOM. GPU-only capacity is an observed endpoint, not a failed mandatory 256K benchmark. ACO-disabled controls have no hot-pool knob. CPU limits likewise remain truthful if stock MTP cannot admit 256K.
- Sparse anomaly flags pause for assessment. First distinguish actual context scaling, MTP acceptance, cohort differences and resource activity. An evidenced explanation uses the continue action; an unresolved anomaly uses full-curve once on the identical profile. Reproduced runtime errors use a truthful finding or a source-directed setup repair; they do not cause blind whole-family retries. Never weaken the benchmark or silently change server settings.
- Luna High owns the single GPU run queue and returns summaries/counts/next action only. Escalate compact evidence first; inspect full raw data only if the Luna agent cannot classify it. Save raw artifacts only outside Git. Preserve Jellyfin/ffmpeg and 8091/8092; pause instead of killing a protected client. Restore previously active gateway/model services after each series or interruption.

Chunk map: `/srv/ai/paged-kv/fixtures/repo-prefill-a5a1110-32k-v1/map.json`; SHA-256 `5ca53650661d610ff2632eae262cb794bceb5b72efaf47364b8bcd43247dcf4f`.
Template SHA-256: `0103d81679a0cc43ce416da910e7db86a04c0e8548d7df3f41564c98a203deb7`. The completed ACO run reached C=261,121 within L=262,144. Existing control endpoints at C=256,000 or 258,048 are retained as measured points within their 256K allocations; this plan does not invent a new control endpoint solely to copy the ACO ladder.

Pristine master: `/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server`, SHA-256 `527c34de411d4cad6290e828deb17a71816ae2b9620a3f1166b833e1267e0f0f`, runtime source `d2407c33f9595d1badccfd9a82a33c4c5ceda490`. Fork: `/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server`, SHA-256 `90426c19b1d3d2e6f322907a35447adcb9a1278341a587f09137d242c8ffab8c`, runtime source `8a79008b5473676bb99a54cc1505d11a4e107357`. Later documentation/measurement-tool commits do not require a new inference build.

## Historical controls reused without new decode

Master GPU/MTP-on already has full four-mode points through its verified maximum. Fork GPU/MTP-on includes complete 105-10 (C6144), 105-13 (C30000), and 105-16 (C61440) plus Nov1 C12288/C78848; these are not missing merely because their x coordinates differ from the latest ACO ladder. Fork GPU/MTP-off already has its complete four-mode curve through C120832. Keep actual occupancy and provenance labels; do not interpolate missing values or silently relabel 61440 as 62464. Valid legacy CPU off-mode requests explicitly disable thinking, so their off graphs count.

Do not add arbitrary intermediate 77K/86K/98K CPU tests or a near-full 261120 endpoint just to imitate the new ACO grid. The old control graphs at their recorded 250K/252K occupied endpoints remain valid 256K-allocation results, labeled precisely. Repeated old prefills alone do not invalidate decode measurements.

## Reuse audit and actual remaining scope

A cell is one prompt at one reasoning mode and one occupied-context point, with three measured repeats. Existing graph points are reused at their actual C, not discarded for differing from a new ladder. The remaining selections below are missing cells, not blanket family reruns.

| Variant | Existing results retained | Initial new decode cells |
| --- | --- | ---: |
| Master GPU, MTP on | All six completed four-mode points | 0; ingest-only same-map curve |
| Fork GPU, MTP on, ACO off | All five completed four-mode points | 0; ingest-only same-map curve |
| Fork GPU, MTP off, ACO off | Complete curve through C120832 | 12, only conditional C130048 endpoint |
| Master GPU, MTP off | All complete cells, including full C78848/C100352 points | 42 holes + conditional C130048; checkpoint-free target-only profile |
| Fork CPU Q4, MTP off | Historical curve plus the completed R4 full 256K fill and repaired 192K Bash cell | 0 |
| Fork CPU Q4, MTP on | Historical off curve plus completed 6K/192K/250K modes | 36 missing middle-point mode cells + 1 bad Bash cell |
| Master CPU Q4, MTP on | Five off-mode points and ten completed 192K cells | 45 missing mode cells + 2 incomplete 192K cells + 1 flagged cell recheck |
| Master CPU Q4, MTP off | All seven off-mode points through C258048 | 63 low/medium/xhigh cells |
| ACO enabled, MTP on | Completed November4 eleven-point four-mode curve | 0 |
| ACO enabled, MTP off | Completed November4 eleven-point four-mode curve | 0 |

Remaining plan: 214 selected cells, at most 856 requests including warmups. Zero routine repetitions of validated historical cells; one explicit MTP-variance recheck. The GPU/MTP-off C130048 endpoints are conditional on genuine admission and successful execution. The master C6144 crash is not an allocation limit; its MTP-off recovery disables optional checkpoints while retaining full legacy target-state restore. R4 MTP-on target-file-restored series lack the intended full draft prefix and are diagnostic only. Older CPU F16 graphs remain separate from Q4_0 controls.

The same-master CPU-Q4 MTP-on L262144 startup OOM is already recorded at `upstream-cpu-mtp-on/C256000/session-01/reasoning-off/cell.json`. Reuse that capacity finding; start its gap run at the listed lower allocation instead of retrying the failed 256K allocation.

## Numbered runs

### 1. fork-cpu-mtp-on — gaps decode

Retain the complete historical off-mode curve and the 6K/192K/250K non-off rows. Add only low/medium/xhigh at the four middle points and validate the bad 192K off-mode Bash cell.

Primary: L=262,144; prefill checkpoints `[6144, 12288, 28672, 61440, 126976, 196608, 256000]`; decode checkpoints `[12288, 28672, 61440, 126976, 196608]`; 148 benchmark requests if this profile completes.

```bash
env LD_LIBRARY_PATH=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -t 16 -tb 16 -ctk q4_0 -ctv q4_0 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/comparison-completion-20261104-r5/fork-cpu-mtp-on/attempt-001-L262144/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --kv-pager off --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4 --spec-draft-kv-device gpu
```

Predeclared startup-OOM fallback: L=200,704; prefill checkpoints `[6144, 12288, 28672, 61440, 126976, 196608]`; decode checkpoints `[12288, 28672, 61440, 126976, 196608]`; 148 benchmark requests if this profile completes.

```bash
env LD_LIBRARY_PATH=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 200704 -np 1 -t 16 -tb 16 -ctk q4_0 -ctv q4_0 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/comparison-completion-20261104-r5/fork-cpu-mtp-on/attempt-002-L200704/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --kv-pager off --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4 --spec-draft-kv-device gpu
```

Predeclared startup-OOM fallback: L=131,072; prefill checkpoints `[6144, 12288, 28672, 61440, 126976]`; decode checkpoints `[12288, 28672, 61440, 126976]`; 144 benchmark requests if this profile completes.

```bash
env LD_LIBRARY_PATH=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 131072 -np 1 -t 16 -tb 16 -ctk q4_0 -ctv q4_0 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/comparison-completion-20261104-r5/fork-cpu-mtp-on/attempt-003-L131072/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --kv-pager off --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4 --spec-draft-kv-device gpu
```

Run: `python3 .wiretail/execution/benchmarks/run-comparison-series.py --step 1 --profile 0 --attempt 1`.

### 2. master-cpu-mtp-on — gaps decode

Retain the five historical off-mode points and valid Nov2 192K cells. Complete only missing non-off reasoning curves and the two incomplete 192K cells. Do not repeat the already-recorded clean 256K-allocation startup OOM on this same master binary and B/U settings.

Primary: L=200,704; prefill checkpoints `[6144, 12288, 28672, 61440, 126976, 196608]`; decode checkpoints `[6144, 12288, 28672, 61440, 126976, 196608]`; 192 benchmark requests if this profile completes.

```bash
env LD_LIBRARY_PATH=/srv/ai/paged-kv/builds/master106-d2407c33f959/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 200704 -np 1 -t 16 -tb 16 -ctk q4_0 -ctv q4_0 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/comparison-completion-20261104-r5/master-cpu-mtp-on/attempt-001-L200704/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4
```

Predeclared startup-OOM fallback: L=131,072; prefill checkpoints `[6144, 12288, 28672, 61440, 126976]`; decode checkpoints `[6144, 12288, 28672, 61440, 126976]`; 184 benchmark requests if this profile completes.

```bash
env LD_LIBRARY_PATH=/srv/ai/paged-kv/builds/master106-d2407c33f959/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 131072 -np 1 -t 16 -tb 16 -ctk q4_0 -ctv q4_0 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/comparison-completion-20261104-r5/master-cpu-mtp-on/attempt-002-L131072/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4
```

Run: `python3 .wiretail/execution/benchmarks/run-comparison-series.py --step 2 --profile 0 --attempt 1`.

### 3. master-cpu-mtp-off — missing-mode-curves decode

The historical off-mode curve is already complete through the 256K allocation. Run only the never-completed low/medium/xhigh curves, not off again.

Primary: L=262,144; prefill checkpoints `[6144, 12288, 28672, 61440, 126976, 192512, 258048]`; decode checkpoints `[6144, 12288, 28672, 61440, 126976, 192512, 258048]`; 252 benchmark requests if this profile completes.

```bash
env LD_LIBRARY_PATH=/srv/ai/paged-kv/builds/master106-d2407c33f959/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -t 16 -tb 16 -ctk q4_0 -ctv q4_0 --no-kv-offload -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/comparison-completion-20261104-r5/master-cpu-mtp-off/attempt-001-L262144/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --spec-type none
```

Run: `python3 .wiretail/execution/benchmarks/run-comparison-series.py --step 3 --profile 0 --attempt 1`.

### 4. fork-gpu-mtp-off — gaps decode

Only the previously requested conditional 127K occupied endpoint: lower points already have all four modes. If 128K allocation cannot fit, record the verified lower ceiling and collect prefill only.

Primary: L=131,072; prefill checkpoints `[6144, 12288, 28672, 62464, 78848, 88064, 100352, 120832, 130048]`; decode checkpoints `[130048]`; 48 benchmark requests if this profile completes.

```bash
env LD_LIBRARY_PATH=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 131072 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/comparison-completion-20261104-r5/fork-gpu-mtp-off/attempt-001-L131072/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --kv-pager off --spec-type none
```

Predeclared startup-OOM fallback: L=122,880; prefill checkpoints `[6144, 12288, 28672, 62464, 78848, 88064, 100352, 120832]`; decode checkpoints `[]`; 0 benchmark requests if this profile completes.

```bash
env LD_LIBRARY_PATH=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 122880 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/comparison-completion-20261104-r5/fork-gpu-mtp-off/attempt-002-L122880/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --kv-pager off --spec-type none
```

Run: `python3 .wiretail/execution/benchmarks/run-comparison-series.py --step 4 --profile 0 --attempt 1`.

### 5. master-gpu-mtp-off — gaps decode

Fill only missing cells, using the source-directed checkpoint-free MTP-off profile and full legacy target-state restore. Native MTP is absent, so target-only files are complete state. Keep quants, B/U, threads, CUDA graphs, FA and binary unchanged.

Primary: L=131,072; prefill checkpoints `[6144, 12288, 28672, 62464, 78848, 88064, 100352, 120832, 130048]`; decode checkpoints `[6144, 12288, 28672, 62464, 88064, 120832, 130048]`; 216 benchmark requests if this profile completes.

```bash
env LD_LIBRARY_PATH=/srv/ai/paged-kv/builds/master106-d2407c33f959/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 131072 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/comparison-completion-20261104-r5/master-gpu-mtp-off/attempt-001-L131072/slot-state --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --kv-unified --spec-type none
```

Predeclared startup-OOM fallback: L=122,880; prefill checkpoints `[6144, 12288, 28672, 62464, 78848, 88064, 100352, 120832]`; decode checkpoints `[6144, 12288, 28672, 62464, 88064, 120832]`; 168 benchmark requests if this profile completes.

```bash
env LD_LIBRARY_PATH=/srv/ai/paged-kv/builds/master106-d2407c33f959/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 122880 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/comparison-completion-20261104-r5/master-gpu-mtp-off/attempt-002-L122880/slot-state --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --kv-unified --spec-type none
```

Run: `python3 .wiretail/execution/benchmarks/run-comparison-series.py --step 5 --profile 0 --attempt 1`.

### 6. master-gpu-mtp-on — prefill-only decode

Retain every completed historical decode cell. Collect only the uniform mapped-chunk ingest curve at a safe 64K GPU allocation, so all three prefill metrics have a same-method comparison.

Primary: L=65,536; prefill checkpoints `[6144, 12288, 28672, 62464]`; decode checkpoints `[]`; 0 benchmark requests if this profile completes.

```bash
env LD_LIBRARY_PATH=/srv/ai/paged-kv/builds/master106-d2407c33f959/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 65536 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/comparison-completion-20261104-r5/master-gpu-mtp-on/attempt-001-L65536/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4
```

Run: `python3 .wiretail/execution/benchmarks/run-comparison-series.py --step 6 --profile 0 --attempt 1`.

### 7. fork-gpu-mtp-on — prefill-only decode

Retain every completed historical decode cell. Collect only the uniform mapped-chunk ingest curve at a safe 64K GPU allocation; no canonical prompt/mode generations are repeated.

Primary: L=65,536; prefill checkpoints `[6144, 12288, 28672, 62464]`; decode checkpoints `[]`; 0 benchmark requests if this profile completes.

```bash
env LD_LIBRARY_PATH=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 /srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 65536 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/comparison-completion-20261104-r5/fork-gpu-mtp-on/attempt-001-L65536/slot-state --cache-ram 12288 --ctx-checkpoints 8 --checkpoint-min-step 8192 --cache-idle-slots --kv-unified --kv-pager off --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4 --spec-draft-kv-device gpu
```

Run: `python3 .wiretail/execution/benchmarks/run-comparison-series.py --step 7 --profile 0 --attempt 1`.

## Artifact and completion rules

Each external attempt is stored under `/srv/ai/paged-kv/results/comparison-completion-20261104-r5/<step-id>/attempt-NNN-<profile>/`. Commands above contain the concrete first-attempt artifact path for each profile. On an explicit retry, only that attempt/slot directory changes; the machine plan's `{run_root}` substitution is restricted to artifact paths. The launcher verifies generated argv against the frozen plan before stopping services.

A series is complete only for the measurements it actually made. A prefill-only fallback can legitimately have zero new decode requests and reuse historical complete decode cells; it must not claim new decode or MTP measurements from its tiny fill acknowledgment. Keep valid partial rows when a later genuine capacity boundary occurs. Resolve setup mistakes and rerun the clean configuration instead of publishing them as speed results.

The script `run-comparison-series.py --step N --profile K --attempt M --full-decode` is the only permitted sparse-anomaly escalation; it uses the same server profile but selects all prompt/mode cells at every checkpoint. Record why it was invoked. No inference-code changes, model changes, new batch sizes, or additional benchmark families are authorized by this plan.

After assessment, use `python3 .wiretail/execution/benchmarks/run-comparison-series.py --queue --resume-queue --assessed-step N --assessment-action retry --assessment-note "concrete setup repair or cause"`. For complete measurements with an evidenced statistical explanation, choose `--assessment-action continue`; for an unresolved anomaly choose `--assessment-action full-curve`; for a real limitation choose `--assessment-action record-finding` and retain its missing cells as gaps. Never edit progress JSON to fake completion or escape a blocked retry.

Validated historical cell index: [COMPARISON_COVERAGE_20261104.json](COMPARISON_COVERAGE_20261104.json). It contains compact per-cell metrics and source hashes, not raw requests or payloads. Saved legacy request bodies prove that those Q4 CPU curves used reasoning off. The earlier exact-grid `comparison-gap-inventory-20261104.json`, `remaining-matrix.json`, and their broad rerun suggestions are superseded and do not drive this campaign.

