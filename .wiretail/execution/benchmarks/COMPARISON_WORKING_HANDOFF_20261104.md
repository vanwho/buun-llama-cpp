# Comparison completion — current handoff

## User-confirmed method

Normal server prefix caching, no benchmark disk save/restore/erase. Each
canonical question uses the fixed filled repo base plus its current question;
prior benchmark questions/answers do not accumulate in that base. Only new
mapped disjoint repository chunks grow the ingest frontier. One fill per
configuration series, never per prompt or reasoning mode. Keep valid historical
decode results even if their legacy fill protocol was inefficient; label it.

## Current execution

- Branch plan/attention-aware-kv-paging, no inference code/binary changes.
- Active plan COMPARISON_COMPLETION_20261104.json revision7, ready; output
  /srv/ai/paged-kv/results/comparison-completion-20261104-r7.
- Runtime owner /root/comparison_runtime_prep (Luna High) must launch durable
  queue with nohup setsid, one GPU/model and /tmp/ai-pager-benchmark.lock.
- Root checked all11 profiles: exact argv/driver SHA, B1024/U256,t16/tb16,
  Turbo4 GPU target/draft or Q4_0 CPU target, MTP nmax2, T0/seed42, one warmup40
  and three measured400. Same binaries/model/template/map. No build required.
- Tool commit da21b657d passes context_checkpoints through all driver layers;
 29 focused tests pass. Profiles32(native default), except forkGPUoff ingest8.
- R7 selects147 genuinely missing feasible cells +3 independent outlier
  sessions =600 maximum canonical requests. Conditional startup profiles replace
  a failed startup;448 catalogue requests are NOT additive planned executions.
- Order: remaining master CPU-off; master GPU-off L122880 gaps; fork CPU-on
  gaps; master CPU-on gaps; three independent C28K/off/Bash rechecks; fork
  GPU-off L122880 ingest-only. Predeclared CPU startup fallbacks filter cells
  beyond their L; never shrink batches or quietly relabel the requested C.
- First repeated C126976 question must keep the recent prefix. Monitor native
  checkpoint creates and cached/backend token counts before letting cases repeat.

## Why R6 paused

Eight native checkpoints were created, newest C61441; none after appends to
C94224/C126977. C126K warmup cached126977 and processed61; measured repeat
restored61441, reprocessed65597. No CUDA/OOM; driver stopped before more repeats.
The8-cap plus protected retention floor is the leading explanation, not a
proven exact skip predicate.32 native checkpoints cost about3.51 GiB extra
for the observed target images, versus~50 GiB host available; native-MTP
companion snapshots are additional and must be observed, not assumed zero.
No benchmark slot files are the recovery. R5/R6 plans are audit archives only;
old driver SHA are superseded, so do not resume those queues.

## Preserved evidence

- ACO current on/off curves complete: evidence/ACO_FULL_CURVES_20261104.json;
 264 cells,792 measured+264 warmups, all pass. Never rerun them unnecessarily.
- R4 forkCPUoff repaired192K Bash3x400, min/median/max3.6813/3.7312/3.7395.
- R4 fork/master CPU-MTP-on target-file-restored data quarantined, external raw
  retained; target-only restore drops native draft prefix/carry.
- R5 masterCPUoff C6144/C12288/C28672 each36/36 pass; operator-held idle driver
  exit143 was intentional, not a model fault. Valid partial-series triplets kept.
- R6 masterCPUoff C61440 passed36/36. C126K cache-miss attempt diagnostic only.
- R6 GPU-MTP-on master/fork mapped C62465 fills passed; weighted backend/net
 1346.87 vs910.57 tok/s, wall1341.45 vs908.05. No canonical decode repeated.
- R6 fork/masterCPU-MTP-on mapped fills completed C256000/C196608, respectively.
- R6 GPU-off L131072 had real fill-time exhaustion: fork completedC100352 then
  failedC120832, master completedC120832 then failedC130048. GPU-free min9MiB,
  host available~50.7GiB. VBR allocator really failed, but main-target F16
  reservation is conservative; not an intrinsic context ceiling for every config.
- Normal masterGPU C6 fault source: comparison-completion-20261104/
  master-gpu-mtp-off/attempt-002-L131072. base_fileNone/resident_prefix_reuse;
 9 passes then cache miss and CUDAillegal access. R2/3/4 file protocols separate.

## Reports / remaining follow-up

- Current preview /srv/ai/paged-kv/results/comparison-final-20261104-preview-r6:
 774 valid cells/2322 reps,171 missing declared cells.24 are conditional GPU
 C130048 endpoints beyond the verified lower allocation;147 feasible gaps R7.
- Reporter maps all10 variants only using verified source manifest/map SHA.
  No historical variable chunks silently pooled. Primary graphs interval-weight
  all new chunks; backend/net-new/backend-time/HTTP-wall rates remain separate.
- Nominal and actual C separate; min/median/max from complete triplets; no
 interpolation, best-sample mixing, source-cohort pooling, or hidden missing cells.
- Old masterCPUon C28672/off/Bash15.9977/11.2388/11.6034tok/s correlated with
 63.25/29.98/29.98% MTP acceptance. Identical target text and full cache_n0
  processing; native proposals greedy, so neither RNG nor restored carry explains
  it. R7 has three independent clean normal-cache sessions, no cherry-picking.
- Source-conditioned dense GPU prefill difference note:
 engineering-notes/dense-turbo4-prefill-cohort-difference-20261104.md.
  Fused-vs-materialized dispatch remains unconfirmed; do not change measured code.
- After allR7 primary/fallback work, execute tracked
 profile-comparison-dispatch.py --execute once. It profiles frozen R6 master/fork
 GPU-MTP-on throughC28672 only, no decode; no changes to their argv/binaries.
  SHA118259d142fa7e9da10424d6d32ff7a8e12d1212a72d5418e663afc51980c012.
  Empty CUDA trace =unmeasured, not zero. Trace timings never replace primary data.
- Docs agent /root/aco_docs_refresh_20261104 is idle; followup ONLY when timings
  idle to export/render and inspect final charts. Matplotlib venv
 /srv/ai/paged-kv/venvs/comparison-report, Agg, OMP/OpenBLAS/MKL1.
- Protect Jellyfin/ffmpeg and8091; never touch8092. Restore gateway/model health.
  Raw payloads/traces/binaries external; commit/push compact reports/CSV/graphs
  and separate local benchmark helpers from portable inference changes.
