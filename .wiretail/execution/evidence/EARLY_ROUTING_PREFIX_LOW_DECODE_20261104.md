# Low-reasoning decode comparison

Run: `candidate-full256k-lowdecode-20261104T011740487741661Z`; complete, 12/12 cases passed.

The comparison is to the previous ACO campaign at C=256,001; this candidate used C=261,121. One fresh source fill reached actual C=261,121, then all prompts reused the same resident prefix. Each prompt had one 40-token warmup and three measured requests capped at 400 tokens. Prefix reuse was `resident_prefix_reuse`; every measured case passed.

## Results

| Prompt | Candidate decode tok/s (3 runs) | Median ± sample SD | Previous ACO median | Speedup | Generated tokens | MTP accepted/drafted | Acceptance | Previous ACO MTP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 59.06, 59.03, 60.84 | 59.06 ± 1.04 | 33.82 | 1.75× | 291, 295, 274 | 553/608 | 90.95% (median 90.00%) | 78.14% |
| P2 | 43.87, 44.45, 47.67 | 44.45 ± 2.05 | 27.73 | 1.60× | 400, 400, 400 | 644/1103 | 58.39% (median 56.12%) | 55.29% |
| P3 | 57.47, 55.49, 54.57 | 55.49 ± 1.48 | 31.35 | 1.77× | 400, 400, 400 | 744/906 | 82.12% (median 81.25%) | 67.06% |

Prompts: P1 Python sorted-list merge function; P2 one-paragraph mmap versus read explanation; P3 original bash directory watcher prompt.

## Configuration and identity

- L=262,144; requested target=261,120; actual occupied C=261,121; admitted hot pages=256 × 256 = H65,536; batch=1,024, ubatch=256; 16 threads.
- GPU placement; Turbo4 target and draft; native MTP with max draft 2; LOW reasoning.
- Build source equivalent to `6068d4a003c6ad0e117a4808097f34f0056e0e85`; report-time HEAD `ce1d8bab01122b93684e9b1a576e95c4cf227648` is docs/evidence only. Server SHA256 `f0f44fdefb920245811908f9a75e96ec5fce782f962836e024bc59f10d837cd0`; loaded DSOs are in the JSON.
- Runtime route log recorded `state_bytes=9,805,824` (9.35 MiB), `target_query_passes=1`; final fill append had 31,630 tokens.

## Resources and restoration

- GPU-wide peak used 15585.0 MiB, minimum free 361.0 MiB; sampled server GPU use max 15534 MiB.
- Server VmHWM 14322600 KiB; sampled VmSwap 0 KiB. Minimum system MemAvailable 44146568 KiB and SwapFree 6130724 KiB.
- Restore completed with exit 0: gateway and managed llama service active, Jellyfin PID 1000914 unchanged, :8091 PID 76881 unchanged, :8092 untouched/unbound.

## Raw evidence

- Row summary: `/srv/ai/paged-kv/results/candidate-full256k-lowdecode-20261104T011740487741661Z/cell/C261120/campaign/row.json`
- Per-request records: `/srv/ai/paged-kv/results/candidate-full256k-lowdecode-20261104T011740487741661Z/cell/C261120/campaign/records.jsonl`
- Startup log: `/srv/ai/paged-kv/results/candidate-full256k-lowdecode-20261104T011740487741661Z/cell/server-startup-20261104T011741Z.log`
- Loaded DSO capture: `/srv/ai/paged-kv/results/candidate-full256k-lowdecode-20261104T011740487741661Z/cell/C261120/campaign/runtime-captures/runtime-20261104T011746-2029558.json`
- Resource samples: `/srv/ai/paged-kv/results/candidate-full256k-lowdecode-20261104T011740487741661Z/cell/resource-samples.jsonl`
- Restore record: `/srv/ai/paged-kv/results/candidate-full256k-lowdecode-20261104T011740487741661Z/restoration.txt`

## Comparison provenance and limits

Previous row: `/srv/ai/paged-kv/results/forward/benchmark-reruns-corrected-20261103/aco-mtp-on-H65536-L262144-reasoning-low-C256000-external-interference-rerun/C256000/campaign/row.json`. It completed one warmup and three measured runs per prompt at C=256,001 with the same model, L262144/H65536, B1024/U256, GPU Turbo4 target/draft and MTP2. The new C=261,121 run is 5,120 tokens (about 2%) fuller. These are historical build/content comparisons, not isolation of one change. No decode regression was observed: medians increased by 74.65%, 60.30%, and 76.98%. Compare **median** MTP acceptance (90.00%, 56.12%, 81.25%) against earlier medians; the pooled accepted/drafted ratios in the table are a separate statistic.

The tested runtime is source-equivalent to `6068d4a00`. Explanation and first validation are committed as `ce1d8bab0`. The full reasoning-mode matrix remains paused; this follow-up covers only the user-requested low-mode canonical prompts.
