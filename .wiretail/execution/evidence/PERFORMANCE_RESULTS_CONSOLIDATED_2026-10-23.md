# Consolidated performance results: upstream master and ACO

Compiled 2026-10-23 from benchmark artifacts in this repository and /srv/ai/paged-kv/results.
This is a results consolidation plus a bounded missing-cell audit. It includes the
standardized 105-series three-prompt matrix, fresh-chunk context-ingest curves, the
upstream-master GPU and CPU-Q4/MTP controls, and the October 20 ACO replay campaign.
Raw request streams remain in their original result roots; source paths and binary
identities below make the numbers auditable.

## Reading these results

These datasets use different protocols and must not be blended:

1. Canonical three-prompt speed matrix (105 series): one warmup and three measured
   repetitions per prompt, 400-token output budget, four reasoning modes, and response-local
   MTP acceptance. The full per-configuration median tables are reproduced in Appendix A.
2. Fresh-chunk ingest curves (October 21): every turn adds one disjoint, previously unused
   repository-source chunk; prior conversation remains in history and earlier source text is
   not resent as the new user turn. Tables below show measured new tokens divided by that
   request's prompt-processing time. Decode values come from a three-token acknowledgment
   and are diagnostic only, not a decode benchmark. MTP acceptance in this test is also based
   on short replies and is not a substitute for the 400-token, three-prompt matrix.
3. The October 20 replay/fill campaign is retained separately. Repeated follow-up questions
   can involve cached-prefix replay/rewind, so those per-question figures are not the same as
   fresh repository ingestion. Two other October 21 cumulative-prompt runs are explicitly
   excluded by the project method note because they resent prior source content.

## Executive coverage and headline findings

| Requested comparison | Available evidence | Result / limit |
|---|---|---|
| Upstream master, all-GPU Turbo4 KV, MTP off | Fresh-chunk curve, Oct 21 | Completed through C=86,016: new-content prefill 1,734.25 to 962.42 tok/s; next rung failed near C=96,228 with an F16 dequant scratch reservation failure. No 128K result. |
| Upstream master, all-GPU Turbo4 KV + GPU MTP | 106-01 GPU-only matrix | Complete 48-request rows at empty L=77,824 and filled L/C=8,192/6,144; 16,384/14,336; 32,768/30,000; 65,536/61,440. Full decode, MTP, and prefill tables are in `evidence/MASTER106_GPU.md`. |
| Upstream master, CPU Q4_0 target KV, no MTP | Fresh-chunk curve, Oct 21 | Completed through C=262,000: new-content prefill 1,166.74 to 331.66 tok/s. |
| Upstream master, CPU Q4_0 target KV + GPU MTP | Historical canonical campaign, Oct 16 | Completed at L=8K, 16K, 32K, 64K, and 128K; 256K failed at startup. At L=128K, canonical decode medians were 10.06 / 8.16 / 9.26 tok/s and MTP acceptance 93.38% / 67.24% / 81.25% for prompts 1/2/3. |
| ACO-branch build, all-GPU Turbo4 + MTP | 105-series GPU-only controls | Canonical three-prompt results at L=8K, 32K, 64K, and 77,824; complete tables are in Appendix A. These use an ACO-built binary with pager off, not upstream master. |
| ACO-branch build, all-GPU Turbo4, MTP off | Exact 77,824-token baseline attempted Oct 23 on two ACO candidates | No requests completed. Both candidates failed during KV-cache startup: the allocator reported 2,508.1 MiB needed with 1,085.9 MiB free. This remains an unmeasured cell; the selective ACO MTP-off curve is not a substitute. |
| ACO-branch CPU offload | 105-series CPU-F16-KV + GPU-MTP controls; Oct 16 CPU-Q4_0+GPU-MTP campaign; Oct 21 CPU-Q4_0/no-MTP fresh-chunk curve | Canonical F16+MTP results are in Appendix A; Q4_0+MTP canonical campaign completes through 128K (256K startup failed); Q4_0/no-MTP fresh-chunk curve completes to 256K. |
| ACO enabled: Turbo4 GPU hot KV + CPU-hosted cold history | 105-series canonical matrix; Oct 21 fresh-chunk curves with MTP on and off | Canonical prompt medians are in Appendix A. The fresh-chunk ACO+MTP-on and ACO+MTP-off runs both reached C=262,000. Requested hot capacity was 131,072 tokens, but startup admitted only 249 pages / 63,744 tokens with MTP on, and 466 pages / 119,296 tokens with MTP off. |

### Most useful fresh-ingest endpoints

| Build / mode | MTP | C at endpoint | New-content prefill at endpoint | Decode sample at endpoint |
|---|---:|---:|---:|---:|
| Upstream master, all-GPU Turbo4 | off | 86,016 (next rung failed near 96,228) | 962.42 tok/s | 33.46 tok/s (3-token diagnostic) |
| Upstream master, CPU Q4_0 KV | off | 262,000 | 331.66 tok/s | 3.30 tok/s (3-token diagnostic) |
| ACO build, selective Turbo4, admitted hot 63,744 | on | 262,000 | 368.99 tok/s | 26.93 tok/s (3-token diagnostic) |
| ACO build, selective Turbo4, admitted hot 119,296 | off | 262,000 | 363.31 tok/s | 7.75 tok/s (3-token diagnostic) |
| ACO build, CPU Q4_0 KV | off | 262,000 | 327.95 tok/s | 3.25 tok/s (3-token diagnostic) |

The short-output decode samples above are not apples-to-apples decode benchmarks. Use the
canonical three-prompt tables for decode comparisons. The fresh-ingest prefill column is
recomputed as newly_evaluated_tokens_reported divided by prompt_ms; this measures new
repository-content growth per request time. It is lower than the stored server-reported
prompt tok/s on some ACO selective rows because the server timing prompt_n does not match
the harness's net-new-token count on those rows. Both values are retained in the detailed
tables below rather than presenting the larger rate as new-content speed.

## Fresh-chunk method and build identity (October 21)

Each conversational turn adds one fresh, disjoint repository-source chunk. The input file
and manifest records the source path/hash and span; earlier source text is not put in the
new user message again. Context checkpoints were 32,768; 65,536; 71,680; 77,824; 86,016;
98,304; 131,072; 163,840; 196,608; 229,376; and 262,000 where admitted.

Binary identities:

- Upstream-master binary path:
  /srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server
  SHA-256 527c34de411d4cad6290e828deb17a71816ae2b9620a3f1166b833e1267e0f0f.
- ACO binary path:
  /srv/ai/paged-kv/builds/aco-preview-20261015-release/bin/llama-server
  SHA-256 c9041ca96611ed4c9aa06be7b9a23443aa11d4b3bb30983d7dd2558c582659ec.
  The MTP-off run used a differently named build directory but the same recorded binary
  SHA-256 as the ACO MTP-on run.
- The 105-series model SHA-256 is
  40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199. Per-run identity
  summaries remain the authority for their model/binary fields.

A fresh-content rate is computed here as:
newly_evaluated_tokens_reported * 1000 / prompt_ms.
The source field newly_evaluated_tokens_reported is usage.prompt_tokens minus the reported
cached-prefix tokens. The server-reported rate is preserved alongside it. For ACO-selected
GPU runs, those two token-count bases diverge on some cache-present turns; do not silently
treat the server rate as the net-new repository-text rate.

### ACO selective Turbo4, native MTP on

Context allocation L=262,144. Requested hot capacity was 512 pages / 131,072 tokens.
The initial runtime snapshot admitted only 249 pages / 63,744 hot tokens. Binary SHA:
c9041ca96611ed4c9aa06be7b9a23443aa11d4b3bb30983d7dd2558c582659ec.
Source summary:
 /srv/ai/paged-kv/results/release/aco-incremental-turns-mtp-on-20261021/summary.json

| Occupied C | New input tokens | Prompt time ms | New-content tok/s | Server PP tok/s | Decode tok/s* | Short MTP acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 32,736 | 31,849.744 | 1,027.83 | 1,027.83 | 25.33 | 100.00% (2/2) |
| 65,536 | 32,765 | 85,430.627 | 383.53 | 766.94 | 43.40 | 100.00% (2/2) |
| 71,680 | 6,141 | 15,663.130 | 392.07 | 783.43 | 25.33 | 33.33% (2/6) |
| 77,824 | 6,139 | 16,049.516 | 382.50 | 764.32 | 36.71 | 75.00% (3/4) |
| 86,016 | 8,187 | 20,627.350 | 396.90 | 793.27 | 38.32 | 75.00% (3/4) |
| 98,304 | 12,283 | 29,499.840 | 416.38 | 832.38 | 37.95 | 75.00% (3/4) |
| 131,072 | 32,763 | 74,811.226 | 437.94 | 875.74 | 36.10 | 75.00% (3/4) |
| 163,840 | 32,763 | 77,996.126 | 420.06 | 839.98 | 26.63 | 50.00% (15/30) |
| 196,608 | 32,737 | 81,675.757 | 400.82 | 801.49 | 34.87 | 82.61% (19/23) |
| 229,376 | 32,737 | 84,860.001 | 385.78 | 771.41 | 37.82 | 100.00% (20/20) |
| 262,000 | 32,593 | 88,331.464 | 368.99 | 737.83 | 26.93 | 57.14% (16/28) |

*Decode is a three-token acknowledgment probe, not a canonical decode result. MTP acceptance
also has tiny, varying denominators; use the standard 400-token prompt benchmark for MTP
quality conclusions. Source chunks vary in size at the small 6K/8K/12K checkpoints.

### ACO selective Turbo4, native MTP off

Context allocation L=262,144. Requested hot was 512 pages / 131,072 tokens; startup admitted
466 pages / 119,296 tokens. Binary path:
 /srv/ai/paged-kv/builds/aco-no-mtp-replay-20261020/bin/llama-server
SHA-256 matches the ACO binary above.
Source summary:
 /srv/ai/paged-kv/results/release/aco-incremental-turns-mtp-off-20261021/summary.json

| Occupied C | New input tokens | Prompt time ms | New-content tok/s | Server PP tok/s | Decode tok/s* | MTP |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 32,736 | 28,214.950 | 1,160.24 | 1,160.24 | 21.27 | off |
| 65,536 | 32,766 | 102,164.803 | 320.72 | 641.33 | 34.45 | off |
| 71,680 | 6,142 | 26,459.612 | 232.13 | 463.84 | 33.72 | off |
| 77,824 | 6,142 | 28,580.156 | 214.90 | 429.42 | 33.35 | off |
| 86,016 | 8,190 | 41,121.468 | 199.17 | 398.06 | 32.88 | off |
| 98,304 | 12,286 | 67,885.381 | 180.98 | 361.80 | 32.00 | off |
| 131,072 | 32,765 | 126,630.990 | 258.74 | 517.40 | 7.76 | off |
| 163,840 | 32,755 | 87,893.531 | 372.67 | 745.21 | 7.82 | off |
| 196,608 | 32,754 | 88,861.384 | 368.60 | 737.07 | 7.15 | off |
| 229,376 | 32,766 | 89,040.245 | 367.99 | 735.86 | 7.68 | off |
| 262,000 | 32,607 | 89,749.351 | 363.31 | 726.50 | 7.75 | off |

*Three-token acknowledgment decode only. ACO MTP-off fresh-content rates show an atypically large
latency at some 6K-12K top-up turns; those rows are preserved exactly, not smoothed or dropped.

### Upstream master, CPU Q4_0 target KV, MTP off

Context L=262,144, CPU main KV; no GPU hot-pager pool.
Source summary:
 /srv/ai/paged-kv/results/release/master-cpu-q4-no-mtp-incremental-256k-20261021/summary.json

| Occupied C | New input tokens | Prompt time ms | New-content tok/s | Server PP tok/s | Decode tok/s* | MTP |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 32,736 | 28,057.550 | 1,166.74 | 1,166.74 | 10.08 | off |
| 65,536 | 32,766 | 38,414.121 | 852.97 | 852.97 | 9.06 | off |
| 71,680 | 6,404 | 9,012.854 | 710.54 | 710.54 | 8.60 | off |
| 77,824 | 6,404 | 9,406.956 | 680.77 | 680.77 | 8.15 | off |
| 86,016 | 8,452 | 12,845.446 | 657.98 | 657.98 | 7.61 | off |
| 98,304 | 12,548 | 20,081.800 | 624.84 | 624.84 | 6.92 | off |
| 131,072 | 33,028 | 58,962.388 | 560.15 | 560.15 | 5.68 | off |
| 163,840 | 33,028 | 69,162.253 | 477.54 | 477.54 | 4.84 | off |
| 196,608 | 33,028 | 79,231.655 | 416.85 | 416.85 | 4.16 | off |
| 229,376 | 33,028 | 89,311.301 | 369.81 | 369.81 | 3.68 | off |
| 262,000 | 32,884 | 99,149.262 | 331.66 | 331.66 | 3.30 | off |

*Three-token acknowledgment decode only.

### Upstream master, all-GPU Turbo4 target KV, MTP off (partial)

L=131,072, all-GPU Turbo4 KV, MTP off.
Source summary:
 /srv/ai/paged-kv/results/release/master-gpu-turbo4-no-mtp-incremental-128k-20261021/summary.json

| Occupied C | New input tokens | Prompt time ms | New-content tok/s | Server PP tok/s | Decode tok/s* | Status |
|---:|---:|---:|---:|---:|---:|---|
| 32,768 | 32,736 | 18,876.192 | 1,734.25 | 1,734.25 | 31.56 | pass |
| 65,536 | 32,766 | 26,369.292 | 1,242.58 | 1,242.58 | 34.99 | pass |
| 71,680 | 6,142 | 5,873.236 | 1,045.76 | 1,045.76 | 34.18 | pass |
| 77,824 | 6,142 | 6,128.563 | 1,002.19 | 1,002.19 | 34.21 | pass |
| 86,016 | 8,190 | 8,509.839 | 962.42 | 962.42 | 33.46 | pass |
| 98,304 | — | — | — | — | — | scratch allocation failure near C=96,228 |

*Three-token acknowledgment decode only.

### Upstream master, all-GPU Turbo4 target KV + native GPU MTP (106-01)

This is the master-build all-GPU MTP-on control that was previously misclassified as
missing. It completed the full 48-request canonical schedule (12 warmups, 36 measured)
at each row. Configuration: GPU Turbo4 target K/V and GPU Turbo4 draft K/V, native MTP
n-max=2, B/U=1024/256, temperature 0, seed 42. The 77,824 row is empty-context; the
other rows use the stated occupied contexts. `MASTER106_GPU.md` contains the complete
per-prompt/per-reasoning decode, MTP-acceptance, and fresh-prefill median tables.

| L / C | Cases | Decode median range tok/s | MTP median range | Fill prefill median tok/s |
|---:|---:|---:|---:|---:|
| 77,824 / empty | 48/48 | 80.5–107.3 | 56.4–93.1% | — |
| 8,192 / 6,144 | 48/48 | 74.2–105.9 | 50.8–94.9% | 1,510.57 |
| 16,384 / 14,336 | 48/48 | 75.1–101.6 | 54.5–92.1% | 1,455.59 |
| 32,768 / 30,000 | 48/48 | 71.2–96.0 | 55.1–90.1% | 1,352.18 |
| 65,536 / 61,440 | 48/48 | 65.7–91.9 | 57.2–95.5% | 1,110.22 |

Binary SHA-256: `527c34de411d4cad6290e828deb17a71816ae2b9620a3f1166b833e1267e0f0f`;
master source commit `d2407c33f9595d1badccfd9a82a33c4c5ceda490`. This is a master build,
not an ACO-branch result.

### ACO all-GPU Turbo4, MTP-off baseline admission attempt (Oct 23)

The requested counterpart to the ACO all-GPU MTP-on baseline was attempted at
L=77,824, empty context, B/U=1024/256, GPU Turbo4 target K/V, pager off, MTP off.
No benchmark request was sent. The Oct 15 ACO release binary
(`c9041ca96611ed4c9aa06be7b9a23443aa11d4b3bb30983d7dd2558c582659ec`) and the
current `build-cuda` candidate (`cd8f33f898d7662ba130ae56a07a965e2cf89151069e32040adaae85dd5c5529`)
both failed at context initialization. The server log reported a 2,508.1 MiB allocation
need with 1,085.9 MiB free and failed KV-cache allocation. Raw commands and logs are under
`/srv/ai/paged-kv/results/forward/missing-matrix/aco-gpu-off-20261023/attempt-01/` and
`attempt-02/`. This is a capacity/setup failure, not a measured speed result; do not
replace it with smaller-context or changed-batch numbers in the 77,824 cell.

### ACO build, CPU Q4_0 target KV, MTP off

This uses the ACO-built binary via the ordinary CPU-main-KV path; it is not selective ACO
paging and has no GPU hot-pager pool.
Source summary:
 /srv/ai/paged-kv/results/release/aco-cpu-q4-no-mtp-incremental-256k-20261021/summary.json

| Occupied C | New input tokens | Prompt time ms | New-content tok/s | Server PP tok/s | Decode tok/s* | MTP |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 32,736 | 28,331.661 | 1,155.46 | 1,155.46 | 9.62 | off |
| 65,536 | 32,766 | 38,828.851 | 843.86 | 843.86 | 8.80 | off |
| 71,680 | 6,148 | 8,939.814 | 687.71 | 687.71 | 8.47 | off |
| 77,824 | 6,148 | 9,339.310 | 658.29 | 658.29 | 7.70 | off |
| 86,016 | 8,196 | 12,790.720 | 640.78 | 640.78 | 7.57 | off |
| 98,304 | 12,292 | 20,022.803 | 613.90 | 613.90 | 6.90 | off |
| 131,072 | 32,772 | 59,095.641 | 554.56 | 554.56 | 5.64 | off |
| 163,840 | 32,772 | 69,239.794 | 473.31 | 473.31 | 4.79 | off |
| 196,608 | 32,772 | 79,436.805 | 412.55 | 412.55 | 4.10 | off |
| 229,376 | 32,772 | 89,607.403 | 365.73 | 365.73 | 3.61 | off |
| 262,000 | 32,628 | 99,491.083 | 327.95 | 327.95 | 3.25 | off |

### ACO build, CPU Q4_0 target KV + native GPU MTP (Oct 16)

This is the ACO build's ordinary CPU-main-KV route (`--no-kv-offload`, target Q4_0),
with native GPU MTP and Turbo4 draft K/V. It is distinct from selective ACO paging.
Canonical three-prompt campaign; all listed contexts have full three-run measurements.
The 256K server failed during startup, so it has no occupancy or speed result.
Source summary:
 /srv/ai/paged-kv/results/release/cpu-q4-offload-20261016-full/summary.json

| L / C | Fill prefill tok/s | P1 prefill / decode / MTP | P2 prefill / decode / MTP | P3 prefill / decode / MTP |
|---:|---:|---:|---:|---:|
| 16,384 / 12,288 | 917.46 | 33.28 / 26.95 / 91.30% | 34.72 / 22.06 / 63.43% | 33.95 / 26.15 / 86.60% |
| 32,768 / 28,672 | 873.69 | 27.15 / 22.87 / 93.38% | 27.49 / 17.48 / 58.90% | 27.49 / 20.08 / 74.38% |
| 65,536 / 61,440 | 760.65 | 21.15 / 17.57 / 92.65% | 20.87 / 13.47 / 60.63% | 20.63 / 15.23 / 76.19% |
| 131,072 / 126,976 | 598.28 | 14.18 / 11.95 / 93.38% | 14.15 / 9.39 / 62.50% | 14.21 / 10.91 / 81.19% |
| 262,144 | — | — | — | — (startup failed) |

MTP percentages are ordered prompt 1 / prompt 2 / prompt 3. This is CPU Q4_0 target KV
with GPU-resident native MTP, not GPU-resident target KV.

### Upstream master CPU Q4_0 target KV + native GPU MTP (Oct 16)

This campaign used the canonical three prompts and longer measured generations, not the
three-token fresh-chunk acknowledgment. Context settings / occupancies:
8,192/6,144; 16,384/12,288; 32,768/28,672; 65,536/61,440; 131,072/126,976;
262,144/258,048. Each completed context has three measured prompt runs per prompt;
262K failed during startup.

| L / C | Fill prefill tok/s | P1 prefill / decode / MTP | P2 prefill / decode / MTP | P3 prefill / decode / MTP |
|---:|---:|---:|---:|---:|
| 8,192 / 6,144 | 935.87 | 968.22 / 29.02 / 96.93% | 973.05 / 21.78 / 58.65% | 971.64 / 24.95 / 79.22% |
| 16,384 / 12,288 | 929.16 | 942.40 / 24.55 / 93.38% | 948.20 / 20.07 / 64.62% | 947.09 / 22.47 / 76.27% |
| 32,768 / 28,672 | 842.13 | 883.20 / 20.74 / 93.38% | 881.30 / 16.25 / 63.16% | 859.58 / 11.60 / 29.98% |
| 65,536 / 61,440 | 757.61 | 756.51 / 15.00 / 91.30% | 772.34 / 12.22 / 64.10% | 770.36 / 13.80 / 77.88% |
| 131,072 / 126,976 | 608.73 | 611.41 / 10.06 / 93.38% | 610.14 / 8.16 / 67.24% | 609.22 / 9.26 / 81.25% |
| 262,144 / 258,048 | — | — | — | — (startup failed) |

MTP order in each cell is prompt-specific. Source:
 /srv/ai/paged-kv/results/release/cpu-q4-master-control-20261016-mtp-safe/summary.json
and its matching manifest.json.

### October 20 ACO selective, MTP-off full-context fill/replay (separate protocol)

Source aggregate:
 /srv/ai/paged-kv/results/release/aco-turbo4-no-mtp-replay-20261020/aggregate.json.
This campaign used ACO selective Turbo4, MTP off, B/U=1024/256, cache-ram=12288,
8 checkpoints, and canonical 1 warmup + 3 measured repetitions per prompt. Each context
had its own fill/replay setup. Follow-up prompts sometimes reprocessed/replayed a large
prefix; their prompt_n is not the amount of newly introduced repository text. Treat fill
curve below as the context-fill data and follow-up prompt rates as separate cached/replay
behavior.

| L / target C | Fill requests | Initial → final fill pp tok/s | Min fill pp tok/s | Effective end-to-end fill tok/s | Peak GPU used MiB |
|---:|---:|---:|---:|---:|---:|
| 8,192 / 6,144 | 1 | 1,486.45 → 1,486.45 | 1,486.45 | 1,459.07 | 13,579 |
| 16,384 / 12,288 | 1 | 1,508.24 → 1,508.24 | 1,508.24 | 1,491.97 | 13,613 |
| 32,768 / 28,672 | 2 | 1,415.72 → 1,373.96 | 1,373.96 | 1,369.15 | 13,753 |
| 65,536 / 61,440 | 4 | 1,470.50 → 935.58 | 906.61 | 1,001.73 | 14,037 |
| 131,072 / 126,976 | 8 | 1,442.00 → 816.78 | 681.86 | 830.33 | 14,615 |
| 196,608 / 192,512 | 12 | 1,410.88 → 1,033.81 | 652.47 | 638.96 | 14,598 |
| 262,144 / 258,048 | 16 | 1,393.37 → 1,077.13 | 651.73 | 389.96 | 14,605 |

| L | P1 reported prefill / decode tok/s | P2 reported prefill / decode tok/s | P3 reported prefill / decode tok/s | Approx median current-input tokens | Approx cached rows |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 1,510.10 / 43.02 | 1,503.07 / 43.19 | 1,503.39 / 43.02 | 6,187 | 0 |
| 16,384 | 67.49 / 40.12 | 48.63 / 40.39 | 67.20 / 40.30 | 45–51 | 12,281 |
| 32,768 | 39.20 / 35.81 | 34.55 / 35.74 | 38.30 / 35.55 | 45–51 | 28,665 |
| 65,536 | 30.56 / 25.56 | 27.03 / 25.25 | 26.76 / 25.17 | 45–51 | 61,433 |
| 131,072 | 93.86 / 14.14 | 92.78 / 14.18 | 92.96 / 14.10 | 286–292 | 126,728 |
| 196,608 | 1,041.61 / 13.89 | 1,038.89 / 13.92 | 1,038.71 / 13.85 | 50,554–50,560 | 141,996 |
| 262,144 | 1,077.84 / 13.97 | 1,077.84 / 14.00 | 1,077.07 / 13.98 | 108,418–108,424 | 149,668 |

These follow-up fields are reproduced for audit, not to claim that cached rows were newly
ingested. Do not use them as a clean incremental-input speed curve.

## Standardized matrix and functional findings

Appendix A reproduces the complete 105-22 standardized matrix, including all per-prompt
decode, MTP acceptance, and prefill median tables for 15 configurations (105-06 to 105-20),
across four reasoning modes. The matrix used three measured runs per prompt and one warmup.
Its MTP values are proper-length response metrics, unlike the fresh-chunk three-token probes.

The 105-21 retrieval check returned 2/2 semantic-correct answers and 2/2 verified physical
promotions. It is a functional result, not a speed benchmark.

## Provenance and exclusions

- 105-22 aggregate source:
  /srv/ai/paged-kv/results/forward/105-matrix/105-22/attempt-01/MATRIX105_FINAL.md
  and MATRIX105_FINAL.json.
- Fresh-chunk method:
  .wiretail/execution/engineering-notes/incremental-repo-context-fill-only.md
- The method note explicitly excludes the October 21 directories
  aco-128k-hot-256k-fill-20261021 and aco-no-mtp-128k-hot-256k-fill-20261021:
  they sent cumulative single-user prompts and are invalid as fresh-chunk ingest curves.
- Early failed or incomplete runs are not averaged into successful medians. Failures that
  affect a headline comparison remain explicit above. 105-20 had 3 execution-incomplete
  rows among 48 scheduled rows due host-global OOM; they are excluded from its median.
- The upstream-master all-GPU MTP-on comparison is measured in 106-01 and summarized
  above; it is not a coverage gap. ACO all-GPU MTP-off remains unmeasured because both
  exact-configuration startup attempts failed before sending any benchmark request.

## Appendix A — full 105-series report

# MATRIX105 final — context, speed, recall, and prefill curve

**Status:** complete measurement aggregation. No rows were rerun.

## Cohorts and identity

- Original cohort 105-06–105-14: candidate `6b5b225b4791c1c5a9402304207c74e1a119e5c7f51a5300a891909dbdae34c7`, source `7a7e4befe175d050348375b4e6a862b47c416872`, model `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`; frozen manifest SHA-256 `3a978d416a65b2092346c6a66872642c4deb02949709a492d231954fd21cda53`. Loaded DSO identities are in JSON.
- Corrected scratch cohort 105-15–105-20: candidate `136142968618a512c81e481f780a2c5ae8709c72d1125ce2e26068253e045b4d`, source `ac9d7e0d7e7a24864b0e0777d0985a3415fb102a`, model `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`; scratch freeze SHA-256 `97225ae50ed611924fff25fc99fd518c00b780dce4943bf32bb54c0b857cda68`. Loaded DSO identities are in JSON.
- The source/DSO repair boundary changes the binary identity. Cross-cohort ratios are descriptive only. Both cohorts use Turbo4 target/draft K/V, B/U=1024/256, the same model hash, four reasoning modes and three prompts.

## Findings

- 14 of 15 configurations completed all 48 scheduled requests. 105-20 has 3 host-global-OOM incomplete measured requests; those are excluded from medians.
- The cold-file recall check returned 2 semantic-correct answers and 2 verified physical promotions; no recall query was incorrect or had unknown promotion telemetry.
- MTP cells without three valid measured percentages: 3; these are marked incomplete/not observed rather than zero.

## Per-configuration median tables

Medians use the three measured repetitions only; warmups are excluded. `incomplete` means fewer than three completed measured values, and `not observed` is not zero. Setup and execution incompleteness remain separate from performance.

### 105-06 GPU-only Turbo4 — 77824 L, unfilled C; cohort original

Configuration: target K/V `turbo4/None`, placement `as frozen`, pager `off`, router `None`, hot pages `not applicable`; status `complete`.
Workload audit: {"completed_rows": 48, "execution_incomplete_rows": 0, "measured": 36, "warmups": 12}.

#### DECODE MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 106.08 | 80.94 | 100.11 |
| low | 104.01 | 78.93 | 92.24 |
| medium | 101.09 | 79.47 | 92.82 |
| xhigh | 99.45 | 79.60 | 77.90 |

#### MTP ACCEPTANCE MEDIAN %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 93.57% | 59.38% | 84.62% |
| low | 90.81% | 57.96% | 74.92% |
| medium | 86.94% | 57.99% | 75.95% |
| xhigh | 85.11% | 58.52% | 55.85% |

#### PREFILL MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 95.04 | 111.83 | 106.47 |
| low | 95.26 | 110.82 | 106.56 |
| medium | 93.65 | 93.24 | 95.86 |
| xhigh | 95.28 | 110.63 | 106.71 |

### 105-07 CPU-F16-KV — 77824 L, unfilled C; cohort original

Configuration: target K/V `f16 in CPU RAM via --no-kv-offload/None`, placement `as frozen`, pager `off`, router `None`, hot pages `not applicable`; status `complete`.
Workload audit: {"completed_rows": 48, "execution_incomplete_rows": 0, "measured": 36, "warmups": 12}.

#### DECODE MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 30.97 | 22.34 | 28.43 |
| low | 29.23 | 22.00 | 25.44 |
| medium | 29.21 | 22.08 | 25.81 |
| xhigh | 27.39 | 22.21 | 21.10 |

#### MTP ACCEPTANCE MEDIAN %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 93.16% | 53.66% | 84.56% |
| low | 86.94% | 53.81% | 72.09% |
| medium | 86.01% | 53.65% | 71.34% |
| xhigh | 79.21% | 55.22% | 48.40% |

#### PREFILL MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 37.86 | 38.97 | 39.07 |
| low | 38.77 | 39.52 | 37.45 |
| medium | 38.46 | 37.42 | 38.34 |
| xhigh | 38.83 | 38.46 | 39.31 |

### 105-08 ACO candidate — 77824 L, unfilled C; cohort original

Configuration: target K/V `turbo4/None`, placement `as frozen`, pager `selective`, router `probe-rerank`, hot pages `256`; status `complete`.
Workload audit: {"completed_rows": 48, "execution_incomplete_rows": 0, "measured": 36, "warmups": 12}.

#### DECODE MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 102.04 | 73.54 | 99.61 |
| low | 103.31 | 77.47 | 88.82 |
| medium | 97.94 | 73.36 | 90.16 |
| xhigh | 99.16 | 84.79 | 78.49 |

#### MTP ACCEPTANCE MEDIAN %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 90.77% | 51.79% | 86.59% |
| low | 92.83% | 58.75% | 73.15% |
| medium | 84.80% | 51.53% | 74.92% |
| xhigh | 87.40% | 67.19% | 58.74% |

#### PREFILL MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 312.71 | 282.91 | 292.69 |
| low | 99.13 | 93.23 | 98.96 |
| medium | 295.02 | 264.16 | 275.72 |
| xhigh | 105.09 | 94.69 | 94.79 |

### 105-09 ACO candidate — 8192 L, 6144 C; cohort original

Configuration: target K/V `turbo4/None`, placement `None`, pager `selective`, router `probe-rerank`, hot pages `16`; status `complete`.
Workload audit: {"declared_total_rows": 48, "execution_incomplete_rows": 0, "measured_complete": 36, "recorded_rows": 48, "warmups": 12}.

#### DECODE MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 90.83 | 65.87 | 77.16 |
| low | 90.14 | 68.44 | 80.48 |
| medium | 84.85 | 62.36 | 78.06 |
| xhigh | 82.70 | 66.65 | 71.76 |

#### MTP ACCEPTANCE MEDIAN %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 91.13% | 52.25% | 69.67% |
| low | 89.16% | 56.29% | 75.39% |
| medium | 81.79% | 46.96% | 71.12% |
| xhigh | 77.73% | 52.90% | 61.34% |

#### PREFILL MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 1435.76 | 1430.56 | 1430.80 |
| low | 1427.29 | 1426.02 | 1426.11 |
| medium | 1438.94 | 1430.06 | 1433.30 |
| xhigh | 1432.48 | 1430.10 | 1434.03 |

### 105-10 GPU-only Turbo4 — 8192 L, 6144 C; cohort original

Configuration: target K/V `turbo4/None`, placement `None`, pager `off`, router `None`, hot pages `None`; status `complete`.
Workload audit: {"declared_total_rows": 48, "execution_incomplete_rows": 0, "measured_complete": 36, "recorded_rows": 48, "warmups": 12}.

#### DECODE MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 102.30 | 74.91 | 87.75 |
| low | 100.24 | 75.83 | 88.13 |
| medium | 100.00 | 74.10 | 89.93 |
| xhigh | 97.56 | 72.53 | 79.67 |

#### MTP ACCEPTANCE MEDIAN %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 90.81% | 53.89% | 71.12% |
| low | 87.32% | 54.46% | 71.56% |
| medium | 87.89% | 52.58% | 73.91% |
| xhigh | 83.61% | 50.00% | 60.11% |

#### PREFILL MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 1601.14 | 1600.86 | 1601.57 |
| low | 1595.04 | 1595.35 | 1596.88 |
| medium | 1602.32 | 1600.08 | 1600.97 |
| xhigh | 1598.03 | 1599.45 | 1599.61 |

### 105-11 CPU-F16-KV — 8192 L, 6144 C; cohort original

Configuration: target K/V `f16/None`, placement `CPU RAM (--no-kv-offload)`, pager `off`, router `None`, hot pages `None`; status `complete`.
Workload audit: {"declared_total_rows": 48, "execution_incomplete_rows": 0, "measured_complete": 36, "recorded_rows": 48, "warmups": 12}.

#### DECODE MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 24.93 | 18.25 | 21.41 |
| low | 23.65 | 19.27 | 22.33 |
| medium | 23.33 | 18.49 | 21.70 |
| xhigh | 22.09 | 17.96 | 19.87 |

#### MTP ACCEPTANCE MEDIAN %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 92.14% | 54.44% | 72.09% |
| low | 84.64% | 59.01% | 78.14% |
| medium | 84.46% | 55.23% | 74.14% |
| xhigh | 78.72% | 53.54% | 61.80% |

#### PREFILL MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 874.30 | 857.31 | 867.20 |
| low | 871.70 | 870.17 | 873.96 |
| medium | 870.10 | 870.59 | 870.87 |
| xhigh | 871.22 | 871.05 | 872.70 |

### 105-12 ACO candidate — 262144 L, 30000 C; cohort original

Configuration: target K/V `turbo4/None`, placement `Turbo4 KV with 16,384 hot tokens on CUDA0; encoded paged history retained on host`, pager `selective`, router `probe-rerank`, hot pages `64`; status `complete`.
Workload audit: {"declared_total_rows": 48, "execution_incomplete_rows": 0, "measured_complete": 36, "recorded_rows": 48, "warmups": 12}.

#### DECODE MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 72.29 | 49.94 | 61.92 |
| low | 67.61 | 47.60 | 62.59 |
| medium | 69.91 | 52.94 | 58.49 |
| xhigh | 65.62 | 53.56 | 48.11 |

#### MTP ACCEPTANCE MEDIAN %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 85.03% | 44.29% | 65.51% |
| low | 76.03% | 39.46% | 67.55% |
| medium | 80.92% | 49.13% | 59.78% |
| xhigh | 74.29% | 50.00% | 40.55% |

#### PREFILL MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 1316.69 | 1315.70 | 1315.67 |
| low | 1313.26 | 1312.53 | 1313.33 |
| medium | 1316.04 | 1315.87 | 1315.42 |
| xhigh | 1313.98 | 1313.83 | 1312.62 |

### 105-13 GPU-only Turbo4 — 32768 L, 30000 C; cohort original

Configuration: target K/V `turbo4/None`, placement `GPU CUDA0 Turbo4 target KV; pager off`, pager `off`, router `None`, hot pages `None`; status `complete`.
Workload audit: {"declared_total_rows": 48, "execution_incomplete_rows": 0, "measured_complete": 36, "recorded_rows": 48, "warmups": 12}.

#### DECODE MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 95.31 | 74.66 | 86.80 |
| low | 93.65 | 72.12 | 85.62 |
| medium | 91.70 | 71.49 | 88.42 |
| xhigh | 89.07 | 73.40 | 78.81 |

#### MTP ACCEPTANCE MEDIAN %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 91.76% | 61.93% | 80.07% |
| low | 90.38% | 58.89% | 78.39% |
| medium | 87.24% | 57.41% | 82.12% |
| xhigh | 83.02% | 60.25% | 68.45% |

#### PREFILL MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 1242.27 | 1241.89 | 1241.41 |
| low | 1239.21 | 1238.74 | 1239.25 |
| medium | 1241.10 | 1239.98 | 1240.42 |
| xhigh | 1238.98 | 1238.20 | 1238.52 |

### 105-14 CPU-F16-KV — 32768 L, 30000 C; cohort original

Configuration: target K/V `f16/None`, placement `CPU RAM F16 K/V via --no-kv-offload`, pager `off`, router `None`, hot pages `None`; status `complete`.
Workload audit: {"declared_total_rows": 48, "execution_incomplete_rows": 0, "measured_complete": 36, "recorded_rows": 48, "warmups": 12}.

#### DECODE MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 14.32 | 11.29 | 13.13 |
| low | 13.82 | 10.32 | 12.77 |
| medium | 13.88 | 10.93 | 13.18 |
| xhigh | 13.49 | 11.15 | 11.05 |

#### MTP ACCEPTANCE MEDIAN %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 93.86% | 62.50% | 81.52% |
| low | 88.55% | 52.88% | 78.39% |
| medium | 88.54% | 59.62% | 82.39% |
| xhigh | 86.18% | 61.25% | 61.80% |

#### PREFILL MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 757.63 | 754.73 | 756.62 |
| low | 756.11 | 753.71 | 753.95 |
| medium | 756.06 | 755.98 | 754.93 |
| xhigh | 752.44 | 753.22 | 752.36 |

### 105-15 ACO candidate — 262144 L, 61440 C; cohort scratch

Configuration: target K/V `turbo4/turbo4`, placement `GPU CUDA0`, pager `selective`, router `probe-rerank`, hot pages `None`; status `complete`.
Workload audit: {"declared_total_rows": 48, "execution_incomplete_rows": 0, "measured_complete": 36, "recorded_rows": 48, "warmups": 12}.

#### DECODE MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 59.55 | 35.22 | 50.08 |
| low | 57.22 | 38.04 | 48.98 |
| medium | 54.25 | 44.42 | 47.69 |
| xhigh | 53.91 | 40.62 | 41.68 |

#### MTP ACCEPTANCE MEDIAN %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 88.85% | 33.73% | 67.95% |
| low | 86.30% | 40.71% | 65.51% |
| medium | 78.96% | 55.15% | 62.43% |
| xhigh | 77.59% | 47.06% | 48.28% |

#### PREFILL MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 1102.83 | 1102.05 | 1101.55 |
| low | 1101.40 | 1101.51 | 1101.25 |
| medium | 1100.29 | 1101.61 | 1101.48 |
| xhigh | 1101.19 | 1099.85 | 1090.80 |

### 105-16 GPU-only Turbo4 — 65536 L, 61440 C; cohort scratch

Configuration: target K/V `turbo4/turbo4`, placement `GPU CUDA0`, pager `off`, router `None`, hot pages `None`; status `complete`.
Workload audit: {"declared_total_rows": 48, "execution_incomplete_rows": 0, "measured_complete": 36, "recorded_rows": 48, "warmups": 12}.

#### DECODE MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 79.03 | 57.25 | 70.09 |
| low | 78.71 | 58.43 | 72.26 |
| medium | 80.25 | 59.78 | 72.23 |
| xhigh | 72.00 | 58.96 | 54.65 |

#### MTP ACCEPTANCE MEDIAN %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 80.39% | 46.02% | 66.18% |
| low | 79.86% | 47.20% | 69.88% |
| medium | 82.67% | 49.50% | 69.67% |
| xhigh | 69.92% | 47.98% | 41.19% |

#### PREFILL MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 915.83 | 915.77 | 915.62 |
| low | 915.39 | 915.10 | 915.45 |
| medium | 915.56 | 915.71 | 915.72 |
| xhigh | 915.47 | 915.42 | 915.27 |

### 105-17 CPU-F16-KV — 65536 L, 61440 C; cohort scratch

Configuration: target K/V `f16/f16`, placement `CPU RAM via --no-kv-offload`, pager `off`, router `None`, hot pages `None`; status `complete`.
Workload audit: {"declared_total_rows": 48, "execution_incomplete_rows": 0, "measured_complete": 36, "recorded_rows": 48, "warmups": 12}.

#### DECODE MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 9.19 | 6.21 | 8.04 |
| low | 8.61 | 6.50 | 8.02 |
| medium | 8.76 | 6.60 | 8.02 |
| xhigh | 7.86 | 6.15 | 6.50 |

#### MTP ACCEPTANCE MEDIAN %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 90.08% | 45.88% | 73.60% |
| low | 81.75% | 50.00% | 73.60% |
| medium | 84.80% | 50.89% | 72.62% |
| xhigh | 70.91% | 44.74% | 49.87% |

#### PREFILL MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 616.43 | 616.34 | 616.30 |
| low | 615.36 | 615.87 | 615.86 |
| medium | 616.04 | 616.56 | 616.70 |
| xhigh | 615.12 | 615.04 | 615.59 |

### 105-18 ACO candidate — 262144 L, 131072 C; cohort scratch

Configuration: target K/V `turbo4/turbo4`, placement `ACO selective pager with a 256-page GPU hot budget and host backing`, pager `selective`, router `probe-rerank`, hot pages `256`; status `complete`.
Workload audit: {"declared_total_rows": 48, "execution_incomplete_rows": 0, "measured_complete": 36, "recorded_rows": 48, "warmups": 12}.

#### DECODE MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 39.78 | 26.66 | 32.44 |
| low | 37.93 | 25.48 | 32.79 |
| medium | 37.35 | 26.28 | 35.13 |
| xhigh | 31.60 | 27.75 | 34.07 |

#### MTP ACCEPTANCE MEDIAN %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 87.89% | 42.44% | 63.07% |
| low | 82.12% | 40.45% | 67.55% |
| medium | 80.07% | 44.21% | 71.34% |
| xhigh | 62.43% | 46.38% | 67.95% |

#### PREFILL MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 815.44 | 814.26 | 814.21 |
| low | 814.03 | 814.88 | 813.97 |
| medium | 816.01 | 815.34 | 815.23 |
| xhigh | 814.76 | 814.83 | 815.68 |

### 105-19 ACO candidate — 262144 L, 196608 C; cohort scratch

Configuration: target K/V `turbo4/turbo4`, placement `ACO selective pager with a 256-page GPU hot budget and host backing`, pager `selective`, router `probe-rerank`, hot pages `256`; status `complete`.
Workload audit: {"declared_total_rows": 48, "execution_incomplete_rows": 0, "measured_complete": 36, "recorded_rows": 48, "warmups": 12}.

#### DECODE MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 36.92 | 24.87 | 35.29 |
| low | 33.95 | 24.06 | 31.07 |
| medium | 33.90 | 22.96 | 29.35 |
| xhigh | 32.42 | 23.99 | 29.28 |

#### MTP ACCEPTANCE MEDIAN %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 84.80% | 39.87% | 78.79% |
| low | 72.34% | 39.77% | 62.71% |
| medium | 75.95% | 40.55% | 65.51% |
| xhigh | 70.09% | 38.61% | 59.94% |

#### PREFILL MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 817.64 | 817.63 | 816.98 |
| low | 817.94 | 817.92 | 818.50 |
| medium | 818.49 | 811.44 | 806.74 |
| xhigh | 818.31 | 818.83 | 818.73 |

### 105-20 ACO candidate — 262144 L, 256000 C; cohort scratch

Configuration: target K/V `turbo4/turbo4`, placement `ACO selective pager with a 256-page GPU hot budget and host backing`, pager `selective`, router `probe-rerank`, hot pages `256`; status `completed_with_execution_incomplete`.
Workload audit: {"declared_total_rows": 48, "execution_incomplete_rows": 3, "measured_complete": 33, "recorded_rows": 45, "warmups": 12}.

#### DECODE MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 35.09 | 23.55 | incomplete |
| low | 34.19 | 25.66 | incomplete |
| medium | 32.75 | 25.02 | 27.85 |
| xhigh | incomplete | 23.27 | 31.56 |

#### MTP ACCEPTANCE MEDIAN %

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 86.30% | 41.03% | incomplete |
| low | 80.66% | 49.62% | incomplete |
| medium | 76.75% | 46.02% | 56.99% |
| xhigh | incomplete | 39.26% | 71.34% |

#### PREFILL MEDIAN TOK/S

| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---:|---:|---:|
| off | 793.06 | 805.36 | incomplete |
| low | 807.25 | 805.74 | incomplete |
| medium | 806.74 | 807.27 | 805.67 |
| xhigh | incomplete | 806.05 | 803.13 |

## Fresh-ingestion prefill curve

Each row is an actual measured seed/append chunk. Fresh ingestion tok/s is computed from executed fresh tokens and that chunk’s prompt-evaluation duration; cached history is recorded separately and excluded. No interpolation is performed. Checkpoints use binary K (Ki) and exact occupied C; chunks are included only at the exact listed C where available.

| Exact C (tokens) | Available rows | Task / setup | Actual chunk fresh tokens | Cached history tokens | Fresh tok/s |
|---:|---:|---|---:|---:|---:|
| 8,192 | 0 | — | — | — | — |
| 32,768 | 0 | — | — | — | — |
| 61,440 | 5 | 105-15 105-15 | 10,771 | 54,190 | 910.44 |
| 61,440 | 5 | 105-16 105-16 | 7,250 | 54,190 | 602.04 |
| 61,440 | 5 | 105-17 105-17 | 7,250 | 54,190 | 441.19 |
| 61,440 | 5 | 105-18 105-18 | 10,771 | 54,190 | 511.59 |
| 61,440 | 5 | 105-19 105-19 | 10,771 | 54,190 | 511.53 |
| 65,536 | 0 | — | — | — | — |
| 131,072 | 2 | 105-18 105-18 | 4,387 | 126,926 | 551.52 |
| 131,072 | 2 | 105-19 105-19 | 4,387 | 126,926 | 553.04 |
| 196,608 | 1 | 105-19 105-19 | 7,888 | 191,335 | 541.85 |
| 256,000 | 1 | 105-20 105-20 | 3,640 | 252,527 | 441.08 |

All actual chunk rows, including occupied C values between named checkpoints, are retained in the JSON `all_actual_chunks` array. No cached historical token is treated as fresh ingestion.

## 105-21 cold-file recall and physical promotion

Report status `complete`; proof `near_full_256k_cold_file_recall_and_promotion_result`. Exact recall outcome and row-level semantic/promotion telemetry are preserved from the compact report below.

```json
{
  "configuration": {
    "C": 256000,
    "H": 65536,
    "L": 262144,
    "batch": 1024,
    "draft_kv": "turbo4",
    "max_tokens": 400,
    "mtp": "gpu",
    "mtp_n_max": 2,
    "page_tokens": 256,
    "reasoning": "off",
    "router": "probe-rerank",
    "seed": 42,
    "target_kv": "turbo4",
    "temperature": 0,
    "ubatch": 256
  },
  "notes": [
    "Exactly two scheduled natural queries were issued, one per distinct file, after separate slot erase and same-base max_tokens=0 restore. Neither query was retried.",
    "The 250Ki base prompt is server-tokenized at exactly 256000 tokens. Each natural query appended its question within the 6144-token reserve.",
    "Promotion is marked verified only because pre-query page inventory identified the natural-proof target as host-backed/nonresident and post-query natural_proof authenticated cold selection, H2D completion, mapping publication and target-graph use.",
    "The /slots valid_rows metric is target-resident row accounting; logical occupied token counts are reported separately from target/host residency."
  ],
  "outcome_summary": {
    "promotion_not_observed": 0,
    "promotion_unknown": 0,
    "promotion_verified": 2,
    "semantic_correct": 2,
    "semantic_incorrect": 0
  },
  "rows": [
    {
      "http_status": 200,
      "logical_occupied_tokens_after_response": 256089,
      "logical_occupied_tokens_before_query": 256000,
      "path": "tools/tokenize/tokenize.cpp",
      "physical_promotion": "verified",
      "promotion_evidence": {
        "after": {
          "content_version": 256,
          "generation": 6,
          "host_backed": true,
          "logical_page_id": 2,
          "position_begin": 512,
          "position_end": 768,
          "resident": true,
          "summary_ready": true,
          "valid_length": 256
        },
        "before": {
          "content_version": 256,
          "generation": 6,
          "host_backed": true,
          "logical_page_id": 2,
          "position_begin": 512,
          "position_end": 768,
          "resident": false,
          "summary_ready": true,
          "valid_length": 256
        },
        "content_version": 256,
        "logical_page_id": 2,
        "natural_proof": {
          "attention_layer": 3,
          "candidate_authenticated": true,
          "candidate_was_cold": true,
          "catalogue_epoch": 517461,
          "content_version": 256,
          "draft_event_sequence": 68,
          "draft_graph_used": true,
          "draft_use_epoch": 517776,
          "draft_use_query_generation": 1020,
          "h2d_aligned_bytes": 34603008,
          "h2d_completed": true,
          "h2d_event_sequence": 11,
          "h2d_queued": true,
          "h2d_useful_bytes": 34603008,
          "host_ready": true,
          "logical_page": 2,
          "mapping_published": true,
          "page_generation": 6,
          "physical_slot": 22,
          "promotion_published": true,
          "publication_event_sequence": 12,
          "published_epoch": 517504,
          "query_commit_admitted": true,
          "query_generation": 1020,
          "query_position": 256028,
          "selected_in_last_graph": true,
          "selector_event_sequence": 10,
          "selector_published": true,
          "selector_rank": 0,
          "sequence_generation": 1,
          "sequence_id": 0,
          "target_event_sequence": 18,
          "target_graph_used": true,
          "target_use_epoch": 517601,
          "target_use_query_generation": 1020
        },
        "page_generation": 6,
        "selected_in_last_graph": true
      },
      "query_prompt_tokens_expected": 256041,
      "semantic_rationale": "Correctly states the redirected-Windows-stdout fallback writes the original string with printf and returns without the console wide-character path.",
      "semantic_recall": "correct",
      "sequence": 1,
      "span": [
        31,
        94
      ],
      "status": "complete"
    },
    {
      "http_status": 200,
      "logical_occupied_tokens_after_response": 256181,
      "logical_occupied_tokens_before_query": 256000,
      "path": "tools/batched-bench/README.md",
      "physical_promotion": "verified",
      "promotion_evidence": {
        "after": {
          "content_version": 256,
          "generation": 1061,
          "host_backed": true,
          "logical_page_id": 7,
          "position_begin": 1792,
          "position_end": 2048,
          "resident": true,
          "summary_ready": true,
          "valid_length": 256
        },
        "before": {
          "content_version": 256,
          "generation": 1061,
          "host_backed": true,
          "logical_page_id": 7,
          "position_begin": 1792,
          "position_end": 2048,
          "resident": false,
          "summary_ready": true,
          "valid_length": 256
        },
        "content_version": 256,
        "logical_page_id": 7,
        "natural_proof": {
          "attention_layer": 3,
          "candidate_authenticated": true,
          "candidate_was_cold": true,
          "catalogue_epoch": 1035233,
          "content_version": 256,
          "draft_event_sequence": 208,
          "draft_graph_used": true,
          "draft_use_epoch": 1035779,
          "draft_use_query_generation": 2067,
          "h2d_aligned_bytes": 34603008,
          "h2d_completed": true,
          "h2d_event_sequence": 79,
          "h2d_queued": true,
          "h2d_useful_bytes": 34603008,
          "host_ready": true,
          "logical_page": 7,
          "mapping_published": true,
          "page_generation": 1061,
          "physical_slot": 31,
          "promotion_published": true,
          "publication_event_sequence": 80,
          "published_epoch": 1035269,
          "query_commit_admitted": true,
          "query_generation": 2067,
          "query_position": 256021,
          "selected_in_last_graph": true,
          "selector_event_sequence": 78,
          "selector_published": true,
          "selector_rank": 0,
          "sequence_generation": 1,
          "sequence_id": 0,
          "target_event_sequence": 86,
          "target_graph_used": true,
          "target_use_epoch": 1035352,
          "target_use_query_generation": 2067
        },
        "page_generation": 1061,
        "selected_in_last_graph": true
      },
      "query_prompt_tokens_expected": 256034,
      "semantic_rationale": "Gives both documented formulas: separate prompts N_KV=B*(PP+TG), shared prompt N_KV=PP+B*TG.",
      "semantic_recall": "correct",
      "sequence": 2,
      "span": [
        5,
        34
      ],
      "status": "complete"
    }
  ]
}
```

Semantic correctness and physical promotion are independent outcomes. Unknown promotion telemetry stays unknown; a correct answer alone is not evidence of page promotion.

## Incomplete or unobserved outcomes

- 105-20 completed with execution incomplete: its report records 45 completed rows and 3 host-global-OOM incomplete requests (measured seq 10, 24, 38); affected table cells are `incomplete` with null medians. They are not low-speed samples and were not retried.
- Any MTP cell lacking three reported response-local percentages is shown as `not observed` or `incomplete`; missing is never counted as zero.
- No semantic scoring is inferred from benchmark response text. Semantic misses are reported only where the recall report explicitly records them.
- Other configuration statuses and protocol anomalies appear with their individual tables above.

## Provenance

The JSON contains SHA-256 and byte size for both freeze manifests, every compact per-configuration report, every corresponding raw-artifact manifest, and the 105-21 report/manifest. Raw response streams remain external.
