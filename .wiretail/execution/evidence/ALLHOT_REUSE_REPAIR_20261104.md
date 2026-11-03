# ACO all-hot replay repair and fresh 256K prefill — 2026-11-04

Runtime fix `f12d3ff14cd1305107c5bbda7c174b14ead47eb2`. Configuration unchanged: L262144 / H65536, B1024/U256, CUDA0, Turbo4 target/draft, full-L GPU MTP2, one slot, 16 threads, FA on, no context shift. Each request appends new repository bytes in at most 32768-content-token groups. No canonical decode benchmark ran.

The defect was in all-hot history commit. `pager_->residency(seq).pages()` returned a reference into a temporary filtered snapshot destroyed before range iteration in C++17. History identities could be corrupted and then pruned, causing the actual-view certificate to request replay even though full prefix coverage passed. The fix keeps one snapshot alive across selection and pruning. No authentication or replay correctness checks were bypassed.

Controlled diagnostic baseline: capacity65536 / physicalpages256 / candidate1 / full1, yet second6144-token append evaluated12276tokens and replayed once (640.22 net-new tok/s). With the snapshot fix the certificate matched: selected24 historical pages,48 total view pages,changed0,replay0. The final uninstrumented build repeated this with one evaluation per new token. Temporary diagnostic code was removed.

The Release attention-view test had disabled assertions. Its target now uses -UNDEBUG; enabled checks exposed an obsolete F16-rejection expectation and an invalid physical slot in a two-slot fixture. These fixtures were corrected to reflect existing supported types/valid geometry. The existing overlapping-generation rejection also exposed premature success in the coverage helper; it now checks every intersecting record before accepting coverage. The assertion-enabled test passes.

## Hot-only validation

| Actual occupied C | New tokens | Backend tok/s | Net-new tok/s | Request-wall tok/s |
|---:|---:|---:|---:|---:|
| 6,144 | 6,144 | 1,457.53 | 1,457.53 | 1,451.07 |
| 12,288 | 6,144 | 1,208.00 | 1,208.00 | 1,202.78 |
| 28,673 | 16,385 | 950.13 | 950.13 | 948.30 |
| 62,465 | 33,792 | 610.40 | 610.40 | 609.11 |

All 5 append requests had zero replay, with backend prompt-token count equal to new occupancy growth.

## Fresh full allocation curve

One new clean session reached **261,121 occupied tokens**, leaving **1,023** tokens inside L262144. Earlier prefixes were reused exactly: all cached-token counts matched the prior occupied frontier.

| Actual occupied C | New tokens in interval | Backend tok/s | Net-new tok/s | Request-wall tok/s | Before net-new | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 6,144 | 6,144 | 1,446.10 | 1,446.10 | 1,440.97 | 1,450.72 | -0.3% |
| 12,288 | 6,144 | 1,199.29 | 1,199.29 | 1,194.56 | 633.67 | +89.3% |
| 28,673 | 16,385 | 948.54 | 948.54 | 946.55 | 491.85 | +92.9% |
| 62,465 | 33,792 | 609.33 | 609.33 | 606.91 | 314.30 | +93.9% |
| 78,849 | 16,384 | 818.18 | 409.24 | 408.48 | 408.66 | +0.1% |
| 88,065 | 9,216 | 821.86 | 411.20 | 409.73 | 410.09 | +0.3% |
| 100,352 | 12,287 | 833.71 | 417.06 | 415.85 | 416.05 | +0.2% |
| 130,049 | 29,697 | 870.18 | 435.18 | 434.45 | 435.21 | -0.0% |
| 196,608 | 66,559 | 813.47 | 406.84 | 405.69 | 407.42 | -0.1% |
| 256,001 | 59,393 | 755.69 | 377.92 | 376.96 | 378.27 | -0.1% |
| 261,121 | 5,120 | 590.16 | 295.43 | 291.82 | 295.88 | -0.2% |

Weighted complete-fill rates: **792.05 backend / 449.94 net-new / 448.61 request-wall tok/s**. 459,657 backend evaluations for 261,121 new occupied tokens; 15 appends. Backend counts all prompt work; net-new divides new occupancy by prompt time; wall uses HTTP request duration. Append sizes vary at checkpoints.

Full-run replay evidence: 5 unchanged-history requests and 10 requests with replay. Do not infer retrieval quality or MTP acceptance from one-token fill ACKs. The 32K-chunk completion does not repair the separate 64K-new-query zero-history-budget publication failure.

## Runtime identity and restoration

Executable SHA256 `70986416002c8d107a4f1251c4aefd69d1058b44d4d3069a82bd3126daa327f8`; fixed libllama SHA256 `66fa6490bf9765ce5879d9ad90acd17f6bc364cadb265eab75743a87eb81a347`. The executable SHA alone does not distinguish this shared-library repair; baseline libllama was `a549daf28b0ffad791c6f208478307bc1fad1f4bf454e31303e9dda8dcc5b298`. All full-run captures used the same DSOs. Source bytes were verified against the fix commit.

Sampled full-run resources: {'samples': 138, 'peak_gpu_used_mib': 15589.0, 'minimum_gpu_free_mib': 357.0, 'peak_server_rss_kib': 8696788, 'peak_server_hwm_kib': 14322560, 'maximum_server_swap_kib': 0}. These are observed five-second sample extrema; GPU free is the driver-reported counter.

Raw hot-only validation: `/srv/ai/paged-kv/results/forward/allhot-reuse-repair-20261104/production`. Raw full run: `/srv/ai/paged-kv/results/forward/allhot-reuse-repair-20261104/full256k`. Assertion-enabled fixture/build summary: `/srv/ai/paged-kv/results/forward/allhot-reuse-repair-20261104/production/validation-summary.md`. Hashes and detailed per-append counters are in the companion JSON.

Managed gateway/server restored after each run. Jellyfin stayed running at PID1000914 and healthy;8091 preserved;8092 untouched. The full decode/reasoning suite remains paused.

Exact full-run server argv:

```bash
/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp --jinja --chat-template-file /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/benchmarks/qwen38-shared-prefix.jinja -ngl 999 --device CUDA0 --fit off -fa on -c 262144 -np 1 -t 16 -tb 16 -ctk turbo4 -ctv turbo4 -b 1024 -ub 256 --poll 0 --host 0.0.0.0 --port 8080 --api-key-file /srv/ai/config/llama/api-keys --metrics --reasoning-preserve --no-context-shift --slot-save-path /srv/ai/paged-kv/results/forward/allhot-reuse-repair-20261104/full256k/cell/slot-state --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots --kv-pager selective --kv-page-size 256 --kv-hot-pages 256 --kv-router probe-rerank --kv-pin-recent 0 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4
```
