# HOTPATH29 summary

Result: partial. This is the compact 29-01/29-02 finding for 30-01. The
primary selective path is operational at L8192/H4096 with native Turbo4/GPU
MTP and accepted live drafts. It has physical host-backed pages, but all
completed natural/offload candidates report zero useful H2D and zero faults;
therefore real cold promotion/use and cold-miss latency are not proved.

`pp` is server prompt/prefill tokens per second, `tg` is predicted/decode
tokens per second, and TTFT is SSE first-token latency. `C` is occupied or
requested prompt tokens, `H` is physical hot capacity in tokens, and `A` is
observed selected/attended rows. B/U is configured batch/ubatch; effective U
was 64. MTP cells are proposed/accepted target tokens.

| profile | L/C/H/A; B/U | placement and physical cold backing | natural promotion/use | pp / tg / TTFT; output; MTP | peak memory | status and raw |
|---|---|---|---|---|---|---|
| primary q0 | 8192/6144/4096/4096; 128/64 | target Turbo4/GPU; native full-L Turbo4/GPU draft; host 3840 rows / 64.88 MB | not proved; H2D 0, faults 0 | 212.17 / 33.66 / 28.967 s; 100; 84/57 | target pool 69.21 MB; charged 15.207 GB; reserved 1.310 GB | pass; `/srv/ai/paged-kv/results/hotpath29-primary-direct-20260913b/raw-q0-measured-1.sse` (`d983868a…`) |
| primary q0 repeat median | 8192/6144/4096/4096; 128/64 | same; host 3840 rows / 64.88 MB | not proved; H2D 0, faults 0 | 220.71 / 31.78 / 27.846 s; aggregate; acceptance pair retained by companion q0 | same | pass, 3 repeats; `/srv/ai/paged-kv/results/hotpath29-primary-q0x3-20260913/` |
| incremental final | 8192/6148/4096/4096; 128/64 | same; 15 host pages / 3840 rows | not proved; H2D 0, faults 0, evictions 0 | 148.08 / 33.67 / 4.799 s; 34; 28/21; live frontier 6183 | same | pass, five turns; `/srv/ai/paged-kv/results/hotpath29-primary-incremental-20260913b/raw-04.sse` (`60ba947e…`) |
| natural cold recall | 8192/6216/4096/4096; 128/64 | same; prior live host backing 15 pages | not proved; H2D 0, faults 0; returned `cedar-orbit-17` | 117.49 / 28.13 / 0.292 s; 9; 8/6 | same | partial, not cold-miss latency; `/srv/ai/paged-kv/results/hotpath29-primary-incremental-20260913b/raw-cold-topic.sse` (`a4557778…`) |
| all-GPU control | 8192/6144/N/A/8192; 128/64 | dense CUDA target; native full-L Turbo4/GPU draft; pager off, host telemetry N/A | N/A | 1364.15 / 102.05 / 4.512 s; 128; 88/82 | pager allocation telemetry N/A | pass control; `/srv/ai/paged-kv/results/hotpath29-control-allgpu-q0-20260913/raw-q0-measured-1.sse` (`04c076c7…`) |
| 32K first append | 32768/1200/16384/1203; 128/64 | target/draft Turbo4/GPU; host 4 pages / 1024 rows / 17.30 MB | not proved; H2D 0, faults 0 | 648.66 / 65.61 / 1.853 s; 4; 2/2 | target pool 276.82 MB; charged 15.265 GB; reserved 1.368 GB | pass, partial population; `/srv/ai/paged-kv/results/hotpath29-scale-32768-h64-20260913/raw-00.sse` (`7f36251e…`) |
| 32K second append | 32768/1203/16384/1203; 128/64 | same host backing | no valid result; fault before proof | pp/tg/TTFT/output/MTP null; 1400 requested, 136.155 s | same | failed: illegal device memory access at synchronize; `/srv/ai/paged-kv/results/hotpath29-scale-32768-h64-20260913/raw-01.sse` (`d2ca4ed9…`) |
| 128K allocation + smoke | 131072/512/65536/515; 128/64 | target/draft Turbo4/GPU; host 2 pages / 512 rows / 8.65 MB | not proved; H2D 0, faults 0 | 866.10 / 21.41 / 0.593 s; 2; 2/0 | target pool 1.107 GB; charged 15.470 GB; reserved 1.573 GB | partial allocation/smoke only; `/srv/ai/paged-kv/results/hotpath29-scale-131072-h256-smoke-20260913/raw-00.sse` (`e4bc1c92…`) |

The all-GPU row is an arithmetic control, not an offload ratio: it uses
A=8192 versus selective A=4096. The CPU-main-KV/GPU-MTP control remained null
because a second weight-resident process would not fit safely. No standalone
synthetic CUDA-kernel throughput record is in the named 29 artifacts; the
focused CPU/CUDA fixture is route/lifetime proof without a token-rate result.
Direct-route/q0 measurements are not relabelled as kernel wins.

The main measured critical path is control/movement churn: the primary trace
rebuilds/submits graphs with no replay, changes the table repeatedly, reads
large summary volumes, and seals pages to host. Wait and queue counters overlap
and are not added or attributed as CPU cost. Scale adds the concrete source
blocker: the second L32768 append aborts at
`ggml_backend_cuda_synchronize` after 136.16 s.

Remaining issues, limited to three: repair that continuation/lifetime fault;
make the natural recall path perform authenticated H2D and publication; and
reduce selected-path summary/seal/table churn or restore graph replay. The
cheapest reproducers are the existing L32768 two-append sequence, the existing
L8192 checkpoint plus cedar-orbit recall (assert H2D > 0), and the existing
L8192 q0 three-repeat counter trace, respectively. H is a useful capacity
tradeoff, not proof of populated context; the 128K result remains smoke-only.

Full-L MTP is configured at every scale, but only live small/first-32K drafts
were accepted. The 128K smoke accepted 0/2, so no full-L populated MTP claim is
made. Sparse selection is bounded approximate ranking (top-k 8, explore 1),
not normalized attention mass; exact hybrid CPU attention and successful cold
promotion remain outside the proved result.
