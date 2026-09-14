# Source and evidence audit, 2026-09-14

Audited integration tip: `44458ad37` (project branch and task48-01 start).
No live benchmark was launched in this audit. No service was changed. This
document corrects interpretation of existing raw evidence; it does not rewrite
historical files or claim the following repairs have been implemented.

## What the runs actually show

Phase36-04 raw SSE timings, **6144 actually evaluated prompt tokens**, one
measured trial per prompt, B128/U128 campaign:

| Mode | q0 pp / decode tok/s | q1 pp / decode tok/s | q2 pp / decode tok/s |
| --- | --- | --- | --- |
| Selective, H4096/A4096 | 214.23 / 43.34 | 215.64 / 39.30 | 214.91 / 42.00 |
| CPU main KV control | 723.49 / 12.28 | 729.51 / 12.29 | 733.70 / 12.54 |
| All-GPU control | 1436.45 / 39.98 | 1445.61 / 39.80 | 1440.24 / 40.01 |

Raw roots under `/srv/ai/paged-kv/results/v9/36-04/`:
`20260915T010000Z-selected-h4096-a4096-u128`,
`20260915T013000Z-cpu-kv-h4096-control`,
`20260915T020000Z-allgpu-h8192-control`; each contains
`raw-q{0,1,2}-measured-1.sse`. Directory timestamps are historical labels,
not a claim the audit ran tomorrow. Values above are the returned
`timings.prompt_per_second` and `timings.predicted_per_second`; some old
summaries instead divide128 by a duration measured over127 decode intervals.
Do not mix these conventions in ratios.

The prefill gap is real for these recorded workloads: roughly6.7x behind
all-GPU, not just a small-context comparison error. However controls report
`draft_n=126,draft_n_accepted=0`, whereas selective accepts55–60 drafts out of
66–72. Decode is therefore **not a fair validated native-MTP comparison**.
These rates also do not prove current-Q cold-page promotion. Investigate the
control acceptance problem before advertising MTP speed ratios.

Phase47's newer q0 selective rates are667.72 pp /34.25 decode, versus CPU
222.81/10.07 and all-GPU651.14/36.64. Those requests contain **30** prompt
tokens, not6144. q1 and q2 contain27 and28. The launcher command uses
`--kv-hot-pages 8` (2048 tokens at page256), not the4096-token hot size in the
summary. These are short smoke results. Phase47 lacks matching q1/q2 CPU and
GPU controls; ratios against q0's scalar control are inadmissible. Keep all
measured rows, including repeated q0, and match by actual workload identity.

More importantly, phase47 still loaded the phase43 bundle at
`/srv/ai/paged-kv/results/v9/43-02/20260914T021611Z-candidate`.
Phase43 and phase47 report the same binaries, including:

- `libllama.so.0.3.0`: `99d46d8e2a8e1a7bca0c6a67a8a2553b2949e80b93eaa35be49856dc904e5a91`
- server: `ebc2cac869de80023daa278f020b40588d33c64028e53baff61a1cdf61896d3c`

The manifest nevertheless changes its claimed source commit from
`09e62c8b3bb53e3833ad633d1535be2c058d7058` to
`c2b9f1d82280c3198d0c6b50e156841396a2ddd4`. The cause is
`write_bundle_manifest()` in `tools/server/bench/run-pager-profile-benchmark.py`:
it hashes the supplied old bundle but stamps **the current checkout's HEAD**.
`bundle_identity_errors()` primarily checks membership of loaded paths, not
proof that source changes were built. Thus later source repairs were not
reliably tested. Hashing an old binary more often cannot repair this.

The phase47 natural-cold incremental root is
`/srv/ai/paged-kv/results/v9/47-02/20260914T033641Z-natural-cold-incremental`.
It reaches C5004/5050 and reports no candidate/promotion/use chain. That is a
real absence of proof, but testing an old bundle and structurally ineligible
cold metadata cannot tell us that attention-aware retrieval itself is futile.

## Concrete source faults and cost risks

1. **Cold pages have zero valid rows.** `llama_kv_pager::exact_page_records()`
   and `routing_inventory()` synthesize a default `llama_kv_page_record` for
   host-only pages without assigning `valid_length` (default0).
   `llama_kv_cache_context::set_kv_page_select_inputs()` copies that value;
   CUDA `page_select_eligible()` rejects `valid_length <= 0`. Existing fixtures
   hand-construct valid records and miss the production conversion. This is
   a direct explanation for cold pages never qualifying, not a quality limit.
2. **The router uses incompatible coordinates.** `pager_routing_summary_build()`
   bounds dequantized stored Turbo4 coefficients without inverse FWHT.
   `build_attn_mha()` invokes `Qcur_routing` before the attention backend's
   transform, and `build_kv_page_select()` passes Q directly to a dot-product
   kernel. Derive router-Q from the same rotation/calibration as mature FA;
   test dot equivalence rather than assuming unrotated Q bounds are useful.
3. **GPU summaries exist in the plan, not the efficient live implementation.**
   `pager_routing_summary_build()` dequantizes rows on CPU, sometimes with
   tensor readback, repeatedly for each head. `seal_ready_pages()` loops
   layers/heads at a synchronization boundary. `set_kv_page_select_inputs()`
   rebuilds and uploads the full historical bounds catalogue every graph
   update; `can_reuse_kv_page_select()` keys capacity to growing inventory.
   These are plausible major prefill/append costs. Profile attribution is
   still required; do not claim an exact percentage saved from source alone.
4. **Cadence does not skip the expensive work.** CUDA `page_select_scores()`
   scores every page before rank's refresh/eligibility filtering. Current
   cadence increments query generations per microbatch, not accepted tokens.
   `page_select_rank()` repeats scans and previous-winner comparisons. Skip
   work early and use bounded parallel top-k, not per-token full-catalogue
   rebuilds or a one-thread sort.
5. **Instrumentation changed computation.** `build_attn_mha()` creates an
   extra `kv_packed_page_mass_producer` custom paged FA node to obtain mass.
   It cannot find nonresident pages and contradicts the mature-FA fast-path
   design. Remove it from normal selective execution. Instrument the actual
   router and copy/publication/use edges instead.
6. **Snapshot identity/epoch and consumption need end-to-end proof.**
   `apply_pager_live_policy()` synchronously reads per-layer outputs, mixes
   resident/cold ranks, and interprets indices against reconstructed inventory.
   `seal_ready_pages()` can change table epoch before consumption. Freezing
   exact submitted IDs and separating source validity from publication epoch
   is safer than either dropping every result or removing stale checks.
   Phase47's readiness repair is not disproved by its old-binary run, but
   neither is it validated.
7. **First overflow is a scheduler/ownership test, not a long benchmark.**
   The `KV pager batch write reservation failed`/`no_victim` boundary needs
   an exact page-crossing reproducer. Batches cannot reserve future write
   pages while every usable slot remains leased or host-dirty indefinitely.
   Chunk at legal write frontiers, reap completed events, and leave safe
   working reserve; do not add a device-wide barrier per token or reduce L
   until the error disappears.
8. **Generic geometry is still incomplete.** Fixed
   `VBR_SELECTED_PAGE_TARGET_LAYERS` live host transport assumptions need
   isolation from immutable legacy snapshot formats. Dynamic live unit
   descriptors must follow actual attention layers and KV heads.

## Direction and likelihood

Keep the compressed hot-pool/mature-FA architecture. Do **not** spend another
phase adding telemetry-only attention kernels, random recall loops or long
matrices to compensate for a broken selector. Fix metadata and coordinates,
prove a controlled selection→upload→actual use, then put summaries and their
incremental updates on GPU and remove CPU/synchronization overhead.

Significant improvement over CPU-main-KV decode remains plausible, supported
by the small occupied-context decode findings, but no measured complete fast
cold-retrieval result yet exists. Near all-GPU prefill and usable256K remain
unproved. Full-L GPU MTP, weights, graph scratch and copying all layer units
per promoted bundle constrain H and PCIe traffic. The design cannot promise
dense256K accuracy or arbitrary-focus GPU speed. First optimize an honest
small physically paging run; increase A/refresh only for demonstrated quality
needs. Independent per-layer physical promotion is a later optimization only
if measured bundle amplification dominates after these fixes.

## Primary references informing the design (not claims of our speed)

- [Quest paper](https://arxiv.org/abs/2406.10774) and
  [reference implementation](https://github.com/mit-han-lab/Quest): query-aware
  page bounds can choose candidate pages without dense attention. Bounds are
  a heuristic; adapting them to compressed coordinates and hybrid/recurrent
  Qwen architecture needs our own correctness tests.
- [NVIDIA asynchronous transfer guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#asynchronous-and-overlapping-transfers-with-computation):
  actual overlap requires appropriate streams, pinned host memory and correct
  dependencies. Merely naming an upload asynchronous does not prove overlap.

Tasks49–52 fold the useful context into new packets. They do not require
reopening any old acceptance ledger. All historical rates remain findings,
not proof that the latest code achieved the goal.
