# Phase-100 hot-path continuation: source map and decisions

Revision: `hotpath-v10-20260914`. Amendment: `post100-hotpath-20260930`.
This is the compact design input for 100-03d through 100-04. The old phase-14
acceptance journals and 93-series campaigns are provenance, not task context.

## Measured boundary, not an inference

The same-candidate 100-03b L=8192/H=4096, B=1024/U=256, all-Turbo4,
native-GPU-MTP campaign measured selected fresh-prefill medians of 179.16,
182.06 and 182.76 tok/s for the canonical three prompts (minimum goal 500),
MTP acceptance 20.74%, 0%, 0%, and selected/CPU-main-KV decode ratios 0.848,
1.089, 0.702. These are a *failed goal result*. The selected-placement
12-row cumulative counters were 3,296 graph captures/rebuilds, 225,401
packed-copy updates/8,591,491 copied rows, 929,381,376 host-seal D2H bytes,
20,608 seal-copy calls and 46,256,256 us reported waits. Cumulative counters
do not attribute wall time to a request or prove which operation is dominant.
100-03c removed a repeated row/query scan in `src/llama-graph.cpp`; its one
post-edit prompt-2 diagnostic was 240.64 fresh tok/s, 31.73 decode tok/s and
94/226 MTP accepted. It is not a paired speed result or a solved goal.

## Buun implementation and next exact owners

* `src/llama-kv-attention-execution.cpp::llama_kv_attention_packed_row_capacity`
  rounds the *current selected extent* to a page. `llama-graph.cpp::
  llm_graph_input_attn_kv::can_reuse` rejects a graph on every row-capacity
  change. `llama_kv_attention_packed_cache::find_or_create` includes capacity
  in the structural key, allocating new K/V owners; old owners drain. Inspect
  `ggml_backend_tensor_copy_async` calls in `refresh_selected_data`. The
  likely repeated graph/packing cost needs request-local measurement before
  choosing an optimization. Stabilize capacity within an *admitted, charged*
  attention-workspace bucket; never allocate a free full-H duplicate.
* `src/llama-kv-attention-execution.cpp::select_route` already admits packed
  and direct GPU Turbo4 routes. 99-01's matched fixture favored packed for its
  tested shape, not every shape. Profile preparation + CUDA kernel + graph
  lifecycle together. Never silently select `selected_reference` in production,
  and never materialize selected history in CPU/F16.
* `src/llama-kv-cache.cpp::seal_kv_pager_pages` polls and checks maintenance
  each compute boundary. `src/llama-kv-pager.cpp::seal_ready_pages` uses a
  maintenance queue with full-scan fallback; completed, immutable pages are
  host-sealed asynchronously when a lane is available. `host_->wait()` is
  allowed only on real queue backpressure. Preserve inclusive canonical host
  Turbo4 pages and epoch/content correctness while removing avoidable fence,
  scan, D2H or wait costs. Measure transferred bytes against newly completed
  pages; do not call all D2H waste.
* `tools/server/server-context.cpp` owns final-user query checkpoint/restore,
  replay-once and request cancel. `src/llama-kv-cache.cpp::apply_pager_live_policy`
  owns mailbox admission and frozen history. `common/speculative.cpp` and
  `src/llama-memory-hybrid.cpp` own MTP carry and GDN/recurrent state. For low
  acceptance, compare same-state target/draft logits/proposal accounting at
  the first divergence before altering speculative settings. Sparse target
  versus dense draft may inherently disagree; quantify separately from an
  actual state or bookkeeping defect.

## Local KVMem comparison: borrow mechanisms, not the implementation

`/srv/repos/kvmem-llama.cpp/docs/architecture.md` and
`src/adapter/llama-memory-kvmem.cpp` show a bounded GPU block-slot pool with
canonical host blocks, packed GPU-format stage-in, stage-in/out *diffs*, and
unchanged-selection/all-resident replay skip. See `stage_in` around 1733,
graph-reuse handling around 2504, and writeback/stage-in around 4341–4428.
This supports avoiding per-token full-view rebuild and repeated H2D. It does
not license importing KVMem's separate pager: its target/draft codecs and
architecture differ from this all-Turbo4 target. Preserve original native
positions/RoPE and the final committed page map during replay. Check whether
Buun already implements an equivalent fast path before adding another one.

## Incoming fork master

Local `master` was fetched to `ab22bc538502534c96041ccb62162f7278476ee8`
on 2026-09-30, 435 commits beyond this plan branch's shared base. 100-03d
merges this *pinned* SHA; no later task assumes pre-merge line numbers. Review
in particular `8d4e76ab1` sparse CUDA attention composition, `fcd45e49e`
batch-ext migration, `f214aedb3` dual MTP graph arenas, and the incoming MTP
draft CUDA-graph/sampled-proposal changes. Preserve Buun's fork dispatch and
our Turbo4/page-table ownership; test only affected fixtures and one short
live request after conflicts. Do not interpret an upstream sparse-attention
interface as an already integrated Turbo4 paged kernel.

## Optimization order and gates

100-03d merge -> 100-03e one bounded attribution profile -> 100-03f stable
packed owner/graph and delta-copy -> 100-03g host seal/transfer overlap ->
100-03h exact MTP first-divergence repair -> 100-03i minimal matched speed
decision -> 100-04 small integrated replay/cancel proof -> phase 101 scaling.
For each code task: one deterministic affected fixture, at most one short
live request only if its claim requires the actual 27B GPU route. Missing
measurements/configuration are repaired in-task, not called a performance
failure. A measured poor result may complete its measurement task, but it
must create a source-directed repair before scaling. No 36/48-row loop, 256K
fill or 10-trial campaign inside a hot-path edit.

Production H, L, G, A and B/U remain tunable and derived from device/model
budgets. Test L=8192/H=4096 and B=1024/U=256 unless a named fixture needs
fewer pages. Full-L GPU Turbo4 MTP draft and Turbo4 target are non-negotiable.
Keep successful candidate loaded, exactly one 27B model process, and retain
site-specific paths only in `.wiretail` or `/srv/ai`.
