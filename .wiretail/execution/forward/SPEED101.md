# Phase 101: GPU execution repairs before capacity scaling

Revision: `hotpath-v10-20260914`. Amendment: `gpu-execution-101-20260930`.
This compact specification supersedes the scheduling and repair directions in
HOTPATH_POST100, REPAIR100 and completed phase-100 packets. They are provenance,
not startup context. The architecture in OVERVIEW remains authoritative.

For the next implementation after101-12g, use `forward/PREFILL101.md` and
tasks101-13–18 plus101-12h. That source-directed amendment supersedes the generic repeated
repair direction below: page-cache visitation, catalogue payload/digest work,
actual CUDA graphs and capacity-driven fences are now explicit owners.
102-01 follows101-12h, not the unsuccessful101-12e speed-review boundary.
Upcoming task startup loads the compact new amendment instead of this completed
implementation history; this file is retained as provenance.

## Assessment of the Luna diagnosis

The diagnosis correctly rejects scaling from the current result. The retained
100-03l row has 4,326 fresh tokens, 20.438454 s prompt processing, 211.660
tok/s prefill, 28.580 tok/s decode and prompt-2 MTP 96/337 = 28.487%. It is one
row, not a final three-prompt campaign. Its nine prompt-marked decode calls
account for 20.139032 s. CPU preparation scopes overlap; do not add them to
wall time. The 415.2 ms graph-build delta and 324 rebuilds cover the measured
request including generation, not an isolated prefill interval. Three us of
explicit server wait says nothing about waits *inside* llama_decode/backend
execution. The handoff's H8192 conflicts with its packet's H4096; the run-config
reports hot_tokens=3957. Recover actual physical pages, writable/history reserve
and occupied hot rows from the candidate, not that handoff sentence or bytes.

100-03f/g's no-change conclusions came from a 27-fresh-token request. They do
not establish the cost of packing/sealing a 4K prompt. Measured checkpoint and
batch-assembly repairs were real, but barely affected ~212 tok/s. 100-03h fixed
non-finite sparse-history MTP verification and its short prompt-1 accepted
47/54 proposals; it did not solve the final acceptance/speed campaign.

100-04a repeatedly tested callback branches and final-map CUDA attention in
isolation. It never proved actual checkpoint restore + real target/draft replay
against a one-pass control. Its useful transition helper and regression are
preserved in source commit bbc80bf86 (resolve full SHA with git). The task is
deferred, not passed; 101-10 owns the executable integrated proof. Another
audit of the same missing receipt is not useful work.

## Concrete code findings and chosen repairs

1. **Contiguous fused Turbo attention is unreachable.** In
   `ggml/src/ggml-cuda/fattn.cu::ggml_cuda_flash_attn_ext`, `turbo_kv` means
   either side is Turbo. The fused guard requires `!turbo_kv` together with
   `turbo_matched || turbo_fused_asym || turbo1_tcq_matched`; every supported
   match requires a Turbo side. Thus even Q=1/2/3/4 matched Turbo4 packed/dense
   attention falls through to the GPU materialize-to-F16 route. Do not remove
   the guard for every mixed tier: the comment records prior non-finite mixed
   GQA results. Restore **matched Turbo4 only**, with encoded-byte/mask parity.
   Direct-paged dispatch is separate and already has a fused GPU loader.
   `git blame` traces the disabling guard to this feature branch's ae02b1bb70
   (2026-09-21); origin/master retains the reachable fused guard. This is a
   branch regression, not an inherent cost of upstream Turbo4 or CPU backing.
2. **Batched packed Turbo4 still expands attended K/V into device F16.**
   `ggml_cuda_turbo_prefill_attend` expands the selected range and calls mature
   MMA. It is GPU computation, not CPU attention, but repeats expansion and
   consumes bandwidth/scratch. The contiguous fused ncols dispatcher currently
   asserts Q<=4; it cannot simply have that guard lifted. Tile arbitrary Q
   across grid.z with existing Turbo4 MMA loaders, using bounded legal tiles,
   or reuse the already batched direct-paged MMA path. Choose from measured
   **preparation + attention + append** cost, not names/counters. Never gather
   CPU/F16 history for production. Small on-chip tile dequantization is expected.
3. **'Fused GDN' does not mean parallel prefill.**
   `gated_delta_net_cuda` loops over all n_tokens serially within each block.
   `build_recurrent_attn` sets K=n_rs_seq+1 for MTP rollback snapshots.
   `ggml_cuda_gdn_fla_ptx_supported` rejects keep_rs, only allows cc800/860/1200,
   and requires >=512 tokens (508 on cc1200). This Ada cc890/U256 setup cannot
   use that cubin path even after dropping snapshots. Do not just whitelist a
   GPU or disable rollback. Use the existing parallel chunked GDN arithmetic
   for a large prompt prefix, then exact short fused tail for the last K
   snapshots; keep decode/verification on the original exact snapshot path.
   Enable the new prefill dispatch only if a matched CUDA operation benchmark
   establishes a gain. The runtime scheduler must keep chunk operations on GPU.
4. **Hybrid graph reuse duplicates an incomplete cache check.**
   `llm_graph_input_mem_hybrid::can_reuse` does not call the ordinary attention
   reuse helper, checks masks against changing valid extent, and bypasses its
   rebuild-reason instrumentation. Packed ownership is rounded to current
   selected extent in `llama_kv_attention_packed_row_capacity`; growth changes
   structural keys/allocations. Share the correct checks; use an admitted stable
   capacity bucket with mutable valid rows/masks, retaining all recurrent and
   allocation-generation safety checks and charging A in the memory ledger.
5. **Current packed writes quantize twice.** `build_attn` in llama-graph.cpp
   writes canonical GPU K/V, then SET_ROWS quantizes the same current K/V into
   packed storage. Add graph-ordered encoded byte row-copy from canonical
   GPU storage; GET_ROWS on a quantized tensor is not a byte copy. Preserve
   cache K/V names/domain, padding, scales and speculative rollback versions.
6. **Page boundaries still create global fences.** `llama_context::decode`
   calls synchronize at page waves; at U=P=256 this can fence each microbatch.
   Host capture starts from `seal_kv_pager_pages` at that fence. Replace only
   the pager's avoidable waits with compute-ready/copy-done events and bounded
   versioned source leases. Wait on a victim only when capacity requires reuse;
   do not delete fences protecting graph storage/MTP host reads. Preserve the
   original once-per-completed-page inclusive RAM store. No speculative sealing.
7. **MTP's apparent device handoff still has host dependence.** Target decode
   always copies h_nextn rows to embd_nextn host memory; native MTP process
   keeps host verify_h and pending_h. `set_embeddings_nextn_device` binds only
   the active graph's last microbatch tensor, so B>U cannot refer to all B
   rows. Use bounded device output staging and a versioned carry; keep draft
   full-L GPU Turbo4, process every prompt token with the right shifted hidden
   input, and retain rollback data only for actual speculative verification.
8. **Profiling is accidentally enabled by pager mode.**
   `llama_context_hotpath_profile_enabled` returns true for selective/exact
   without LLAMA_HOTPATH_PROFILE. Make detailed timing opt-in. Do not add a
   per-layer CPU callback, per-token CUDA readback, or synchronous metrics to
   fix missing evidence. Use nsys (installed) for one bounded diagnostic row.
   The mode-enabled timing helper was introduced by branch commit162f2c0e35;
   remove that diagnostic overhead without assuming it explains sevenfold loss.

The code already rejects host target tensor storage for a bounded pager in
llama-kv-cache construction. No reviewed evidence proves CPU attention is the
current main bottleneck; CPU metadata and GPU waits can still serialize it.
Confirm actual op/device placement once in 101-01. Likewise the old 64-query
model-ubatch throttle has already been removed: prefill_ubatch_size uses physical
admission, not the kernel tile size. Do not reimplement that completed repair.

Buun's merged master is ab22bc5385; existing improvements include no-copy batch
views, dual MTP graph slots, GPU Turbo MMA/paged loaders and the GDN chunk/cubin
paths. Reuse these rather than introduce another pager. KVMem's local
`src/adapter/llama-memory-kvmem.cpp` and
`kvmem/src/host/kvmem_runtime.cpp` provide selection-diff and bounded GPU-slot
inspiration. Its explicit synchronize/stage-out paths are not a justification
for additional per-token transfers. Do not import NVMe, a second runtime, or
its snapshot accounting into this branch.

## Execution order and priorities

101-01 measures a real matched 4K input and turns profiling off by default.
101-02/03 fix and measure fused Turbo4 decode and batched attention.
101-04 proves exact rollback snapshots for a split GDN prefill and measures
whether the graph decomposition wins. If it loses,101-04a owns a fused CUDA
prefix before101-05 proceeds.
101-05/06 fix stable graphs and redundant packed encoding.
101-07 completes overlap/bounded metadata maintenance without changing selection.
101-08/09 fix device MTP handoff and diagnose the first real disagreement.
101-10 proves integrated server replay with one weights allocation.
101-11 proves natural promotion/ring on the final implementation.
101-12 runs the final canonical short comparison and releases scaling only on
actual success; real misses create targeted repair successors before phase102.
Old phase101 capacity tasks are now 102-01/02/03; old phase102 review is103-01.

Each task has one owner, explicit code procedure and focused executable proof.
Routine implementation probes use 256–1024 fresh tokens and <=80 output;
one 4K row is reserved for attribution, one representative row after material
prefill changes. <=16K fresh input is an upper bound, not a required test size.
No long occupancy until 101-12 has released it. Correctness fixtures can use
synthetic pages and explicitly controlled selection; **natural** live proof
cannot force nomination/promotion. Short benchmark contexts are L8192/H4096;
if admission reports a different H, repair launcher/budget and record the actual
geometry, do not silently label it4K. Product sizes remain model/device-derived.

The 1300–1500 tok/s historical number is not a matched current long-prefix
baseline. 101-01 measures pager-off GPU with the same model/weights/codec/MTP,
prefix/B/U/candidate before attributing a sevenfold loss to paging. 500–750 is
still the requested selected-prefill target, not a guarantee based on inspection.
Sparse target attention can inherently disagree with dense full-L MTP draft;
fix state/mask defects first, and report residual distribution disagreement
separately rather than manufacture an acceptance percentage.

## Shared completion and context contract

Load this file, TESTING, the current task/cluster and its **one** named compact
predecessor handoff. Source pointers mean inspect symbol-sized regions using rg
and sed, not inject whole multi-megabyte source files at startup. Never load raw
Codex JSONL/WORK_LOG/full historical evidence. Full raw profiler data stays under
/srv/ai; summaries carry request geometry, times, first divergence, changed
symbols, executable command/hash and the next concrete action.

Use gpt-6-luna High for state/kernel work and Medium for final procedural runs;
task assessment pins remain Luna High. Respect existing project-wide runner
model overrides; do not edit the universal runner's defaults. Cluster boundaries
follow GPU attention, recurrent execution, storage, MTP and live lifecycle work,
with no arbitrary task-count cap.

Each new task emits `evidence/V10_<id>.json/.md` with schema_version=1,
task ID, full source_commit, actual dirty/build identity, and checks for its
state.required_proofs. Every check has the real executable argv, exit_code=0,
status=pass and SHA256 artifacts. Then run its completion_check before
`/srv/wiretail/task_state.py complete`; refresh current_task normally. A passing
receipt validator is not a computation proof. Measurement tasks can complete
with an honest speed miss **only after inserting a concrete repair dependency
before scaling**. Do not leave a done task with no receipt or forge runtime gates.

Build only affected targets; CPU TurboQuant execution refusal is not failure
of CPU byte storage. Use CUDA for inference and numerical Turbo4 tests. Use
sudo -n for managed service changes. Reuse the exact candidate if identity and
geometry match; otherwise reload through the existing lifecycle. Only one Qwen
weights allocation at a time. Keep a successful candidate loaded; never touch
8092. Do not use rigid answer formatting or small response caps as promotion
gates. Code/device paths/knobs must be generic; host service paths remain local.
