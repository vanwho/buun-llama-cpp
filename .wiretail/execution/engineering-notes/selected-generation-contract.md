# Selected generation: current source and measured findings

Revision: hotpath-v10-20260914. Updated October7 for105-02 onward.
Current summary, not an attempt diary. Raw runs remain external. The 105-02a
selected/dense exact-fact comparison succeeded semantically with response-local
MTP. Identity-bound transfer/target-use witness remains unknown; this is not a
semantic failure or completion gate. Root owns task state/receipt.

## Architecture and preserved repairs

GPU Turbo4 hot target K/V, canonical Turbo4 host storage, full-L GPU Turbo4
MTP. GPU coarse shortlist -> exact encoded-key rerank -> bounded common R ->
atomic promotion -> query-only replay -> frozen generation. CPU RAM stores
pages, not historical attention computation. B1024/U256, automatic fast
routes, one retrieval decision per user query. No per-token rerank/H2D,
F16 draft, global reference route or expected-answer injection.

- `52d04942c`, `src/llama-graph.cpp::build_attn`: packed copies consume the
  completed layer SET_ROWS result. ggml nested offsets are absolute in the
  slab; rebase relative to the write result and preserve owner/bounds/type.
  Zero encoded K AND V padding once per shared-slab execution.519/768 leaves
  249 rows;4089/4096 leaves7. Negative-infinity masks cannot sanitize NaN
  norms/V. CUDA uses stream-ordered captured memset, not F16 history/fences.
  Reuse compares flattened allocation owners and absolute offsets.
- `538b32727`, `common/speculative.cpp::common_speculative_rollback_dft`:
  successful paired accepted-prefix replay retains authenticated carry and
  its idempotent guard. The old target_restored_without_draft event cleared
  the full draft prefix on the next process. Real unpaired restores still
  invalidate carry. Accept precedes target/draft rollback.
- `230a366fb`, `llama_kv_cache::apply_pager_live_policy`: reusable streaming
  chunk staging is not entire-transaction storage. Keep hot/history/slot,
  eight-page/event/max-transfer admission; do not cap by ring_bytes/page_bytes.
- `06e6b9473`, `llm_graph_input_mem_hybrid::can_reuse`: run the recurrent
  child's existing identity/binding guard. Do not copy an unconditional
  false can_reuse implementation or globally disable CUDA graphs.
- `llama_context::prepare_kv_attention_graph` consumes ALL admitted frozen
  R pages plus required query/current/resident-output rows. Nomination is
  provenance, not another filter. No additional all-layer union allocation.
- Current routing-summary source fixes address dirty-map pointer lifetime,
  zeroed metadata holes, incomplete-bounds retries and transactional full-grid
  publication. Keep holes invalid until complete bounds are published.
- `cdfb1b795` fixes summary lifetime; `7374e74a3` enables packed automatic
  MTP when capacity permits (prefill unchanged, direct fallback retained).
  `c803dc14b` is obsolete after SET_ROWS/padding fix `52d04942c`.

## Confirmed I32 -> I64 position defect

`llama_pos` is I32; direct CUDA query/native tensors and the paged kernel use
I64. Four direct/exact-wave uploads sent4bytes per element, leaving half the
positions stale. Device query0 became4344+(4345<<32)=18661632905464; host
masks hid the malformed upload from earlier snapshots.
The old first mismatch at row7 was the halfway point of the malformed
upload, NOT evidence of a4351/4352 page-boundary kernel bug.

Root correction in `src/llama-graph.h/.cpp`:

- direct_query_positions_host/direct_native_positions_host are persistent
  vector<int64_t>, widened from metadata/ubatch positions before upload.
- ordinary direct query/native and exact-wave query/native paths send8bytes
  perposition. Host *_uploaded vectors remain I32 comparison keys only.
- No new kernel, full-H/L copy, fence, F16 representation or diagnostic work
  is introduced into generation. Ordinary direct uploads only its Q vector.

The built comparison reads exactly4344..4357, maps all14 current rows and
captures7392bytes each of K/V, byte-equal across replay/control. SAME-question
logits/hidden and all emitted layer taps are exact0/0. This producer failure
is fixed, not bypassed. Compiled libllama SHA256:
`a78b2e0924374e43c0a4c8da9260f56c44d2d4e8b60c25803e741533e71c42a1`.

## Post-fix measurements

Historical candidate `a78b2e09…` only: exact prompts, one40-token warmup and
three400-token maxima each, temperature0/reasoning off. All12HTTP200, coherent,
fresh slot and cached input0. This matrix predates the current DSO and cannot
be claimed as its canonical result.

| Prompt | Median decode tok/s | Median request MTP | Aggregate accepted/proposed |
| --- | ---: | ---: | ---: |
| Python sorted merge/docstring |104.88|91.86%|712/772|
| mmap vs read paragraph |82.32|61.04%|283/460|
| Bash watcher |102.28|86.59%|426/492|

Tiny prompts are NOT bulk-prefill measurements. Do not relabel candidates or
claim an isolated patch gain without an old/new ablation.

### Completed 105-02a 32K trajectory

The selected-32k run is complete and reusable under source
`d4826061b7656ed788aea3467fd9d613fc855413`, server
`d52e7ec7054377fafdb38700b5539ede60edf12d6f111f41c073adab2c393b74`,
libllama `3d350db1723cf7ef0209531458e06c062a5b481d7ac3679c88d44506eaa6dfca`,
model `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
L32768/H16384/B1024/U256; four HTTP200 requests; committed C=26984 exceeded
H=16384 and target26000. Do not repeat the trajectory.

| Stage | Fresh | Prompt n/ms | Reported tok/s | Fresh-only tok/s | Decode tok/s | MTP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | 3890 | 3890/2831.383 | 1373.89 | 1373.89 | 92.63 | 225/284 (79.23%) |
| B | 11901 | 23791/17900.838 | 1329.04 | 664.83 | 74.15 | 240/278 (86.33%) |
| C | 9650 | 19289/15644.224 | 1232.98 | 616.84 | 70.62 | 166/206 (80.58%) |
| A2 | 129 | 248/1411.104 | 175.75 | 91.42 | 63.56 | 231/335 (68.96%) |

Reported prompt rates use prompt n and can include replay. A2 is short cached
input, not bulk prefill. It partially recalled the Windows-console
invalid-UTF8 fact (`<fffe>`, `invalid_utf8=true`) but did not provide all
redirection/non-Windows tests for `write_utf8_cstr_to_stdout`; without the
matched dense control this is not a ranking-defect finding. The measured 40%
MTP acceptance floor is a finding.

### Completed current-candidate canonical and selected/dense recall

The fresh canonical matrix is complete under candidate hashes recorded in
`evidence/PRODUCER105_02_FINDINGS.json` (including
`libllama-server-impl.so`). All12 requests were HTTP200; measured rows used
`cache_prompt:false`, reset slots and cached input0. The three original prompts
had median decode speeds105.25,82.83,103.94tok/s with median request MTP
91.86%,61.04%,88.27%. Inputs30/27/28 tokens and prefill medians202.14/203.06/
199.83ms are not bulk-prefill measurements. The earlier cached-hit attempt
with two Python HTTP500 responses is superseded diagnostic history, not the
current canonical result.

The matched exact-fact selected/dense control is also complete. Both returned
the requested fact with HTTP200/EOS. In selected mode, page5 (seq0, generation1,
page generation9, content version256, positions1280..1536) was cold and
host-backed before the query; it was eligible/summary-ready at zero-based rank
1/8, nominated/admitted, mapping published, and resident plus host-backed in
the after snapshot. Query104 covered5503..5557. Dense returned the same fact
with prompt5569/cached5494/new75, 141 outputs, 1541.8ms decode, MTP84/112.
Selected returned134 outputs, prompt5569/cached5501/new125, MTP79/112.
However, formal identity-bound transfer/target-use proof is unknown/incomplete;
`target_graph_used=false` leaves target-use certification unproven, not evidence
by itself that semantic execution failed. Record witness tags as unknown where
unsupported; no witness or bitwise
parity gate applies. No forced page ownership or retired diagnostic route is
required.

The tested candidate was built from source HEAD
`d4826061b7656ed788aea3467fd9d613fc855413` plus source diff
`3e99195feb7ceb29eecc40b17a0b0440c63360ec9b4be535023e45b61ed3cdab`, now
production commit `ccf07a9a6b7bbd32f85b6cab61a789d7c020214d`; do not report raw
build HEAD as ccf. Reuse the completed 32K and canonical results without
repeating unchanged runs. The cached
`cache_prompt:true` recapture returned HTTP200 with zero cached tokens and
re-evaluated the user query; its optional LCP journal check is not a completion
gate. Mixed identities remain disallowed within paired comparisons/campaigns,
not across separately labeled historical findings.

Earlier 105-02 route-fix candidate (distinct from current 105-02a matrix): source root
`/srv/ai/paged-kv/results/forward/105-02/source-review-fixes/routefix-current-20261007T190142Z/`;
server SHA256 `0c84115796002414733ff3f235af7f21188cc50312792c5b6ae9e27bb1672d63`,
libllama SHA256 `2a075159887d9c93e3f0736ecbab03cdb4a08390cf20ec8fa22cc51d3b4fb857`.
One A/B/A: all HTTP200. A1: 2155 fresh tokens, prefill1305.59tok/s,
decode155 tokens, MTP93/124 (75%). B: usage5498, cached2310; prompt n6366,
4575ms, 1391tok/s is raw kernel/replay-inclusive, not clean input rate.
Final A: 131 tokens, normal stop, fact-correct, no filler; MTP73/116 (62.93%),
decode72.17tok/s. Page5 was cold before/hot after; it was nominated/admitted
at zero-based exact-rank index1 for the current query. Prior-query witness
accounting is fixed; the full stage chain remains unvalidated. Server request
generation3 and trace generation106 use different counter domains. The witness
fix passed CPU metadata tests; latest disk hashes are server `d52e7ec7`,
libllama `3d350db1`, not a new model measurement.

## Remaining numerical boundary and exact next seam

CPU prefetch/residency/speculative-state and production-shape/padding CUDA
checks pass (24-Q/4-KV, Q1/Q3/Q256, causal gaps, generation roll, poisoned
padding). Cancellation, native frontiers, frozen map and next-MTP2/2 pass.
Full-fixture bitwise parity remains false; current max logit/hidden deltas are
.334331/.475013. The 12x248320 comparison is finite, top1 agrees12/12, KL
mean/max .000856/.001750 and TV mean/max .01544/.02658. Next-MTP is2/2 and
maps are exact. Main-question parity was proven by the I64 fix; bitwise
quantized batch/scalar equality is not a gate. After packed-MTP commit7374e74a3,
layer4 conv/recurrent deltas are token4360 .00580704/.000310144 and token4361
.00580704/.000388846. No additional owner defect is confirmed.

## Cold ranking: avoid false bottlenecks

`llama_kv_query_cold_rank_width` is min(cold_pages,64). L16K/H4K has coarse
cold width48, not eight. `retain_pager_coarse_shortlist` retains those records
and `prepare_router_query_layers` processes the full authenticated list.
Eight is a separate later full-K/V H2D transaction limit. Default encoded
query/key domains agree by source audit; no compensating transform needed.

Generation106's older trace missed page5 at exact bundle3. The final selected
query found it at rank1/8 and both selected/dense answered the exact fact. Keep
older measurements attached to their own identities. The 105-02a 32K/H16K,
canonical matrix and selected/dense semantic controls are complete. Next task
105-03 is the one novel occupied-context trajectory; do not rerun unchanged
small/canonical/32K experiments or reopen retired diagnostic routes.

## Artifacts and continuation

Compact findings: `evidence/PRODUCER105_02_FINDINGS.json`. Executable receipt
`evidence/V10_105-02a.json` passed; measured outcome is `goal_miss`, distinct
from task completion. Raw artifacts remain under
`/srv/ai/paged-kv/results/forward/105-02/source-review-fixes/`.

Keep one Qwen under the lifecycle lock; exact argv is in105-02. Immich ML
remains operator-stopped; never restart it automatically. 105-02a findings and
receipt validation are complete. The optional cache LCP journal check is not a
completion gate. 105-03 owns new scale findings only. Keep raw tensors/binaries
external and uncommitted.
