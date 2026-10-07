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

### 105-03 attempt01 — partial scale curve and GGML metadata-arena failure

External summary:
`/srv/ai/paged-kv/results/forward/105-03/attempt-01/runtime-failure-summary.json`.
Attempt01 ran repo HEAD `503aec68e4c920a4586941fdb7c60446a6ce0d82`,
backend version `b14455-d4826061b`, with L262144/H51200, 200×256-token hot pages, B1024/U256,
Turbo4 target and full-L GPU MTP.

| Stage | C | Fresh/cached | Processed/fresh tok/s | Decode tok/s | MTP accepted/drafted |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 4258 | 3890/0 | 1376.59/1376.59 | 84.45 | 215/306 |
| B | 16545 | 11888/4258 | 1250.94/625.70 | 88.30 | 247/304 |
| Continuation2 | 28800 | 11856/16545 | 939.28/469.90 | 76.21 | 233/330 |
| Continuation3 | 41103 | 11904/28800 | 754.67/377.50 | 80.78 | 251/295 |

All200 pages/51200 tokens were admitted. Last complete snapshot had65
host-backed/resident pages, no faults/evictions; GPU free was roughly626–686
MiB across observed snapshots. Request4 failed while capturing checkpoint at
uncommitted C=52998; committed C stayed41103. `ggml_new_object` in
`server_context_impl::update_slots`, via `ggml_view_4d` in
`llm_graph_context::build_attn`, exceeded the GGML CPU metadata arena by368
bytes. This was not VRAM OOM. The held-out recall and canonical prompts were
not reached; attempt01 is incomplete, not scale proof.

Root repair `a079452f56788049109cbef6404f233a41f75e08` sizes `graph_max_nodes`
using checked structural allowance of eight objects per page/attention layer
and fixed budgets from the physical plan, independent of L. It changes no GPU
KV, routing, B or U policy. CPU 200×16 fixture/build and two focused tests pass.
### 105-03 attempt02 — completed large trajectory and fact recall

The repaired candidate completed 22 fill requests plus A2 without runtime
fault, ending committed C=257445. A2 correctly recalled that invalid bytes
`ff fe` print as `<ff fe>` (space-separated hex bytes) on a Windows console,
and `invalid_utf8=true`. A2 MTP was77/112 (68.75%); aggregate fill MTP was
4434/6048 (73.31%). The fact is correct; large-context physical rank/promotion
remains unknown and is not inferred from the answer.

The compact 23-row numeric curve, exact candidate/DSO identity and raw artifact
SHA references are tracked in `evidence/FORWARD105_OCCUPANCY.md`; external raw
request/SSE artifacts remain append-only.

Report both processed and genuinely fresh input rates: aggregate 744.0 versus
375.1 tok/s (about2x replay). At C=248771 the tail was645.0 processed /322.6
fresh tok/s; final C=257248 was611.8/306.1. Decode was about38 tok/s. Final
observed GPU use was15357 MiB with about590 MiB free. Canonical12 was not sent:
an obsolete guard reserved11072 tokens although rendered C=257477 plus the
40-token ceiling fit L262144 with4699 remaining. Root corrected the guard to
exact rendered length + output ceiling + MTP3 +4096. Wrapper is now restored
to16K/H16K; do not rerun the large trajectory. Post-load canonical rows are
deferred to105-03b. The final large rank/promotion stage is unknown; no claim
of physical promotion is justified.

Attempt02 was built from repo HEAD
`503aec68e4c920a4586941fdb7c60446a6ce0d82` plus graph-arena repair delta
`a079452f56788049109cbef6404f233a41f75e08`, now that source delta's commit;
it was not built from a079 HEAD. Executable hash starts `9c95e267`, libllama
`5a03a591`, server-impl `efa611a6`, CUDA `f8b66c3f`; complete DSO identity is
in attempt02 `candidate-identity.json`. Preserve attempt01's separate partial
curve and CPU metadata-arena failure finding (368 bytes short, not VRAM OOM).

105-03a validates the existing split-document-query layout at L16K/H8K with
one A1/B/A2 sequence per arm (six total). 105-03b, only after that passes,
owns one novel improved L256K/H51200 curve, primary recall, occupied-base
canonical12 with history cached, and two extra short earlier-document facts
before teardown. Canonical requests are not fresh250K prefill or empty-slot
resets; 105-02a's empty-slot matrix remains separate. Use ordinary user-message
source turns and a short question-only final user message. Keep the 4096-token
margin, preserve completed curve points and frontier evidence, and report
processed versus fresh throughput separately. No unchanged retry, bit-parity
gate, forced route, or formal witness gate.

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
fix passed CPU metadata tests; that historical build's hashes were server
`d52e7ec7`, libllama `3d350db1`. They are not the latest disk/runtime identity.
The arena-repair build is server `9c95e267c7daf768ecf77f2cb03c2a23e4e2f100b9db12773f66369f36d05b65`,
libllama `5a03a591c7d0125a9469274bc747cf819fa801b9d580bda4294fa885b841e4bb`,
server-impl `efa611a672ab73931f8d834cd6d7017fc4e702acc2307f9ccebb1c1384f6b1c0`,
CUDA `f8b66c3fa9b17cc77e08678dba8407b4c2998362278e6e796f9d5cf67de42867`; its
build-time source delta is now commit `a079452f56788049109cbef6404f233a41f75e08`.
Full loaded DSO map is recorded in attempt02 `candidate-identity.json`.

### 105-03a — paired client-layout proof

The same A1/B/A2 source files/ranges and questions completed both legacy and
split-document arms: six HTTP200 requests at L16384/H8192, HOT32, B1024/U256,
Turbo4 target, native full-L GPU MTP, probe-rerank and no context shift. The
split arm places the documents in earlier ordinary user messages and leaves
only the short question in final A2. Receipt
`evidence/V10_105-03a.json` records the validated checker pass.

| B arm | Fresh / processed | Prompt ms | Fresh / processed tok/s | MTP accepted/drafted |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 5353 / 10695 | 7612.341 | 703.20 / 1404.96 | 230/338 (68.05%) |
| Split | 5358 / 5455 | 4332.828 | 1236.61 / 1258.99 | 237/321 (73.83%) |

Split reduced B prompt time43.08% and raised genuinely-fresh rate75.86%;
processed prompt tok/s remains a different measure. Both final A2s were
correct, natural-stop, `<ff fe>` and `invalid_utf8=true` (155 legacy /149 split
output tokens). Legacy B and split A1/B ended at the normal400-token ceiling;
these are valid partial code completions, not retry or transport failures.
Physical rank/probe telemetry remains unknown/optional, not a gate. The
post-load canonical matrix remains unmeasured, so this is not full completion.

Campaign source HEAD was `4e915425751f65fd2c91ea2864277b6cc3e6348d` plus dirty
fingerprint `6e340b9…`; the candidate was the existing build from HEAD
`503aec…` plus graph-arena delta `a079452…`, not built from campaign HEAD.
Exact compact findings and raw artifact hashes are in handoff
`handoffs/105-03a.md` and receipt `evidence/V10_105-03a.json`.

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
canonical matrix and selected/dense semantic controls are complete. The 105-03
large trajectory and primary fact are also executed; do not repeat them
unchanged. 105-03a validates the split layout, then 105-03b owns only novel
scale/recall/canonical evidence. Do not rerun unchanged small/canonical/32K
experiments or reopen retired diagnostic routes.

## Artifacts and continuation

Compact findings: `evidence/PRODUCER105_02_FINDINGS.json`. Executable receipt
`evidence/V10_105-02a.json` passed; measured outcome is `goal_miss`, distinct
from task completion. Raw artifacts remain under
`/srv/ai/paged-kv/results/forward/105-02/source-review-fixes/`.

Keep one Qwen under the lifecycle lock; exact argv is in105-02. Immich ML
remains operator-stopped; never restart it automatically. 105-02a findings and
receipt validation are complete. The optional cache LCP journal check is not a
completion gate. 105-03's large curve and primary fact are executed; 105-03a/b
own remaining layout validation and novel post-layout capability/canonical
findings. Keep raw tensors/binaries external and uncommitted.
