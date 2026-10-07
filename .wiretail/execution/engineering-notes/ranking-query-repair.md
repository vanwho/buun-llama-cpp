# Query capture, ranking and retrieval repair — 2026-10-07

Revision: `hotpath-v10-20260914`. Supersedes October6/failed retry instructions.
Load this compact note, current task and immediate handoff, not raw transcripts.
Main and experiment remain distinct until selective integration in105-01g.

## Verified defects, not the last agent's pointer hypothesis

1. **Wrong native-position layout.** `llama_batch_allocr::ubatch` in
   src/llama-batch.cpp writes SECTION-MAJOR M-RoPE positions:
   `pos[section*n_tokens + row]`. Qwen text batches have four planes.
   Pager/attention/MTP additions read `pos[row*n_pos]`, confusing another
   row/plane with the temporal position. Repeated/scalar fixtures hid this.
   This corrupted causal masks, final-user probe detection and target hidden
   row/MTP frontier metadata BEFORE ranking. Matching host/device readbacks
   did NOT prove the captured temporary-pointer theory.
   `llama_ubatch::pos0(row)` now supplies plane0 in llama-context.cpp,
   llama-graph.cpp and all selector/capture/accumulator consumers in
   llama-kv-cache.cpp. Other planes still feed RoPE normally.
2. **Unpadded selected-dense Turbo4 FA views.** Exact partial-tail lengths
   violated the ordinary CUDA FA tile/mask-stride contract. Dense eligibility
   exposes `padded_row_count=GGML_PAD(rows,256)`, with K/V and masks padded
   together in llama-kv-attention-op.cpp/llama-graph.cpp. Existing page storage
   is reused; padding remains masked. No F16 historical cache.
3. **Pinned metadata rewrite at publication.** Host-catalog discovery
   overwrote host_valid on retained current pages; the transaction returned
   all_pinned. Boundary construction authenticates host flags only for cold
   pages. `llama_kv_live_policy_prepare_query_target` preserves the published
   resident's lifecycle flags/pins/leases after identity/slot/version checks.
   Real pinned eviction, stale content and unsafe transfers still fail.
4. **Stale physical ownership in cold catalogue.** `reconcile_live_target`
   drops GPU residency while catalogue records can retain the old slot.
   Policy's local repair did NOT protect selector inputs/exact rerank plans.
   `llama_kv_pager::exact_page_records` and `routing_inventory` now normalize
   every catalogue-only entry to host-cold/no physical slot, clearing GPU-only
   pins/leases while retaining immutable content/summary identity.
   A reused slot must never supply another page's encoded keys.

## Indexed capture and diagnostic isolation

Experiment: /srv/repos/vanwho/buun-llama-cpp-ranking-v1.
Product12038ee142c6941bf78798aa5eee8bf66632d9f2 preserves accumulated work.
Current experiment source repair997b818d9; harness1b9a7d5af. Generic main
repaird5a93544a. Final fixture/live binaries were built from the identical
then-dirty source contents before997b818d9: retain their original build/runtime
identities, do not retroactively replace raw hashes. Full MAIN CUDA rebuild
was stopped before completion; it is not claimed verified.105-01g owns the
integrated MAIN build after porting, avoiding a redundant full CUDA build now.

- ggml_kv_query_probe_params / ggml_kv_query_probes_indexed carry generation,
  interval and four actual Q row indices in immutable op params. CPU/CUDA
  gather Q directly: no production position tensor/scans/uploads. Dynamic op
  stays as independent oracle. Indexed row selection uses corrected pos0.
- pager_query_probe_rows, set_kv_query_probe_inputs, can_reuse_kv_query_capture
  validate indices/bounds/sequence/owner. Scheduler fixture covers CUDA+CPU
  fallback, U256/U37, graph replacement/replay and generation reset.
  CPU supports_op's obsolete control tensor dereference was repaired.
- Removed per-layer capture readbacks and abandoned persistent per-shape
  position buffers. Direct page-mass/split workspace allocates only for an
  explicitly enabled bounded diagnostic, not ordinary telemetry.
  Failure logs are bounded and failure-only.
- Indexed capture alone passed numerical replay but natural output remained
  incoherent. Do NOT claim it alone fixed semantics/MTP.
- The old four probes sampled only `query_end-{1,2,4,8}`: all can fall inside
  generic trailing answer instructions and miss the filename/question. The
  production `ggml_kv_query_probes_spread` now uses final/quarter/mid/three-quarter
  user-span positions, keeping four channels and the same storage. Shared
  `ggml-kv-query-probes.h` defines targets for host/CPU/CUDA; indexed graph
  keys include the actual rows/generation/span. Cross-U256/U37 capture keeps
  earlier probes alive until the final probe arrives; duplicates in tiny spans
  are invalid, not double-weighted. Legacy dynamic/indexed suffix oracles stay
  unchanged. There is no text matching, filename injection or per-token scan.

## Small live results and how to use them

Raw: /srv/ai/paged-kv/results/forward/105-01e/verification/ (never load wholesale).
L16384/H4096, B1024/U256, Turbo4 target/full-L GPU MTP, reasoning off,
one managed Qwen; frozen PY_MERGE_03 A/B/A schedule from105-01c.

- Old selected: slash filler at A1; B HTTP500/nonfinite probe. Invalid useful
  speed evidence, not a score-quality observation.
- Dense control: correct preallocation fact in normal prose; A2 48 drafts,
  33 accepted (68.75%), ~73 decode tok/s. Filename-only scorer falsely
  rejected it. Facts now score normally; acknowledgement syntax is not a gate.
- Correct positions/padding: coherent A1/B ~1302/~1396 prefill tok/s;
  A2 exposed pinned publication failure. Short replies are NOT canonical
 400-token acceptance evidence.
- Correct resident lifecycle: all requests HTTP200; A2 ~55.67 decode tok/s,
  43/84 drafts (51.19%) but semantic miss. Zero cold keys staged exposed stale
  catalogue slot ownership, not a reason to tune score formulas.
- Correct cold catalogue: A2 now actually runs48 cold-reader graphs and
  promotes/publishes cold pages, then target uses them. One run had89.63% MTP
  despite an incorrect/repetitive answer. Acceptance does NOT prove retrieval.
- Matched selected all-resident isolation (H16128/C~5K): correct preallocation
  fact;35/50 drafted accepted (70%),~71.29 decode tok/s. This is NOT offload
  evidence. Dense also answers correctly. The low-H failure is not explained
  by general model execution alone.
- Spread-query capture: A1/B~1296.68/~1392.80 fresh prefill tok/s. A2 retrieves
  preallocation/two-pointer information initially, then emits long slash
  filler,37/374 MTP accepted (9.89%),~37.89 decode tok/s. Its83-token cached
  recall prefill129.46 tok/s is NOT a bulk prefill measurement. Actual cold
  page5 publication/H2D~4.3MB/target use occurred, but the requested fact page
  was not proven cold. Do NOT call unrelated-page promotion requested recall.
- Final current-source model replay `query-replay-final.json` PASS: resident16,
  cold48,MASS16 graphs,288 records; cold catalogue normalization, changed/
  unchanged replay, canonical target/draft KV parity and both cancellations.
 2 MTP proposals/0 accepted here are not a benchmark. CPU/CUDA spread scheduler,
  distinct M-RoPE planes, FA padding and pinned retention fixtures PASS.
- Harness accepts facts in prose; empty marker sets do not pass. Long slash
  tails after a coherent prefix are flagged separately, not rescued by matched
  words. No output-budget reduction or exact filename-only gate.

## Remaining generation boundary: do not guess a repair

On the spread low-H A2, ordinary target decode used packed while multi-row MTP
verification used direct. Different route names alone do NOT prove numerical
disagreement. `llama-kv-attention-execution.cpp::planned_route`,
`llama-graph.cpp::build_attn` and `ggml-cuda/fattn.cu` consume the same published
page identity/native causal mask, but synthetic replay uses repeated token1
and is weaker than diverse-language generation.105-01f must extend the existing
CUDA attention oracle with varying nonzero encoded K and V, Q1/Q3, noncontiguous
physical slots, partial tails and a rolled generation page; compare direct vs
packed on identical bytes and native positions. Keep fixture work small, GPU,
outside production loops. If it passes, do NOT force a route or undo transport
repairs. Coverage is limited to its declared shapes/dispatch. See the current
[selected-generation contract](selected-generation-contract.md):105-01f's2:1
materialized control is not Qwen6:1 production-seam parity. Approximate target
history also differs legitimately from full-L draft history.
If it fails, fix the first K/V addressing/mask/WHT/output-layout producer, not
the acceptance counter.105-02 owns MTP generation/carry work after integration.

## Architecture, outcome branches and execution order

Keep independent final-user probes, diverse GPU coarse K-only candidates,
then exact encoded Turbo4 key rerank. Cold staging is bounded (64 candidates
per layer); full-K/V promotion has a separate budget. CPU RAM stores canonical
pages; Q/attention/rerank execute GPU. Promotion/publication/replay once per
user query; generation history then freezes. No per-token rerank, catalogue
copy/readback, CPU attention or F16 target/draft substitution.

HTTP500: fix producing invariant from bounded failure reason, not relaxed
authentication or empty-success. Dense misses: repair ambiguity. Coherent
selected misses: verify actual cold ownership, shortlist membership/encoded-key
identity, exact order, publication, replay and consumer in that order.
Do not change score formulas while reading another page's keys/mask.

Order:105-01e supplied capture proof ->105-01f small numerical/recall outcome
->105-01g selective main integration ->105-02 generation repair/proof
->105-02a small paired recall/canonical MTP ->105-03 curve
->105-04/05 review. Main already contains generic position/padding/residency
fixes. Port ranker by symbol preserving newer main prefill batching/fences,
graph reuse, sealing and diagnostic isolation. No whole experiment overwrite.
105-01d was NO ADOPTION. Completed semantic misses may close an outcome task
with explicit goal_miss and named repair ownership; HTTP500/nonfinite or absent
measurements may not. No receipt turns the current poor low-H A2 into success.

Build incrementally --parallel16. One Qwen under existing lifecycle lock;
verify executable/DSOs/model/argv, leave successful matching profile loaded,
never touch8092. No unchanged256K failure or nine-sequence retry loop.
