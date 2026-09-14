# V10 test ladder: identify work, prove promotion, then measure speed

Revision: `hotpath-v10-20260914`.

## Universal run contract

- Candidate identity is established **when building**, not by stamping the
  current source HEAD on an arbitrary existing binary. Bundle every required
  project DSO with a source/build fingerprint and file hashes. Verify actual
  loaded executable/DSOs by endpoint-owning PID. Keep requested, observed and
  immutable build identities separate. Old/unknown provenance is diagnostic.
- Resolve model through existing `/srv/ai/config/llama` profile and
  `/srv/ai/models/text/current.gguf`; record resolved file/hash/GGUF metadata.
  Use exact profile runner/credential-file boundaries already installed; never
  print secrets. `sudo -n true` then `sudo -n systemctl` for authorized named
  Qwen lifecycle. Another loaded GPU model is not "CUDA unavailable".
- Use `tools/server/bench/run-pager-profile-benchmark.py` and existing
  `/srv/ai/benchmarks/run-profile-benchmark.sh` after their identity/workload
  fixes. Reuse `run-final-curve.py` and `run-incremental-recall.py` where
  applicable; repair shared helpers instead of forking another benchmark
  stack. Store raw requests/responses, timings, metrics deltas and manifest
  under `/srv/ai/paged-kv/results/v10/<task>/<UTC-run-id>`.
- Server flags/observations must prove main `-ctk turbo4 -ctv turbo4`, native
  MTP `-ctkd turbo4 -ctvd turbo4 --spec-draft-kv-device gpu`, full-L draft
  capacity and same model/weights. CPU-main-KV control legitimately uses
  `--no-kv-offload` plus explicit GPU draft. Do not misclassify it as all-CPU.
- Never infer C from `--ctx-size` or `--context` launcher arguments. Record
  tokenizer counts, observed evaluated tokens, reused cache tokens, output
  tokens and actual occupied frontier. A 30-token prompt at L8192 is not C6144.
  Derive H/A/page bytes from observed configuration, not a requested label.

## Three distinct promotion proofs

### T0 — deterministic production-chain CUDA fixture (49-06)

Use real Turbo4 bytes with head dimension divisible by128 and several KV
heads/GQA query groups. Fill eight logical pages with only three/four device
slots through the actual pager reserve/write/seal/evict API. Preserve canonical
host bytes and summaries. A cold page must be uniquely highest ranked for a
fixed nontrivial query under a CPU reference derived from **quantized** stored
bytes. Generate a fixed seed, inspect the reference score margin, and fail if
it is not decisive; do not assume arbitrary original vectors survive packing.

Run the production inventory converter and `GGML_OP_KV_PAGE_SELECT` on CUDA,
then the same submission owner, mailbox decoder, residency transaction and
packed mature-FA route used by the target. Do not supply candidate IDs or
manually publish a ready mailbox as the positive proof. Explicit mock/fake
tests remain useful but are separately named.

Assertions: cold page has valid length and summary; rank result identifies
that logical page and correct layer; cold budget is independent of resident
budget; upload was queued and completed; old table remains usable until the
event; published physical payload matches canonical K/V checksums; a completed
target attention graph uses that mapping. Compare output to a same-selected-
pages all-resident reference. An isolated fixture V-perturbation on the winning
cold page must change output; restore the original bytes afterward. This
proves consumption without adding attention-mass computation to production.

Include shuffled logical IDs/slots, GQA, partial-tail exclusion, generation
rollback, graph replay, no-refresh, first overflow, two layer candidate sets,
and inventory growth after submission. Test both ordinary and native-MTP-like
multi-query shapes. Respect causal query position; a future Q must not select
pages for an earlier query in its microbatch.

### T1 — actual-model controlled-residency challenge (51-01)

Capture a bounded real target-Q/catalogue fixture from the installed model via
a **test-only** hook. At a safe fixture boundary make a high-ranked sealed
nonmandatory page nonresident, while keeping its canonical bytes/summaries.
Replay that Q through the normal selector and production transfer/consumption
chain. Residency setup is controlled; ranking and promotion are not injected.
Use a bounded captured attention-layer fixture and sequential reference/device
evaluation; do not load a second27B model or two full-L caches for comparison.
No public forced-promotion API, no candidate ID from the client, and no
production behavior conditional on a benchmark topic. Record it explicitly as
`controlled_model_query`, never organic discovery or answer-quality evidence.

### T2 — organic cached conversation, C > H (51-01)

L8192/H4096, start A2048 or4096 as already proven legal. Put randomized
nontrivial facts in early history outside pinned sink/system/recent pages;
append unrelated topics until actual C6144 and verify those pages are cold.
Use the exact same cached prefix, bounded appends and native MTP throughout.
Ask about an earlier topic without repeating its answer. Output32–64 tokens
and allow at most two configured refresh periods plus completed upload/target
graph boundaries before concluding no promotion. Fixed seeds and raw request
token arrays make this reproducible.

Sample only the relevant query: eligible cold count, top-k IDs/scores,
submission/rollback generations, requested/queued/completed/published/used
IDs, byte counts and reasons for each rejection. Target use means the completed
target graph references the mapping, not merely that a table epoch changed.
An attention-mass sample or a correct answer alone is not that proof.

If eligible cold >0, cold budget>0 and Q is finite, an empty cold result is a
pipeline bug to reproduce with T0, not a reason to run 24 new recall cases.
If T0/T1 pass but T2 chooses an irrelevant page, preserve the small fixture and
treat it as policy/quality evidence; report nonzero physical promotions
separately from useful recall. Do not block all performance work on one exact
answer once the physical chain works. Conversely never call zero promotions
successful paging performance.

### T3 — two-document cold-to-hot round trip (51-01)

This is the preferred human-readable organic promotion scenario. Generate two
local deterministic documents outside portable source, each containing several
pages of filler plus unique facts/nonces that do not occur in the other
document. Tokenize them first. Choose their sizes so
`pages(document_A) + pages(document_B) + required system/recent overhead > H`
while each document is large enough to span multiple pages. Set the bounded
test's recent/pinned allowance explicitly and record it; do not assume that a
requested H means document A is cold. A cold boundary is valid only after the
actual inventory shows A host-backed and B/current pages resident.

Run one cached conversation in this order:

1. Ingest document A, ask its unique fact, and retain the exact cached prefix.
2. Ingest document B plus only enough deterministic filler to cross the
   measured hot capacity. Ask B's unique fact to verify the current path.
3. Ask A's unique fact again without including A's contents or answer in the
   request. Allow the normal accepted-token refresh cadence and asynchronous
   copy/target-consume boundary.

Use the actual model tokenizer and original request transport; do not send a
client-side `page_id`, call a force-promote API, or rebuild the context from
scratch between steps. Capture C/cache_n, document token/page ranges, residency
before each query, cold eligibility/ranks, stable IDs, queued/completed/published/
used events, H2D bytes and output. The third query is a physical promotion
proof only when A's page ID appears in the completed target graph after being
cold; a correct A answer by itself can come from MTP/recurrent state and is
not sufficient. If A never became cold, mark setup invalid and fix sizing or
recent/pin accounting; do not call it a promotion. If A is cold/eligible but
not selected, preserve it as policy evidence and diagnose T0/T1 wiring first.

Use distinct fact values and a prompt that requires the value, not a broad
"summarize the file" request. Record answer correctness separately. One
reverse-order A/B run is optional only if the first round trip is mechanically
valid; do not turn this into a large document corpus or 24-case matrix.

## Speed methodology, deliberately small for iteration

1. Repair iteration: one original q0, one cold fill and one64/128-token cached
   append/decode segment. One warmup and one measurement, no statistical claim.
   Change one mechanism at a time; sample GPU/CPU timing only in diagnostic run.
2. Stable comparison: original q0/q1/q2 from `/srv/ai/benchmarks`, identically
   tokenized/padded using existing original long-context helper to C6144,
   paired across selective, CPU-main-KV/GPU-MTP, all-GPU/GPU-MTP. One warmup
   plus three measured trials per row. No 48-case corpus. Record independent
   cold prefill, cached append (64 and256 new tokens) and committed decode.
   Prefix reset/reuse is explicit; verify `cache_n` rather than assume it.
3. The pair key includes bundle/model, prompt-token SHA, template/sampling,
   L/C/B/U, MTP settings and output length; H/A/mode differ intentionally and
   are reported. Never reuse q0 controls for q1/q2. Missing partner => null
   ratio. Report medians/range/sample counts and every raw trial, including
   failures; use the server's rate fields consistently and separate client
   TTFT/end-to-end wall rates. Do not double-count cached prefix as new prefill.
4. Compare feature-off MTP acceptance on identical prefix/token/sampling state.
   If all drafts reject systematically only in controls, diagnose draft state,
   placement and decode rollback first. Record MTPoff as a diagnostic paired
   isolation, not as the requested final configuration.
5. B/U/A sweep is bounded and sequential: baseline U128, U64, and U256 only
   if peak scratch+headroom permits. Start B=U; raise B independently only if
   profiling shows host scheduling benefit. Never silently change B/U between
   mode comparisons. Smaller U saves scratch but can reduce GEMM throughput;
   report both, choose by measured append/decode and useful H, not instinct.
6. Scale only after small physical-paging smoke passes. Pilot32K/H16K, then
  128K/safeH with cached incremental input. Dry-run allocation ledger and
   short forward-progress probe first; use wall budget/ETA, not just per-request
   timeout. If recent throughput predicts hours without answering a new
   question, stop at the last successful frontier, emit measured finding and
   return to the specific hot-path cost. Continued HTTP progress is not an
   obligation to burn hours. Never replace a capacity failure by a16K result
   under a128K label.
7. Test256K allocation/startup first and bounded append separately; a loaded
   L262144 is not a full occupied C262144 run. Only after a practical pilot
   schedule the final20K/40K/60K/100K/175K/256K curve as findings, not gates.
   Report full-L MTP and actual H at each coordinate; max128K active pilot and
   broader256K objective must remain distinct. No YaRN work in this revision.

## Minimal required findings, not costly telemetry routes

Record effective L/C/H/A/B/U, target/draft K/V types/devices/capacities,
prefill/append/decode, MTP attempted/accepted, total and category peak VRAM,
host committed/pinned bytes, summary update bytes/time, catalogue uploads,
Q-router/score/rank time, snapshot readback, waits, selected-row copies,
new-row copies, graph reuse, H2D/D2H and actually used promoted page IDs.
GPU timing can be a sampled separate run. Baseline timing has optional traces
disabled. Reuse allocation ledgers/counters; do not create another FA pass.

Receipts describe tests actually run. A successful checked-in source edit,
generic CPU test or healthy service does not replace a rebuilt candidate and
the named live proof. Capture failures once with sufficient diagnosis; future
tasks read a compact handoff, not the full log history.
