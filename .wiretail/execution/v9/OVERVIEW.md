# V9 — fast bounded attention with asynchronous cold KV

Authoritative execution revision: `hotpath-v9-20260913`. Applies to phases
33–38 and their explicitly scheduled successors. Old phases are history, not
requirements to reopen. This document replaces all earlier plan amendments.

## Outcome and scope

Keep model computation on the accelerator, a configurable hot target KV cache
in VRAM, canonical compressed target history in CPU RAM, and native-MTP KV
**entirely Turbo4 on the GPU with capacity equal to resolved target context**.
Use attention-related selection to fetch useful cold pages without scanning
or synchronizing the entire target cache on every token. Measure prefill,
cached append, and committed decode separately against sequential all-GPU and
CPU-main-KV/GPU-MTP controls. Sparse attention is approximate; do not call it
dense-equivalent or promise arbitrary-context near-GPU speed.

Current live campaign: 8K logical / 4K hot, then 32K / 16K, then up to 128K
with measured maximum safe H. These are test coordinates, not production
constants. The wider 256K objective remains a capacity/design extension; this
revision does not reinstate a mandatory six-point long-prefill campaign. The
final review records 256K feasibility and remaining work. YaRN, broad accuracy
suites, exact CPU/GPU attention, independent per-layer physical pools, custom
paged MMA, VBR precision ladders, and upstream publication are later options.

## Decisions already made — do not redesign these in implementation tasks

1. **Prefer mature Turbo4 Flash Attention.** Contiguous legal selected view
   first; otherwise persistent, bounded **Turbo4** packed working tensors.
   This applies to prefill, decode, and MTP verification. Existing custom
   paged direct/scalar/MMA routes are diagnostics, not automatic production
   fallbacks. Do not repair their speculative performance paths in this phase.
2. **Separate L, H and A.** L = logical capacity, C = occupied context,
   H = device-resident target capacity, A = selected attention workspace rows
   per layer. A may be much smaller than H. Large H must not silently cause
   every layer/token to attend, repack, or materialize H or L rows.
3. **Stable compressed workspaces.** Keep per-layer A-sized K/V owners across
   graphs. Keep page slots stable and copy only newly selected pages and dirty
   committed/current rows. Current-microbatch KV must reach attention through
   a real graph dependency, not a host copy made before KV was written.
4. **Cold discovery uses current target Q and an all-history catalogue.**
   Maintain compact min/max summaries of sealed Turbo4 K pages in GPU memory;
   score them cooperatively on GPU, independently of the attention kernel.
   No CPU readback of full Q/K/attention matrices. No serial one-thread GPU
   scan. No attention-mass telemetry requirement on the mature FA hot path.
5. **Selection is intentionally asynchronous.** Use completed previous-query
   snapshots for the next boundary. Per-layer selected sets are bounded by A;
   cold promotion candidates share the existing whole-layer page bundles.
   Start with an eight-accepted-token refresh cadence, request/page-boundary
   refreshes, and at most two cold bundles per refresh. These are tunable
   generic defaults, not correctness assertions about attention quality.
6. **Reuse the existing pager and two-slot mailbox.** Wire a real producer;
   do not add a second scheduler/residency framework. Promote on a copy stream,
   publish after completion, pin in-use pages, and keep old valid selections
   while copies run. No synchronous miss-driven CPU attention in selective mode.
7. **CPU RAM is canonical.** A newly written GPU tail may be dirty until its
   asynchronous host copy completes; it cannot be evicted before host-ready.
   Rejected speculative writes never become committed canonical history.
   Evicting a clean page drops its mapping; it does not copy it back again.
8. **Instrument only to answer a question.** Persistent counters and sampled
   CUDA events, no full probability output or forced split attention kernel
   just to count pages. Transfer/dispatch counters do not prove useful recall.
9. **Generic implementation.** Derive geometry, bytes, budgets, scratch and
   supported shapes from model/backend metadata. No host paths, GPU name,
   fixed layer IDs, 77K constant, credentials or benchmark coordinates in
   upstream-facing code. This machine's commands belong under `/srv/ai` or
   `.wiretail`, never in portable runtime source.

## Why the approach changed (sufficient historical context)

At code SHA `4af26bd68f0875db852fe6973f30b98be7f0f158`, phase 29 measured
q0 at C6144/L8192/H4096: **220.71 prefill / 31.78 decode tok/s**, versus
**1364.15 / 102.05** for its matched executable/DSO all-GPU control. Phase 31
isolated q0/q1 reached roughly **205 / 36.4**, but repeated requests and
recall still crashed. A real 4,325,376-byte H2D bundle copy occurred; a stable
current-Q → cold page → promotion → target attention → answer chain did not.
There is no matched CPU-main-KV/GPU-MTP speedup claim to inherit.

Specific causes found: production query capture is a no-op; mailbox production
publication is absent; automatic dispatch favors a slower custom direct path;
split scratch prevents its advertised MMA dispatch anyway; its latent MMA
policy has unproved multi-KV-head addressing and Q rotation; caller-side
physical-row failure can fall back to a logical row; packed-cache keys include
changing page contents; current copy/graph lifetimes require stronger tests.
These are implementation faults, not evidence that PCIe inherently costs 6x.

Fork master and upstream master are pinned at
`da458765dde0e0414fc530e2e577d9a8e2ac6795`; updating master did **not** merge
the paging integration branch. Phase 33 performs that real merge first.

## Execution map

| Phase | Work | Exit artifact |
| --- | --- | --- |
| 33 | Pinned upstream merge, conflicts, feature-off and MTP compatibility | `V9_INTEGRATION.json` |
| 34 | Row safety, stable packed mature FA, crash repair, memory sizing | `V9_FOUNDATION.json` |
| 35 | All-history GPU summaries, real Q producer, async promotion | `V9_COLD_PROOF.json` |
| 36 | Append/graph/MTP tuning and small matched speed comparison | `V9_SMALL.json` |
| 37 | Conditional 32K and 128K findings, compact benchmark summary | `V9_SUMMARY.json/.md` |
| 38 | Assess only V9 summary; close or schedule actual remediation tasks | `V9_REVIEW.json/.md` |

Tasks use an explicit risk-based recommendation: bounded setup and benchmark
collection use **Luna Medium**, while merge, CUDA/pager ownership, asynchronous
promotion, capacity diagnosis, and benchmark-review tasks use **Luna High**.
All tasks opt into `retry1_reasoning: high`. Wiretail shared defaults and
model-family policy are unchanged.

## Context and completion rules

Read the current packet, its new V9 cluster, the current state entry and only
its `context_files`. Read source by symbol and range. Do not load the old
canonical plan, V5–V8 documents, all historical handoffs, raw JSONL transcripts,
full WORK_STATE/WORK_LOG, or recursively follow old acceptance references.
The necessary old findings are folded into this revision. New cluster IDs
ensure no old Codex session is resumed. Source findings override stale line
numbers; a source symbol rename is not permission to reload old plans.

Implementation tasks finish only when their named implementation tests pass.
A benchmark task may finish with a measured failure or a justified bounded
`not_run` finding, **not** with a fabricated performance/capability pass. A
summary must distinguish this. Review 38-01 must add real, ordered task/state/
cluster entries before marking itself done if the capability is not met.
Ending the list with only a prose recommendation is forbidden.

Cluster assignments are context-area decisions, not a fixed task-count rule:
keep related tasks together for cached context until a real boundary or
context-budget concern warrants a fresh cluster. The runner's default context
guardrails are disabled, so a cohesive cluster may continue across many tasks;
explicit limits or provider boundaries still cause rotation. Details are split into
`INTEGRATION.md`, `FAST_PATH.md`, `ROUTING.md`,
`BENCHMARKS.md`, `OPERATIONS.md`, and `REVIEW.md`. Packets choose which to read;
do not automatically read them all.
