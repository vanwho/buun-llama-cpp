# Active corrective implementation contract: 85-09 onward

Revision: `hotpath-v10-20260914`. Amendment: `repair85-20260921`.
This supersedes conflicting execution directions in the old V10 overview,
testing text and phase-83–85 summaries for unfinished tasks. It does not
rewrite historical evidence. Load this file, REPAIR85_TESTING, RECEIPTS and
the packet/cluster; load the audit only where explicitly listed.

## Goal and architecture decisions (already made)

- GPU target inference with an H-sized Turbo4 physical pool, canonical full-L
  Turbo4 K/V in RAM, dynamic attention-aware promotions from RAM. CPU handles
  bounded control/storage only. Native MTP K/V is Turbo4 and fully on GPU,
  with capacity derived from resolved **per-sequence L**, not H or A.
- Reuse existing pager, capture ring, selector, mailbox, backend and graph
  owners. Fix the production chain; no parallel framework or benchmark-only
  promotion path. GPU-written pages become host-valid after D2H completion;
  clean host-backed eviction drops a mapping without another D2H copy.
- Fast consumer: existing contiguous mature Turbo FA when the selected
  physical layout genuinely qualifies; otherwise **corrected page-table MMA
  for multi-query prefill and short-query fused GPU attention**. Frozen-table
  reference is a test oracle. Do not force native MTP onto gather/reference.
  Persistent compressed packing is a bounded compatibility path for genuinely
  unsupported shapes, never a silent replacement for this target's fast path.
- No full-L F16 K/V staging, no per-token copy of all H/A, and no CPU attention
  in selective production. Tile-local decode and bounded A/workspace fallback
  may be measured explicitly; a flag/counter must identify fallback, and it
  cannot be labeled direct production success.
- Target K uses Turbo4's transformed domain: transform Q once, load decoded
  transformed K/V tiles, untransform the result's V domain once. Routing
  uses the same K-domain transform and scales. Physical rows, KV-head strides,
  global query-head groups and native causal positions have separate meanings.
- Hold a stable selected mapping/generation for one speculative transaction.
  Verification of each candidate uses its own causal position, not future
  rows. Publish promotions at a safe boundary; no per-proposal CPU sync or
  duplicate target pass for measurements. Recurrent and draft state rollback
  follows the accepted target frontier, once, from real success results.
- GPU summary catalogue is keyed by logical page/layer/head/content version
  and survives eviction. Update only changed sealed pages using the existing
  GPU operator. Refresh selector on first use, committed page crossing and a
  configurable accepted-token cadence (initially 8); limit promotion rate and
  exclude mandatory/current/read-pinned slots. Sparse attention's quality
  limitation is explicit; do not confuse a selector miss with transfer failure.
- Size H after model weights, full-L draft, recurrent state, B/U compute,
  A workspace, catalogue, graphs, transfer ring and a safety reserve. All
  sizes are model/backend-derived and tunable; test values are not defaults.
  More batching may improve prefill substantially: measure a small safe U
  increase, not a universal tiny-U production cap. Chunked input does not
  limit conversation L; it trades scratch/throughput and API scheduling costs.

## Ordered work

| Tasks | Scope | Implementation exit |
| --- | --- | --- |
| 85-09 | Fix direct stride, GQA addressing and Q transform | Numerical CUDA fixture actually exercises fast MMA |
| 85-10 | Repair trustworthy dense/native control and short test setup | Finite coherent all-GPU baseline and recoverable launch |
| 85-11 | Capture/graph/slot ownership and canonical publication | Async CUDA chain completes without capture errors/races |
| 85-12 | Integrate fast native MTP with correct causal/rollback semantics | Greedy target-equivalent output and meaningful draft acceptance |
| 85-13 | Persistent GPU summaries; true maintenance cadence | No host KV decode/catalogue rebuild in steady critical path |
| 85-14 | Efficient page lookup, short-query dispatch and prefill batching | Small matched timing with attributed hot-path cost |
| 85-15 | End-to-end promotion with MTP | Completed C>H CUDA workload; cold bytes consumed |
| 85-16 | Original three prompts and small hot/cold speed tests | Valid matched speed/acceptance results |
| 85-17 | Conditional context scaling | Measured safe H/scratch and explicit 256K feasibility/occupancy |
| 85-18–19 | Compact summary and review | Results-driven decision; actionable successors if still needed |

Clusters follow source/context overlap, not a task-count cap. New `repair85-*`
cluster names start fresh sessions so the old workaround is not inherited as
an instruction. Keep related tasks in the same session when useful.

## Completion and autonomy

The source audit identifies bugs; it is not a claim that every future failure
has a known cause. Packets specify exact localization branches where evidence
is incomplete. On a failed assertion, fix that first failing invariant and
rerun that small test. Do not replace it with an unrelated passing CTest or a
receipt-format validation. A config/auth/port/batch mismatch is a repairable
setup error: resolve it within the task and rerun, not a successful finding.
Never silently shrink L/H in a row labeled with the old values.

Implementation tasks cannot complete with required CUDA/live proofs deferred.
Benchmark tasks may complete with valid measured bad performance, not with
missing/setup-invalid rows disguised as speeds. A source-level numerical
regression remains implementation work, not a benchmark finding to skip.
An actual external barrier is documented with the failing command and bounded
distinct recovery attempts; do not claim guarantees about sudo/hardware.

Use Luna High for CUDA, MTP and ownership repairs; Medium for tightly specified
measurement/summary work. First substantive retry uses High as already
configured in project state. Do not change global Wiretail defaults.
Read CONTRIBUTING; keep server paths/secrets and this metadata out of portable
code/PRs. Wiretail owns implementation and metadata commits separately.

Handoff limit is a guideline, not truncation of needed facts: prefer one short
file with changed symbols, exact tests/results, immutable bundle/root pointers
and the one next action. Do not paste logs or append repeated recovery prose.
