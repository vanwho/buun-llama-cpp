# Active forward implementation: turn-boundary retrieval and query replay

Revision: `hotpath-v10-20260914`. Amendment: `forward-turn-retrieval-20260927`.

This is the authoritative contract for phases 94–102. The current amendment is
`forward-generation-admission-20260929`; read `REPAIR100.md` for the latest
failure boundary and executable repair order. It implements the fully
read Sol handoff retained in `SOURCE_FORWARD_PLAN.md`; that long reference is
not automatically loaded into task sessions. By the user's 2026-09-27 direction,
the remaining legacy 93-11n campaign is deferred, not passed. 93-11o's completed
selector repair and the existing uncommitted source work are preserved. Start
from the first unfinished task named by `WORK_STATE.json`; after this repair
that is 100-02a. 100-02 is deferred with its query-page implementation and
partial measurements preserved; it is not a passed campaign. Prior94–99
implementation checkpoints and completed100-01 are preserved;95-03 is intentionally
deferred. 94-01a repaired authenticated selection-to-admission under the new
turn-boundary contract. Do not repeat the legacy A/B/A or require its failed
receipt before implementing the current architecture. The unstarted 93-12/93-13
packets remain removed. Final replay parity, cancellation and natural-promotion
integration proof is owned by 100-04, before any phase-101 capacity scaling.
Supersede accepted-token historical reselection and fixed route preferences.

Read `POLICY_ADMISSION_BASELINE.md` for the compact attempt-13 finding. The
unchanged Sol source handoff is research provenance; its instruction to finish
93-11n first is overridden by this explicit scheduling amendment.

## Architecture decisions

Reuse `llama_kv_pager`, residency/transactions, host store, routing catalogue,
GPU selector, Turbo4 attention and native MTP. No second KVMem pager/library.
The target is Qwen3.8-27B UD-IQ4_XS, with 16 full-attention layers and 48
Gated DeltaNet layers. Derive that geometry from the model; never put model,
card, service paths or these counts in portable production logic.

- L = logical history capacity; C = occupied tokens; H = physical GPU target
  KV capacity; R = frozen historical allocation; G = minimum generation tail;
  A = separate attention view/workspace if needed; B/U = batch/microbatch.
- R + G <= H. G is a guaranteed minimum with spare slots borrowable; it is
  not a maximum generation length or a permanently empty partition.
- Target K/V and native-MTP draft K/V stay Turbo4. Draft capacity equals L,
  and stays entirely GPU-resident. Draft paging and VBR are optional later work.
- Complete sealed target KV remains canonical and inclusive in CPU RAM.
  Promotion retains the host copy; clean eviction drops residency without D2H.
- The final user span is located in the actual rendered/tokenized request.
  Checkpoint immediately before it; process it provisionally; capture target Q
  over that user span; choose history; H2D misses only; publish; restore and
  replay only the query if the historical set changed; then freeze that set.
- Query replay must restore GDN/convolution/frontier/MTP carry and provisional
  writes while retaining the newly published historical map. Never restore an
  old page table over the new selection or serialize full-L KV per user turn.
- No historical reselection, H2D or eviction during assistant generation.
  The historical set is stable across speculative verification/rollback.
  Generation mapping epochs may advance as a separate rolling tail changes.
- Generation may recycle only its oldest completed, clean, host-backed,
  unpinned pages. Current, in-flight, speculative-pinned and frozen historical
  pages are excluded. Spilled output becomes retrievable history next turn.
- GDN remains exact and GPU-resident and processes every committed token.
- Pageable host storage plus a bounded pinned transfer ring is the initial
  default. Seal D2H once per completed page and overlap next-chunk compute;
  selection diff and batched H2D happen before generation.
- Compare direct paged Turbo4 and compressed GPU-packed mature FA using the
  same selected IDs, encoded bytes and native positions. Choose by shape and
  total preparation+kernel+append cost; no CPU or F16-history fallback.
- Summary scoring remains in Turbo4's actual transformed domain. User-span
  mean Q is the first candidate; final-row is a diagnostic baseline; compare
  current min/max scoring with Mean-K before choosing a new default.

## Execution order

| Phase | Tasks | Result |
| --- | --- | --- |
| 93 | 93-11o done; 93-11n deferred | Selector repair retained; legacy promotion unproven |
| 94 | 94-01, 94-01a, 94-02 | Retrieval epoch; authoritative admission; frozen history/MTP |
| 95 | 95-01–02; 95-03 deferred | Exact final-user span and bounded checkpoint/replay implementation; final parity/cancel proof moves to 100-04 |
| 96 | 96-01–02 | Whole-user-span target Q; measured Mean-K/current selection |
| 97 | 97-01–02 | Protected history, generation ring, sealing/rollback |
| 98 | 98-01–02 | Inclusive host authority and batched async promotion |
| 99 | 99-01–02 | Matched GPU kernel comparison; shape-aware production routes |
| 100 | 100-01 done; 100-02 deferred; 100-02a–f; 100-03/04 | Budget/ownership and batch-ring repair; hot-path cost reduction; MTP/retrieval proof; staged paired speeds; cost-based remediation; final replay/cancel proof |
| 101 | 101-01–03 | 32K/16K, 128K, then full 256K occupancy and memory proof |
| 102 | 102-01 | Goal assessment, useful final curve, concrete next remediation |

Optional follower-MTP and first-attention-Q one-pass experiments are fully
specified in `OPTIONAL_ADVANCEMENTS.md`. Schedule them as phases 103/104 (or
next unused phases) only after the primary architecture is proven and the
measured savings justify them. They do not delay primary success.

## Evidence and release rules

Read only the current packet/cluster and explicit context. Use compact
predecessor handoffs; full histories, source reference and raw JSONL stay out
of loaded context. Clusters follow source/lifecycle boundaries, not task count.
All tasks stay on `gpt-6-luna`. Use the per-task reasoning pin in WORK_STATE:
Medium only for bounded procedural harness/benchmark/occupancy tasks; High for
kernel/state-machine changes, algorithm or route decisions, integration proof,
diagnosis and goal assessment. Retry assessments remain Luna High. No task in
this sequence may use Terra, Sol or Astra.

Implementation completion requires the specified executable correctness proof.
Fix config/auth/port/identity errors and retry the smallest affected request.
Benchmarks can complete with valid measured poor performance; those findings
create source-directed remediation tasks before scaling. A missing measurement
is not a speed finding. Keep physical promotion, answer quality and MTP
acceptance as separate fields. No exact filename/YES/NO output-format gates.

The new repair tasks supersede the failed100-02 campaign, not its source work.
Query-page capacity repair is already committed separately. Do not diagnose
an absent prefill spill subsystem from the old `no_victim` label: the latest
run prefills beyond H and fails during generation. Trace query-commit ownership
and production batch preflight first. A fixed number of valid campaign rows
is not a lifetime retry budget. Update scheduling/receipts before declaring done.

The initial query replay and natural promotion work has partial live evidence;
its complete one-pass parity and cancellation checks are delayed until the
attention implementation and first speed campaign settle. Short-path speed
results before 100-04 are diagnostic; do not start context scaling until that
integrated correctness task passes. This order allows scorer/kernel work and
useful short-path measurements to proceed without repeatedly rerunning an
early full integration fixture.

Primary goal remains 256K full logical/host history, bounded H, natural
promotion, correct replay/freeze/ring, native GPU MTP, selected fresh prefill
>=500 tok/s per canonical prompt (750 preferred), and materially faster decode
than ordinary CPU-KV offload. Speeds below target must remain visibly failed
goal findings and must cause actual repair tasks, not another unchanged audit.
Use 3x CPU-KV decode as the reporting target; final acceptance requires a
positive matched speed advantage and the architectural no-decode-PCIe proof.
Full-resident Turbo4 identity and feature-off controls remain unchanged.

Server-specific artifacts stay under `/srv/ai` or `.wiretail`. Preserve
uncommitted source work; Wiretail owns separate implementation/metadata
commits. Follow CONTRIBUTING; publication and human-authored issue/PR posts
are not part of these tasks.
