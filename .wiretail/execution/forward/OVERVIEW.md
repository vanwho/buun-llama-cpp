# Active forward implementation: turn-boundary retrieval and query replay

Revision: `hotpath-v10-20260914`. Amendment: `gpu-execution-101-20260930`.

This is the authoritative architecture contract through phase103. The current
execution amendment is `gpu-execution-101-20260930`; read `SPEED101.md` for the
code findings, measured failure boundary and specific implementation order.
Completed phase100 and HOTPATH_POST100/REPAIR100 are provenance, not startup
instructions. Start at current_task in WORK_STATE; after this amendment it is
101-01. The blocked100-04a audit is deferred, not passed: its useful code is
preserved and101-10 owns actual integrated replay parity. Phase101 now repairs
GPU execution before scale; former101 capacity tasks move to102, and the final
review to103. Legacy93/95 incomplete proofs remain deferred and must not be
reopened as gates. Preserve the already implemented turn-boundary architecture.

POLICY_ADMISSION_BASELINE and the long Sol handoff are historical research,
not required startup context. Their legacy campaign/release ordering is
overridden by this explicit scheduling amendment.

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
| 100 | 100-01 done; 100-02 deferred; 100-02a–f; 100-03/a/b/c done; 100-03d–i; 100-04 | Merge synced master; request-local attribution; measured packed/host/MTP repair; minimal matched speed decision; small final replay/cancel proof |
| 101 | 101-01–12 | GPU attribution, fused Turbo4, parallel GDN, stable graphs, encoded copy, overlap, MTP device/alignment repair, real replay/promotion proof, canonical speed release |
| 102 | 102-01–03 | 32K/16K, 128K, then full 256K occupancy and memory proof after GPU101_RELEASE passes |
| 103 | 103-01 | Goal assessment, useful final curve, concrete next remediation |

Optional follower-MTP and first-attention-Q one-pass experiments are fully
specified in `OPTIONAL_ADVANCEMENTS.md`. Schedule them as phases 104/105 (or
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

Phase100 speed remains a goal miss (~212 fresh tok/s in the latest4K row).
101-10 replaces the missing integrated parity proof;101-11 observes natural
promotion on that implementation, and101-12 produces GPU101_RELEASE. Do not
scale until this release actually passes. A measured miss inserts concrete
101-12a/b... repairs and a renewed speed decision before102-01, never another
unchanged audit of100-04a. No completed historical task substitutes for this
current implementation's runtime proof.

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
