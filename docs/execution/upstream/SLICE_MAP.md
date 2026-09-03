# Upstream slice map

## Scope and provenance

This map is an extraction plan for the implementation currently in this Buun
worktree. It does not create upstream prose, issues, pull requests, review
replies, commits, or merges. The source implementation is tested in the Buun
fork; generic candidates are marked as candidates until maintainers choose an
API and destination.

Recorded synchronization points after read-only fetches on 2026-09-03:

| Repository | Base used by this work | Current fork tip | Current upstream default | State |
| --- | --- | --- | --- | --- |
| Buun | `cb703be37e3628dadb71912f3b3b25b82090555b` | `8ac64b1714b1f4c29898a3450619e8d53df295cb` | `cb703be37e3628dadb71912f3b3b25b82090555b` | synchronized before this implementation |
| llama.cpp fork | `67a17c17caa95742186f8b1ecadd1b5abd6d5ebb` | `67a17c17caa95742186f8b1ecadd1b5abd6d5ebb` | `de8656bd94f1163188125542534e4bcbc9f9fb1f` | fork is 11 commits behind upstream; re-sync required |

The Buun implementation range is therefore `cb703be37e...` to
`8ac64b1714...`. The `llama.cpp-kv-pager` worktree is based at
`67a17c17c...` and contains an uncommitted generic draft-placement port. It
has no reviewable tip SHA yet; the map does not present that dirty worktree as
an upstream commit.

Implementation SHAs below are the task `feat` commits. Their immediate parent
is the extraction base for that task. The following task-completion and merge
commits contain execution metadata and are not part of an upstream slice.

## Existing direction and destination rules

- [Discussion #21961](https://github.com/ggml-org/llama.cpp/discussions/21961)
  is the existing generic paged-KV design discussion.
- [Draft PR #22569](https://github.com/ggml-org/llama.cpp/pull/22569) is still
  a draft one-commit paged-KV implementation. Its review already identifies
  the need for smaller slices and separates the CUDA/backend concern from the
  design questions.
- [Issue #28115](https://github.com/ggml-org/llama.cpp/issues/28115) is open
  and tracks MTP draft KV remaining in RAM when target KV is offloaded. It is
  an input for the independent draft/MTP placement slice, not a destination
  for Buun VBR or Turbo4 code.
- [Buun issue #109](https://github.com/spiritbuun/buun-llama-cpp/issues/109)
  is open and covers multi-slot dynamic-VBR context accounting. Selective
  pager authority remains single-slot, so this issue is a dependency/risk for
  Buun server integration rather than an upstream generic issue.

Destination policy:

- `llama.cpp` means a candidate generic slice for maintainer discussion. It
  must be rebased from current `upstream/master` before any fork branch is
  prepared.
- `Buun fork` means `spiritbuun/buun-llama-cpp` or its authorized `vanwho/*`
  fork. These slices own VBR, Turbo4, Qwen3.8, and server-specific behavior.
- `execution only` means the file is evidence or operator metadata and must
  not be included in an upstream code range.

## Dependency-ordered slices

### U01 - independent draft-KV placement

**Destination and ownership.** The normalized `auto|gpu|cpu` draft placement
contract is a candidate for `llama.cpp`. Turbo4 aliases, native MTP logging,
and the Buun server adapter stay in the Buun fork. This separates the generic
placement decision from VBR and pager ownership.

**Source range.** Generic prepared worktree: base
`67a17c17caa95742186f8b1ecadd1b5abd6d5ebb` to an uncommitted working-tree
state, with no tip SHA. Buun port: `c003031954db6e88796727c614fe9711c187898a`
to `8b13597b24f2792e12da4c1967752b1dae901d53`.

**Dependencies.** None for the generic contract. Buun native-MTP use depends
on the later MTP reservation slice B08.

**Exact files and hunks.**

- Generic common seam in `common/common.h`: the
  `common_speculative_draft_kv_device` enum, the draft parameter field, and
  conversion declarations.
- `common/common.cpp`: the
  `common_speculative_draft_kv_offload`, availability, and name helpers.
- `common/speculative.cpp`: normalized draft conversion and availability
  checks in `common_base_params_to_speculative()` and speculative context
  initialization.
- `common/arg.cpp`: only the `--spec-draft-kv-device` parser option for the
  generic candidate. The `kv_cache_type_from_str()` Turbo4 alias hunk is
  Buun-only.
- `tests/test-arg-parser.cpp`: placement truth-table, parser, invalid GPU,
  environment, and precedence cases.
- Generated option tables in `tools/cli/README.md`,
  `tools/completion/README.md`, and `tools/server/README.md` follow the
  generic parser. `docs/speculative.md` and the native-MTP hunk in
  `tools/server/server-context.cpp` remain Buun-side follow-up hunks.

**Tests and platform.** The generic and Buun parser/build paths passed the
task-00-05 checks; the deterministic matrix covers target CPU/GPU by draft
`auto|gpu|cpu`. The external-draft CUDA launch remains deferred because no
separate compatible draft artifact is provisioned. Scope is CPU/common and
optional CUDA placement; `auto` must preserve existing behavior and an
unavailable GPU request must fail closed.

**Risks and compatibility.** Do not copy the Buun Turbo4 spelling into
generic llama.cpp unless that type exists at the rebased destination. Keep
target placement immutable and apply the normalized value before fit and
context construction. The current generic worktree must first be committed by
the authorized fork owner after the llama.cpp re-sync.

**Issue/PR disposition and rebase.** Link the generic candidate to #21961 and
#22569 for maintainer direction; do not create or update either item here.
Rebase a new generic branch from current `upstream/master`, then use the
uncommitted worktree diff as the review source. Extract the Buun aliases and
server adapter as a separate fork-only patch on top of U01.

### B02 - pure page identity and budget primitives

**Destination and ownership.** Candidate generic internal primitives for
`llama.cpp`, first maintained and reviewed in the Buun fork. No public API is
required. The policy replay file is intentionally deferred to B07 so this
slice remains identity and accounting only.

**Source ranges.** Page table:
`65180956d49af390ed5bdca824673008b47340c2` to
`f02add48e012527371e4626e2411f4cf1f041f8d`. Budget arithmetic:
`7f699a9225b78d42a7b2b935e88e65e1e731e224` to
`b36ee8e99603d4e87d8d473788aadd4528f467fd`.

**Dependencies.** The semantic contract in execution metadata task 01-01 is
the design prerequisite. U01 is independent. B03 consumes these types.

**Exact files and hunks.**

- `src/llama-kv-residency.h/.cpp`: `llama_kv_page_id`, page states and
  records, immutable snapshots, epoch-checked transactions, page-count and
  identity validation.
- `src/llama-cache-budget.h/.cpp`: admission input/result, overflow-safe
  rounding, target-page capacity, full-length MTP reservation arithmetic,
  scratch/staging/headroom charges, and typed fail-closed refusals.
- Existing target registration in `src/CMakeLists.txt` and focused targets in
  `tests/CMakeLists.txt`.
- `tests/test-kv-residency.cpp` and the budget test target cover tails,
  duplicate/gap maps, pins, rollback, stale publication, and budget edges.

**Tests and platform.** `test-kv-residency` and `test-cache-budget` passed in
CPU-only builds; the complete focused CPU/fake matrix later passed as recorded
in `docs/execution/evidence/CPU_TEST_MATRIX.md`. Platform scope is portable
CPU arithmetic and immutable metadata. No device allocation is claimed.

**Risks and compatibility.** Preserve `VBR_GENERATION_PAGE_CELLS` as the
existing 256-cell identity boundary, keep logical and physical identity
separate, and do not expose Qwen3.8 constants as a public generic contract.
Turbo4 MTP sizing is a Buun admission specialization until a generic type
contract exists.

**Issue/PR disposition and rebase.** A generic page identity/table can be
  discussed in #21961, but the Buun budget categories and VBR generations do
  not belong in #22569 wholesale. Rebase the candidate from current upstream,
  apply only the two implementation commits, then run the existing test
  targets before B03.

### B03 - host backing and transactional residency

**Destination and ownership.** Buun fork. The selected capture seam consumes
  Buun VBR representation descriptors and the host catalog stores Turbo4
  cross-layer pages; the separate residency pool is the possible generic
  boundary. Do not merge the VBR representation pool with pager policy.

**Source ranges.** Apply in this order:

1. `0ffec87e32a49b65185cb03b60f3f89c1198f8cc` to
   `45a1a795a91f771d9674681b6c85a39b26b0d9d3` (bounded VBR page capture).
2. `a687880e2c043a37644a75490fcf289469eccdf6` to
   `e531addc23916ef1f4b66324c0daaeec9ea3ae06` (host catalog and accounting).
3. `53d1f37c1f394d17751cfc0c4483a77346e4608c` to
   `ea12da5d08bfaca54fe3676dbedc2842255de80a` (async transfer seam).
4. `b50a11691d01ff4dad60707bbbff2a428d81e69f` to
   `4a81b10bf8282331b4033966e6b5ab4a524da4a6` (transaction publication and
   rollback).
5. `a4f8697534a61cdbb37a9453d5d1d7c174aa99ae` to
   `ab8e08920ce2d7a810317ac70fd94a2cfd7b44b5` (fixed 304-page proof).

**Dependencies.** B02, then each numbered sub-range in order. B07 and B08
  consume the callback and publication contracts; no attention kernel is
  needed to build this slice.

**Exact files and hunks.**

- `src/llama-vbr-artifact-capture.h/.cpp`: selected-page projection,
  authenticated 32-unit Turbo4 capture, bounded pinned-ring streaming, and
  terminal generation recheck. The catalog and transfer commits add hunks to
  these same files; extract them in task order.
- `src/llama-vbr-artifact-catalog.h/.cpp` and
  `src/llama-cache-accounting.h/.cpp`: immutable pageable host catalog,
  checksums, aliases, invalidation, and transactional accounting cells.
- `src/llama-kv-residency-transfer.h/.cpp` and
  `src/llama-vbr-artifact-stage.h/.cpp`: compatible layer/side runs, H2D/D2H
  reservations, event completion, cancellation, stale cleanup, and bounded
  byte counters.
- `src/llama-kv-residency-transaction.h/.cpp`: snapshot, reserve, pin,
  reseal, clean-drop, restore, recheck, publish, rollback, and retire phases.
- `src/llama-kv-fixed-window.h/.cpp`: derived Qwen geometry, 304-slot manual
  window, clean-eviction zero-D2H rule, and scalar ledger.
- Repeated `src/CMakeLists.txt` and `tests/test-kv-residency.cpp` hunks register
  the modules and fake transfer/transaction/fixed-window coverage.

**Tests and platform.** `test-vbr-artifact-capture` and `test-kv-residency`
  passed with CPU/fake providers. The tests cover capture stale rejection,
  catalog corruption and budget, transfer failure at each phase, rollback,
  pinned limits, 17-token tails, 304 residents, and clean eviction with zero
  D2H. CUDA VMM/event and device row geometry remain deferred. The production
  boundary is backend-neutral callbacks; the proof is not hardware evidence.

**Risks and compatibility.** This is the highest Buun-specific extraction
  risk: host payloads are VBR/Turbo4 typed, and `test-vbr-artifact-adopt` has a
  pre-existing include-path failure. Keep pageable host backing canonical,
  keep pinned staging bounded, and leave `ggml_vbr_vmm_pool` representation-
  owned. Preserve old whole-prefix artifact serialization unchanged.

**Issue/PR disposition and rebase.** Not a direct #22569 patch. A generic
  residency pool may be discussed after a maintainer accepts the separate
  ownership boundary; the VBR capture/catalog and Qwen geometry remain Buun.
  Rebase each sub-range independently, resolve shared capture/CMake hunks in
  dependency order, and run both focused targets after every sub-range.

### U04 - selected view and sparse metadata

**Destination and ownership.** Candidate generic internal metadata shape for
  #21961/#22569, with the currently proven implementation in the Buun fork.
  The view and operator metadata carry no KV payload and do not allocate a
  whole-cache F16 gather. Qwen/Turbo4 capability checks stay explicit at the
  consumer boundary.

**Source ranges.** View:
`c1f41c2c4363f7fa39b996815f7ac3dde3182989` to
`d8224c7338361689c06ffa2466e6a2719c46b6fb`. Operator metadata:
`6a5776f24962be5ae56bfb6a8f7356debea24042` to
`f2908b775d7465de7772e9893b62bd2c9d565244`.

**Dependencies.** B03. The CUDA operator C05 consumes this metadata.

**Exact files and hunks.**

- `src/llama-kv-attention-view.h/.cpp`: immutable compact selected rows,
  source-slot map, native position/mask expansion, tail sizing, graph fence,
  and duplicate/absent/overflow refusal.
- `src/llama-kv-attention-op.h/.cpp`: validated page-table operator metadata,
  native query positions, GQA dimensions, causal flag, Turbo4 type checks,
  backend capability result, and table-epoch reuse key.
- `src/llama-graph.h`: `allow_reuse` epoch key additions associated with the
  metadata contract.
- `src/CMakeLists.txt`, `tests/CMakeLists.txt`, and
  `tests/test-kv-attention-view.cpp` registration and deterministic coverage.

**Tests and platform.** `test-kv-attention-view` passed with residency/view
  CPU fixtures for permutations, gaps, tails, masks, GQA, off mode, and
  unsupported backends. No backend executes the metadata in this slice.

**Risks and compatibility.** The current names and graph key are provisional
  internal APIs. Generic extraction must not make `get_n_kv()` equal the
  logical frontier or silently turn unsupported backends into dense fallback.
  Preserve feature-off behavior.

**Issue/PR disposition and rebase.** This is the closest generic design
  material for #21961/#22569, but no upstream PR is authorized. Rebase from
  current llama.cpp, compare the metadata to any maintainer-selected paged
  block table, and retain the Buun type checks if the generic destination
  cannot represent them.

### C05 - Turbo4 CUDA direct page attention

**Destination and ownership.** Buun fork only. This is a Turbo4, FWHT, native
  position, Qwen-shaped CUDA kernel and is not a generic llama.cpp slice.

**Source range.** `f4011b2c009aa2c20152570561c8950629e063b6` to
`3c5458f8ab88a62ab411d319e3bf2860e2bfad47`.

**Dependencies.** U04 and B03 metadata. C06 graph lifecycle consumes the
  direct capability callback.

**Exact files and hunks.**

- `ggml/src/ggml-cuda/fattn-paged-turbo4.cuh`: direct paged Turbo4 launch
  declaration and metadata/loader boundary.
- `ggml/src/ggml-cuda/fattn.cu`: `ggml_cuda_op_fattn_paged_turbo4` dispatch,
  validated page table, direct Turbo4 K/V row load, causal native-position
  filtering, GQA, and optional page-mass reduction.
- `ggml/src/ggml-cuda/fattn.cuh`: CUDA declaration/dispatch surface.
- `tests/test-cuda-fattn-paged-turbo4.cu` and `tests/CMakeLists.txt`: direct
  kernel fixture and registration.

**Tests and platform.** CUDA build and `test-cuda-fattn-paged-turbo4` passed
  on RTX 4080 sm_89; memcheck and racecheck logs are recorded in
  `docs/execution/evidence/`. The bounded fixture covers tails, permutations,
  native causal masks, GQA, and no F16 full-cache allocation. Broader shapes,
  graph capture, and live model execution are deferred.

**Risks and compatibility.** The launch is intentionally narrow: batch 1,
  causal one-token decode, Dk/Dv 256, GQA 4, Turbo4 K/V. It must fail closed
  for other shapes and must not be advertised as a generic sparse kernel.
  Keep optional mass reduction output separate from attention output.

**Issue/PR disposition and rebase.** Do not append this to #22569. It is a
  Buun experiment until a generic kernel and maintainer direction exist.
  Rebase only after U04 metadata is settled, and run the CUDA fixture plus
  sanitizer evidence on the destination GPU.

### C06 - selected graph and prefill/decode lifecycle

**Destination and ownership.** Buun fork initially. This slice wires the
  selected view to Buun graph/context lifecycle and therefore follows C05; a
  generic port would need a separately accepted memory interface.

**Source range.** `4aa124f1b5a735e8123fe5447d4c455c5366bad4` to
`122f29e63176434db74edcc9a713c9596e67b71f`.

**Dependencies.** U04, B03, and C05.

**Exact files and hunks.**

- `src/llama-kv-attention-execution.h/.cpp`: prefill page admission,
  explicit tail finalization, route selection, direct/reference/refusal
  decisions, graph epochs, and bounded scratch sizing.
- `src/llama-kv-attention-op.h/.cpp`: lifecycle additions to consume the
  selected metadata and direct loader.
- `src/llama-graph.h/.cpp`: table, selected-content, representation, and
  shape epoch reuse fencing.
- `src/llama-context.h/.cpp`: selected-view lease release after scheduler
  synchronization.
- `src/CMakeLists.txt`, `tests/CMakeLists.txt`, and
  `tests/test-kv-attention-execution.cpp`: target registration and lifecycle
  fixture.

**Tests and platform.** CPU execution fixtures passed for multi-chunk prefill,
  short tails, route fallback, graph reuse/rebuild, concurrent leases,
  feature-off, observe, refusal, and scratch bounds. The direct CUDA fixture
  and `llama` target also passed. Full model graph capture is deferred.

**Risks and compatibility.** This is a graph integration boundary, not proof
  of live pager execution. Retain dense/off behavior, table-epoch fencing, and
  immutable view lifetime. Do not let logical context length enter compact
  selected scratch sizing.

**Issue/PR disposition and rebase.** No generic upstream submission is
  proposed yet. Rebase after C05 and compare the memory/graph API to whatever
  #21961 direction maintainers select; keep Buun-specific context hooks in the
  fork.

### B07 - routing, telemetry, policy, and prefetch

**Destination and ownership.** Buun fork. These modules implement attention-
  aware selection around Buun page identity and Turbo4 page mass. Only the
  pure algorithmic portions might later be generalized; no generic upstream
  policy is implied by this slice.

**Source ranges.** Apply in dependency order:

1. Pure replay policy: `2794b1cfb06407539433fa798ee47e6c5929ef88` to
   `4c8975ddf80e9299cc44a36f2182542fa992eaab`.
2. Routing summaries: `4662995c8d3c06cd013d81b998182a177ee20982` to
   `3ac062c330741b502843f4bde034b50d004dd2bf`.
3. Attention telemetry: `b0c3322a219e9c013c86981bf71b6252dfdd807e` to
   `f340a5ff43f31d4c6ba088fe0927c60ce968ad03`.
4. Hot-set controller: `a92a00ac18c5574c4653cc252d01238f5798aed6` to
   `cda8d1c10e3b33b78b94a06f494ae1b86d7278c7`.
5. Prefetch/backpressure: `7e449d5ac22ed2933e92e8b49f86ba499dbe2bf0` to
   `0d398d6b5ab32dd0e6b8e3d582daa3e0e65d038422`.

**Dependencies.** C06 and B03. The controller depends on routing and
  telemetry; prefetch depends on the controller and transfer callbacks.

**Exact files and hunks.**

- `src/llama-kv-policy.h/.cpp`: replay retrieval/retention, normalized
  evidence, mandatory/pinned admission, hot-set partition, reasons, and
  hysteresis. The 01-04 and 04-03 hunks must be extracted as one coherent
  history.
- `src/llama-kv-routing-summary.h/.cpp`: bounded all-page representatives,
  identity/version checks, score ordering, and byte accounting.
- `src/llama-kv-attention-telemetry.h/.cpp`: completed page-mass publication,
  EMA/peak normalization, stale/invalid drops, and fixed-size controller copy.
- `src/llama-kv-prefetch.h/.cpp`: bounded queue, double-buffered staging,
  generation cancellation, readiness fallback, clean eviction, dirty reseal,
  and shutdown drain.
- Repeated `src/CMakeLists.txt`, `tests/CMakeLists.txt`,
  `tests/test-kv-policy.cpp`, `tests/test-kv-routing-summary.cpp`, and
  `tests/test-kv-attention-telemetry.cpp` registration/test hunks.

**Tests and platform.** CPU/fake tests passed for all-page scoring, cold-needle
  retrieval, stale identity, EMA/peak behavior, pin overflow, reordered
  completion, bounded queues, cancellation, readiness fallback, clean
  eviction, and dirty reseal. Live Qwen3.8 calibration, device scoring, and
  transfer overlap remain deferred.

**Risks and compatibility.** Selection is approximate and must not turn
  unobserved cold pages into zero attention. Keep routing retrieval separate
  from resident retention. No production default coefficients or enablement
  should be inferred from synthetic traces.

**Issue/PR disposition and rebase.** Keep this in Buun until a generic sparse
  metadata and policy owner is accepted. Do not use #28115 as a policy issue;
  it only informs draft placement. Rebase each sub-range in order and retain
  the CPU/fake evidence as the boundary proof.

### B08 - hybrid, MTP, server, and speculative lifecycle

**Destination and ownership.** Buun fork only. This slice is Qwen3.8/VBR,
  native Turbo4 MTP, recurrent companion, slot, checkpoint, and server
  lifecycle integration. It must not be sent to generic llama.cpp as one
  change.

**Source ranges.**

- Hybrid atomicity: `c084b0d36f2b63ffb74dc3ac3cc29acb88e71f94` to
  `e801caadb4f64612dd5cd92efac6cdd792d3369f`.
- Full-context MTP placement: `ff1b7dc8d0e56b5c3103bcb12c9000fc2853b764` to
  `824e18cae174f33ad826647057f78ddaced1b5d1`.
- Server slot generations: `9c748f91c0cb6f0116b06ae02517e68d687a8ab2` to
  `0156ae6aac7a14c8c0c70ec6a31711ba156a168b`.
- Checkpoint/speculative rollback: `a41f244363f301d5056d8df4e6cdab4b30597d8d` to
  `ede62ff96e9c5b4861fa83510a9125f18512bf05`.

**Dependencies.** B02, B03, B07, and U01. Hybrid atomicity precedes MTP;
  MTP precedes server slot lifecycle; rollback follows slot generations.

**Exact files and hunks.**

- `src/llama-memory-hybrid.h/.cpp`, `src/llama-memory-hybrid-idx.cpp`, and
  `src/llama-kv-cache.h`: composite attention/recurrent/index operations.
- `common/fit.cpp`, `common/fit.h`, `common/common.cpp`,
  `common/speculative.cpp`, `common/speculative.h`, and
  `src/llama-cache-budget.cpp/.h`: full trained-frontier MTP reservation,
  static Turbo4 MTP context, GPU residency refusal/logging, and budget
  additions. Extract only the MTP hunks; draft-placement hunks belong to U01.
- `tools/server/server-context.cpp/.h`: native MTP setup, slot session
  generations, single-slot selective admission, queue cancellation, cache
  debug fields, checkpoint restore, and speculative rollback boundaries.
- `tests/test-recurrent-state-rollback.cpp`, `tests/test-cache-budget.cpp`,
  `tests/test-arg-parser.cpp`, `tests/test-server-prompt-cache.cpp`, and
  `tests/test-mtp-vocab-trim.cpp`: corresponding deterministic regression
  hunks.

**Tests and platform.** Hybrid/server, cache-budget, parser, MTP trim, and
  prompt-cache tests passed. CPU/fake lifecycle evidence is recorded in the
  handoffs and CPU matrix. Full Qwen3.8 CUDA placement, MTP pressure, live
  transfers, multi-slot selective behavior, and long-context teardown are
  deferred. MTP GPU placement is directly relevant to open #28115; Buun
  multi-slot authority remains constrained by #109.

**Risks and compatibility.** Keep MTP outside pager victim eligibility and
  reserve it before target capacity. Preserve ordinary off/observe multi-slot
  behavior while refusing unsupported selective multi-slot authority. All
  checkpoint, recurrent, target, draft, and carry state must publish or reset
  together on generation mismatch.

**Issue/PR disposition and rebase.** Fork-only. Record #28115 and #109 as
  regression inputs in any human-authored future discussion; do not write that
  discussion here. Rebase shared `common/speculative.cpp`, budget files, and
  `server-context.cpp` by symbol, not by whole-file cherry-pick, after B07.

### E09 - exact all-page page-wave reference

**Destination and ownership.** Buun fork first. The online-softmax merge and
  coverage ledger are generally useful concepts, but the current executor and
  partial Turbo4 CUDA output depend on the Buun page/catalog and kernel
  contracts. A generic utility extraction requires separate maintainer review.

**Source range.** `c55e3d3f753ec195937469e661cfc6b244386909` to
`76ac9366d6d005a1903ee53fb24575da429e02b9`.

**Dependencies.** B03, U04, C05/C06, and the deferred Phase 06 acceptance
  decision. The exact implementation is last among code slices.

**Exact files and hunks.**

- `src/llama-kv-attention-exact.h/.cpp`: online `(m,l,o)` state, stable merge
  and normalization, logical-page wave plan, native/tail/identity validation,
  coverage ledger, and bounded serial/double-buffer executor callbacks.
- `src/llama-kv-attention-execution.h/.cpp`: exact route and executor seam
  additions.
- `ggml/src/ggml-cuda/fattn-paged-turbo4.cuh` and
  `ggml/src/ggml-cuda/fattn.cu`: optional per-head partial `[m,l,o[256]]`
  output and partial-only launch path. Extract these hunks after C05 and
  preserve the direct-kernel behavior.
- `tests/test-kv-attention-exact.cpp`, additions to
  `tests/test-cuda-fattn-paged-turbo4.cu`, and CMake registration.

**Tests and platform.** Exact CPU merge/coverage fixtures passed for one page,
  tails, permutations, gaps, 305-page coverage, serial and double-buffer
  schedules, and refusal cases. The bounded CUDA partial fixture and memcheck
  passed; raw evidence is in `docs/execution/evidence/EXACT_REFERENCE.md` and
  `exact-cuda-fixture.log`. Full 256K model execution, graph reuse, and dense
  logits comparison remain deferred.

**Risks and compatibility.** Do not add a CPU Turbo4 fallback that does not
  exist. Keep all states unnormalized until every page has merged, require
  exactly-once logical coverage, and retain separate cold staging. No quality
  or throughput claim follows from the bounded fixture.

**Issue/PR disposition and rebase.** Fork-only until the generic sparse-page
  and exact operator contracts are accepted. Do not append this to #22569.
  Rebase the executor after C06 and re-run both exact CPU and partial CUDA
  fixtures; compare range-diff against the C05 kernel extraction.

### D10 - documentation, operator guidance, and evidence metadata

**Destination and ownership.** Execution metadata and evidence are fork-local
  and excluded from all upstream code ranges. User-facing option documentation
  follows U01/B08 ownership. The benchmark adapter remains Buun execution
  tooling, not a release claim or generic upstream feature.

**Source coverage.** The complete execution package is the documentation
  range from `cb703be37e...` through `8ac64b1714...` under
  `docs/execution/`, including handoffs, CPU/CUDA matrices, exact reference,
  and deferred Phase 06 acceptance. The relevant non-execution files are:

- U01/B08 option docs: `docs/speculative.md`, `tools/cli/README.md`,
  `tools/completion/README.md`, and `tools/server/README.md`.
- Benchmark adapter: `tools/server/bench/README.md`,
  `tools/server/bench/run-pager-profile-benchmark.py`, and
  `tools/server/bench/run-pager-profile-benchmark.sh`.
- Execution helper tooling: `tool/codex/run_clustered.sh` and
  `tool/codex/task_state.py`.

**Dependencies.** All code slices and their evidence. Task 07-02 owns the
  final evidence index and operator-document consistency pass.

**Tests and platform.** Validate Markdown links and JSON, run
  `python3 tool/codex/task_state.py validate`, `git diff --check`, and the
  changed benchmark adapter syntax checks. Existing raw CPU/CUDA logs and
  final acceptance JSON remain the source of truth. The selective model-backed
  gates are explicitly deferred in `FINAL_ACCEPTANCE.md` and must not be
  upgraded by documentation.

**Risks and compatibility.** Do not copy `/srv/ai` paths, service state,
  model filenames, or RTX assumptions into production code. Evidence may point
  to external raw artifacts, but it must label unavailable runtime telemetry
  and deferred hardware/model checks. Generated option tables must be
  regenerated from the actual parser rather than hand-edited.

**Issue/PR disposition and rebase.** No upstream destination for execution
  metadata. Human maintainers may later select individual operator-doc changes
  alongside their accepted code slice. Keep all issue/PR prose human-owned.

## Boundary verification matrix

| Slice | Boundary test/evidence | Result at source boundary | Deferred |
| --- | --- | --- | --- |
| U01 | `test-arg-parser`; generic/Buun build paths | passed | external-draft model CUDA launch |
| B02 | `test-kv-residency`, `test-cache-budget` | passed | device allocation |
| B03 | `test-vbr-artifact-capture`, `test-kv-residency` | passed with CPU/fake providers | live CUDA transfer/VMM and hardware geometry |
| U04 | `test-kv-attention-view` | passed with CPU fixtures | real backend operator |
| C05 | `test-cuda-fattn-paged-turbo4`, memcheck/racecheck | passed on RTX 4080 sm_89 | full model graph and broader shapes |
| C06 | `test-kv-attention-execution`, view/residency tests | passed | model-backed graph capture |
| B07 | policy, routing-summary, and telemetry tests | passed with CPU/fakes | live trace calibration and overlap |
| B08 | server prompt-cache, rollback, MTP trim, budget/parser tests | passed deterministically | Qwen3.8 CUDA/MTP/long-context lifecycle |
| E09 | exact CPU fixture and CUDA partial fixture | passed bounded reference | full 256K exact run and logits comparison |
| D10 | JSON, state validator, syntax/link checks | required for task 07-02 | external raw evidence availability |

## Rebase and range-diff procedure

1. Keep the Buun extraction base fixed at `cb703be37e...` for fork review and
   keep the current implementation tip recorded as `8ac64b1714...`.
2. Before any generic slice, fast-forward a clean `vanwho/llama.cpp` fork
   worktree from current `ggml-org/llama.cpp` `upstream/master` at
   `de8656bd9...`; the old `67a17c17c...` base is historical evidence only.
3. Create one branch per slice from the synchronized destination default.
   Apply the listed task implementation commits or manually extract their
   listed hunks. Exclude `docs/execution/**`, completion/state commits, and
   unrelated generated or local files.
4. For shared files, apply hunks by symbol in dependency order: common draft
   conversion before MTP, page capture before catalog/transfer, policy replay
   before hot-set control, and direct CUDA launch before exact partial output.
5. Run the boundary test in the matrix after each slice. If a destination
   cannot compile independently, retain the smallest dependency slice and
   record the missing prerequisite rather than widening the slice.
6. Compare the resulting branch to its synchronized destination base with
   `git range-diff <destination-base>..<source-range> <rebased-base>..<rebased-tip>`
   and inspect `git diff --check`. A range-diff must show only the listed
   source hunks; do not use a whole-plan diff as proof of slice cleanliness.
7. Preserve the current exact reference and acceptance evidence as fork-local
   validation. It is not a substitute for maintainer direction or upstream
   review.

## Non-claims and human gates

- No upstream issue, discussion, pull request, review response, or merge was
  created by this work.
- The implementation has deterministic CPU/fake coverage and bounded CUDA
  evidence, but no live 256K selective pager runtime, production transfer
  ledger, or selective quality/throughput result.
- The generic destination is not ready for extraction until the 11-commit
  llama.cpp fork/upstream divergence is resolved and the uncommitted generic
  draft-placement worktree receives a reviewable tip SHA.
- Human maintainers must choose whether any generic page-table or scheduler
  material belongs in #21961/#22569, and a human must write any future
  GitHub-facing text.
