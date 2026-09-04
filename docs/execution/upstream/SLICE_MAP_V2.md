# Upstream slice map V2

## Scope and synchronization

This is a fork-local extraction map for the completed Qwen3.8 attention-aware
Turbo4 pager. It is not an issue, pull request, review reply, commit message,
or upstream proposal. The map records source ranges, destinations, tests, and
known exclusions so a human owner can prepare small branches later.

Checked 2026-09-04 from the `codex/task-15-01` worktree:

| Ref | SHA | Meaning |
| --- | --- | --- |
| Recorded Buun base | `cb703be37e3628dadb71912f3b3b25b82090555b` | `WORK_STATE.json` base and parent of current upstream tip |
| `origin/master` | `cb703be37e3628dadb71912f3b3b25b82090555b` | fork default remains at the recorded base |
| `upstream/master` | `3823c9eb6541725bffa70fe3f8508c55b3b5ca7b` | current upstream tip, `cuda: fix Qwen4 HC combine ordering on SM90+` |
| integration tip | `39dfb98c88e40ebd9acc0b9de84d66322412f974` | current phase-14 integration branch |
| common ancestor | `cb703be37e3628dadb71912f3b3b25b82090555b` | merge base of integration and current upstream |

The recorded base is an ancestor of the integration tip. Current upstream is
one commit ahead of that base and is not yet an ancestor of the integration
tip: `git rev-list --left-right --count upstream/master...HEAD` returned
`1 199`. The fetched upstream commit changes only
`ggml/src/ggml-cuda/dsv4-hc.cu` and `tests/test-backend-ops.cpp`; neither path
is changed by the phase-08–14 implementation range. The expected re-sync
forecast is therefore a clean path-level apply, followed by the backend-op
focused test and the release smoke. The task agent did not rebase, merge,
commit, push, switch branches, or change remotes.

## Upstream duplicate/direction check

The current public records were checked only to detect overlap; no GitHub
content was created or changed.

- [Discussion #21961](https://github.com/ggml-org/llama.cpp/discussions/21961)
  remains the generic paged-KV design discussion.
- [Draft PR #22569](https://github.com/ggml-org/llama.cpp/pull/22569) remains a
  one-commit paged KV/block scheduler proposal. Its own review calls out the
  large scope and the need to split backend and scheduler work.
- [Issue #28115](https://github.com/ggml-org/llama.cpp/issues/28115) remains an
  open MTP draft-KV placement issue and has no development branch or pull
  request. It overlaps the independent MTP-placement motivation, not the
  Buun-specific VBR/Turbo4 pager implementation.
- [Buun issue #109](https://github.com/spiritbuun/buun-llama-cpp/issues/109)
  remains open and concerns multi-slot VBR context accounting. It is a
  lifecycle dependency/risk, not a duplicate implementation branch.
- No active upstream issue or pull request was found that duplicates the
  complete Buun selective pager, Qwen3.8 lifecycle, or exact-wave series.

Any future issue, discussion, PR, or review text remains human-owned.

## Extraction rules

Each phase-08–14 `feat(<task>)` SHA below is a source range whose parent is the
extraction base for that commit. A commit is assigned to one slice, except that
explicitly identified documentation hunks in a mixed commit belong to the
tests/docs slice. The task-state, handoff, evidence, plan, merge, and chore
commits under `docs/execution/**` are execution metadata and are excluded from
all upstream-bound slices.

The phase-00–07 foundation slices remain as the smaller ranges in
`SLICE_MAP.md`: U01 draft placement, B02 page/budget primitives, B03 host
backing and transactions, U04 selected metadata, C05/C06 Turbo4 attention and
graph lifecycle, B07 routing/policy/prefetch, B08 hybrid/MTP/server lifecycle,
E09 exact reference, and D10 docs/tooling. Their source ranges are preserved
below as the prerequisites for the live slices rather than re-cherry-picking
the old execution package.

## Dependency-ordered slices

### S01 — independent draft placement and dynamic MTP

**Destination:** generic draft placement is a candidate for `llama.cpp`; the
Qwen/Turbo4 MTP reservation and server fit wiring remain Buun-specific.

**Exact ranges and hunks:**

- Foundation U01: generic prepared worktree `67a17c17caa95742186f8b1ecadd1b5abd6d5ebb`
  to the uncommitted draft-placement state; Buun port
  `c003031954db6e88796727c614fe9711c187898a` to
  `8b13597b24f2792e12da4c1967752b1dae901d53`.
- `f7e498653d6ffe31db255ca0293e67cb4ccad129..c242dd746a0ad8321c0bc8b56f5a9ac914376de4`:
  `common/speculative.{h,cpp}` resolved target-context MTP rows and
  target/draft conversion; `common/common.cpp`, `common/fit.h`,
  `src/llama-cache-budget.h`, and `tools/server/server-context.cpp` reserve
  and report the resolved native MTP geometry. Parser and budget regression
  hunks are in `common/arg.cpp` and the three corresponding test files.

**Dependencies:** none for the normalized generic placement contract; the
live Buun reservation consumes the sizing ledger in S02.

**Tests:** `test-arg-parser`, `test-cache-budget`, and `test-moe-cache-fit`;
the model-backed context ladder is evidence in `DYNAMIC_CAPACITY_LADDER`.

**Risks/exclusions:** keep `auto|gpu|cpu` placement independent of target KV;
never use trained context as an MTP allocation floor; do not move Turbo4
aliases, native-MTP logs, or server adapters into a generic branch.

### S02 — generic sizing, accounting, and page core

**Destination:** portable internal primitives are candidates for a future
generic branch; live pager ownership and Qwen geometry stay in the Buun fork.

**Exact ranges and hunks:**

- Foundation B02: page identity
  `65180956d49af390ed5bdca824673008b47340c2..f02add48e012527371e4626e2411f4cf1f041f8d`;
  budget arithmetic
  `7f699a9225b78d42a7b2b935e88e65e1e731e224..b36ee8e99603d4e87d8d473788aadd4528f467fd`.
- `a729b097c0d18e84939163fd180d6eea2097ca6d..630256e890c3d4d52b98f8be20d0ecba34a924ec`:
  overflow-safe ledger admission, runtime target-page capacity, MTP/scratch/
  headroom charges, and capacity-relative policy/telemetry input in
  `src/llama-cache-budget.*`, `src/llama-kv-policy.*`, and
  `src/llama-kv-attention-telemetry.*`.
- `3ecf06943c6bf5b12884c026bbc242e93c146953..4c36ba20083a842202070e75b76cbf9e8cbd482d`:
  normalized pager configuration and feature gate in
  `src/llama-kv-pager-config.*`, context construction, and argument/API seams.
- `e66089e0fd5c31297acb800d5ed26f481315fbf7..15346779b649ec7fde83e8e31294372e03c26d6a`:
  production pager owner, compact physical storage, geometry/accounting,
  CMake registration, and deterministic owner tests.
- `c8c2b84daea6491e58695e49ff14faa94754c1f4..39adfde835ca8d7f87d6c33a1b0e36464cc6ea19`:
  target writes, valid-row/tail tracking, sequence mutation completion, and
  pager/cache/memory integration in `src/llama-kv-pager.*`,
  `src/llama-kv-cache.*`, `src/llama-memory*`, and the pager test.
- Calibration remains a separate policy slice in S05; the runtime capacity
  formula and all fixed-count removal belong here.

**Tests:** `test-kv-residency`, `test-cache-budget`, `test-kv-pager`, parser
tests, and the CPU/fake matrix. No device allocation is implied by the generic
candidate.

**Risks/exclusions:** preserve logical/physical identity and the 256-cell
  generation boundary; do not expose Qwen numeric geometry as a public generic
  contract; absolute page counts are valid only in explicit fixtures.

### S03 — canonical host backing and CUDA residency

**Destination:** Buun fork. A generic residency pool can be extracted later,
but VBR capture/catalog ownership and Qwen layer geometry are fork-specific.

**Exact ranges and hunks:**

- Foundation B03: capture `0ffec87e32a49b65185cb03b60f3f89c1198f8cc..45a1a795a91f771d9674681b6c85a39b26b0d9d3`;
  catalog `a687880e2c043a37644a75490fcf289469eccdf6..e531addc23916ef1f4b66324c0daaeec9ea3ae06`;
  transfer `53d1f37c1f394d17751cfc0c4483a77346e4608c..ea12da5d08bfaca54fe3676dbedc2842255de80a`;
  transaction `b50a11691d01ff4dad60707bbbff2a428d81e69f..4a81b10bf8282331b4033966e6b5ab4a524da4a6`;
  fixed-window proof `a4f8697534a61cdbb37a9453d5d1d7c174aa99ae..ab8e08920ce2d7a810317ac70fd94a2cfd7b44b5`.
- `a37aa52e02f333045d9c087ec01a5b5092474d1d..ca2650c72cc7d7bbf35b43bc7d2e6801fa3f64ef`:
  live host sealing, bounded capture rings, authenticated representation
  identity, and clean-eviction/partial-page publication in the pager/cache.
- `25da0a1745f0009f1289142220311e7feb4c5fa3..59fbbb4ee97f3f780cd70d4aa9ca1625f299b5b0`:
  real tensor-view H2D/D2H adapter, bounded staging, event completion,
  cancellation, stale cleanup, transaction publication, and transfer counters.
- `a360bb35ebbbf1ce5e3096efff0287af0bf1ede4..2da10d15719ae937272be6e6eb4afcbf6430b4b0`:
  transfer/staging overlap and packed projection refinements in
  `src/llama-vbr-artifact-stage.cpp` plus its existing capture/adopt tests.

**Tests/evidence:** `test-vbr-artifact-capture`, `test-vbr-artifact-adopt`,
`test-kv-residency`, CUDA VMM/event evidence, and `09-05`/`13-03` raw logs.

**Risks/exclusions:** host RAM is canonical Turbo4 backing; clean eviction is
zero-D2H; partially transferred pages cannot publish; `/srv/ai` paths and
hardware/service facts remain execution-only.

### S04 — Turbo4 attention and graph execution

**Destination:** Buun fork only for the current Qwen-shaped Turbo4 operator;
generic extraction requires a separate accepted sparse-attention contract.

**Exact ranges and hunks:**

- Foundation U04/C05/C06: selected view
  `c1f41c2c4363f7fa39b996815f7ac3dde3182989..d8224c7338361689c06ffa2466e6a2719c46b6fb`;
  operator metadata `6a5776f24962be5ae56bfb6a8f7356debea24042..f2908b775d7465de7772e9893b62bd2c9d565244`;
  direct CUDA `f4011b2c009aa2c20152570561c8950629e063b6..3c5458f8ab88a62ab411d319e3bf2860e2bfad47`;
  graph lifecycle `4aa124f1b5a735e8123fe5447d4c455c5366bad4..122f29e63176434db74edcc9a713c9596e67b71f`.
- `daa4d924314f503289b357a26b00983a45019dcf..71bf9ac4948f2df9631b970cf94ca83bae0e6064`:
  selected-all-pages reference attention reaches the live Qwen graph with
  bounded selected rows and graph/table fencing.
- `eca69faa673effd09db1243874f0189b3420c677..0b88846cbc5fb516a5ac19a6106ac09b1694b86b`:
  direct paged Turbo4 Flash Attention dispatch, raw I8 page slab views,
  native positions, and fail-closed shape checks in ggml/CUDA.
- `835a929e3ff08337f548ef8f4db4d16e3a90366f..e59eb3f7380cbb69d06104c95c74866dc107fc6e`:
  graph epoch/replay/rebuild metrics and execution-lifetime fencing.
- `af8bd0c8c617435c9eb189c5ebe513622fabc736..70247ed7ac095769ef9f822c52835d001650b5e4`:
  bounded selected Turbo4 prefill and physical-row scratch/admission bounds.
- `d9717589a93a91e4d0620e92b7dbd018e96479dd..e9e73c63257b2ae09bfc05305fa4aa393c2e39ef`:
  reusable staging/table upload optimization and graph input reuse fencing.

**Tests/evidence:** `test-kv-attention-view`, `test-kv-attention-execution`,
`test-cuda-fattn-paged-turbo4`, graph-key tests, CUDA memcheck/racecheck, and
the bounded decode/prefill profiles.

**Risks/exclusions:** direct support is batch-1, causal, Qwen/Turbo4-shaped;
never gather a full-cache F16 tensor or silently dense-fallback an unsupported
shape. The Qwen model graph and CUDA measurements are not generic proof.

### S05 — routing, telemetry, dynamic policy, and prefetch

**Destination:** Buun fork. Only pure summaries may become generic after
maintainer selection of a metadata and policy owner.

**Exact ranges and hunks:**

- Foundation B07: replay policy `2794b1cfb06407539433fa798ee47e6c5929ef88..4c8975ddf80e9299cc44a36f2182542fa992eaab`;
  routing `4662995c8d3c06cd013d81b998182a177ee20982..3ac062c330741b502843f4bde034b50d004dd2bf`;
  telemetry `b0c3322a219e9c013c86981bf71b6252dfdd807e..f340a5ff43f31d4c6ba088fe0927c60ce968ad03`;
  controller `a92a00ac18c5574c4653cc252d01238f5798aed6..cda8d1c10e3b33b78b94a06f494ae1b86d7278c7`;
  prefetch `7e449d5ac22ed2933e92e8b49f86ba499dbe2bf0..0d398d6b5abde0e6b8e3d582daa3e0e65d038422`.
- `226d951a8354b24796b9fb8f228db2f1c0c3ec1d..27efb0e0a5e8457217b47b68009c3b3c4020fb6c`:
  bounded Turbo4 representatives and sealed-page summary ownership.
- `c3957a5c1ce1ba8e5a465c743bb38376deafde2d..9c92e5a77ea27b3cd4463f17b89ad8bfb7cb0d00`:
  all-page query retrieval and structural-union API.
- `b70658a615ff9e5f8080ada2c7ddbb8c62b21f8c..ef6e59c6f5d01525656a0a67acd3df038f7c25b3`:
  optional GPU page-mass output, post-fence aggregation, cadence, and graph
  telemetry publication.
- `61f5675e987d41190436e18f9fe31c6ee85842e1..d838f6eb0293298ce8e4b7c943998894e6d88ce7`:
  live hot-set publication and occupied-slot promotion ordering.
- `291690e30b12149845e804c95841047b12d5e488..7aff8460275d5d31e070e61c4441accd37a5fbf8`:
  predictive prefetch, bounded queue/coalescing, and lifecycle callbacks.
- `2d9721215c483f91b6ec14badf032781b5126228..1e38028802bcf4e69cc72d413b0760eb6a3bed0d`:
  calibrated release ratios, normalized policy defaults, and small-capacity
  fail-closed tests. The one policy accounting correction in
  `1988b072cc0e3466c739a1762c4e5f44b00a59a4..f118f270b4a298500d9675dd688eed3dab374976`
  also belongs here.

**Tests/evidence:** routing summary/retrieval, telemetry, policy, pager,
residency, and lifecycle fake tests; `ROUTING_RETRIEVAL_CHECKPOINT`,
`LIVE_ATTENTION_TELEMETRY_CHECKPOINT`, and policy calibration evidence.

**Risks/exclusions:** cold retrieval and resident retention remain separate;
policy scales from admitted `H`; coefficients are tied to normalized evidence,
not synthetic tuning; live quality/speed failures in phase 14 remain failures.

### S06 — Qwen, MTP, and server lifecycle

**Destination:** Buun fork only. This slice owns the Qwen hybrid/recurrent,
native MTP, slot, checkpoint, and server boundaries.

**Exact ranges and hunks:**

- Foundation B08: hybrid atomicity `c084b0d36f2b63ffb74dc3ac3cc29acb88e71f94..e801caadb4f64612dd5cd92efac6cdd792d3369f`;
  MTP `ff1b7dc8d0e56b5c3103bcb12c9000fc2853b764..824e18cae174f33ad826647057f78ddaced1b5d1`;
  slot generations `9c748f91c0cb6f0116b06ae02517e68d687a8ab2..0156ae6aac7a14c8c0c70ec6a31711ba156a168b`;
  rollback `a41f244363f301d5056d8df4e6cdab4b30597d8d..ede62ff96e9c5b4861fa83510a9125f18512bf05`.
- `291690e30b12149845e804c95841047b12d5e488..7aff8460275d5d31e070e61c4441accd37a5fbf8`:
  source lifecycle owner and server-safe cancellation/teardown callbacks.
- `d20b71bac529a82c4495092be79813e2862c5ef7..e5be946675679d93a711e2133a787c0a5baa37d5`:
  production pager diagnostics and configuration summaries in context/config;
  the `tools/cli`, completion, and server README hunks are assigned to S08.
- `4f03687dc484c77a7caf88c26f31c2b81dad2f54..396981d666bc608113124ef1dfef7a4971d0e861`:
  metrics/ledger publication, server task metrics, benchmark telemetry
  ingestion, and teardown snapshots.
- `8a550d30cf50e3b171153f3c54abe6bcaf4dc10a..281ff43345aae0936b67aad444d508c8fbc3f4fb`:
  generation-safe pin release, full-sequence reset, and current-page cleanup.

**Tests/evidence:** recurrent rollback, prompt-cache, MTP trim, budget/parser,
server lifecycle, and the dynamic context/MTP ladder. Exact live catalog and
callback refusal are recorded by S07.

**Risks/exclusions:** native MTP is Turbo4/GPU-resident and never a target
victim; selective authority remains single-slot; no generic branch receives
Qwen geometry, service paths, profiles, or multi-slot claims.

### S07 — exact page-wave mode

**Destination:** exact executor source remains Buun fork-only. Tests and
portable option docs are assigned to the separate S08 package. All execution
metadata is fork-local and excluded from upstream slices.

**Exact ranges and hunks:**

- Foundation E09: exact reference
  `c55e3d3f753ec195937469e661cfc6b244386909..76ac9366d6d005a1903ee53fb24575da429e02b9`.
- `dad8ab4bf4afdc92a161bca3e78679835ddbc70f..45e2096961c024969bf411c9197a0c2e33078229`:
  live canonical page inventory, resident/cold exact planning, bounded
  page-wave preflight, and exact ledger source in the catalog, pager,
  context, and executor.

**Tests/evidence:** exact CPU merge/coverage and CUDA partial fixtures. The
exact tests and CUDA test additions are assigned to S08; the acceptance
records remain evidence, not claims that blocked quality/speed gates passed.

**Risks/exclusions:** no CPU Turbo4 exact fallback is invented; exact waves
require complete coverage and configured callbacks; tests/docs, benchmark raw
output, model/profile names, absolute paths, service state, and
`docs/execution/**` remain outside this source slice.

### S08 — tests, benchmark/docs, and acceptance evidence

**Destination:** tests and portable option documentation follow their owning
source slice; benchmark adapters and acceptance evidence remain fork-local.

**Exact ranges and hunks:**

- `58fdbc9dd3175ef5187a84a83ae726523ff8fc9c..ff5107d4763b1f55f56c1667802ef27db3b4333c`:
  frozen corpus generator, benchmark contract/validator, and dry-run adapter
  in `tools/server/bench`.
- Test hunks from `dad8ab4bf4afdc92a161bca3e78679835ddbc70f..45e2096961c024969bf411c9197a0c2e33078229`:
  exact ledger and live catalog assertions in `tests/test-kv-attention-exact.cpp`
  and `tests/test-kv-pager.cpp`.
- `5a867e524cd7a344f04ed11ad9f0844b4aafcf97..13ac2195824f552910693342e3777df36ebb98a6`:
  pager mode proof and fail-closed lifecycle regression test.
- Documentation hunks from `d20b71bac529a82c4495092be79813e2862c5ef7..e5be946675679d93a711e2133a787c0a5baa37d5`:
  generated CLI, completion, and server README updates.
- Test and generated-help hunks from
  `1988b072cc0e3466c739a1762c4e5f44b00a59a4..f118f270b4a298500d9675dd688eed3dab374976`:
  phase-14 policy/view/cell/architecture regression fixes and help-table
  refresh. The source policy line is assigned to S05.

**Tests/evidence:** benchmark Python compile/JSON validation, generated help
checks, release CPU/CUDA/fault matrices, and all phase-14 evidence manifests.
The evidence records preserve blocked quality/speed decisions and do not
promote them to acceptance passes.

**Risks/exclusions:** do not put raw benchmark output, model/profile names,
absolute paths, service state, or any `docs/execution/**` file in an
upstream-bound implementation branch.

## Commit coverage and exclusions

The phase-08–14 source coverage is complete with the following assignment:

| Task commits | Slice |
| --- | --- |
| `08-02:c242dd746`, `08-03:630256e89` | S01, S02 |
| `08-04:ff5107d47` | S08 |
| `09-01:4c36ba200`, `09-02:15346779b`, `09-03:39adfde83` | S02 |
| `09-04:ca2650c72`, `09-05:59fbbb4ee`, `13-03:2da10d157` | S03 |
| `10-01:71bf9ac49`, `10-02:0b88846cb`, `10-03:e59eb3f73`, `10-04:70247ed7a`, `13-04:e9e73c632` | S04 |
| `10-05:45e209696` | S07/S08 as detailed above |
| `11-01:27efb0e0a`, `11-02:9c92e5a77`, `11-03:ef6e59c6f`, `11-04:d838f6eb0`, `11-05:7aff84602`, `13-05:1e3802880` | S05/S06 as detailed above |
| `12-01:e5be94667`, `12-02:396981d66`, `12-03:13ac21958` | S06/S08 as detailed above |
| `14-01:f118f270b`, `14-02:281ff4334` | S05/S06/S08 as detailed above |

`08-01`, `08-05`, `13-01`, `13-02`, and `14-03` through `14-05` contain only
execution analysis, evidence, state, or handoff material at their completion
boundaries. Their files are intentionally not in an upstream implementation
slice. All merge/chore/state commits and every path under `docs/execution/**`
are likewise excluded.

## Boundary verification matrix

| Slice | Local boundary | Result |
| --- | --- | --- |
| S01/S02 | parser, budget, pager, residency, policy, telemetry tests | passed in CPU/fake and release matrices |
| S03 | capture/adopt/residency tests; VMM/event and transfer evidence | passed at deterministic boundary; live acceptance is recorded separately |
| S04 | view/execution, direct CUDA, graph-key, sanitizer tests | passed for bounded fixtures |
| S05 | routing, retrieval, telemetry, policy, lifecycle fakes | passed; phase-14 model quality/speed gates remain blocked by their recorded failures |
| S06 | server/MTP/rollback and dynamic context ladder | passed at local boundaries; selective MTP request rollback remains a phase-14 risk |
| S07 | exact CPU merge/coverage and CUDA partial fixtures | passed for bounded exact boundary |
| S08 | benchmark syntax/JSON, generated help, release smoke, and acceptance manifests | passed for locally executable checks; manifests retain blocked decisions honestly |

## Re-sync and review procedure for the authorized outer Git owner

1. Preserve the task metadata and create a clean fork integration worktree at
   the fetched `upstream/master` tip `3823c9eb6`; do not include
   `docs/execution/**` in an upstream-bound branch.
2. Apply S01 through S08 as separate dependency-ordered branches. For mixed
   commits, extract only the symbols listed in the slice; do not cherry-pick a
   whole task merge or execution commit.
3. Run `git range-diff` against the synchronized destination base for every
   branch and inspect `git diff --check`. The fetched Qwen4 CUDA fix has no
   changed-path overlap with the local implementation, but its backend-op test
   must still be rerun after integration.
4. Run the listed boundary test after each slice, then the Release CPU/CUDA
   smoke. Re-run full model-backed phase-14 gates separately; the current
   `QUALITY_ACCEPTANCE_V2` and `PERFORMANCE_ACCEPTANCE_V2` records are blocked,
   not portable acceptance passes.
5. Human maintainers choose whether S02/S04 metadata belongs in #21961/#22569;
   S01 MTP placement may reference #28115. S03, S05, S06, and exact Turbo4
   work remain Buun-specific unless maintainers explicitly accept a new
   generic boundary.

## Portability and exclusion audit

- `git diff --check` on the current worktree has only pre-existing blank-line
  diagnostics in historical execution files and one execution-era header; the
  task-local state/docs edits introduce no whitespace errors.
- Secret-pattern scanning found no credentials in production changes.
- Absolute `/srv/ai` paths, service commands, API-key defaults, and profile
  names occur only in the benchmark adapter or operator documentation; they
  are excluded from production extraction.
- `src/llama-kv-pager.cpp` and `src/llama-kv-residency-transfer.cpp` are large
  source implementations, not generated or machine artifacts. No generated
  binary, model, build directory, or raw benchmark output is in a source
  slice.
