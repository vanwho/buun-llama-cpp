# Phase 23 benchmark summary

Overall status: **not demonstrated**. Phase 23 provides partial deterministic
implementation evidence, but no coherent repaired runtime campaign proves the
Turbo4 selective-pager product goal. This summary consumes only the compact
phase-23 receipts and their indexed raw artifacts; it does not replay an
earlier phase or start a campaign. Machine-readable evidence is
[`PHASE23_BENCHMARK_SUMMARY.json`](PHASE23_BENCHMARK_SUMMARY.json).

## Identity and evidence policy

The repaired source identity is `e981d13e`; the local repaired executable is
`7968b152…3c65f` with indexed candidate DSOs. The preserved deployed runtime
is the distinct 20-07 bundle: manifest `653cda…e7401` and binary
`d534fd…46d8`. The Qwen model is `40fac4…e6199` and pager-corpus-v4 is
`cece9e…a0bd`. All 40 unique artifacts indexed by the phase-23 receipts were
present and matched their recorded SHA-256 values. `[receipt_23_02,
receipt_23_03, receipt_23_04, receipt_23_05, receipt_23_06, receipt_23_07,
receipt_23_08, receipt_23_09, receipt_23_10, candidate_binary,
candidate_libggml_cuda, candidate_libllama, candidate_server_impl,
deployed_binary, deployed_bundle_manifest, target_model, target_corpus]`

The local repaired candidate was not deployed as the preserved runtime. The
phase-23 quality receipt has no deployed tokenizer/chat-template identity, and
no phase-23 receipt supplies one immutable target configuration hash. Those
missing identities are rejected; historical template or policy values are not
used to manufacture a target/control match. `[receipt_23_05, receipt_23_07,
receipt_23_08, receipt_23_09]`

## Deterministic repairs

The admission ledger/retry, compact checkpoint identity, Qwen 24:4 direct-route
geometry, and authenticated multi-page residency fixtures passed their local
checks. These are implementation and fixture results, not live full-context
acceptance. `[receipt_23_01, receipt_23_02, receipt_23_03, receipt_23_04]`

## Base 262,144-token functionality

| Area | Status | Result |
|---|---|---|
| Allocation | `not_measured` | The geometry fixture covers 262,144 tokens, 1,024 logical 256-token pages; no fresh automatic budget-derived production allocation was run. |
| Occupied context | `not_measured` | The requested 262,136-token population with eight-token generation reserve was not sent to the repaired candidate. |
| Retrieval and continuation | `not_measured` | No early/middle/late/recent/focus retrieval or continuation response exists. |
| Base gate | `failed` | Allocation, near-full population, retrieval, continuation, and native GPU MTP residency were not jointly demonstrated. |

`PHASE23_FULL256K.json` explicitly sets `base_goal_demonstrated` to false.
`[receipt_23_06]`

## Placement, paging, identity, and parity

The required target and draft K/V codecs remain Turbo4, canonical CPU backing
remains the required policy, native MTP is required to be GPU-resident Turbo4,
and hot capacity is required to be budget-derived. None of those placements
was observed in a fresh repaired-release full-context allocation. `[receipt_23_05,
receipt_23_06, receipt_23_09]`

The residency fixtures measured cold-inventory selection, authenticated
multi-page H2D construction, publication, rollback, and slot behavior.
Production attention-driven cold retrieval was **not measured**: no logical
page, canonical checksum, transfer event/fence, physical slot/generation,
route, answer, or post-request counter chain exists. `[receipt_23_04,
receipt_23_09]`

Production selected-all/exact parity was **not measured**. Both denominators
are zero, and no lossless full-context identity/parity claim is made.
`[receipt_23_05, receipt_23_06]`

## Quality, curve, and controls

The frozen quality matrix planned 72 requests and executed zero. Exact,
selected-all, and selective numerators and denominators are all zero because
the repaired candidate was not deployed and required route/page/movement/
placement telemetry was absent. The frozen corpus has no multi-hop case, so
that family remains explicitly excluded. `[receipt_23_05, target_corpus]`

The original-three-prompt curve covers 20K, 40K, 60K, 100K, 175K, and 262,144
contexts, but has zero completed target rows. All speed, occupancy, movement,
route, and memory values are null. No release-matched CPU-KV/GPU-MTP or dense
all-GPU Turbo4/MTP control row exists, so all ratios and component bottlenecks
are **not measured**. Historical 3x/5x/70% comparisons remain annotations,
not denominators or gates. `[receipt_23_07, receipt_23_08]`

## Physical movement and lifecycle soak

The planned warm-focus, known-cold-promotion, focus-shift, churn,
cancellation/drain, checkpoint restore, restart, and slot-reuse segments are
all **not measured**. No physical movement, checksum/fence/slot-generation
correlation, resource high-water/plateau, leak, or lifecycle claim is made.
Zero movement is not interpreted as no movement. `[receipt_23_09]`

## YaRN stretch

YaRN status is **`not_attempted_base_goal_unmet`**. Because the base gate failed,
Qwen RoPE/scaling metadata inspection, positional configuration, >256K
occupancy, memory, shared-length quality, and speed were intentionally not
attempted. `[receipt_23_10, receipt_23_06]`

## Ranked gaps

1. **Coherent repaired deployment and telemetry preflight — failed.** The
   repaired executable/DSOs, template identity, and configuration identity
   were not deployed as one measured bundle. `[receipt_23_05, receipt_23_07,
   receipt_23_08, receipt_23_09]`
2. **Base 262,144 occupied-context proof — not measured.** Automatic
   allocation, near-full population, retrieval, continuation, and full GPU
   native-MTP residency remain unproven. `[receipt_23_06]`
3. **Physical cold movement and lifecycle correlation — not measured.** The
   required checksum/event/fence/slot/generation/route/answer chain is absent.
   `[receipt_23_04, receipt_23_09]`
4. **Speed curve and matched controls — not measured.** No completed target row
   exists from which a speed ratio or bottleneck share could be computed.
   `[receipt_23_07, receipt_23_08]`

## Deferred verification

Deploy the repaired executable and all loaded project DSOs as one immutable
bundle, recording model, template, and configuration hashes. Keep port 8092
and unowned processes untouched. Pass request telemetry preflight, then
complete the 262,144-token base allocation/population/retrieval/continuation
and GPU MTP proof. After that, run the physical soak, six-point curve, and
release-matched controls. Revisit YaRN only after the base gate is demonstrated.
`[receipt_23_05, receipt_23_06, receipt_23_07, receipt_23_08, receipt_23_09,
receipt_23_10]`
