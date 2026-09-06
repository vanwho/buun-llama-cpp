# Phase 22 benchmark-only review

Verdict: **not reached**. The phase-21 summary SHA-256 is
`b21201bf1cc0bdf1341c5c7bd56271c4d44a5b89103fb9309f9901e800dfc085`.
Its compact receipt hashes and the representative allocation, population,
startup, curve and soak raw hashes all match their index. Mixed or telemetry-
incomplete phase-21 inputs remain excluded.

## Goal assessment

| Goal | Status | Finding |
|---|---|---|
| 262,144 allocation and occupied context | Failed | Allocation required fixed four-page recovery; auto admission OOMed. Near-full population stopped at 15,360 processed tokens after 1,200.08 seconds without a response. |
| Canonical CPU Turbo4 and bounded GPU target pages | Failed | Host-canonical rows and four GPU pages were observed only at 6,401 tokens. Full-context CPU backing was not populated, and four pages were not budget-derived. |
| Attention-driven cold retrieval | Not measured | No cold logical page was correlated through query selection, host checksum, H2D event/fence, physical slot and same-operation attention. |
| Full-context GPU Turbo4 MTP | Measured, allocation only | The recovery reports 262,144 native GPU MTP rows and 276,955,000 bytes with Turbo4 K/V; occupied-context generation is absent. |
| Lossless residency / selected-all / exact parity | Failed | Selected-all passed 6/6 bounded rows, but full-context lossless identity and exact parity were not measured. |
| Quality tradeoff | Failed | Bounded selective stopped at 8/19 correct; no telemetry-gated full matrix or 262K retrieval result exists. |
| Physical churn and lifecycle | Failed | The second pressure request aborted during checkpoint save before promotion, churn, restore and slot-reuse checks. |
| Speed and controls | Not measured | No target trial or release-matched control row completed, so no honest speed ratio can be calculated. |
| YaRN stretch | Not measured | Correctly skipped because the base goal is unmet; it does not affect this verdict. |

The historical 3x/5x CPU-KV and 70% all-GPU comparisons are not used as pass
gates. There is simply no valid phase-21 curve to compare. Long-context prefill
is the first observed speed blocker, but its cost cannot yet be separated from
model compute, fallback attention, graph churn or synchronization.

## Ranked remediation

1. `23-01` tests whether hot-page admission occurs before late recurrent and
   graph/compute reserves, using 175K/262K auto startup and fixed-four-page
   control. Success changes auto admission from OOM to a reconciled nonzero
   budget-derived pool.
2. `23-02` tests whether dense logical offsets are applied to compact reused
   slot tensors during checkpoint save. The two-request reproducer becomes
   three save/restore/continuation cycles with no bounds fault or state loss.
3. `23-03` isolates the exact live Qwen direct-route rejection or redundant
   fallback graph churn. This does not repeat the completed three-to-16-token
   tile change; it measures actual route eligibility and graph/sync counts.
4. `23-04` tests whether the fresh-query policy transaction is missing before
   selected-table construction, requiring nonzero useful H2D plus page/checksum/
   fence/slot/route/answer correlation.

Tasks `23-05`–`23-11` then repeat bounded quality, full 262,144 functionality,
the original-three-prompt curve, matched controls, physical soak, conditional
YaRN and compact summary on new provenance. `24-01` is the same Sol High
benchmark-only procedure and reads only that phase-23 summary.

## Deferred verification

The scheduled GPU-runtime repairs and measurements are deferred to phase 23;
the new compact-summary review is deferred to phase 24. No unavailable hardware,
credentials, hosted service or human upstream action blocks that work.
