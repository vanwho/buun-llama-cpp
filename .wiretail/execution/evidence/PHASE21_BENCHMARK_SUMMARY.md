# Phase 21 benchmark summary

Overall status: **not demonstrated**. This summary consumes the compact
phase-21 receipts and their raw indexes; it does not replay historical
acceptance audits or start another campaign. Machine-readable evidence is
[`PHASE21_BENCHMARK_SUMMARY.json`](PHASE21_BENCHMARK_SUMMARY.json).

## Identity and evidence policy

The accepted target measurements use the 20-07 bundle, server binary
`d534fd99…646d8`, Qwen model `40fac405…e6199`, and corpus file
`cece9edd…a0bd`. The receipts also retain the bundle manifest, runtime
template, quality-template, and profile-policy hashes. Commit labels in the
21-06 allocation receipt differ, so binary SHA-256 and bundle identity—not
those labels—are used for matching. `[receipt-21-01, curve-binary,
curve-model, curve-corpus, soak-bundle-manifest, receipt-21-06]`

Mixed or incomplete inputs are rejected from claims: the 21-05 managed-service
candidate lacks a release manifest and required pager telemetry; the current
22016-token sentinel has a different declared corpus/context; and 20-06
controls use a different release and protocol. Missing telemetry remains
missing, never zero-filled. `[receipt-21-05, quality-live-preflight,
controls-current-control-config, controls-current-control-record,
controls-historical-controls]`

## Base 256K functionality

| Area | Result | Evidence |
|---|---|---|
| Allocation | Exact 262144-token capacity and 1024 logical pages allocated after fixed-four-page recovery; automatic budget-derived hot-page admission hit CUDA OOM. | `21-06-allocation-run-config`, `21-06-allocation-records` |
| Population | Incomplete: 262136 locally rendered prompt tokens with an eight-token reserve; server timed out after 15360 processed tokens at 1200.0778666429687 seconds, with no response. | `21-06-population-summary`, `21-06-population-slots`, `21-06-population-metrics` |
| Retrieval | Not measured. | `receipt-21-06` |
| Continuation | Not measured. | `receipt-21-06` |
| Base gate | Not demonstrated; no shorter-context substitute was used. | `receipt-21-06`, `21-06-population-summary` |

Target and draft K/V are Turbo4, and native MTP is GPU-resident Turbo4. The
allocation receipt reports 262144 MTP rows and 276955000 bytes, but this is
allocation-only and is not full-context generation proof. `[receipt-21-06,
21-06-allocation-records, soak-stable-01-response]`

Canonical CPU Turbo4 backing and a four-page hot set were observed in the
bounded pressure run: 26 logical pages, three host pages, and 768 host-valid
rows at high water. These are bounded observations; they do not prove the
same state was populated at 262K. `[receipt-21-06, soak-resource-timeline,
receipt-21-09]`

Route counters were recorded, but no logical-page ID, host checksum,
promotion, physical-slot, transfer-event, or fence correlation exists. H2D /
D2H useful-byte, fault, and eviction counters were zero in the partial soak;
that is an observation, not proof of absent paging. `[receipt-21-01,
receipt-21-06, soak-resource-timeline, soak-stable-01-response,
soak-metrics-089]`

## Quality and speed

Bounded sentinels passed for dense all-fit (1/1) and selected-all (6/6), but
the selective sentinel stopped after 19 records with 8 passes and 11 answer
mismatches. No full-262K retrieval quality score exists because population
did not complete. `[receipt-21-01, receipt-21-06, 21-06-population-summary]`

The automatic startup ladder passed only at 20K, 40K, 60K, and 100K. Startup
failed at 175K during recurrent-state allocation and at 262144 during
compute-buffer allocation. The 20K warmup processed 3072/19600 prompt tokens
in 134.30 seconds and produced no completed trial; therefore all three
questions × three trials at every curve point have null speed data. `[receipt-21-07,
curve-startup-175k-journal, curve-startup-262k-journal, curve-20k-campaign]`

No release-matched target row completed, so CPU-KV/GPU-MTP and dense all-GPU
controls, ratios, ablations, and component decomposition are all unavailable.
Historical and shorter controls remain explicitly unpaired. `[receipt-21-08,
controls-curve-evidence, controls-current-control-config,
controls-historical-controls]`

## Pressure soak and YaRN

The 6401-token, four-hot-page soak completed one 6393-token warm-focus request
(348.2877993530128 seconds; four draft and four accepted tokens), then the
next equal-work request hit a checkpoint-save tensor-bounds assertion and
disconnected. Cold promotion, churn, cancellation/drain, restore, slot reuse,
and page/fence correlation were not measured. The exact fixed-four-page 262K
profile was restored healthy afterward, which does not pass the soak. `[receipt-21-09,
soak-stable-01-response, soak-stable-02-response, soak-restored-systemd]`

YaRN was intentionally not attempted because the base gate was not proven.
There are no YaRN position, occupancy, memory, quality, or speed measurements.
`[receipt-21-10, yarn-base-256k-receipt]`

## Ranked measured blockers

1. Long-context prefill/context execution: 20K warmup and near-full 262K
   population made insufficient progress without a response. `[curve-20k-campaign,
   21-06-population-summary]`
2. Native-MTP high-context allocation headroom and automatic hot-budget
   admission: 175K/262K startup failures and 262K auto-hot OOM. `[curve-startup-175k-journal,
   curve-startup-262k-journal, receipt-21-06]`
3. Checkpoint-save lifecycle assertion under repeated pressure. `[soak-stable-02-response,
   soak-restored-systemd]`
4. Measurement cannot separate model compute from pager, transfer, or
   synchronization cost because no matched target/control row or physical
   movement correlation exists. `[receipt-21-07, receipt-21-08, receipt-21-09]`

## Deferred verification

Repair automatic hot-budget admission, native-MTP high-context allocation, and
checkpoint save; then complete near-262K population, retrieval, continuation,
physical transfer correlation, the six-context speed curve, and release-matched
controls. Revisit YaRN only after the base gate is proven. No unavailable
hardware, hosted service, or human action caused these omissions; the remaining
work is local runtime repair and measurement. `[receipt-21-06, receipt-21-07,
receipt-21-08, receipt-21-09, receipt-21-10]`
