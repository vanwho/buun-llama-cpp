# V10 phase-61 benchmark summary

- Result: **current findings**
- Revision: `hotpath-v10-20260914`
- Scope: phase-61 receipts and bounded raw manifests only

## Identity, geometry, and placement

The requested matched coordinate was L8192/C6144/H8192/A4096/B128/U64. The
first exact-token smoke used L8192/H4096/B128/U64, admitted the request, then
stalled before producing a response or occupied frontier. The repaired full-L
fixtures separately measured L262144 allocation, and the VBR fixture reached
C1200 at L262144/H30208/B128/U128. Allocation is not occupied C262144.

The observed candidate was the phase-61-02 bundle for the full-L packed-graph
fixture. Target Turbo4 K/V was on CUDA and native-MTP Turbo4 K/V was on GPU;
the full-L draft capacity was 262144 rows. Requested, observed, and immutable
identity fields are retained separately in the JSON.

## Requested rows and rates

| row | status | reason |
|---|---|---|
| matched q0/q1/q2 (27 rows) | not_run | stopped at first invalid small-path smoke |
| cached append 64 / 256 | not_run | no usable frontier |
| controlled T1 promotion | not_run | campaign stopped before T1 |
| organic T2/T3 promotion | not_run | campaign stopped before T2/T3 |
| answer quality | not_run | no quality prompt completed |
| L32768 / L128K pilots | not_run | smoke prerequisite failed |
| occupied C262144 | not_run | no valid occupied full-context frontier |

Cold prefill, cached append, and committed decode rates therefore have zero
samples and null rates. Native MTP denominators were not observed because no
response completed; they are not represented as zero.

## Interpretation

Phase 61 validates the two preceding repair fixtures and preserves the first
post-repair revalidation stall as a concrete finding. It makes no speed,
acceptance, promotion, answer-quality, pilot, or occupied-C262144 claim.

See `V10_SUMMARY_61.json` for source artifact paths, hashes, byte geometry,
request status, and the separate allocation-versus-occupancy boundary.
