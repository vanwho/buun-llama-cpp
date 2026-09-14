# V10 review 56

Revision: `hotpath-v10-20260914`. Reviewed input:
`V10_SUMMARY_55.json/.md` only.

## Decision

`goal_met` is false. The summary measures the 8K selective geometry and real
cold-prefill, cached-append, and committed-decode rates, and it measures
Turbo4 target KV on CUDA plus a 262144-row Turbo4/GPU full-L draft. It does not
contain a complete immutable bundle manifest, request-scoped native-MTP
denominators, either control mode, controlled or organic physical promotion,
useful answer quality, 32K/128K pilots, or occupied `C262144`.

The measured rates are cold prefill 186.035 tok/s (9 samples), cached append
456.823 tok/s (2 samples), and committed decode 13.104 tok/s (9 samples) at
L8192/C6144/H4096/A2048/B128/U128. These are findings, not a paired practical
speed claim: the required controls and MTP denominators are missing.

The full-L result is allocation/startup plus a bounded C1207 request. It is
not full occupancy. The phase-55 summary's explicit `not_run` boundaries are
preserved rather than inherited from phase 53.

## Ordered next work

Phase 57 repairs the curve parser's dropped MTP counters and the missing
immutable bundle identity, then runs one bounded matched/control,
promotion/quality, and full-L occupancy campaign. It produces a phase-57-only
summary. Phase 58 reviews that summary. Every repair packet contains an exact
symbol, hypothesis, minimal fixture, invariant, numeric discriminator, and
stop condition; this is not an evidence-only loop.

## Deferred verification

The following require the authorized candidate GPU endpoint after the two
contract repairs: native-MTP denominators and controls, controlled/organic
physical promotion with completed target use, answer quality, and occupied
`C262144`. Until then no practical comparison, full-context occupancy, or
quality claim is made.
