# Phase-84 diagnostic summary

Revision: `hotpath-v10-20260914`. This is a diagnostic handoff, not an
overall-goal acceptance ledger.

## Stable coordinate

Use `L8192/C6143/H4096/A4096/B128/U128` for further work. `B128/U64` is a
measured secondary shape and is slower in the selected route. The identity is
the Qwen3.8-27B UD-IQ4_XS model with Turbo4 target K/V and GPU Turbo4 native
MTP K/V. See `V10_84-02.json` and `V10_84-03_speed_attribution.json`.

## MTP and token parity

All-GPU native MTP generated drafts but accepted none: `123/0` per measured
trial, or `0.0%`. Its output matched the same-placement feature-off control;
no native-only first divergent token was observed. The zero-acceptance state
is therefore an unresolved proposal/target verification or common model/output
diagnostic, not evidence of an MTP-only token divergence. The CPU-main-KV
native control has the same `123/0` result. Selected native MTP accepted
`40/44`, `39/47`, and `39/47` drafts (90.91%, 82.98%, 82.98%).

## Route and cold promotion

Automatic selected attention uses `selected_direct` only for the proven
one-page shape. Multi-page prefill is deliberately `selected_reference` after
a live direct experiment reached a CUDA illegal-memory boundary. Packed Turbo4
is explicit diagnostic-only; automatic packed-copy updates were zero.

The cold page chain does complete for the organic A/B/A run: eligibility,
selection, host readiness, H2D, table publication, and completed target-graph
use are all recorded. A-again used logical pages
`[0,1,2,6,14,17,23,24]`, transferred 4,325,376 useful bytes, and set
`target_graph_used=true`. This is physical promotion evidence, separate from
answer quality.

## Cost attribution and implementation brief

Selected cold prefill is dominated by reference-route/selector work, host
summary/seal movement, and graph churn. The cached +64 append built 64
summaries (8.66 MB), issued 96 host-seal D2H calls (4.325 MB), and rebuilt and
submitted 24 logical graphs. Selected native trial 3 reported 573 graph
rebuilds. Native MTP rejection dominates the all-GPU and CPU-main-KV control
diagnostic; U64 is also slower than U128. Fine-grained kernel time is null
because CUDA event spans were not enabled.

The highest-value single code change is to give the selected cold/reference
working set a stable graph-friendly direct or packed CUDA execution plan,
removing repeated reference summary/seal work and graph rebuilds while
preserving the measured 64/64 append behavior.

## Prohibited work

Do not claim full occupied `C262144`, run occupancy advancement, or publish a
broad multi-prompt speed ratio until the all-GPU/CPU-main-KV zero-acceptance
state and proposal-target seam are explained and rerun. Keep automatic
multi-page direct and packed promotion prohibited until the illegal-memory
boundary has an independent proof. No missing diagnostic row was silently
converted to a ratio; `missing_rows` is empty.

Sources: `V10_84-01.json`, `V10_84-02.json`, `V10_84-02.md`,
`V10_84-03.json`, `V10_84-03_speed_attribution.json`, and the phase-83 summary.
