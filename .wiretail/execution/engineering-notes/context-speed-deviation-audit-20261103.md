# Adjacent-context performance deviation audit — 2026-11-03

The checkpoint ladder is unchanged. L is server allocation, C is actual occupied
context, and H is requested GPU hot capacity. In particular, the saved 61K point
is C=62,464, the 77K point is C=78,848, and H=65,536 in the ACO comparison.

## Evidence and screening method

The audit uses completed canonical rows in:

- `/srv/ai/paged-kv/results/forward/benchmark-reruns-20261101`
- `/srv/ai/paged-kv/results/forward/benchmark-reruns-corrected-20261103`

Compare the same build, placement, MTP state, reasoning mode and original prompt
across adjacent occupied-context points. Use three-measurement decode medians.
Context-ingest speed comes from `campaign/schedule.jsonl` and the row's context
fill summary, not the short question suffix's prompt-processing rate. Do not
compare raw prompt-token count with server allocation or mix original-template
and shared-prefix-template streams without recording the protocol difference.

A screen flags a >=25% drop when C grows by <=35%, or an isolated >=25% dip and
rebound. This is a triage threshold, not a validity or task acceptance gate.
Exclude the known two-token `I` answers at C=196,608/P3/off from CPU decode
comparisons. Do not discard valid data simply because it exposes a slow route.

## Confirmed ACO boundary cliff

All points below used the same fork executable SHA-256
`00aba66bd1d8e4db5d833bfb8321f6b8f515a287dfab5bebd9a23df69a34ca31`,
L=262,144/H=65,536, Turbo4 K/V, B=1024/U=256. Both compared cells completed
12/12 requests and all nine measurements; their cached base frontiers were
62,465 and 78,849. These are reasoning-off medians for the original prompts:

| MTP | Prompt | Decode at C=62,464 | Decode at C=78,848 | Change |
|---|---|---:|---:|---:|
| off | 1 | 42.32 | 14.86 | -64.90% |
| off | 2 | 42.33 | 15.02 | -64.51% |
| off | 3 | 42.21 | 14.93 | -64.63% |
| on | 1 | 79.77 | 42.75 | -46.41% |
| on | 2 | 59.48 | 34.74 | -41.59% |
| on | 3 | 74.41 | 41.10 | -44.76% |

The same cliff occurs in the independently filled low/medium/xhigh MTP-on
series and in low MTP-off. Across modes, the median of prompt medians drops
68.54 -> 35.98 tok/s with MTP (-47.5%) and 42.26 -> 14.86 without (-64.8%).
MTP-off remains around 14.8–15 tok/s at later C values, rather than recovering.
This is a repeatable change near C>H, not an isolated noisy repetition.

The corresponding context-fill medians do not have a cliff: approximately
292.4 -> 302.9 tok/s with MTP and 308.8 -> 328.2 without. The decode cost must
not be described as a simultaneous collapse in measured ingest speed.

MTP acceptance also does not explain it. In reasoning-off, P1 acceptance is
93.5% -> 90.1%, P2 57.3% -> 64.5%, and P3 84.8% -> 85.4%. Those changes are
small relative to the throughput drop; the MTP-off cliff is larger still.

Existing five-second GPU samples support a utilization change: the MTP-off
reasoning-off interval has median utilization 95% -> 57% and power
259.8 -> 167.9 W, with used VRAM 14,541 -> 14,553 MiB. These interval medians
include idle gaps and are not request-only GPU busy time. The recorded compute
process lists show only the tested server in MTP-off and MTP-on low/medium/xhigh.
MTP-on reasoning-off's primary sampler failed; its fallback lacks process lists,
so its isolation evidence is weaker. Do not invent a resident-page census from
these resource samples or cumulative promotion counters.

Exact row evidence is under
`aco-mtp-off-H65536-L262144-reasoning-off-fixed-helper/C{62464,78848}/campaign/`
and `aco-mtp-on-H65536-L262144-reasoning-{off,low,medium,xhigh}/C{62464,78848}/campaign/`
within the corrected root above.

## Source-directed explanation and remaining uncertainty

`build-cuda/common/build-info.cpp` records source `a5a1110ca`. The relevant C++
files are unchanged between that commit and this audit. No runtime code,
build, batch geometry, KV type, or forced attention route was changed here.

1. `llama_kv_attention_execution::planned_route()` in
   `src/llama-kv-attention-execution.cpp` first checks complete causal history.
   It selects `selected_dense` for eligible, fully resident contiguous history.
   When an old causal page is missing, decode and MTP verification prefer
   `selected_packed`; batched prefill can instead choose selected direct.
2. `llama_kv_attention_dense_view_check()` in `src/llama-kv-attention-op.cpp`
   requires contiguous logical and physical page order. Reusing/evicting slots
   can therefore remove the mature dense-view eligibility after crossing H.
3. `llm_graph_context::build_attn()` in `src/llama-graph.cpp` copies every
   selected K and V page for every attention layer on every packed graph
   execution. The source explicitly says to copy every selected page, including
   current rows, through the cache-write dependencies. This preserves the
   correctness repairs; it adds per-page device copy nodes and dispatch work
   that selected-dense views avoid.
4. `llm_graph_input_attn_kv::can_reuse()` also rejects a packed graph when the
   physical/layout key or page-rounded selected capacity changes. A physical
   membership change can rebuild a graph; row-tail growth alone is excluded
   from the physical key. Do not claim a rebuild on every token without counts.

This is a concrete mechanism for a hot-boundary performance change and a
strong candidate for the observed cliff. Existing artifacts do not measure
how much is copy/dispatch, graph rebuilding, host metadata work, or GPU kernel
execution. Do not blame host transfers alone, assume a reference fallback,
change precision, or switch production routes speculatively.

## Apparent 256K prefill outlier resolved by matching intervals

The standalone low/MTP-on replacement's whole-run fill median was
272.18 tok/s, while other 256K checkpoint summaries were about 183 tok/s.
Those summaries covered different ranges: the standalone replacement includes
all appends from zero, while the continued series' 256K stage covers only the
last interval after C=196,609. Comparing those aggregate medians is invalid.

Using only appends wholly after C=196,608 and summing new tokens / prompt time:

| Mode | Actual compared C range | Weighted net-new ingest tok/s |
|---|---|---:|
| off | 196,609–256,001 | 222.77 |
| low replacement | 199,169–256,001 | 218.96 |
| medium | 196,609–256,001 | 223.19 |
| xhigh | 196,609–256,001 | 223.06 |

The replacement is within about 2% of the other runs on this matched late
interval. Its first crossing chunk is excluded rather than prorating time.
Retain its valid decode/MTP data; no whole 256K rerun is justified by this
aggregation difference. Future curve reports should retain actual interval
bounds and use weighted tokens/time alongside per-append observations.

## Ordinary controls

No non-ACO group met the adjacent-drop or isolated-dip screen in context-fill
or decode. For example, master GPU/MTP-off moves from 42.89 -> 41.19 tok/s
decode and 1,301.1 -> 1,156.8 context-fill tok/s between C=62,464 and 78,848.
Fork GPU/MTP-off moves 42.99 -> 41.36 decode at the same occupied points.
These are much smaller changes than ACO's cliff. CPU curves with large gaps
between recorded C values cannot locate a sharp local boundary; their slower
large-context results alone do not establish an anomalous dip.

## Focused repeat and outcome rules

Two focused investigations are frozen in the remaining-matrix manifest:
`aco-hot-boundary-mtp-off` and `aco-hot-boundary-mtp-on`. Each starts clean,
uses L262144/H65536/B1024/U256, fills incrementally to C62464 and then C78848,
and runs reasoning-off with the original three prompts, one 40-token warmup
and three 400-token-budget measurements per prompt. That is 48 canonical
requests total across both MTP settings. The main checkpoint ladder is unchanged.

The harness now persists the before/after snapshots it already collected and
records request wall-clock epochs. Snapshot file writes are outside the timed
request. No CUDA hot-path profiling, forced routes, extra per-token scraping,
or model rebuild is enabled. Read route counters/selected-row fields if present;
missing diagnostic fields do not invalidate otherwise complete timing results.

- If the cliff repeats with a route-family change, keep the real slow results
  and attribute the boundary at route-family level. Kernel/copy/CPU fractions
  remain unproven until a separate bounded profiler observation.
- If it disappears, compare exact argv, environment, source selection and
  active route before deciding which older cells need replacement.
- If a foreign compute client or setup/cache failure occurs, reject only that
  attempt, correct setup and repeat within the declared recovery budget.
- If acceptance remains stable while decode falls, prioritize selected-route
  overhead rather than another speculative MTP-accuracy repair.
- Do not optimize by removing the cache-write dependency or reusing stale
  packed content. Candidate improvements are batched encoded-page gathers and
  safely retained per-layer packed working sets, subject to measured time and
  a realistic VRAM budget. They are not implemented by this audit.

### Narrow optimization candidates after the frozen measurements

The packed graph creates two copy nodes per selected page per attention layer.
At 256 selected pages that is 512 graph copy nodes per attention layer, compared
with no such copies in the dense view. This is a source-level node count, not
a measured CUDA launch count; backend fusion/capture and timings must not be
invented from the count.

First consider coalescing adjacent `layer.copies` spans during graph construction
in `src/llama-graph.cpp`: merge only when both source K/V byte ranges and compact
destination ranges are adjacent, representations/strides match, and no invalid
tail gap is crossed. Keep the existing dependency through `cache_k_write` and
`cache_v_write` and the encoded tail initialization. That preserves byte-exact
Turbo4 data and the current one-layer workspace while potentially replacing
many per-page copy nodes with a few contiguous-span copies. Do not assume how
many runs the real physical map has without inspecting its descriptors.

If physical fragmentation prevents useful coalescing, a batched encoded-page
gather can copy all descriptors with one/few launches per layer. It should copy
opaque Turbo4 blocks without dequantization/requantization, use persistent
descriptor addresses for graph replay, and run after that layer's cache writes
and before its mature FA consumer. A single gather for all model layers before
their current rows exist would violate dependencies. This approach retains the
same bounded workspace rather than duplicating H-sized persistent buffers for
every layer. It is a candidate optimization, not a change authorized or measured
by this audit.

Focused raw results are under
`/srv/ai/paged-kv/results/forward/final-remaining-20261103/aco-hot-boundary-mtp-{off,on}/`.
The completed repeat findings are appended below.

## Completed focused repeat

Both series exited successfully and all 48 canonical requests passed. Configuration was unchanged between the two points of each series; no runtime rebuild or forced route was used. Actual C was 62,465 and 78,849.

| MTP | Prompt | Decode at 61K | Decode at 77K | Change |
|---|---|---:|---:|---:|
| off | 1 | 42.36 | 14.91 | -64.81% |
| off | 2 | 42.37 | 15.00 | -64.60% |
| off | 3 | 42.21 | 15.00 | -64.47% |
| on | 1 | 78.85 | 43.09 | -45.35% |
| on | 2 | 56.25 | 30.98 | -44.94% |
| on | 3 | 72.25 | 40.09 | -44.52% |

All measured after-snapshots identify **selected dense** at 61K and **selected packed** at 77K. At 61K, active target rows ranged about 62.7K; at 77K about 65.0K, with the same 256-page physical capacity. Thus the attended target extent grows modestly while decode time per token nearly triples without MTP.

MTP acceptance stayed close across the repeat: P1 91.67% -> 90.12%, P2 51.74% -> 49.37%, P3 81.25% -> 80.92%. MTP verification at 77K also uses selected packed.

Measured requests at 77K had 0–34,603,008 H2D useful bytes. In particular, P2 repetition 3 had zero H2D transfer in both MTP settings yet remained slow. This rules out cold-page transfer as the complete explanation for steady-state decode. It does not time or eliminate all pager policy/metadata costs.

| MTP | Actual late fill window near 61K | Net-new prefill / wall tok/s | Actual late fill window near 77K | Net-new prefill / wall tok/s |
|---|---|---:|---|---:|
| off | 46,361–62,465 | 284.18 / 282.22 | 62,465–78,849 | 325.84 / 322.14 |
| on | 46,361–62,465 | 265.91 / 264.06 | 62,465–78,849 | 298.49 / 294.69 |

The fill table uses weighted new tokens / measured prompt time, and separately new tokens / request wall time, over similar late-ingest windows. It is not the tiny question suffix processing rate.

Five-second resource samples filtered to measured requests show GPU utilization falling 97% -> 55% without MTP and 90% -> 78% with MTP. GPU memory is nearly flat across the boundary; no OOM occurred. Sampling is sparse (11/36 samples without MTP, 7/15 with MTP), so these are observations, not a CUDA timeline. Detailed copy/kernel/CPU timing counters were intentionally disabled and are unmeasured.

Keep the earlier complete slow rows. The rerun corroborates them; neither smoothing them away nor repeating the entire matrix would resolve the route cost. The next performance work should target the packed path above and preserve byte-exact Turbo4/cache-write ordering. The ordinary remaining matrix remains available and the focused investigations are now marked complete in their external state files.

Machine-readable paired results, interval definitions and hashed row paths are in `context-speed-deviation-audit-20261103.json`.
