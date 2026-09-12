# Interactive 26-04 primary

The implementation connects hybrid/iSWA attention telemetry to the live pager,
canonical Turbo4 routing summaries, query capture, stale-identity handling,
graph reuse metadata, and bounded selected attention. It also adds per-reason
drop counters and a compact trace record, while keeping resident EMA separate
from cold-page observation.

Live candidate: PID 1163817, CUDA binary on port 8080, context 8192, page 256,
H=8, B=128, U=64, selective pager, no pinning or forced page, and native GPU
Turbo4 draft MTP. Startup logged the MTP reservation for 8192 rows. Port 8092
was not touched.

The clean query-aware primary returned HTTP 200 with 5834 prompt tokens and
128 committed output tokens. It completed 145/145 graphs, produced 89
attention samples covering 5154 sampled tokens and 643 sampled pages, read
106560 summary entries, and ended with 8 resident plus 7 host-backed pages.
Telemetry recorded 44 cadence skips, 1 no-output drop, and 11 nonfinite drops;
stale snapshot and stale identity drops were zero. The response and checksum
are recorded in `INTERACTIVE26_04_PRIMARY.json`.

The current query-aware run selected a direct-resident route and therefore has
zero H2D in its final counters. The separately retained same-profile natural
primary (`raw-primary-q0-final-fixed.json`) recorded 4,325,380 useful/aligned
H2D bytes, one fault, one eviction, 8 resident pages, 7 host pages, and the
expected ORBIT-417 fact. This distinction is explicit; the direct run is not
presented as a cold-promotion transfer.

Verification passed:

- CPU focused routing/telemetry CTest: 3/3.
- CUDA focused routing/telemetry CTest: 3/3.
- CPU and CUDA `llama-server` builds.
- `git diff --check`.

## Deferred verification

The existing routing-recall availability metric remains `not_measured`; no
recall percentage is claimed. The current query-aware final boundary had no
additional H2D, while the retained same-profile cold transition provides the
completed transfer evidence.
