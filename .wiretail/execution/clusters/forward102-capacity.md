# Cluster forward102-capacity

Revision: `hotpath-v10-20260914`. Amendment: `repo-context-scale-20261002-16k`.

Tasks in this context area: `102-03`, `102-04`.

Purpose: use the validated repo-text occupancy driver to progress from the
16K/8K baseline through 32K/16K and 128K, preserving one candidate-bound
frontier and measuring actual host-backed target KV, hot GPU KV and full-L
GPU Turbo4 MTP.

For the 128K logical test, target C=120,000 committed repository tokens,
leaving 11,072 tokens for the final query/output/replay/MTP reserve. If the
measured reserve is larger, stop at the safe frontier and record the shortfall;
never trigger compaction to reach the target.

Every selected-mode stage must cross its configured hot boundary by at least
2,048 committed tokens at 16K/32K and at least eight full pages at 128K/256K.
The live pager snapshot must also show positive host-valid rows and bytes plus
a nonresident host-backed page. Request accounting that only shows C>H is not
proof that data reached host backing. If OOM requires reducing H, preserve L
and all other settings, record the new H, and apply the same crossing rule.

Use the repo-content fixture/generator introduced by 102-02. Do not use
neutral filler for these capacity runs. Same Qwen3.8-27B UD-IQ4_XS build,
CUDA model execution, target/draft Turbo4, GPU full-L MTP, B=1024/U=256, one
slot and fixed runtime settings. CPU target-KV controls are required at the
16K baseline only; do not run CPU target-KV at 32K, 128K or 256K. At higher
contexts, compare all-GPU target KV only when a read-only admission estimate
shows it fits with scratch; otherwise mark that control not admitted and run
selected mode.

Keep a per-request candidate fingerprint and atomic occupancy checkpoint. Each
new request adds no more than 16K fresh rendered tokens to the same cached
conversation. Select a corpus stopping frontier from actual Qwen-rendered
tokens, not bytes or word counts. Before each request record
`reserve_tokens = max(final rendered query + requested output, measured
replay/MTP verify margin, minimum safety margin)` and enforce
`rendered_prompt_tokens + output_reserve <= L - safety_gap`. Use a safety gap
of at least 2K at 32K and at least 8K at 128K/256K; increase it when measured
query/replay/verify needs require. Verify context shifting/truncation is off.
The saved C is the actual committed frontier, never the requested token target.

Hot geometry: 32K uses H=16K. 128K starts H=59,904 tokens, the largest
256-token-page-aligned value <=60,000; maximum H is 60,000. If a selected run OOMs or runs out of scratch,
reduce only H in 256-token increments. Keep L, B/U, batch, ubatch, threads,
model, codecs, full-L draft and every other server setting unchanged. Recompute
derived G/R, relabel the measured H, and retry only the affected selected row.
Never shrink L or change batch geometry to make a capacity result appear to
pass. If full-L draft allocation fails independently of H, preserve that
distinct admission failure and continue with non-destructive diagnostics; do
not silently move draft KV to CPU or reduce its L.

Measure the first representative chunk's GPU work, CPU gaps, transfers, waits
and utilization. If it is slow but continues to make valid progress, continue
the occupancy proof with candidate-bound checkpoints; low speed alone is not
a reason to abandon the requested near-L history. Stop for a concrete runtime
or correctness failure, or genuine no-progress condition, then add its
source-directed repair and resume without refilling committed history. Do not
run the final multi-context speed curve here. Completion can report a valid
measurement below the performance goal; an actual behavior defect needs an
ordered repair task, not an evidence-only retry loop.
