# Cluster forward102-capacity

Revision: `hotpath-v10-20260914`. Amendment: `repo-context-scale-20261002-16k`.

Tasks in this context area: `102-03`, `102-04`.

Purpose: use the validated repo-text occupancy driver to progress from the
16K/8K baseline through 32K/16K and 128K, preserving one candidate-bound
frontier and measuring speed with hot GPU target KV and full-L GPU Turbo4 MTP.

For the 128K logical test, target C=120,000 committed repository tokens,
leaving 11,072 tokens for the final query/output/replay/MTP reserve. If the
measured reserve is larger, stop at the safe frontier and record the shortfall;
never trigger compaction to reach the target.

Every selected-mode stage must commit fixture/repository content beyond its
configured hot boundary by at least 2,048 tokens at 16K/32K and at least eight
full pages at 128K/256K. This is the required occupancy condition. Host-valid
rows/bytes, page inventory, residency, promotion, and transfer counters are
optional diagnostics: record exposed values, but missing or incomplete
telemetry is never a failure, retry trigger, or completion gate. Do not claim
exact host backing or promotion unless directly observed. If OOM requires
reducing H, preserve L and all other settings, record the new H, and apply the
same crossing rule.

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

Hot geometry: 32K uses H=16K. The completed 102-04 128K run admitted only
H=16,384 from its then-requested 59,904 cap; this is historical evidence, not
the target for the new higher-context runs. Tasks 102-05 and 102-06 must both
fill/use H=51,200 (200 pages). Do not lower H to recover OOM or scratch.
Preserve L, B/U, batch, ubatch, threads, model, codecs and full-L draft; first
verify a clean single-process start, effective options and memory accounting.
If genuine capacity still prevents admission of all 200 pages, record
not-admitted rather than running a smaller hot window.
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
