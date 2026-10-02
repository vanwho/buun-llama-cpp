# Repository-content context and promotion fixture v1

This is a deterministic, repo-grounded A→B→A workload for Qwen3.8-27B.
Unlike the older generated merge/mmap/watcher corpus, its prompt bodies are
read from ordinary tracked files in this repository. The manifest freezes the
source paths and SHA-256 identities; the runner must record the exact rendered
Qwen token counts for the complete conversation, not estimate from words or
bytes.

Turn A loads the listed code files and asks for generated C++ control flow
grounded in the tokenizer implementation. Turn B adds the listed documentation
files and asks about the benchmark sizing contract. Turn A2 does not resend the
code files; it asks for generated C++ tests about a different behavior in the
original tokenizer source. Both A turns require code generation to exercise
MTP on realistic structured output. Use the same `max_tokens=400` cap for all
responses and prompt for substantive answers near 300–400 tokens without
requiring exact lengths. Keep all turns in one server slot and one chat
history.
Record whether the tokenizer-source page was actually host-backed before A2,
whether the natural selector nominated it, whether the policy admitted it,
whether H2D completed and published, and whether the target attention consumed
it. Answer correctness alone is not a promotion proof.

For the 16K/8K baseline, use the complete primary A and B file groups first.
Before any request, render and count the whole prompt using the Qwen tokenizer.
Do not send if the projected context plus the recorded output/query/replay
reserve exceeds L. If the full files leave too little reserve, omit only
optional files in the order listed as supplemental; never trim the primary
tokenizer implementation or the benchmark formulas. Record any omitted file.
To ensure actual paging pressure, append the next non-overlapping scale-corpus
chunks with the B documentation in the same request until the complete
conversation reaches L minus the measured reserve (within one 256-token page).
Use identical bytes/order for the GPU, host and selected comparisons; never
cross the reserve to force H eviction. Require selected C>H and report exact C.

For larger occupancy runs, enumerate tracked files with `git ls-files -z` and
sort paths bytewise. Include readable UTF-8 files under `src/`, `tools/`,
`docs/`, `examples/`, or the root with suffix `.c`, `.cc`, `.cpp`, `.cxx`,
`.h`, `.hh`, `.hpp`, `.hxx`, `.cu`, `.cuh`, `.metal`, `.py`, `.sh`, `.bash`,
`.cmake`, `.md`, `.txt`, `.yml`, or `.yaml`; also include root
`CMakeLists.txt`, `Makefile`, `CONTRIBUTING.md`, and `LICENSE`. Exclude
`.git/`, `.wiretail/`, all `build*`, output, vendor, third-party, generated,
dependency and model trees, binaries, minified assets, symlinks escaping the
repository, and any individual file larger than 1 MiB. At run start, record
the candidate commit and dirty-tree fingerprint plus the ordered paths, exact
source hashes, bytes, included line ranges, and rendered Qwen token counts.
Serialize each included range as a path header plus exact file text. Split
only at UTF-8 line boundaries. Append successive non-overlapping chunks to
the same slot; do not replace prior conversation with a single huge prompt,
repeat content, or use neutral filler to stand in for repository content.
Keep each request's *new* rendered tokens at or below 16K, choosing smaller
chunks when measured to be more efficient or needed to preserve reserve.

The exact prompt wording and acceptable semantic answer points are in
[`prompts.md`](prompts.md); the source ordering and checksums are in
[`manifest.json`](manifest.json). Refresh a source hash only when deliberately
creating a new fixture revision.
