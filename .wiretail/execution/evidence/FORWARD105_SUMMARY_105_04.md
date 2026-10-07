# Forward measured summary snapshot — 105-04

This immutable snapshot records the measured state reviewed by task105-04. The
convenience projection `forward/FORWARD_FINAL_SUMMARY.md` may be refreshed by
later tasks; this evidence copy remains fixed for the 105-04 receipt.

Overall goal status was not assessed because a candidate-matched CPU-KV speed
comparison was missing. That gap is not evidence of a known defect. 105-03b
attempt02 execution completed at L262144/C249293/H51200, B1024/U256 with
full-L GPU Turbo4 MTP; the numeric frontier was707 below requested C250000.
Exact C=L remained unproven. The candidate was built from source HEAD
`8f8faebacedff769e48a3ce80b7544d886a3a049` plus the captured dirty pager
diff; binary, loaded DSO and model identities are fixed in the paired JSON.

The 21 bulk occupancy rows are preserved in the JSON. At the final bulk row,
11878 fresh tokens over16990.276ms measured699.11 useful fresh tok/s;
`timings.prompt_n` was11910 executed tokens, or700.99 processed tok/s. The
short cached A2 tail was62 fresh/115 executed tokens, not bulk prefill.

All12 canonical requests on the occupied prefix completed. Median decode and
request-local MTP acceptance were43.32 tok/s/88.28% for merge,31.17/48.91%
(48.78% pooled) for mmap vs read, and40.75/79.78% for the watcher. The primary
fact and both additional source facts were correct with natural EOS. These
semantic results do not prove physical cold residency, rank, transfer or
target use; that optional identity-bound witness remains unknown. 105-02a
selected and dense exact-fact results are also semantically successful, with
the same physical-witness limitation.

The current route was `probe-rerank`; replay was coherent, but selector,
replay and promotion costs were not measured separately. Candidate-matched
CPU-KV comparison is missing. All15 goal rows and their compact evidence are
preserved in the JSON snapshot.
