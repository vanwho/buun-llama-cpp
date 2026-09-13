# HOTPATH31 summary

Status: partial post-repair evidence; bounded scale is gated.

## Capability

The three repairs are locally supported by their focused CUDA/regression
verification: asynchronous direct metadata publication and ownership fencing,
authenticated layer-aware asynchronous promotion/publication, and stable
selected-direct graph reuse with dirty-driven pager policy. The retained
full-L native-MTP profile also completes isolated primary L8192/H4096 q0 and
q1 requests and a natural retained continuation through occupied C=5000.

The primary path is not proven stable. q0 and q1 each completed one matched
no-warmup cold-prefill request at 205.203/204.932 prefill tok/s and
36.466/36.382 decode tok/s, with 134 MTP proposals and 59 acceptances. The
natural ladder reached 5000 tokens; its 3800->4800 H crossing recorded
4,325,380 useful/aligned H2D bytes and 96 transfer completions.

## Remaining defects

The warm-control q0 measurement, q0 repetition, q2, and authenticated natural
cold recall all hit CUDA illegal memory access at
`ggml_backend_cuda_synchronize` during `llama_context::decode`, aborting with
status 6/ABRT. The two HTTP-success q0/q1 receipts still report one pager fault
and one eviction. Cold recall returned only `The retrieval topic`, not the
expected `cedar-orbit-17`, before the runtime fault.

Therefore there is no stable warm-controlled q0/q1/q2 proof, no validated
current natural recall, and no scale result. The conditional L32768/H16384
pilot and L131072 allocation/population smoke were correctly not run. This
summary makes no 256K, YaRN, quality, or soak claim.

## Exact next action

Isolate and repair the post-prefill CUDA illegal-memory-access at
`ggml_backend_cuda_synchronize`/`llama_context::decode`, then rerun the same
canonical L8192/H4096 warm-controlled q0/q1/q2 and authenticated natural cold
recall. Only a clean rerun validating `cedar-orbit-17` should unlock bounded
scale.

Machine-readable rates, counters, provenance, and evidence pointers are in
`HOTPATH31_SUMMARY.json`; detailed primary and scale-gate records remain in the
31-04 and 31-05 evidence files.
