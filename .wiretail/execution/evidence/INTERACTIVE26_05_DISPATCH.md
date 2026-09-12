# Interactive 26-05 dispatch

The production policy is dense first when the selected pages form a contiguous
causal prefix. Otherwise it keeps direct for prefill query batches up to two
64-token CUDA tiles and for one-query decode/native-MTP verification. Packed
Turbo4 is selected only for larger non-contiguous prefills, where its bounded
duplicate is amortized. The policy keeps the model UBatch and CUDA query tile
separate and never rewrites native RoPE positions.

The RTX 4080 fixture measured direct paged Turbo4 at 22.044 ms for 24 query
heads and 530 selected rows; Q1/Q2/Q3/Q5 decode measured 0.517/0.529/0.591/1.075
ms. The 64-query tile sweep was 2.601/4.045/7.641 ms for tiles 16/32/64;
256-query was 6.275/8.820/14.180 ms and 512-query was 12.360/13.774/21.967
ms. Split-KV capacity 3 was 20.243 ms versus 21.960 ms serial, while the
capacity-16 control was 20.225 ms. These end-to-end/fixture results retain
direct for the normal U64 primary shape.

Three matched automatic primary trials completed HTTP 200 with 5007 prompt
tokens, native GPU Turbo4 MTP, 237 accepted draft tokens out of 286, and
382/382 graph submissions/completions. The final metrics selected direct for
237 prefill decisions and direct for 144 MTP verification decisions, with 8
resident and 7 host pages, zero faults/evictions, and zero additional H2D.
The retained natural-recall artifact remains the quality proof for this same
profile: ORBIT-417, 5995 prompt tokens, 86 committed tokens, 72 MTP drafts and
49 accepted, with one cold fault and one eviction.

The forced packed control used the same selected primary page shape and
completed after view initialization was added for graph-owned page views. It
reported 34,603,000 bytes of bounded duplicate Turbo4 storage, 8032 changed
page-copy updates covering 1,932,530 rows, one in-flight consumer, and no
faults or evictions. Unsupported dense forcing refused explicitly in the
backend-neutral route test. Packed storage is exposed separately from
cumulative pack transfer work so it cannot be mistaken for an unbounded
allocation.

Raw live artifacts are under
`/srv/ai/paged-kv/results/interactive-26-05-dispatch/`, including the three
primary responses, automatic metrics, and forced-packed metrics. The retained
quality artifact is `/srv/ai/paged-kv/results/interactive-26-04-primary/raw-primary-q0-final-fixed.json`.

## Deferred verification

None. CUDA fixture and live primary checks were available and executed. Port
8092 was not changed.
