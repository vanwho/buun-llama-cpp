# 13-01 performance profile

This is the frozen calibration and budget record for phase 13. It contains no
optimization. The machine-readable source is
[`PERFORMANCE_PROFILE.json`](PERFORMANCE_PROFILE.json).

## Build and measurement contract

The Release CUDA build uses `GGML_CUDA=ON`, CUDA graphs, `89-real`, and the
unmodified `-O3 -DNDEBUG` configuration. The server and CUDA library are
unstripped and retain `.symtab`; their hashes, model hash, GPU, driver, and
toolkit are in the JSON record. Sampling is greedy (`temperature=0`, seed 42),
with one warmup and five measured requests per prompt.

The four corrected same-build controls provide 15 measured requests each (three
prompts × five), all with zero errors. Their decode medians are recorded with
raw SHA-256 hashes in the JSON artifact. They establish repeatable control
timing and context sensitivity, but are not selective pager measurements.

## Frozen shapes

The calibration matrix covers all-pages-fit, warm `H<L`, cold promotion, focus
shift, churn, odd 17-token tail, short/long prefill, and exact waves. Runtime
`H`, selected pages/rows, logical gaps, fault batches, and coalesced runs must be
read from each live trace; the matrix never installs a fixed hot-page default.
The bounded CUDA fixture currently available is four query heads, one query,
529 selected rows, and three physical pages. It measured five kernel events with
a 20.952 ms median and is not an end-to-end quality or speed claim.

## Ranked costs and budgets

The current evidence ranks direct attention first, followed by descriptor/table
preparation, page movement/waits, and graph/policy/server overhead. Only the
first is measured on CUDA today; the others are explicitly low-confidence until
the live pager route emits component counters. Budgets and one regression test
per optimization hypothesis are in the JSON record. The three hypotheses are
metadata/run coalescing, validated-geometry vector-load specialization, and a
bounded pinned-ring/batched transfer path.

## Component accounting

Raw controls account for HTTP, prompt, and decode timings. Production pager
counters are wired, but no live selected run populated graph, descriptor,
mass-reduction, policy, D2H/H2D, event-wait, seal, or recurrent timing fields.
Those components are therefore labeled unavailable/unattributed; no total time
decomposition or selective speed claim is inferred.

## Raw paths and deferred verification

Raw benchmark directories remain outside Git under `/srv/ai/paged-kv/results/`.
The intended live profile root is `/srv/ai/paged-kv/results/13-01/`.
Nsight Systems/Compute exports, five-trial traces for every selected shape,
live transfer overlap, and model-backed selective profiles remain deferred to
the configured pager service and later phase-13/14 gates. The direct CUDA
fixture resource and sanitizer evidence remains linked from the existing CUDA
matrix.
