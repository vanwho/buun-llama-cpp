# 49d-mtp-evidence-v10

Revision: `hotpath-v10-20260914`.
Tasks: `49-07`.

Make the existing canonical three-prompt benchmark fail closed when native
MTP was not actually executed or its request-scoped acceptance counters were
not observed. Keep startup placement and per-request acceptance distinct.
This is a benchmark/evidence adapter change; do not alter target attention to
manufacture acceptance or count feature-off rows as MTP.

Read `v10/TESTING.md`, task49-07, the existing profile runner boundary under
`/srv/ai/benchmarks`, and
`tools/server/bench/{run-pager-profile-benchmark.py,pager_benchmark_contract.py}`.
The site runner may be adjusted separately from portable repository code;
never commit credentials or host paths to upstream-facing sources. Keep the
original three prompts and raw journal/SSE artifacts.
