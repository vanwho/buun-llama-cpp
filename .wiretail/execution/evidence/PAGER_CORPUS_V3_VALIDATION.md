# Pager corpus v3 validation

## Result

The checked-in corpus is `tools/server/bench/fixtures/pager-corpus-v3.json`.
The portable validator passed all 24 cases: 12 calibration and 12 held-out.
The corpus hash is `cbf1e3bf727f0f52e8d0e922a68af9cd70bf7ded48b4402ae1ef9328921ac52e`.
The model and tokenizer hash recorded in every case is
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.

The Qwen tokenizer command measured every prompt. Declared lengths cover
4096–22016 tokens; the largest absolute measured/declaration difference is 18
tokens. Logical-page counts cover 16–86, including odd page counts and a
non-page-aligned tail. Needle distances cover 0, 2, 3, 7, 9, 11, 13, 15, 17,
19, and 21 pages. Every expected answer is present in its fixture prompt, and
calibration/held-out prompt hashes are distinct while fixture IDs and answers
remain balanced.

## Dense-control preflight

The deterministic preflight found the expected fact in all 24 prompts and
verified each needle page/distance relation. This is only input reachability;
it is not a model response or a quality score.

Raw prompt source and per-prompt hashes are preserved in the checked-in JSON.
Model-backed dense-control responses for all cases and the selected-all
reference fixture are deferred because `BENCH_ENDPOINT` and
`CANONICAL_BENCHMARK_RUNNER` are not configured in this environment. The
later live run must preserve raw requests/responses and compare dense against
selected-all before pager tuning.
