# Pager corpus v4 validation

## Result

The immutable fixture is `tools/server/bench/fixtures/pager-corpus-v4.json`.
The portable validator passed all 24 cases: 12 calibration and 12 held-out.
The corpus hash is
`37111506c8ddc7f9d04086121797cc9c2b7a2842cc1a56ef4b33f506c2252007`.
The pinned model and tokenizer hash is
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.

The Qwen tokenizer measured every prompt at its declared length: 4,096–22,016
tokens with no rounding slack. Logical-page counts cover 16–86, including odd
counts 57 and 63 and one-token tails. Needle distances cover 0, 2, 3, 7, 9,
11, 13, 15, 17, 19, and 21 pages. Every expected answer and fixture fact is
present in its prompt; partition construction differs only by the partition
identity line, and all prompt/stable hashes are unique.

The V3 corpus remains unchanged. V4 changes only the problematic warm-focus
fixture to an explicit natural-language answer that the model can copy from
the supplied fact. No held-out suffix or post-observation checker change was
introduced.

## Deterministic dense-input preflight

The complete 24-case reachability probe passed facts, expected-answer presence,
page-distance relations, exact declared tokenizer lengths, tails, and partition
identity. Raw prompts, per-case reachability records, provenance, and hashes are
preserved outside Git at:

`/srv/ai/paged-kv/results/17-06-corpus-v4-reachability-20260905T005240Z/`

## Model-backed probe

An isolated CPU-only Qwen3.8 dense server answered the calibration warm-focus
case exactly with the V4 answer. Its raw request/response and partial run are
preserved at:

`/srv/ai/paged-kv/results/17-06-corpus-v4-dense-20260905T004808Z/`

The managed CUDA service was left on its existing selective pager profile. A
probe against it returned refusal text for the two warm-focus cases, so those
responses are retained only as selective-route diagnostics and are not claimed
as dense-control evidence:

`/srv/ai/paged-kv/results/17-06-corpus-v4-dense-gpu-20260905T005113Z/`

