# 18-02 exact rendered-token sizing evidence

## Result

PASS. The three benchmark harness families now use one shared padding-only token fitter. Live rendering is delegated to the target server’s `/apply-template` and `/tokenize` endpoints, so template specials, Unicode, punctuation, tools, and generation reserve are counted before submission. Completion usage is compared with the preflight count and mismatches are rejected.

## Implementation

- `pager_benchmark_contract.fit_prompt` accepts complete messages, desired occupancy, generation reserve, and an injected render/tokenize callable. It returns exact IDs/count, template/tokenizer identity, request hash, and rendered fact offsets.
- `run-quality-corpus.py` splits only the explicit V4 tail padding, fits it, records configured/resolved/occupied/reserve fields, and rejects server usage mismatches before scoring.
- `run-pager-soak.py` uses the same fitter for every synthetic request and cancellation payload; legacy word-shaped text is only an upper-bound padding hint.
- `run-pager-profile-benchmark.py` no longer exports `BENCH_CONTEXT_WORDS`; the site canonical runner uses the server input-token endpoint for exact preflight and compares usage after completion.
- `/srv/ai/benchmarks/run-profile-benchmark.sh` records configured 77,824 versus resolved 22,016 separately and admits requests only when exact prompt plus reserve fits resolved capacity.

## Verification

- 37 focused Python tests passed, including empty padding, mandatory overflow, Unicode/punctuation, tool messages, fact offsets, page-tail capacities 16,384/22,016/262,144, and the 53,797 overflow regression.
- Live target request returned HTTP 200 with local count 32, reserve 16, and server `usage.prompt_tokens=32`.
- Canonical site smoke completed six requests with 6/6 exact preflight/usage matches at resolved capacity 22,016; no request errors and no 8092 reference.
- Quality diagnostic completed two requests with 4,088 local/server prompt tokens plus 8 reserve inside 4,096. The two answer mismatches are semantic diagnostic results, not sizing mismatches.
- Python compilation, site shell syntax, and `git diff --check` passed.

## Raw evidence

Machine-readable receipt: [18-02_SIZING.json](18-02_SIZING.json). Bulk live artifacts are under `/srv/ai/paged-kv/results/18-02-sizing-20260905T0845Z/`, `/srv/ai/paged-kv/results/18-02-site-20260905T0900Z/`, and `/srv/ai/paged-kv/results/18-02-quality-live-20260905T0915Z/`.

## Deferred verification

No human or upstream action is required. The full 22,016-token quality matrix and final speed curve were intentionally not run in this sizing task. The corpus’s declared tokenizer hash is a model-hash placeholder; live server template/tokenization identity is recorded and authoritative.
