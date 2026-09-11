# SPEED25_02_ATTRIBUTION

Result: `pass` for the short speed receipt and paired live control. Both cases used the same read-only bundle, source diff, model, tokenizer/template, L=4096, exact 2048-token prompt fit, zero warmups, one cold trial, and max output 128.

## Critical-path attribution

| Rank | Stage | Off control | Selective | Evidence / uncertainty |
| --- | --- | ---: | ---: | --- |
| 1 | Prefill wall | 1.185 s | 14.065 s | Server timing; selective includes correctness-first selected path |
| 2 | Decode wall | 1.294 s / 128 tok | 14.119 s / 68 tok | Server timing; selective stopped at EOS |
| 3 | Summary builds | not applicable | 274,080 calls / 578.858 MB source | Before/after counter delta; count is not kernel time |
| 4 | Host seal D2H | not applicable | 616 calls / 2.664 GB | Before/after counter delta; bytes are useful page payload accounting |
| 5 | Seal lifecycle | not applicable | 2,666 calls; 23,994 scanned; 17,092 changed | Before/after counter delta; no direct wall span exported |
| 6 | Graph decision | not applicable | 72 us; 59 logical capture/rebuild/submission events | Backend-neutral logical decision timing, not CUDA timing |
| 7 | CUDA capture/update/launch | unavailable | unavailable | Explicit null + reason; CUDA event spans were not enabled |

Throughput derived from the recorded wall times is approximately 1,728 prompt
tokens/s and 99 decode tokens/s for the off control, versus 146 prompt tokens/s
and 4.8 decode tokens/s for the selective request. This is an attribution
diagnostic, not a rate gate; output lengths differ because the selective case
reached EOS at 68 tokens.

The selective result shows the suspected cost center: repeated clean-page
summary construction and host sealing dominate the observed lifecycle counts.
No claim is made that `total_token_us`, graph construction, or any receipt
counter is CUDA kernel time. Quality/page-mass fields were intentionally not
collected by this speed-first suite.

## Reproduction

Driver help:

```text
python3 tools/server/bench/run-final-curve.py --help
```

Measured selective invocation:

```bash
PAGER_BUNDLE_ROOT=/srv/ai/paged-kv/results/25-02-runtime-bundle-20260912T041000Z \
python3 tools/server/bench/run-final-curve.py --suite micro --context 4096 \
  --prompt-tokens 2048 --question-index 0 \
  --output /srv/ai/paged-kv/results/25-02-short-selective-final-20260912T043000Z \
  --endpoint http://127.0.0.1:8080 --api-key-file /srv/ai/config/llama/api-keys \
  --model qwen38-fast-turbo4-mtp --mode selective --warmups 0 --trials 1 \
  --max-tokens 128 --reserve-context 512 --cache-condition cold-prefill \
  --slot-id 0 --server-bin \
  /srv/ai/paged-kv/results/25-02-runtime-bundle-20260912T041000Z/bin/llama-server
```

The feature-off invocation is identical except for `--mode off` and its raw
output root. Resume uses the durable `CaseStateStore` checkpoint and complete
provenance key; raw SSE is flushed and fsynced while it streams.

## Deferred verification

CUDA event spans/nsys profiling were not enabled in this short run, so real
CUDA capture/update/launch timings remain deferred to 25-03. The 262K
population and pressure suite are intentionally deferred; this task has no
such completion gate.
