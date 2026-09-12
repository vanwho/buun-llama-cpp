# SPEED25_06_BATCHING

Status: implementation and locally executable verification pass, with the
unavailable GPU-utilization and bounded-H parity checks deferred explicitly.

## Chosen approach

Whole-model prefill now separates requested batch size, physical writable
capacity, and CUDA query tile size. The effective batch is
`min(requested, physical_pages * page_tokens)`; the query tile remains an
independent CUDA concern. Write pages are bulk-reserved before graph
submission and rolled back in reverse order on failure. Direct prefill takes
precedence over packed routing when its shape is supported, while dense remains
the control route. Target rows and MTP verification rows are counted once at
the shifted pending boundary.

## Verification

- Host focused tests passed:
  `cmake --build build --target test-kv-attention-execution test-kv-pager test-kv-pager-model -j2`
  and `ctest --test-dir build -R 'kv-(attention-execution|pager)' --output-on-failure`.
- Synthetic native-MTP model-free coverage passed:
  `./build/bin/test-kv-pager-model --mtp native`.
- CUDA build passed for `llama-server`, `test-cuda-fattn-paged-turbo4`, and
  `test-kv-pager-model`; the CUDA fixture passed on an RTX 4080.
- Fixture timings and serial/split controls are in
  `raw/SPEED25_06_BATCHING_cuda.txt`.
- The live native-MTP server completed the V6 micro request at requested
  B=256 and B=512. Both runs had effective B greater than 64, one prefill
  subbatch, and complete graph submission/completion counts.

## Measured live runs

| requested B | effective B | physical capacity | graph submissions | prefill direct routes | prompt tok/s | MTP accepted/generated |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 256 | 4096 | 23 | 9 | 463.00 | 18/25 |
| 512 | 512 | 4096 | 18 | 5 | 453.91 | 19/23 |

The B256 driver elapsed time was 5.1749 s (TTFT 4.4509 s); B512 was 5.2222
s (TTFT 4.5400 s). Decode throughput was 43.56 and 46.20 tok/s respectively.
The reported `target_tokens_processed` metrics (2099 and 2096) are cumulative
server-lifetime counters, so they are not presented as request-only counts.

## Source and raw roots

- Working-tree implementation diff SHA256:
  `6d514892f48f6413291cadead99848da8e22ef14999bc5e928fcbdaa1d3f8e49`.
- Runtime bundle:
  `/srv/ai/paged-kv/results/25-06-runtime-bundle-20260912T064000Z`.
- B256 raw root:
  `/srv/ai/paged-kv/results/25-06-short-final-20260912T064000Z`.
- B512 raw root:
  `/srv/ai/paged-kv/results/25-06-short-ub512-20260912T064000Z`.

## Deferred verification

- No trustworthy sampled GPU-utilization trace was available; graph launch
  submission/completion counts and prompt/decode throughput are recorded.
- The deliberately bounded-H standalone model probe exercises the direct path,
  but its dense numerical comparison is not yet a valid parity result because
  bounded residency/cold recall is not a complete equivalence path. No parity
  pass is claimed.
- No 1024 or 256K run was attempted.

## Next command

`PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp python3 /srv/wiretail/task_state.py validate`
