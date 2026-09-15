# V10 72-01 runtime evidence

Revision: `hotpath-v10-20260914`. The bounded controlled-model query ran
against the resolved installed model `/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf`
(requested through `/srv/ai/models/text/current.gguf`), model SHA-256
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.

## Build and fixture identity

The tested source tree was at commit
`49052e9f8845f63d5c0b12027e145435c6adb69c`; the diagnostic JSON serializer
change for this task was present in the working tree. The rebuilt CUDA test
binary and linked project DSOs were:

| file | SHA-256 |
|---|---|
| `build-cuda/bin/test-kv-pager-model` | `6e384f90b85c558a2ea0f7fafbdb45ea12399779d6ca587dc7279cb012980be3` |
| `build-cuda/bin/libllama.so` | `5dbc7432bb54a8fd1b928b9c2f960679d2b437ce047a50bd20fff19c4849aea1` |
| `build-cuda/bin/libggml.so` | `06b1a11fac2674dd6fef5048a440fc9b5b30757426db437194846c1bab03439b` |
| `build-cuda/bin/libggml-cuda.so` | `4737527588aa7a9d9923e3ca58ca52863bbb5e9ad00e8aea6c7b1c04b4e17c42` |
| `build-cuda/bin/libggml-base.so` | `4213509f34262146732909b2279369e3f541401685e872d5c2896295b8c11e83` |
| `build-cuda/bin/libggml-cpu.so` | `a7a77b270aea94f18143b0180cb651b25b7b0c078c156baa7c43dbabd8aaca56` |

The captured token fixture was the existing 6,144-token file
`/srv/ai/paged-kv/results/v10/51-01/20260914T-live-proof-speed-v10/model-tokens.txt`,
SHA-256 `2cd38048ca2e8bacb6f189399b78ad0920a5945c56267fcbcdae2fe9e1e4e7cf`.

## Command and measured proof

The rebuilt test was run with CUDA available and the managed Qwen service
stopped for isolation, then the service was restored. The exact test argv was:

```text
build-cuda/bin/test-kv-pager-model --model /srv/ai/models/text/current.gguf --tokens-file /srv/ai/paged-kv/results/v10/51-01/20260914T-live-proof-speed-v10/model-tokens.txt --context 8192 --n-batch 128 --n-ubatch 64 --hot-pages 16 --generate 16 --mtp off --selected-only --output /srv/ai/paged-kv/results/v10/72-01/20260915T052500Z/controlled-model-query-final.json
```

The existing internal `LLAMA_KV_ATTENTION_ROUTE=packed` diagnostic setting
selected the persistent packed bridge; it is not a public API and no
`--force-page` or client candidate ID was supplied. The executable exited 0.
The final JSON is
`/srv/ai/paged-kv/results/v10/72-01/20260915T052500Z/controlled-model-query-final.json`.

Observed geometry: `L8192/C6144/H4096/A2048/B128/U64`, page size 256,
32 logical pages, 16 physical slots, 15 final host-backed pages, and page
bytes `4325376`. The final route was `selected packed`; target resident bytes
were `69206016`, host pageable bytes `103809024`, and host metadata bytes
`583872`.

The natural controlled-model proof was:

| field | value |
|---|---:|
| query generation / position | `66 / 4097` |
| catalogue epoch / published epoch | `8326 / 8392` |
| logical page / attention layer / selector rank | `0 / 3 / 0` |
| page generation / content version | `5 / 5` |
| physical slot | `15` |
| candidate cold / host ready / promotion published | `true / true / true` |
| selector published / H2D queued / H2D completed | `true / true / true` |
| mapping published / target graph used | `true / true` |
| target-use epoch / query generation | `12462 / 66` |
| useful H2D bytes / aligned H2D bytes | `4325376 / 4325376` |

The run also measured 63 promotion and 63 eviction deltas, with 6,048 H2D
submissions and 6,048 event completions. The bounded rejection counters are
retained in the raw JSON. A first setup batch logged an unsupported packed
shape before the later selected packed route; the executable continued to the
successful bounded proof and returned 0. This diagnostic is retained in the
hashed stderr artifact rather than hidden.

## Separate answer quality

Answer quality was not inferred from this teacher-forced controlled fixture.
The separate phase-70 A/B/A quality result remained measured and correct:
expected `AURORA-CEDAR-161` and `EMBER-OAK-204`; observed A, B, and A-again
matched those values. This is independent evidence and is not substituted for
the physical promotion chain.

## Raw artifacts

| artifact | SHA-256 |
|---|---|
| `controlled-model-query-final.json` | `0447eb7129f5b5216d27303f6c729df680769c64deea6de7d319219b8733381d` |
| `controlled-model-query-final.stdout` | `0447eb7129f5b5216d27303f6c729df680769c64deea6de7d319219b8733381d` |
| `controlled-model-query-final.stderr` | `dca9555f6331e9a40c85ca8a5e4ffbe1dea4c44c4e5483dfba27c7ae05b3a668` |

