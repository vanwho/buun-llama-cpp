# V10 66-01 runtime evidence

Revision: `hotpath-v10-20260914`. Candidate model: `/srv/ai/models/text/current.gguf`,
resolved Qwen3.8-27B-UD-IQ4_XS, model SHA-256
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
The managed candidate was `build-cuda/bin/llama-server` on port 8080 with
CUDA Turbo4 target, native GPU Turbo4 MTP, L8192, H4096, page 256, and hot
capacity 16. No client page ID or public force-promotion API was used.

## Production repair

`src/llama-graph.cpp` now skips optional graph-input writes when the scheduler
did not allocate that input tensor. This closes the post-initialization
model-query crash boundary without changing graph selection.

`src/llama-kv-pager.cpp` now synchronizes live-policy reconciliation with
asynchronous host-seal sideband state. It cancels captures for identities no
longer retained, preserves matching in-flight versions, synchronizes the
state content version from the committed record, and fail-closes stale
in-flight flags after the bounded worker wait. Batch admission also gets the
bounded seal opportunity before reservation.

## Checks and measured rows

* `build-cuda/bin/test-kv-pager`: exit 0.
* `build-cuda/bin/test-cuda-kv-promotion`: exit 0; T0 fixture retained as
  `seed=0x49060001`, winner logical page 0, `copy_bytes=1081344`, mature-FA
  consumption parity pass.
* Controlled T1 command:
  `build-cuda/bin/test-kv-pager-model --model /srv/ai/models/text/current.gguf
  --context 8192 --n-batch 128 --n-ubatch 64 --hot-pages 16 --tokens
  1..5000 --selected-only --mtp off --force-page 0`.
  This is explicitly the test-only `controlled_model_query` fixture. It
  completed with exit 0 and emitted: logical pages 32, physical capacity 16,
  final resident/host pages 16/15, promotion/eviction delta 9/9,
  H2D useful/aligned bytes `38928384/38928384`, submitted/completed
  `864/864`. The natural proof was logical page 12, selector rank 0,
  useful/aligned bytes `4325376/4325376`, and
  `selector_published=true`, `h2d_queued=true`, `h2d_completed=true`,
  `mapping_published=true`, `target_graph_used=true`. The test-only forced
  checksum row was logical page 0, physical slot 15, page generation 2,
  content version 2, with equal host/device checksum
  `4563831312766228892`.

* T3 organic A/B/A used the immutable request files from the phase-64 fixture
  and completed all three SSE responses with HTTP 200 and `[DONE]`:

  | row | response ID | prompt/completion tokens | response bytes | stop |
  |---|---|---:|---:|---|
  | A | `chatcmpl-DQSCTp1sQ20LpMEnyXJT3lncxWQhRrLR` | 2035/16 | 5243 | measured, done |
  | B | `chatcmpl-XgIVRrkipoceo9DrKpWVNeqqADicngOy` | 4073/16 | 5244 | measured, done |
  | A-again | `chatcmpl-J6oyx3HZ8vNd9doPvAe8TYiZaiYWJ2An` | 4110/16 | 5236 | measured, done |

  The production endpoint does not expose page identity per response. The
  captured natural proof immediately after this round was page logical 14,
  layer 3, selector rank 0, query generation 206, query position 4124,
  page/content generation 176/176, physical slot 14, catalogue/published
  epochs 20988/20991, H2D useful/aligned bytes `4325376/4325376`, and
  `candidate_was_cold=true`, `host_ready=true`, `selector_published=true`,
  `h2d_queued=true`, `h2d_completed=true`, `mapping_published=true`,
  `target_graph_used=false`. Therefore the organic physical promotion claim
  is failed/unproven (the page is not counted as physically consumed), while
  the A/B/A transport rows are measured and pass. Aggregate endpoint counters
  were faults/evictions `15/15`, host-seal D2H bytes `168689664`, H2D useful
  bytes `64880640`, and transfer event completions `1440`.

* Separate concise quality request: HTTP 200, response ID
  `chatcmpl-J3BCBtKp0tWhsENNFKMLYA0OXaFZHwG5`, prompt/completion tokens
  83/16, stop `length`; response content was empty and reasoning was
  `////////////////`. Expected `AURORA-CEDAR-161`; quality is failed
  independently of the paging rows.

The first post-repair A request no longer returned the former
`KV pager batch write reservation failed: transaction` error. No segmentation
fault occurred in the final controlled fixture or the final A/B/A sequence.

The final canonical one-case runner against the rebuilt candidate completed
with exit 0: decode 35.2294745694 tok/s, prompt 198.7636898491 tok/s, native
MTP acceptance 3.63636364%, errors 0.
