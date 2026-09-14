# V10 scale pilot findings

The prerequisite small physical-promotion chain and native-MTP matrix passed.
The pilots used the 51-02 candidate, B/U=128/128, page size 256, target and
draft Turbo4 on GPU, and draft n-max=2.

## Results

- `L32768/H16384`: 18/18 requests passed through C=24,580 in 524.0 seconds.
  The run measured warm C=1,200, below-H C=15,240, above-H C=16,644, and the
  next useful frontier C=24,580. Cached tokens advanced on each append. The
  last measured prefill rate was 23.76 tok/s, implying about 340 seconds to
  the logical frontier at that recent rate (about 14.4 minutes total). No
  runtime fault occurred. The run recorded no useful H2D promotion, so it is
  not a promotion proof.
- `L131072/H30208`: full-L startup passed with 131,072 GPU Turbo4 draft rows
  (138,543,104 bytes). A four-request cached-ingest probe passed through
  C=4,204 in 18.15 seconds; the final cached prefix was 4,021 tokens. The
  probe stayed below H and therefore makes no promotion or full-occupancy
  claim. Full-L occupancy was deliberately not run after the short probe.
- The first 128K startup attempt failed because the still-running 77K profile
  left only about 1.1 GiB free; CUDA could not allocate the 13,695,514,488-byte
  model buffer. After stopping that profile, the exact 128K candidate loaded
  successfully. This is recorded as a capacity/setup observation, not as a
  model-capability failure.

The request snapshots export host seal, summary, graph, packed-copy, wait,
scratch, and transfer counters. At the final 32K sample host valid/sealed data
was 21,626,800 bytes and scratch high water was 7,372,820 bytes; at the final
128K sample those values were 4,325,400 and 1,073,200 bytes. Transfer counters
remained zero because no cold page was selected and consumed. The allocation
ledger from tuning charges 16,206,500,000 bytes against 16,720,200,000 usable
bytes with 201,327,000 bytes headroom; it is retained as the measured ledger,
while fields absent from the live snapshot are explicitly not inferred.

Raw runs are under `/srv/ai/paged-kv/results/v10/51-03/`; the complete matrix
and status details are in `V10_SCALE.json`.

## Deferred verification

No hardware or credential verification was deferred. A full 128K occupancy,
promotion-quality, and soak matrix was intentionally not run under the
bounded practical pilot budget. A 256K pilot remains outside this task.
