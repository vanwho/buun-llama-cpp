# Phase-25 / blocked-26-01 findings for the interactive revision

2026-09-12 source/evidence audit; no new live speed benchmark during this
revision. Evidence below is historical and not a common-binary A/B campaign.

## Observations

| Receipt / source | Observed result | Interpretation |
| --- | --- | --- |
| SPEED25_03_INCREMENTAL | roughly 364 prompt tok/s after summary repair versus roughly 146 earlier | large host-work improvement, not proof of final native-MTP cold recall |
| SPEED25_06_BATCHING | B256: 463 pp/s, 43.56 tg/s; B512: 453.91 pp/s, 46.20 tg/s | 23–25 proposal/output-scale samples; larger batch not proven better; resolve legacy B versus -ub from argv |
| SPEED25_08_PRESSURE | L8192, H2048; one eviction/promotion, 4325376 useful H2D bytes and matching checksum; roughly 30 tg/s | deterministic forced recall, MTP OFF |
| SPEED25_13_WHOLE_MODEL | retained 2048-token run: 2343 seal calls, 4672 summary-build calls, 22 host D2H calls, 1.498s wait and 2.508s queue | overlapping timings, not additive attribution; fresh post-MTP-change profile absent |
| SPEED25_15_READY | L8192, H2048, native GPU MTP; about 411 pp/s and 60 tg/s, 128 output tokens, 75/102 accepted/proposed | attention_samples=0 and H2D=0; readiness for natural movement explicitly false |
| SPEED26_01_FULL_CONTEXT | L262144 native Turbo4 GPU MTP startup and short generation succeeded | allocation != populated context proof |
| same long retries | 737 MiB total F16 scratch failure; later 53 MiB reserve failure; context exhaustion after much longer prefill | allocation-owner, route and frontier problems, not established hardware context ceiling |
| final row/page retry | 73216 hot rows passed as --kv-hot-pages | intended 286 pages at P256; reject units before startup |

Raw SPEED25_15 SSE in
/srv/ai/paged-kv/results/25-15-v6-pressure-20260912/raw-q0-measured-1.sse
reports 4107 prompt tokens / 9989.94ms = 411.11 pp/s; predicted=128 and
predicted_ms=2125.44, server rate 59.75233 using its count convention. Receipt
410.00/60.22 uses different counts. Preserve both; V7 records definitions and
exact actual counts rather than promoting a synthetic precise speed ratio.

The normal loaded Qwen service inspected at revision time has -c77824,
-b1024 -ub256 and --spec-type none, despite its mtp alias. Therefore its fit
does not establish a matched 70K native-MTP memory budget. Historical native-
MTP readiness at L262144 reported draft KV=276955136 bytes, consistent with
264 MiB payload plus padding, and H=73216 rows, but failed during long use.

## Scratch interpretation: explicit estimates, not fitted benchmarks

Source: src/llama-kv-cache.cpp prepare_with_slots, vbr_scratch_reserve,
memory_vbr_scratch_bytes_per_token; ggml/src/ggml-cuda/fattn.cu
ggml_cuda_turbo_prefill_attend, kv_dequant_scratch_try and scratch_memory.

Model K+V F16 scratch payload is approximately 4096 bytes * materialized rows
per owning backend pool, shared across attention layers. 737 MiB corresponds
to 188672 rows, and 53 MiB to 13568 rows. Those widths are compatible with
history/watermark-plus-batch reservations, but an error alone does not identify
which consumer required them. Trace target versus MTP owners and real views.
The dirty VBR watermark cap is not yet a validated universal pager fix.

Reducing B from256 to64 saves only 0.75 MiB in this particular 4096*B term.
It may save much more in activations, output and recurrent/MTP buffers, which
must be measured. Eliminating wrongly materialized cold rows is the material
scratch correction. At L128K a legitimate full-width MTP F16 view alone can
be about512 MiB even when target H is small. Dense/packed Turbo prefill really
does use F16 temporary views; direct paging and bounded tiled alternatives
must be compared rather than treating every Turbo4 route as scratch-free.

New async-seal retry reached 210031 input tokens with 820 changed pages,
216481 seal calls and about3.55GB D2H before the server's context-exhaustion
message, output=0. Source inspection shows that message also covers generic
memory-slot/prepare failures after batch reduction to one token:
server-context.cpp n_batch==1 && ret==1, and llama_context::decode returns1
on LLAMA_MEMORY_STATUS_FAILED_PREPARE. A scratch or pin failure may therefore
be mislabeled. This is not established true logical-capacity exhaustion.
Payload volume is near once-per-token KV, but maintenance invocation count
is still per-token scale. A page-version dirty queue, coalesced asynchronous
completion and valid committed/speculative frontiers remain important.

## Minimal source map

- Units/fit/driver: tools/server/bench/run-final-curve.py (_fit_prompt,
  _case_record, _runtime_identity), test_resume_contract.py;
  run-pager-profile-benchmark.py and test_pager_benchmark_adapter.py.
- Site launcher: /srv/ai/benchmarks/run-profile-benchmark.sh and
  /srv/ai/scripts/start-primary-llama-profile.sh; current profile uses BATCH
  and UBATCH. Add checked per-run overrides, don't assume driver flags launch.
- Dispatch: src/llama-kv-attention-execution.cpp prepare chooses dense, then
  direct, then packed on capability; src/llama-graph.cpp packed copies/direct
  page mass; ggml/src/ggml-cuda/fattn.cu CUDA dispatch and materialization.
- Natural telemetry: llama_context::publish_kv_attention_telemetry and graph
  direct_telemetry_snapshot/page_mass handling; attention-telemetry publish
  validation; routing-retrieval and cache routed identities.
- Async host/frame: llama-kv-pager seal_ready_pages/apply_live_policy,
  llama-kv-cache write reservations; server-context.cpp frontier handling;
  common/speculative.cpp native-MTP pending/accepted row handoff.

Old raw roots are indexed by SPEED26_01_FULL_CONTEXT.json. Old 26/27 packets
and handoff are archived under archive/phase26-before-interactive-20260912/.
Do not load their multi-retry narrative into new cluster sessions.
