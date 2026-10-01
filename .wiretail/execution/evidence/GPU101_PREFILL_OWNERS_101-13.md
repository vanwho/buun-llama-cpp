# GPU101 current prefill owners — 101-13

## Workload and identity

- Candidate: `build-cuda/bin/llama-server`, SHA-256
  `1922c6f291a9f6180c686c779581d931112d2d0622719861556a12e2eb105cf7`.
- `libllama.so.0.5.0`: `7e5a9f8bf39dceec3b8162d56cbd022add2659ac57e537893fdbd3bbb75f47d6`;
  `libllama-server-impl.so`: `e6c433ce1a40a6f9fa15e6869621bd36a4821b4d62fb13ce3edfbcddb0e24ed8`.
- Source commit `2d9c5511f72cb9a640d0ef0c2da123eb4984551c`; dirty tracked-source
  diff SHA-256 `f9f7c356432d3016eea918ba753686279f1ccaebc3bf7b5978de3b32fabdb485`.
- Model SHA-256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
- Frozen prompt prefix SHA-256
  `8218b0f427cc931fa38d08f1c29f91ec7d82e12ac309ff7743eb7a60da678d20`.
- RTX 4080, CUDA candidate, selective pager, P256, H4096, L8192, B1024/U256,
  Turbo4 K/V, native GPU MTP, reasoning off. Warmup requested 40 tokens;
  measured request requested 80 and returned HTTP 200 with complete SSE.
  Slot 0 was erased before both requests. Measured incoming prompt 4,112 tokens,
  cached tokens 0; selected-packed route counts: prefill 40, decode 28,
  MTP verify 43.

## Current measured owner table

| Phase/owner | Measured interval or events | Interpretation |
| --- | ---: | --- |
| Fresh prefill | 19,774 ms from task-38 slot launch through final prompt-progress event; server reports 4,112 prompt tokens, 19,953 ms prompt-eval timing | The server timing includes synchronization/fence overhead; the journal boundary is the request-phase interval used for Nsight filtering. |
| CUDA kernels | 32,768 kernels, 1,304.83 ms summed device duration | Device kernel work; sums overlap across GPU streams. Largest owners were `mul_mat_q` (type 23) 367.87 ms / 1,899 launches and `gated_delta_net_cuda` 222.82 ms / 576 launches. |
| CUDA copies | 51,585 memcpy events, 96.02 ms summed device duration; 376 memsets, 0.15 ms | Device-side work. Counts include graph-internal operations observed by Nsight. |
| GPU busy union | 1,400.56 ms of the 19,774.10 ms phase (7.08%) | Union of kernel/memcpy/memset intervals; not the sum of durations. Most wall time is outside those GPU intervals. |
| CUDA graph API | 7 `cudaGraphLaunch` calls and 1 `cudaGraphExecUpdate` during prefill | Nsight API events are actual driver/runtime operations; separate from server graph-decision counters. |
| Host CUDA synchronization | 9,677 `cudaStreamSynchronize` calls, 1,436.79 ms inclusive API duration; 1,536 event sync calls, 1.47 ms | Host API durations may overlap and are not additive with GPU intervals. They identify synchronization activity, not all host work. |
| Pager graph decisions | Slot snapshot: 97 capture/rebuild decisions, 15 replays, 112 submissions/completions, 207 us reported construction | Application bookkeeping counters; not substituted for the Nsight CUDA graph API counts. CUDA event capture/update/launch fields remain `not_measured`. |
| Summary-cache rebuild | `summary_build_calls_delta=0`; no `summary_cache_rebuild` log records | This fresh sequential append did not exercise the host-page summary rebuild path. No read/decode time is attributed to it. |
| Selector sideband | Six selector refreshes in slot telemetry; no `selector_sideband` records | Refresh count is not tensor-set submission count. Per-call CPU/upload time is unmeasured in this run. |

The source has exact-profile (`LLAMA_HOTPATH_PROFILE=1`) records for
`process_ubatch`, fences, summary-cache rebuild, selector sideband, and
catalogue accounting. The fresh append did not emit those named records in the
service journal. Therefore the current attribution uses the measured server
prompt interval, Nsight CUDA activity/API trace, and slot counters only. It
makes no claim about unobserved host summary reads, dequantization, digest work,
or tensor-set completion. Nsight CPU stack sampling and CPU context-switch
tracing were disabled; CUDA API durations are the named host intervals available
in this trace.

## Multi-page locality reproduction

The deterministic provider fixture changed two pages across 64 layer/head
configurations and recorded configuration-major visitation. It built 128
page/configuration summaries; page-major visitation would build 64. This shows
the existing all-page rebuild amplification independently of the live request.

## Artifacts

- Full bounded report: `/srv/ai/paged-kv/results/forward/101-13/attempt-01/nsys-current-prefill-v3.nsys-rep`, SHA-256 `90f48b0e7f9873250be60b039a7d84a62e556d119ffe25fd2050878569cfc250`.
- SQLite export: `/srv/ai/paged-kv/results/forward/101-13/attempt-01/nsys-current-prefill-v3.sqlite`, SHA-256 `63c9b3f0e329287a2f4c6ead2cbea7b92adb4706c105985383475d206eaf539f`.
- Bounded interval/counts: `/srv/ai/paged-kv/results/forward/101-13/attempt-01/raw/prefill-owner-timeline.json`.
- Time-filtered `nsys stats`: `/srv/ai/paged-kv/results/forward/101-13/attempt-01/raw/nsys-prefill-filtered-stats.txt`.
- Exact service timeline: `/srv/ai/paged-kv/results/forward/101-13/attempt-01/raw/profile-request-timeline-nsys2025-v3.log`.
- Successful warmup/measured request and slot files:
  `/srv/ai/paged-kv/results/forward/101-13/attempt-01/raw/retry-v3/`.
- Build, deterministic fixture and absent/zero-profile logs are under
  `.wiretail/build/101-13-*`.
