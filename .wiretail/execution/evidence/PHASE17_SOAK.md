# Phase 17 soak and lifecycle evidence

## Decision

**Lifecycle passed; phase-17 soak acceptance blocked.** The feasible
16,384-token selective/native-MTP profile completed the long-context page
waves, focus shifts, speculative rejection, cancellation, slot reuse,
checkpoint round trip, clear/recovery, concurrent transport, and restart
health checks. The raw pager counters show host-page residency and table/graph
activity, but no physical transfer, fault, or eviction counter advanced. The
concurrent output pair also lacked a deterministic exact-marker oracle. Those
limits are recorded as blocked sub-gates rather than converted into passes.

Machine-readable evidence is [`PHASE17_SOAK.json`](PHASE17_SOAK.json). Raw
evidence is retained at:

`/srv/ai/paged-kv/results/17-10-soak-20260905T021200Z/`

The raw integrity check is `SHA256SUMS` / `SHA256SUMS.check` in that
directory.

## Tested boundary

- Source tree: `5d5276ca5a166cfeb4a05f016d329503d800ee86`
- Binary: `build-cuda/bin/llama-server`, SHA-256
  `d306009f51f0ebb554752d9c41fabb51c796f9d708eef977cc3cc09014e361db`
- Model: Qwen3.8-27B-UD-IQ4_XS, SHA-256
  `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`
- Profile: `qwen38-fast`; context 16,384; page size 256; 64 logical pages;
  Turbo4 target and native draft K/V on GPU; one slot
- GPU: NVIDIA GeForce RTX 4080, driver 595.84, 16,376 MiB

The checked-in V4 corpus identity is recorded for provenance. The lifecycle
workload itself used retained synthetic markers and was not scored as a corpus
quality acceptance run.

## Gate results

| Sub-gate | Result | Evidence |
| --- | --- | --- |
| Long-context page waves | Pass: 3/3 HTTP 200, 3,245 prompt tokens observed | `raw/stable-*` |
| Focus shifts | Pass: 3/3 HTTP 200 | `raw/focus-*` |
| Page churn | Blocked: host pages peaked at 12 and resident pages at 13, but faults, evictions, H2D/D2H, and transfer submissions remained zero | `raw/metrics-after-page-waves.txt`, `raw/metrics-pre-restart.txt` |
| Speculative rejection | Pass: 163 draft, 85 accepted, 83 verification-step deltas; rejection histograms in journal | `raw/metrics-after-speculative-rejection.txt`, `raw/service-journal.log` |
| Cancellation | Pass: client timeout exit 124; slot returned idle | `raw/cancel.*`, `raw/slots-after-cancel.json` |
| Concurrent isolation | Transport/slot release passed; exact output oracle blocked because one original payload returned `Error: No valid data found.` with HTTP 200 and the controlled retry returned refusal prefixes | `raw/concurrent-*.response.json`, `raw/concurrent-retry-*.response.json` |
| Save/erase/restore | Pass: 1,965 tokens and 191,143,964 bytes round-tripped with matching SHA-256 | `raw/slot-roundtrip.json` |
| Clear/recovery | Pass: erase and recovery request HTTP 200; final slot idle | `raw/clear-final.record.json`, `raw/recovery.response.json` |
| Restart | Pass: PID 2897889 → 2902110; 8080 and 8091 both HTTP 200; exact profile/binary retained | `raw/restart.json`, `raw/identity-*`, `raw/health-*` |
| Resource boundedness | Pass for bounded observation: 45 workload samples plus 30-second post-restart idle plateau | `raw/resource-timeline.jsonl`, `raw/resource-idle-after-restart.jsonl` |

Before restart, host pageable bytes peaked at 51,904,500 and host pinned bytes
remained zero. After restart, 30 idle samples held RSS between 2,458,052 and
2,460,296 KiB, GPU memory at 14,707 MiB, host pages/host pageable bytes/host
pinned bytes at zero, and target bytes constant at 276,824,000. This is a
bounded observation and not an unbounded leak proof.

## Verification

- Focused CUDA pager/lifecycle suite: 10/10 passed.
- Generated rollback and state lifecycle suite: 3/3 passed.
- Soak harness Python compilation, `git diff --check`, and raw SHA-256
  verification passed.
- Final `llama-server.service` and `ai-long-memory.service` are active;
  qwen38-fast selective/native-MTP remains loaded for the dependent task.

## Deferred verification

No required hardware or human action was unavailable. A follow-up measured
boundary is required for physical page-fault/eviction/transfer churn and a
deterministic concurrent response oracle. Both remain explicitly blocked in
the JSON artifact; no pager soak acceptance claim is made for them.
