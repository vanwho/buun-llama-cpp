# Review48 — current goal unmet; source and measurement repairs scheduled

User-requested audit completed2026-09-14. Detailed source findings, corrected
raw rates, uncertainty and primary references are in
[V10 assessment](../v10/ASSESSMENT.md); machine record is
[V9_REVIEW_48.json](V9_REVIEW_48.json). No live benchmark run in this audit.

The latest phase47 benchmark used phase43 binaries, short30/27/28-token
requests and observed H2048, not the advertised C6144/H4096 comparison.
Historical phase36 genuine6144-token raw prefill remains about215tok/s vs
1440all-GPU; selective decode39–43 vs12–13CPU-KV is encouraging but the
controls'0% MTP draft acceptance makes a fair native-MTP claim premature.
No complete current-Q→cold promotion→target consumption proof is established.

Direct code bug: both host-only inventory conversions leave valid_length0,
which CUDA explicitly excludes. Router Q/summary basis mismatch, CPU summaries,
whole-catalogue uploads, skipped-cadence scoring and telemetry-only extra FA
also need targeted repair; their exact performance contributions are not yet
profiled. First prove the smallest physical chain, then optimize costs.

Appended17 explicit tasks in phases49–52 with risk-based Luna Medium/High,
High first retry, fresh bounded context-area clusters and named-proof receipt
checks.49-01 is next.49-06 is deterministic production CUDA promotion;
51-01 is actual-model controlled/organic proof;51-02 reports matched original
three-prompt prefill, append and decode. Longer pilots are conditional findings.
52-01 reads only the latest compact summary and schedules real remaining
implementation if needed. Older ledgers remain unchanged and out of context.
