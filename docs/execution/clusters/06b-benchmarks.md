# Cluster 06b — benchmark integration and selective acceptance

Tasks: `06-03`, `06-04`.

Purpose: extend the established harness without losing historical comparability, then decide selective
capacity/quality/speed from raw repeated evidence.

Carry forward:

- `/srv/ai` is dirty user work; prefer a repo-local adapter and never overwrite overlapping changes;
- preserve `run-config.json`, `records.jsonl`, raw response, summary, isolation, interruption, and
  production-profile restore contracts;
- same build/model/corpus controls compare 77,824 all-GPU, ordinary 256K CPU KV with draft placement
  verified, observe, selective focus, cold needles, focus shifts, churn, and feature-off;
- record pp/tg/TTFT, MTP acceptance, VRAM/RAM, ledgers, faults/prefetch/evictions, transfer bytes/times,
  epochs, and errors over warmups plus at least five release trials;
- target gates are >=2x ordinary CPU KV and within 20% of comparable 77K all-GPU after warmup; 3x is a
  stretch, not a movable requirement;
- quality thresholds and corpus are frozen before results; failures block rather than trigger retuning.

Read: plan sections 1–3, 11 Phase 06, 12.4, 16–18; baseline and correctness handoffs; all three
`/srv/ai/benchmarks` files. Do not implement exact mode here.

Exit artifacts: compatible benchmark records and `evidence/FINAL_ACCEPTANCE.md` for selective mode,
including full disclosure of churn/fault degradation and exact inputs for the next cluster.
