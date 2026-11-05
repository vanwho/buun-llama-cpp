# Comparison progress — 2026-11-05T00:35:57Z

Counts are scientific cells: one nominal context × reasoning mode × prompt, with one warmup and three measured repetitions. Active counts will advance.

| Configuration | Remaining cells/work | Nominal context tokens still missing | Status |
|---|---:|---|---|
| Fork CPU Q4, MTP on | 21 | 28,672 (2); 61,440 (9); 126,976 (9); 196,608 (1) | Running; 16 newly complete |
| Master CPU Q4, MTP on | 41 | 6,144 (3); 12,288 (9); 28,672 (9); 61,440 (9); 126,976 (9); 196,608 (2) | Queued; 0 newly complete |
| Master GPU Turbo4, MTP off | 42 | 6,144 (12); 12,288 (10); 28,672 (10); 62,464 (1); 88,064 (6); 120,832 (3) | CUDA runtime fault; diagnostic queued; 0 newly complete |
| Master CPU/MTP-on outlier rechecks | 3 independent cohorts | 28,672 / off / prompt3 | One warmup +3 measured per cohort |
| Fork GPU/MTP-off mapped prefill | One series; no decode repeats | 6,144 →120,832 | Queued |
| Master CPU/MTP-on clean256K startup | One startup-only check | L262,144 | Queued after primary; a pass requires filling missing256Kdecode results next |

## Method and retained data

Fresh disjoint mapped repo chunks and ordinary cumulative Q/A; one continuous fill per configuration series, no prefill per prompt/mode, no benchmark KV saves/restores. B1024/U256 and16threads fixed. Original prompts, reasoning modes,40-token warmup and3×400-token measured caps retained. Record actualC separately from nominal grouping; backend/net-new/wall-prefill rates separate.

Retain completed ACO on/off264cells, master CPU/MTP-off R7, and approved historical controls. R4 target-only restored CPU/MTP-on data omitted draft/carry history and remains quarantined. Do not blanket-rerun good cells.

## Errors and investigations

- Master GPU/MTP-off: C6,144 fill succeeded; first question appended33tokens with rewind0, then CUDA illegal memory access at target synchronize before output. Exact accessing kernel unproven;42missing cells are not complete. Frozen memcheck follows primary measurements.
- GPU/MTP-off130,048-token endpoints: R6 shows only one matching GPUowner and free memory falling to9MiB before allocation failure. Twelve cells per GPUfamily excluded at that unachieved endpoint; L122,880 tests the feasible frontier without batch changes.
- Master CPU/MTP-on256K: old startup really failed allocating1,730MiB onGPU, but lacks prelaunch sole-owner/free-memory proof. Fresh startup-only confirmation is now required before treating it as a clean limit. If it fits, run the missing256Kbenchmark block, keeping smaller valid points.
- Master CPU/MTP-on28K/off/prompt3 has unexplained MTP/decode variation. Three independent clean cohorts are queued; preserve all, not just fastest.
- Last-user reasoning steering moves in the frozen Jinja template, causing bounded old-tail re-evaluation. Current source ranges remain disjoint; separate repair stays isolated during primary comparisons. This does not explain the off-mode CUDA fault with rewind0.
- A systematic master/fork pure-GPU prefill difference awaits a dispatch trace; exact kernel contribution remains unverified.

## Active-plan wording erratum

First six R8 steps carry stale ingest-only descriptive labels despite nonempty decode selections; records confirm decode. Selections/counts are authoritative. Only final fork-GPU/MTP-off is fill-only. Correct wording after primary/diagnostics; do not change frozen active plan hash midqueue.

Plan: `.wiretail/execution/benchmarks/COMPARISON_COMPLETION_20261104.json`. Compact external audit and raw-record source indexes: `/srv/ai/paged-kv/results/comparison-gap-audit-20261105/`. Raw payloads/traces/binaries remain external.
