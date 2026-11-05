# Comparison progress — audited after R9, 2026-11-05 04:42 UTC

R9 has finished with findings. The report now contains **805 usable complete cells / 2,415 measured repetitions** under a narrow saved-response artifact audit. This excludes a whole triplet if any hash-verified response contains at least 64 nonspace characters, no letters or digits across visible content plus reasoning, and at least 80% repetition of one punctuation character. Short EOS and reasoning-only answers remain eligible. This is not a semantic accuracy check; 189 source triplets lacked saved raw responses and are marked unassessed, not rejected.

“Usable complete cells” includes retained historical data. The selected gap-fill columns count only the plan’s previously missing cells; they are not counts of all family measurements. For example, master-GPU/MTP-off retains 54 older cells while none of its 54 newly selected gap-fill requirements has yet qualified. Fork-GPU/MTP-off retains 102 older cells; its 12 selected requirements are the C=130,048 capacity-N/A endpoint.

| Family | Usable complete cells | Selected gap-fill cells recovered | Feasible missing cells | Capacity N/A cells |
|---|---:|---:|---:|---:|
| Master GPU, MTP off | 54 | 0/54 | 42 | 12 |
| Master GPU, MTP on | 72 | no open requirement | 0 | 0 |
| Fork GPU, MTP off | 102 | 0/12 | 0 | 12 |
| Fork GPU, MTP on | 60 | no open requirement | 0 | 0 |
| Master CPU Q4, MTP off | 84 | 63/63 | 0 | 0 |
| Master CPU Q4, MTP on | 31 | 7/48 | 41 | 0 |
| Fork CPU Q4, MTP off | 84 | no open requirement | 0 | 0 |
| Fork CPU Q4, MTP on | 54 | 4/37 | 33 | 0 |
| Current ACO, MTP off | 132 | complete source curve | 0 | 0 |
| Current ACO, MTP on | 132 | complete source curve | 0 | 0 |

## Feasible decode gaps

| Family | Nominal context C | Missing cells | Status |
|---|---:|---:|---|
| Fork CPU, MTP on | 12,288 | 6 | 6 failed the response-quality audit |
| Fork CPU, MTP on | 28,672 | 9 | 9 failed the response-quality audit |
| Fork CPU, MTP on | 61,440 | 9 | 9 failed the response-quality audit |
| Fork CPU, MTP on | 126,976 | 8 | 7 failed the audit; 1 was not completed |
| Fork CPU, MTP on | 196,608 | 1 | Not completed |
| Master CPU, MTP on | 6,144 | 3 | Not measured in retained normal-prefix cells |
| Master CPU, MTP on | 12,288 | 9 | Not measured in retained normal-prefix cells |
| Master CPU, MTP on | 28,672 | 9 | Not measured in retained normal-prefix cells |
| Master CPU, MTP on | 61,440 | 9 | Not measured in retained normal-prefix cells |
| Master CPU, MTP on | 126,976 | 9 | Not measured in retained normal-prefix cells |
| Master CPU, MTP on | 196,608 | 2 | Not measured in retained normal-prefix cells |
| Master GPU, MTP off | 6,144 | 12 | R8 CUDA runtime fault blocks the current comparison series |
| Master GPU, MTP off | 12,288 | 10 | R8 CUDA runtime fault blocks the current comparison series |
| Master GPU, MTP off | 28,672 | 10 | R8 CUDA runtime fault blocks the current comparison series |
| Master GPU, MTP off | 62,464 | 1 | R8 CUDA runtime fault blocks the current comparison series |
| Master GPU, MTP off | 88,064 | 6 | R8 CUDA runtime fault blocks the current comparison series |
| Master GPU, MTP off | 120,832 | 3 | R8 CUDA runtime fault blocks the current comparison series |

The fork GPU/MTP-off **mapped prefill-only series completed** through nominal C=120,832 (actual final C=120,833), with nine append groups and 120,824 net-new tokens. It adds no decode cells.

## Exclusions and runtime outcomes

- **31 whole CPU/MTP-on source triplets** are excluded for repeated-punctuation-only output: 27 in R8 fork CPU/MTP-on and 4 in R9 fork CPU/MTP-on. R8 retains only C=12,288 low P1/P2/P3 as quality-eligible. R9 adds C=126,976 medium P1 as quality-eligible. The response paths and hashes are in the external failure inventory.
- **24 capacity N/A cells** remain separate: 12 each for master/fork GPU MTP-off at C=130,048. They are not completed measurements or runnable gaps.
- The clean master CPU/MTP-on L=262,144 startup check failed before readiness: 1,730 MiB GPU allocation requested with 15,900 MiB free and no other compute owner. Keep primary L=200,704; no synthetic 256K decode cells.
- Master GPU/MTP-off remains a runtime finding, not a capacity result: CUDA illegal access on the first normal appended question after a successful C=6,144 fill.
- Follow-up GPU diagnostics are complete but do not change the 42 missing cells: Compute Sanitizer timed out at 900 seconds after 322 output tokens without locating an invalid access. CUDA-GDB reproduced a warp out-of-range address in a Flash Attention F16-compute/Turbo4-K/V specialization. A graph-disabled reproduction also failed after its 40-token warmup with a misaligned address at stream synchronization. This shows graph capture is not required for the failure, but the bad pointer/source instruction remains unidentified.
- The three independent master CPU/MTP-on C=28,672/off/P3 outlier cohorts each retained two valid measured repetitions; repetition 3 failed with the DFlash target argmax row lookup fault. None forms a complete triplet.
- R8 fork CPU/MTP-on was operator-stopped. Its 30 HTTP-complete triplets are not 30 usable comparison cells: 27 were removed by the response audit. R4 target-only slot-restore CPU/MTP-on cohorts remain quarantined.

## Method and retained data

Keep the ordinary cumulative conversation and fresh mapped repository bytes per series. One context fill is shared across prompts and modes. Group cells by nominal requested C; preserve actual C per repetition. Backend, net-new, and HTTP-wall prefill remain separate. No per-mode fresh fills, benchmark disk KV save/restore, or context shifting.

Compact machine-readable summary: `comparison-final-20261104-quality-audit/comparison-final-20261104.json` (external SHA-256 `f3f9bfef2b39d0945c87ad43e357638553e310fd52b5b7eb9343cab34b99af2b`). Response-quality inventory: `comparison-response-quality-failures-20261104.csv` (external SHA-256 `7cc1a7cb7e068c98501a43b9b0619cb825227d571dcfb86cbedc5de520272efd`). Raw request/response payloads remain external.
