# Release benchmark results

This is the fixed-candidate cohort. It supersedes neither nor mixes with the
archived pre-fix diagnostic cohort. The small placement comparison and focused
natural recall completed; the 256K fill reached its planned C=258,048 frontier.
The synthetic repeated-question follow-up was deliberately stopped at 4/12
completed requests; a separate, append-only conversation check completed as
single observations, not a repeated timing matrix.

## Candidate and protocol

- Candidate server SHA-256: `c9041ca96611ed4c9aa06be7b9a23443aa11d4b3bb30983d7dd2558c582659ec`.
- Embedded build/source commit: `794edc211` (production tree `a59a52ac`), based on the cleaned upstream-integrated release tree.
- Model SHA-256: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
- Reasoning off, temperature 0, seed 42, `draft-mtp` maximum draft length 2, batch 1024, microbatch 256, one slot, 999 GPU layers, flash attention enabled.
- GPU target/draft KV: Turbo4. CPU-target placement: F16 target KV with GPU Turbo4 draft KV. ACO means **Attention-Aware Context Offloading**.
- Three canonical prompts; each placement ran one 40-token warmup and three measured requests capped at 400 tokens per prompt. Small runs used `cache_prompt=false` for every request, so all warmups and measured rows prefetched the same frozen prefix. Warmups are excluded from medians.
- The executed frozen matrix runner is tooling commit `49cb38d18`. Post-run tooling commits `de3c198a3`, `9e84f5f2b`, `5e4a13352`, `4ab98f371`, `80636246a`, and `a3675529e` add the append-history helper, make future fill IDs unique, flag overwritten artifacts, refuse to overwrite raw evidence, and keep the exporter reusable; they do not change server behavior or benchmark measurements. The append observations were run by helper SHA-256 `175829f73a87fb6c556db03fc6a8ee41a5632787cc06faf9e2813bbcc8449497`; its saved streams were subsequently revalidated with the committed strict `[DONE]`/usage/error validator.

Small prefix occupancy was C=6,143 native tokens (planning target 6,144), with L=8,192 and H=4,096. The same frozen prefix hash was used across placements; its full rendered prompt counts were 6,166 / 6,163 / 6,164. Prefix preparation is outside generation wall time. Client HTTP wall time is separate from backend prompt and decode stages.

## 8K placement comparison

Each row below gives per-prompt medians over three measured repetitions. MTP acceptance is the median of the three per-request ratios. The overall table reports the median across all nine measured requests and the pooled MTP counts.

| Placement | Prompt | Wall median (ms) | Backend prompt rate median (tok/s) | Decode rate median (tok/s) | MTP acceptance median |
| --- | --- | ---: | ---: | ---: | ---: |
| GPU-only (Turbo4 target/draft) | Python merge | 7,415.7 | 1,640.0 | 107.0 | 95.52% |
| GPU-only (Turbo4 target/draft) | mmap vs read | 5,420.2 | 1,637.2 | 78.4 | 58.33% |
| GPU-only (Turbo4 target/draft) | Bash watcher | 8,125.8 | 1,639.5 | 91.8 | 76.19% |
| CPU target F16 / GPU draft Turbo4 | Python merge | 19,647.5 | 894.0 | 26.2 | 96.93% median of runs |
| CPU target F16 / GPU draft Turbo4 | mmap vs read | 13,214.3 | 894.5 | 21.9 | 71.05% |
| CPU target F16 / GPU draft Turbo4 | Bash watcher | 23,736.9 | 892.0 | 23.7 | 81.52% |
| ACO | Python merge | 8,490.1 | 1,450.8 | 93.7 | 93.12% |
| ACO | mmap vs read | 6,092.6 | 1,447.0 | 70.1 | 58.47% |
| ACO | Bash watcher | 9,232.5 | 1,447.2 | 80.6 | 73.91% |

| Placement | Overall wall median (ms) | Prompt-stage median (ms) | Decode-stage median (ms) | Pooled MTP accepted / drafted |
| --- | ---: | ---: | ---: | ---: |
| GPU-only | 7,415.7 | 3,759.8 | 3,643.2 | 1,698 / 2,109 (80.51%) |
| CPU target F16 / GPU draft Turbo4 | 19,647.5 | 6,895.0 | 12,738.8 | 1,645 / 1,937 (84.93%) |
| ACO | 8,490.1 | 4,271.6 | 4,214.9 | 1,692 / 2,148 (78.77%) |

These are observations for this candidate, model, host, and prompt set, not a general performance guarantee. The 400-token value is a request cap; EOS can end a response sooner. The benchmark used normal prose/code responses, with no exact output-token gate.

## 256K ACO fill curve

Configured geometry was L=262,144, H=65,536 (256 pages), with planned and achieved system-prefix C=258,048. All 16 fill requests completed. The table distinguishes native occupancy advance from backend work: backend `prompt_n` includes replay, and `cache_n` is reported separately. “Backend rate” is `prompt_n / prompt_ms`; “C advance rate” is native ΔC divided by client HTTP wall. They are different quantities and should not be conflated.

| Actual C | Native ΔC | Backend prompt_n | cache_n | Backend rate (tok/s) | C advance / wall (tok/s) | Wall (s) | RSS after (GiB) | VmSwap (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15,922 | 15,922 | 15,944 | 0 | 1,225.7 | 1,214.9 | 13.11 | 3.87 | 0 |
| 34,858 | 18,936 | 19,206 | 15,674 | 899.9 | 884.5 | 21.41 | 4.50 | 0 |
| 70,550 | 35,692 | 35,962 | 34,610 | 620.9 | 615.0 | 58.03 | 5.59 | 0 |
| 89,970 | 19,420 | 19,690 | 70,302 | 856.4 | 840.1 | 23.12 | 6.23 | 0 |
| 105,174 | 15,204 | 15,474 | 89,722 | 804.0 | 781.8 | 19.45 | 6.82 | 0 |
| 120,304 | 15,130 | 15,400 | 104,926 | 780.3 | 757.7 | 19.97 | 7.20 | 0 |
| 135,378 | 15,074 | 15,344 | 120,056 | 767.6 | 745.6 | 20.22 | 7.48 | 0 |
| 149,916 | 14,538 | 14,808 | 135,130 | 747.1 | 727.2 | 19.99 | 7.77 | 0 |
| 165,563 | 15,647 | 15,917 | 149,668 | 737.1 | 718.6 | 21.78 | 8.07 | 0 |
| 179,469 | 13,906 | 29,823 | 149,668 | 783.5 | 363.4 | 38.26 | 8.37 | 0 |
| 193,103 | 13,634 | 43,457 | 149,668 | 800.0 | 250.0 | 54.53 | 8.67 | 0 |
| 207,010 | 13,907 | 57,364 | 149,668 | 807.2 | 195.1 | 71.30 | 9.08 | 0 |
| 220,626 | 13,616 | 70,980 | 149,668 | 813.2 | 155.6 | 87.52 | 9.35 | 0 |
| 234,592 | 13,966 | 84,946 | 149,668 | 815.3 | 133.7 | 104.43 | 9.61 | 0 |
| 247,694 | 13,102 | 98,048 | 149,668 | 809.7 | 107.9 | 121.43 | 9.79 | 0 |
| 258,048 | 10,354 | 108,402 | 149,668 | 806.7 | 76.8 | 134.74 | 10.02 | 0 |

The cache count plateaus at 149,668 tokens after C=165,563 while later backend prompt work includes an increasingly large suffix replay. Thus the approximately 807 tok/s late backend rate is not unique-source ingestion throughput: the measured native C-advance rate fell to 76.8 tok/s at the final frontier. No configuration change or timeout normalization was applied. Per-frontier counters, rates, RSS, and swap are in the machine-readable long-context artifact.

The runner's request ID for the final two frontiers collided (`r15` for both C=247,694 and C=258,048). Their distinct JSONL measurements and matching server-log prompt/timing records remain available, but the earlier frontier's raw request/SSE files were overwritten by the final frontier. The public export assigns unique ordinal frontier IDs and flags the earlier raw artifact as unavailable; no curve point was rerun or dropped.

## Long canonical requests: two distinct workloads

### Repeated-question replacement-history diagnostic — partial, 4/12

After reaching C=258,048, the frozen system prefix and the same p0 question were sent repeatedly as replacement-history requests, not as a normal append-only conversation. The p0 warmup and three measured p0 requests completed (4 of 12 planned); each used `cache_n=149,668` and backend `prompt_n=108,424`. The warmup wall was 135,702.671 ms; measured p0 walls were 143,603.179 / 143,365.401 / 143,819.806 ms. For the first measured identical p0 request, the full rendered prompt was 258,071 tokens and prompt stage was 134,484.335 ms. The warm cache did not eliminate prompt replay.

The next synthetic warmup had started when this scope was stopped. It was cancelled at server `n_tokens=152,740`; request and partial raw stream are retained privately, and the row is published as `incomplete`, not a pass. No p1 or p2 replacement-history requests completed. These four p0 rows are diagnostic observations only, not a completed three-prompt 12-request matrix and not a median claim.

### Append-only full-history single observations

To separate ordinary conversation growth from the synthetic replacement sequence, the exact completed p0 request and its actual assistant reply were reconstructed. A 40-token recovery warmup duplicated p0 to restore the retained base after cancellation; its actual reply was retained. Then canonical p0, p1, and p2 were sent as new user turns, each with all actual preceding user/assistant turns preserved. Every complete transcript was server-native tokenized before request and checked with a 400-token output budget plus 64-token reserve. All four streams ended with `[DONE]`, had final usage, and had no SSE error events.

| Observation | Native full prompt | Backend prompt_n | cache_n | Prompt stage (ms) | Decode stage (ms) | HTTP wall (ms) | Output tokens | MTP accepted / drafted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recovery warmup (40 cap; duplicate p0) | 258,436 | 105,717 | 152,740 | 131,550.862 | 995.372 | 132,748.585 | 40 | 26 / 26 |
| Append p0 (400 cap) | 258,508 | 54 | 258,475 | 3,632.369 | 2,171.314 | 6,023.435 | 84 | 55 / 56 |
| Append p1 (400 cap) | 258,620 | 47 | 258,591 | 3,591.641 | 7,282.869 | 11,078.492 | 208 | 113 / 188 |
| Append p2 (400 cap) | 258,857 | 49 | 258,827 | 3,553.040 | 9,548.528 | 13,306.048 | 328 | 203 / 248 |

These are one observation per row, not a repeatable timing matrix or medians. The recovery cost is reported separately. The append turns reused nearly the full retained transcript and processed only 47–54 backend prompt tokens; this is distinct from the checkpoint-limited replacement-history replay above.

## Focused natural cold recall

The fixed A1/B/A2 sequence completed on the small ACO configuration. A2 total client wall was 2,342.881 ms; backend prompt and decode stages were 636.489 ms and 1,698.595 ms. This is total turn time, not promotion-only overhead. The tracked page at token positions 1,024–1,280 was host-backed and nonresident before A2, then resident and selected afterward; useful request H2D increased by 34,603,008 bytes. The native singleton witness named page 1 rather than target page 4, so it is reported as a mismatched stronger witness and not used to establish target-page identity. Manual semantic review found the response correctly described the fixture's preallocated-output implementation.

## Reproducibility and artifacts

Run the portable protocol in [`README.md`](README.md). The candidate was built and focused host/CUDA tests passed before this cohort; this report identifies the exact server binary and model hashes above. The append-only helper validates completion markers and final usage before appending any response to the next transcript.

Sanitized artifacts in [`results/fixed/`](results/fixed/) contain allowlisted scalars only: `results.json` / `results.csv` / `summary.json` for the 8K placement and natural-recall rows, plus `long-context.json` / `long-context.csv` for all 16 fill points, the partial replacement-history scope, and append-only observations. They contain no local paths, command lines, request bodies, answers, credentials, or raw SSE. Private raw logs and transcripts remain outside the release tree; the one overwritten frontier payload is explicitly marked in the long export.
