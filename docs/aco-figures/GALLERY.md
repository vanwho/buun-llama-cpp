# ACO graphs and captions

Maximum decode results from the three prompt benchmarks, with reasoning off and a 400-token output budget. Each point uses the fastest of three measured repetitions. The final prefill figure follows fresh repository chunks.

## Maximum decode with MTP on around 64K context

![Maximum decode with MTP on around 64K context](figures/01_mtp_64k_comparison.png)

Around 64K on P1, ACO reached 79.69 tokens/s, GPU reached 80.70, and CPU reached 17.68.

## Maximum decode with MTP on across prompt types

![Maximum decode with MTP on across prompt types](figures/02_mtp_context_comparison.png)

At 250K filled context, ACO measured 61.13, 48.83, and 55.35 tokens/s on P1, P2, and P3, compared with CPU at 7.21, 5.49, and 6.50.

## Maximum decode with MTP on for P1 Python merge

![Maximum decode with MTP on for P1 Python merge](figures/03_mtp_prompt_1.png)

ACO measured 102.48 tokens/s at 6K context and 61.23 tokens/s near full 256K context on P1.

## Maximum decode with MTP on for P3 Bash watcher

![Maximum decode with MTP on for P3 Bash watcher](figures/04_mtp_prompt_3.png)

ACO measured 92.93 tokens/s at 6K context and 55.35 tokens/s at 250K filled context on P3.

## Maximum decode with MTP on at 250K filled context

![Maximum decode with MTP on at 250K filled context](figures/05_mtp_long_context.png)

At 256,000 occupied tokens, ACO measured 61.13, 48.83, and 55.35 tokens/s across P1, P2, and P3.

## Maximum decode by CPU MTP placement

![Maximum decode by CPU MTP placement](figures/06_cpu_mtp_placement.png)

Around 64K on P1, CPU + GPU MTP measured 17.68 tokens/s and master CPU + MTP measured 15.12, a 16.9% gain.

## Maximum decode with MTP off across prompt types

![Maximum decode with MTP off across prompt types](figures/07_non_mtp_context_comparison.png)

At 250K filled context, ACO measured 37.51, 37.57, and 37.45 tokens/s on P1, P2, and P3.

## Maximum ACO decode with MTP on and off

![Maximum ACO decode with MTP on and off](figures/08_aco_mtp_on_off.png)

At 250K filled context on P1, ACO measured 61.13 tokens/s with MTP on and 37.51 tokens/s with MTP off.

## Backend prefill with MTP on across context lengths

![Backend prefill with MTP on across context lengths](figures/09_prefill_mtp_on_context.png)

With MTP on, ACO ingested fresh repository chunks at 1,402 backend tokens/s on the first 8K append and 614 tokens/s near full 256K context.

## Maximum decode with MTP on for P2 mmap vs read

![Maximum decode with MTP on for P2 mmap vs read](figures/10_mtp_prompt_2.png)

ACO measured 82.01 tokens/s at 6K context and 48.83 tokens/s at 250K filled context on P2.
