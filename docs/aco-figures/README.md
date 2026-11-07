# ACO publication figures and data

The [article](../aco-article.md) follows the supplied publication draft and uses the current measured results. This folder includes 17 PNG/SVG chart pairs and the updated cover. The [complete gallery](GALLERY.md) gives every chart and its caption.

| Figure | Results |
|---|---|
| [MTP on at 64K-class context](figures/01_mtp_64k_comparison.png) | On the P1 Python prompt, ACO measured 79.69 tokens/s: 4.5× CPU at 17.68 and 99% of GPU at 80.70. |
| [MTP-on decode across context lengths](figures/02_mtp_context_comparison.png) | Near 250K context, ACO measured 61.13, 48.83, and 55.35 tokens/s on P1–P3, versus CPU at 7.21, 5.49, and 6.50: 8.47–8.89× CPU speed. |
| [MTP on · P1 Python merge](figures/03_mtp_prompt_1.png) | On P1, ACO measured 102.48 tokens/s at 6K, 70.61 at 128K, and 61.23 near 256K. |
| [MTP on · P2 mmap vs read](figures/04_mtp_prompt_2.png) | Near 250K on P2, ACO measured 48.83 tokens/s versus CPU at 5.49: 8.89× CPU speed. |
| [MTP on · P3 Bash watcher](figures/05_mtp_prompt_3.png) | Near 250K on P3, ACO measured 55.35 tokens/s versus CPU at 6.50: 8.51× CPU speed. |
| [MTP off at 62,464 tokens](figures/06_non_mtp_64k_comparison.png) | On P1, ACO measured 42.33 tokens/s versus CPU at 8.48 and GPU at 43.01: 5.0× CPU speed and 98% of GPU speed. |
| [MTP-off decode across context lengths](figures/07_non_mtp_context_comparison.png) | Near 256K, ACO measured about 37.5 tokens/s across P1–P3, versus CPU at about 3.0 tokens/s. |
| [ACO decode · MTP on versus off](figures/08_aco_mtp_on_off.png) | On P1 near 256K, ACO measured 61.23 tokens/s with MTP on and 37.59 with MTP off. |
| [MTP off · P1 Python merge](figures/09_non_mtp_prompt_1.png) | Near 250K on P1, ACO measured 37.51 tokens/s versus CPU at 2.98: 12.6× CPU speed. |
| [MTP off · P2 mmap vs read](figures/10_non_mtp_prompt_2.png) | Near 250K on P2, ACO measured 37.57 tokens/s versus CPU at 2.98: 12.6× CPU speed. |
| [MTP off · P3 Bash watcher](figures/11_non_mtp_prompt_3.png) | Near 250K on P3, ACO measured 37.45 tokens/s versus CPU at 2.98: 12.6× CPU speed. |
| [CPU MTP placement](figures/12_cpu_mtp_placement.png) | Around 64K on P1–P3, CPU + GPU MTP measured 17.68, 13.56, and 15.29 tokens/s; master CPU + MTP measured 15.12, 12.26, and 13.84. Around 128K, CPU + GPU MTP measured 14–19% higher. |
| [MTP-on prefill at 32K target](figures/13_prefill_mtp_on_32k_comparison.png) | At 32K, GPU measured 924 and ACO 832 fresh tokens/s; ACO reached 90% of the GPU rate. |
| [MTP-on prefill across context lengths](figures/14_prefill_mtp_on_context.png) | ACO measured 1,402 fresh tokens/s on the first append and 614 near 256K. Around 64K, ACO measured 817 versus GPU at 559. |
| [Prefill without MTP · 128K context](figures/15_prefill_mtp_off_128k_comparison.png) | Near 128K, ACO measured 850 fresh tokens/s, CPU 559, and GPU 318; ACO reached 2.7× the GPU rate. |
| [MTP-off prefill across context lengths](figures/16_prefill_mtp_off_context.png) | ACO measured 1,589 fresh tokens/s on the first append and 803 near 256K. |
| [Long-context maximum decode](figures/17_long_context_decode_comparison.png) | At 250K filled context, ACO measured 8.47–8.89× CPU speed with MTP on and 12.55–12.59× with MTP off across P1–P3. |

## Data

The renderer reads [chart-data.csv](data/chart-data.csv), containing 309 curated observations: 156 decode points, 111 prefill points and 42 CPU/MTP placement points. Decode values are maximum measured repetitions with reasoning off. Context positions are cumulative and 1K means 1,024 tokens.

The original compact exports remain available as [decode.csv](data/decode.csv), [prefill.csv](data/prefill.csv) and [cpu-mtp-placement.csv](data/cpu-mtp-placement.csv). Recorded decode repetition values remain alongside their maxima. The [manifest](data/manifest.json) defines the metrics and records file hashes.

## Rendering

The renderer reads only bundled data and writes the chart PNG/SVG files and captions. Run it from the repository root:

```bash
uv run --no-project --with matplotlib python3 docs/aco-figures/render.py
```

Alternatively, use `python3 docs/aco-figures/render.py` in an environment with Matplotlib installed.

The cover was updated from the supplied original using the built-in imagegen tool. Its [exact edit prompt](cover-image-prompt.txt) is retained separately from the numerical chart renderer.
