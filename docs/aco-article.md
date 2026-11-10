# Qwen3.8-27B at 256K Context on a 16GB GPU, with 9× CPU-Offload Speed

*Attention-Aware Context Offloading keeps long conversations fast with a 64K GPU hot cache and native MTP.*

![Qwen3.8-27B with ACO on RTX 4080, 256K context, 64K hot GPU context, 9X CPU KV offload with MTP enabled, and 1M Context Soon](aco-figures/figures/cover.png)

Last time, I pushed Qwen3.8-27B past 100 tokens per second on a 16GB RTX 4080. The next challenge was keeping that model useful after loading a repository and working through a long conversation. [The earlier 100 tok/s article](https://medium.com/@paulvanliew/i-made-qwen3-8-27b-run-2-5-faster-breaking-100-tok-s-on-a-16gb-rtx-4080-72b6a99d7a42).

I built **ACO — Attention-Aware Context Offloading** to retain a much larger history in system RAM while keeping the active attention working set on the GPU.

The result is **256K context with a 64K GPU hot cache, on the same 16GB card, with native MTP enabled**. The Python code-generation benchmark reaches **102.48 tok/s at small context and 61.23 tok/s near full context**. At the long-context comparison, ACO delivers **8.5–8.9× CPU-offload decode speed** across the three prompts.

## Keeping GPU-like decode speed as context grows

![Maximum Python code-generation decode speed with MTP across context lengths](aco-figures/figures/03_mtp_prompt_1.png)

The main graph follows one fixed Python prompt across the context curve. Each point is the **maximum measured decode rate** from the prompt benchmark, with reasoning off and a 400-token output budget.

At small context, ACO reaches **102.48 tok/s**, matching the full-GPU result of **102.46 tok/s**. Around 64K, ACO produces **79.69 tok/s**, compared with **80.70 tok/s** for full GPU and **17.68 tok/s** for CPU KV offload. That is **approximately 99% of GPU speed and 4.5× CPU-offload speed**.

![Maximum Python decode speed around 64K with MTP enabled](aco-figures/figures/01_mtp_64k_comparison.png)

The curve continues through the larger context sizes. ACO reaches **70.61 tok/s around 128K**, **66.52 tok/s at 192K**, and **61.23 tok/s near the end of the 256K fill**. The Bash code-generation prompt follows the same pattern, reaching **64.14 tok/s around 128K** and **60.35 tok/s at 192K**.

The three configuration names in the graphs describe target KV placement. **GPU** keeps it in VRAM, **CPU** keeps it in system RAM, and **ACO** retains the full history in RAM with selected pages hot in VRAM. Model weights are on the GPU in all three. GPU and ACO use Turbo4 target KV. The CPU target cache uses Q4_0, paired with GPU-resident Turbo4 MTP in the branch comparisons.

## The long-context difference is substantial

![Maximum ACO and CPU-offload decode speeds with MTP at long context](aco-figures/figures/05_mtp_long_context.png)

At **250K occupied tokens in the 256K configuration**, the three prompts give the following maximum decode speeds.

| Prompt | ACO with MTP | CPU offload with MTP | Speedup |
|---|---:|---:|---:|
| Python sorted-list merge | 61.13 tok/s | 7.21 tok/s | 8.47× |
| mmap versus read | 48.83 tok/s | 5.49 tok/s | 8.89× |
| Bash directory watcher | 55.35 tok/s | 6.50 tok/s | 8.51× |

The Python and Bash prompts exercise code generation, where MTP can accept multiple predicted tokens and accelerate generation. With ACO, that acceleration remains useful even when the retained conversation is several times larger than the GPU hot cache.

For a local coding agent, the difference between roughly **6–7 tok/s and 55–61 tok/s** is immediately noticeable while waiting for code. ACO carries the larger history while preserving much of the generation speed that makes a GPU-hosted model pleasant to use.

## How ACO keeps the working set on the GPU

ACO stores completed historical KV pages in system RAM. At the start of a turn, attention evidence and key reranking help select pages for the incoming request. Needed cold pages are promoted into the bounded GPU hot cache, and the selected historical view remains stable during generation.

New KV rows are written on the GPU. Completed pages acquire host backing so they can be evicted from VRAM and retrieved on a later turn. This keeps the larger history available while target attention operates over the selected GPU working set.

The measured configuration uses **65,536 hot tokens and 262,144 total context capacity**. Turbo4 is used for the target K/V, its offloaded page storage, and native MTP K/V. MTP keeps its own full-context KV allocation on the GPU.

For this model's geometry, the 64K hot target-KV payload occupies approximately **1.03 GiB**, compared with **4.125 GiB** for 256K of fully GPU-resident target KV. That is **75% less GPU memory for the target-KV payload**.

The implementation uses selective attention over the retained history. Its paging, selection, and GPU attention paths are described in the setup guide. [ACO implementation and configuration](https://github.com/vanwho/buun-llama-cpp/blob/plan/attention-aware-kv-paging/docs/aco.md).

## GPU MTP also makes CPU offload faster

A related addition improves conventional CPU KV offload. The branch adds **independent native-MTP KV placement**, exposed through `--spec-draft-kv-device gpu`, so the target KV can remain in CPU RAM while Turbo4 draft KV stays on the GPU.

The master native-MTP control inherits the target's CPU KV placement. The graph shows how the independent GPU placement changes its decode performance.

![CPU KV offload with GPU-resident MTP compared with master native MTP](aco-figures/figures/06_cpu_mtp_placement.png)

Around 64K context, the Python prompt improves from **15.12 to 17.68 tok/s**, a **16.9% gain**. Around 128K, it improves from **10.12 to 12.02 tok/s**, an **18.8% gain**. The Bash prompt gains **10.5%** around 64K and **17.6%** around 128K.

The main ACO comparison already uses that faster CPU-offload configuration. ACO then moves the active target-attention pages onto the GPU, producing the much larger gains across the long-context curve.

## MTP off shows the underlying offload advantage

![Maximum decode speeds without MTP across context lengths](aco-figures/figures/07_non_mtp_context_comparison.png)

With MTP disabled, ACO reaches **37.5 tok/s** across all three prompts at 250K occupied tokens, compared with approximately **3 tok/s** for CPU KV offload. That is **12.6× CPU-offload speed**. With MTP enabled, the same long-context Python benchmark rises to **61.13 tok/s**, which is why the main results use MTP throughout.

## Loading the repository stays fast, too

![Backend prefill speed with MTP enabled across context lengths](aco-figures/figures/09_prefill_mtp_on_context.png)

The prefill curve uses mapped chunks of fresh repository content, added progressively to an occupied context. With MTP enabled, ACO starts at **1,402 tok/s**, measures **817 tok/s around 64K**, and finishes near 256K at **614 tok/s**. At the 64K scale, the full-GPU curve measures **559 tok/s**. These are model prompt-stage rates for the new input chunks. The final ACO append also measures **603 net-new tokens/s over the complete request wall time**.

## Hardware and benchmark settings

The machine has an **RTX 4080 16GB, Ryzen 7950X3D, and 64GB of RAM**, running Linux. The model is **Qwen3.8-27B UD-IQ4_XS**.

The benchmark uses batch/microbatch **1024 / 256**, one sequence, 16 CPU threads, flash attention, greedy sampling, and seed 42. Native MTP uses a maximum draft length of two. Each prompt has a 40-token warmup and three measured repetitions with a **400-token output budget**.

The article and graphs use maximum measured decode speeds with reasoning off, comparing the same prompt across configurations. The three prompts are the original Python sorted-list merge, mmap-versus-read explanation, and Bash directory watcher. Context is loaded from mapped repository chunks and retained for the prompt tests. The data files include the individual measured repetition rates and exact context positions. **1K means 1,024 tokens.**

The accompanying graph pack includes the full curves for all three prompts, ACO with MTP on and off, the CPU MTP-placement comparison, and the prefill curve. [Graphs and plotted data](https://github.com/vanwho/buun-llama-cpp/tree/plan/attention-aware-kv-paging/docs/aco-figures).

## Open code

ACO makes a larger working history practical on a 16GB desktop GPU while keeping code generation close to GPU speed. I am continuing the work toward larger contexts, including the 1M context goal.

The branch, setup guide, and benchmark material are available publicly. [ACO source](https://github.com/vanwho/buun-llama-cpp/tree/plan/attention-aware-kv-paging) · [Setup guide](https://github.com/vanwho/buun-llama-cpp/blob/plan/attention-aware-kv-paging/docs/aco.md).

This builds on Buun/spiritbuun's work and llama.cpp. Credit also goes to the Qwen team, Daniel Han, Unsloth AI, and the NVIDIA CUDA ecosystem. [Buun](https://github.com/spiritbuun/buun-llama-cpp) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [Daniel Han](https://www.linkedin.com/in/danielhanchen) · [Unsloth AI](https://www.linkedin.com/company/unsloth/) · [NVIDIA AI](https://www.linkedin.com/showcase/nvidia-ai/).
