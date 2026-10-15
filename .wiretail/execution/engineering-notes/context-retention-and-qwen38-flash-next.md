# Context retention priorities and future Qwen3.8-Flash-Next support

Date: 2026-10-14.

This is a research and design assessment, not a statement that the proposed changes are implemented. Code line references identify the source inspected during the assessment and may move as implementation proceeds.

## Recommended client/server partnership

The best design is a **partnership: the client describes which context should persist, and buun-llama manages residency and attention selection**. The server can measure attention relevance, but it cannot reliably infer whether a paragraph is the current plan, an obsolete plan, or quoted instructions. The client usually already knows that from how it assembled the request.

The key distinction is between **keeping information readily available** and **making the model attend to it**. We should give durable instructions a residency preference without artificially increasing their attention logits.

A practical division would be:

| Context | Information supplied by client | Server treatment |
|---|---|---|
| System instructions and persistent constraints | Instruction scope and lifetime | Small, bounded protected allocation |
| Active plan, goal, decisions | Active document identity/version; task lifetime | Strong retention preference, released when superseded |
| Currently edited or referenced files | Working-set hint; current-turn relevance | Temporary retention preference |
| Other files, search results, logs | Ordinary or transient context | Attention relevance, reuse, and recency determine residency |

File extensions and names should only be fallback hints. A file called `PLAN.md` might be an obsolete plan or merely an example. Automatically promoting it to instruction status would be unreliable.

NVIDIA already exposes token-range retention priorities and durations in TensorRT-LLM. That establishes a useful client/server interface precedent, although its documented reuse policy does not by itself solve selection within an active sparse-attention conversation. [TensorRT-LLM retention API](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kvcache.md)

## Five implementation properties

### 1. Clients provide optional, small annotations

Each annotation identifies a message or content span, its category, document version, and lifetime—for example, “active plan until replaced.” The server resolves it into actual token ranges after chat-template rendering. Clients should not calculate physical page IDs or assume their token offsets survive rendering.

Existing clients should continue working without changes. The server can derive basic preferences from structured message roles. Richer treatment of plans and working files would require a client adapter or proxy.

### 2. Importance is a bounded retention preference

Active writes and pages used by an executing graph remain protected for correctness. Beyond that, a small critical-instruction allowance and soft preferences for active plans compete within the admitted hot capacity. Declaring many documents important must not pin the entire pool.

Lifetimes should use conversation/task events or accepted-token progress, with explicit invalidation when documents change. Old plans should lose their preference immediately when superseded.

### 3. Attention stays responsible for relevance

The current query should still find and promote highly relevant cold pages even when they have no client annotation. Retention combines current relevance, recent relevance, demonstrated reuse, and recency; client importance prevents durable material from being discarded too casually during temporarily unrelated work.

This balance has research support: H₂O combines frequently attended information with recent tokens, while Quest shows why current-query relevance matters rather than relying solely on accumulated importance. Neither establishes ideal weights for our model; those remain implementation choices to measure. [H₂O](https://arxiv.org/abs/2306.14048), [Quest](https://arxiv.org/abs/2406.10774)

### 4. One policy governs both incoming writes and promotions

Adding importance only to retrieval would leave the prefill victim-selection problem intact. Incoming pages and promoted pages must use the same eligibility and retention metadata. Among otherwise weak candidates, older pages should be evicted first, rather than choosing the first eligible physical slot.

Residency alone also does not guarantee an instruction is visible to selected attention. For our existing approximate selector, a bounded critical-instruction candidate set could ensure availability while ordinary attention weights determine its contribution. That needs explicit treatment; merely keeping the bytes on GPU is insufficient.

### 5. Added work happens at boundaries

Categorize and map spans when requests arrive; update reuse evidence when a query completes; expire preferences lazily. Victim selection should inspect the bounded resident set, with no full-context text classification or catalogue scan on every token.

Shared metadata can be tiny: an additional 16–32 bytes per logical page would occupy only 16–32 KiB for a 256K context with 256-token pages. There is no need for a second model, embeddings, full attention-matrix readback, or CPU tensor scoring to classify plans.

## Existing repository mechanisms

The repository already provides useful building blocks: [rendered message spans](../../../common/chat.h#L156), the pager’s retention attributes, and [retention lineage/lifetime machinery](../../../common/common-retention-sidecar.h#L65). The latter currently manages reusable artifacts and checkpoints, so its identity and expiration mechanisms are reusable, but its artifact scores should not simply be substituted for live-page relevance.

The current prefill admission weakness is visible in the [batch victim-selection path](../../../src/llama-kv-pager.cpp#L3709): eligible victims are collected in physical-slot order. That can repeatedly reuse a limited group of slots while leaving other old pages resident. This is a code-based possible explanation for the 107-02 target remaining resident at larger sizes; it is not proof that attention deliberately preserved that target.

## Qwen3.8-Flash-Next architecture and integration

On **Qwen3.8-Flash-Next**, there is a promising architectural improvement, but also substantial integration work.

The current Qwen3.8-27B already alternates three linear-attention layers with one full-attention layer. Flash-Next’s major relevant addition is **Qwen Sparse Attention (QSA)**: a learned indexer selects small blocks for the attention layers. It therefore supplies a model-trained relevance mechanism, rather than requiring our approximate router to reconstruct relevance from a few probes. [27B configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json), [Flash-Next configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)

For future support, the storage layer should accept candidates from different selectors:

- Current 27B: our existing attention-based router.
- Flash-Next: its native QSA indexer.
- Both: shared host storage, transfers, residency metadata, recency, and optional client retention hints.

**For Flash-Next, the native selected indices should determine required attention data.** Client importance should primarily improve residency and prefetching. Replacing its learned selection with our global page ranking—or silently truncating it to fit the hot pool—could undermine the architecture’s trained behavior.

There are several concrete constraints in the current code:

- [ACO’s capability check](../../../src/llama-context.cpp#L1385) accepts `qwen35`/`qwen35moe`, not Flash-Next’s `qwen4exp`.
- The [QSA implementation](../../../src/models/qwen4exp.cpp#L872) assumes its indexer cache tracks attention-cache cells. An offloaded layout needs explicit logical-position-to-physical-slot mapping.
- [Indexer keys are stored separately in F16](../../../src/llama-memory-hybrid-idx.cpp#L67). They are learned routing features, distinct from Turbo4 attention K/V, and require their own memory accounting.
- The local [Flash-Next MTP implementation uses dense attention](../../../src/models/qwen4exp.cpp#L315). The technical report describes QSA in MTP and reuse of selected indices across draft steps. That discrepancy needs investigation before assuming the published efficiency is available locally. [Qwen technical report](https://arxiv.org/html/2608.30320v1)

Page granularity matters particularly here. QSA selects up to 512 four-token blocks for a 2,048-token budget. If those blocks are scattered across 256-token storage pages, they can touch far more than 2,048 tokens of physical storage. Future support should accommodate smaller transfer/gather units or staging for selected blocks, rather than assume a small attention budget automatically means a small resident footprint.

Finally, Flash-Next is a **125B main model with another 51B parameters of n-gram embeddings**, despite activating only 6B parameters per token. Its architecture is attractive for long context, but it is not a straightforward replacement that fits entirely beside a hot cache on the 16GB card. Weight offloading introduces a separate performance constraint. [Official model description](https://github.com/QwenLM/Qwen3.8-Flash-Next)

## Recommended direction

Implement the shared residency policy and optional client annotations first, while defining the selector interface so QSA can later supply native candidates. That improves the current model and avoids embedding today’s approximate ranking assumptions into the future storage architecture.
