# Prototype and community salvage matrix

This is a read-only reuse map for Qwen3.8-27B/qwen35. It does not recommend
merging either reference branch wholesale. The target implementation must keep
logical page identity separate from VBR representation identity, and must keep
recurrent companions coupled to the same sequence transaction.

| Candidate | Source / evidence | Solved problem and proven tests | Qwen3.8 incompatibilities | Decision | Destination / task |
| --- | --- | --- | --- | --- | --- |
| Logical identity | Matiaslin `0b0f7bd7`, `src/llama-kv-cache-paged.*`; current `src/llama-vbr-identity-digest.h`, `src/llama-vbr-generation.*` | Maps logical token ranges to physical blocks; reference paged unit/e2e tests cover allocation and lookup. | Reference identity is block-table-local; Qwen needs 256-token pages, 1,024 logical pages, sequence/epoch lineage, and Turbo4/VBR digesting. | Reuse concept; adapt code | `src/llama-kv-pager-*`; 01-02 |
| Transaction publication | Current `src/llama-vbr-transaction.h`, `src/llama-vbr-operation.*`, occupied-replacement guard and transaction tests | Validates/adopts mutations and prevents stale occupied-cell replacement. | Existing VBR transactions describe representations/artifacts, not GPU page residency plus recurrent state and attention epoch as one commit. | Reuse concept; adapt code | pager transaction owner; 02-04, 05-01 |
| Policy replay | Current `common/common-cache-plan.*`, `common/common-retention-sidecar.*`, `tests/test-cache-plan-*`, `tests/test-cache-yield.cpp` | Deterministic ranking, budget projection, retention alternatives, and replayable accounting. | Retention artifacts are not demand-resident attention pages; policy must account for router output, recent window, MTP reservation, and pinned pages. | Reuse concept; do not reuse authority | pager policy layer; 01-04 |
| State serializer | Current `common/common.h` checkpoint types and `src/llama-memory.*`; `tests/test-server-prompt-cache.cpp` | Serializes/restores prompt and typed companion state with lineage checks. | A page image needs codec/topology/model digest, page identity, and partial restore; generic checkpoint payloads cannot silently stand in for page backing. | Adapt code | page capture/restore descriptors; 02-01, 05-02 |
| Block manager | Matiaslin `0b0f7bd7`, `src/llama-block-manager.*`, `tests/test-paged-kv.cpp` | Free GPU/CPU block registries, checkout/release, and safety watermarks. | It assumes one F16 type, synchronous copies, one block size, and no host artifact generation or rollback. | Reuse concept only | pure residency table and budget; 01-02, 01-03 |
| Block-table CUDA op | Matiaslin `0b0f7bd7`, `ggml/src/ggml-cuda/pagedattn.*`, `src/llama-graph.*` | Consumes a block table for paged attention; CPU/CUDA tests exercise the prototype path. | Kernel is F16-oriented and dense-block shaped; it does not consume Turbo4 K/V, selected pages, Qwen full-attention cadence, or native positions. | Wait for upstream sync; use as oracle | Turbo4 selected attention; 03-02, 03-03 |
| CPU pool | Matiaslin `0b0f7bd7`, `llama_kv_cache_paged::cpu_registry`; CPU tests | Provides a CPU physical block registry and swap API. | CPU KV is a control baseline, not the desired host Turbo4 artifact store; synchronous swap would stall decode and has no pinned staging policy. | Reject as implementation; retain baseline | CPU-KV comparison only; 00-03 |
| Selected native positions | Current Qwen hybrid/recurrent paths in `src/llama-memory-hybrid.*`, `src/llama-memory-hybrid-idx.*`, and Qwen model builders; current position-index tests | Preserves recurrent/attention position semantics and sparse index metadata. | Community paged prototype has no Qwen3.8 recurrent companion or native MTP position coupling. | Reuse current code; extend page view | selected reference view; 03-01, 05-01 |
| VBR artifact capture | Current `src/llama-vbr-artifact-capture.*`, `src/llama-vbr-explicit-capture.*`, `src/llama-vbr-artifact-validate.*`, and capture tests | Captures, validates, stages, and adopts typed VBR artifacts with identity/codec checks. | VBR generation pages are produced by representation policy and may not correspond to pages currently demanded by attention; artifact extents are not automatically page extents. | Reuse concept; adapt boundary | selected-page capture descriptors and host catalog; 02-01, 02-02 |
| Pinned ring | Current `src/llama-vbr-pinned-ring.*`, pinned-ring tests, and `src/llama-vbr-artifact-stage.*` | Bounded pinned host staging and transfer-lane coordination. | It is an artifact/retention staging primitive, not a page scheduler; capacity, cancellation, and useful-vs-actual byte accounting need page-specific ownership. | Reuse concept; adapt code | transfer ring and batched moves; 02-03 |
| Sparse FA | Current QSA/index and sparse-attention plumbing (`src/llama-kv-cache-dsa*`, `src/llama-vbr-qsa-index.*`) plus upstream sparse-FA history | Represents sparse selection/index metadata and executes sparse attention variants. | Qwen3.8 selective attention is page selection over Turbo4 full-attention layers; QSA indexes and page routing have different correctness and position contracts. | Reuse interfaces only; do not conflate | attention view/router; 03-01, 03-02 |
| Non-contiguous restore runs | llama.cpp commit `2d8d612e4`, non-contiguous KV restore batching; current restore/copy paths | Coalesces contiguous destination runs to reduce restore overhead. | It restores dense cells into an existing cache; page residency needs source/destination physical IDs, transfer cancellation, publication epochs, and rollback. | Reuse algorithm after contract mapping | batched page transfers; 02-03, 02-04 |
| Recurrent companions | Current `src/llama-memory-recurrent.*`, `src/llama-memory-hybrid.*`, Qwen3.8 builders, and checkpoint lineage tests | Keeps recurrent state, attention KV, and sequence lifecycle operations coherent. | Neither community paged branch nor generic block manager models Qwen3.8 hybrid recurrent state or native MTP checkpoint coupling. | Keep as authority; integrate page transaction | hybrid/MTP integration; 05-01 through 05-04 |
| Server checkpoints | Current `common/common-retention-sidecar.*`, `tools/server/server-cache-yield.*`, `tools/server/server-prompt-cache-payload.*`, and server prompt-cache tests | Bounded slot retention, checkpoint accounting, and server lifecycle integration. | Checkpoints are warm/cold retention artifacts, not transparent live CPU K/V offload or attention page residency. | Reuse lifecycle vocabulary only | pager observability and lifecycle adapter; 06-03, 06-05 |

## Explicit boundaries

The old Buun checkout at `/srv/ai/paged-kv/repos/buun-llama-cpp` was inspected
read-only; its VBR work is useful for representation generation and retention,
not proof of demand-resident page attention. The Matiaslin branch at
`/srv/repos/matiaslin/llama.cpp` was inspected read-only at `0b0f7bd7`; its
`tests/test-paged-kv.cpp` and `tests/test-paged-kv-e2e.cpp` are correctness
oracles, not merge targets. The current llama.cpp history entries
`2d8d612e4` and `36b101543` are algorithm references only and remain subject
to deliberate upstream synchronization.

No row treats a VBR generation page as an actual demand-resident data page.
All page movement must publish identity, codec, physical location, and
attention epoch atomically, while recurrent and MTP companions follow the
same sequence mutation.
