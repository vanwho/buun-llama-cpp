# Cluster 09a — production pager owner and compact target memory

Tasks: `09-01`, `09-02`, `09-03`.

Purpose: make the pager a real target-context owner. At cluster exit, Qwen target writes and memory
mutations use compact physical target storage governed by a runtime-derived capacity.

Carry forward:

- add one normalized experimental configuration object shared by CLI/server/context; `off` is default;
- capability is initially Qwen3.8/qwen35 causal Turbo4 K/V on one target-attention CUDA device; refuse
  unsupported cases before partial allocation;
- `C`, page geometry, row bytes, layers, heads, dimensions, and `H` come from target metadata and the
  resource ledger. No server or GPU constant enters source;
- target attention storage has `H * page_tokens` physical rows while logical identity spans
  `ceil(C/page_tokens)` pages; native positions and sequence generations remain logical;
- the current write page is pinned. A page becomes host-authoritative only after all required layer/K/V
  segments and position metadata seal successfully;
- shifts, copy/remove/keep, rewind, clear, prompt reuse, and partial tails update pager and recurrent/MTP
  companions through one preflighted generation transition;
- feature-off constructs the ordinary cache and performs no pager allocation, accounting, or graph work.

Read: plan sections 21–22; handoffs 01-02, 01-03, 02-01, 02-04, 02-05, 05-01;
`src/llama-{context,kv-cache,memory-hybrid}.*`, pager headers/sources, Qwen model builders, common/server
parameter conversion, and existing cache lifecycle tests.

Exit artifacts: parser/config tests, capability diagnostics, a context-owned pager object, compact
allocation/position/write tests, lifecycle generation tests, and memory ledgers showing physical target
storage follows `H` rather than `C`.
