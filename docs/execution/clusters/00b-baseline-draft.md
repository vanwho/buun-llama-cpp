# Cluster 00b — baselines, salvage, and draft placement

Tasks: `00-03`, `00-04`, `00-05`.

Purpose: establish comparable pre-pager evidence, convert old/community work into a reuse map, then land
the one independent prerequisite: draft KV placement in both forks.

Carry forward:

- scope is Qwen3.8-27B/qwen35 and must not expand to unrelated model issues;
- `/srv/ai/paged-kv/repos/buun-llama-cpp` is dirty, read-only reference code;
- `/srv/repos/matiaslin/llama.cpp` is F16/synchronous reference work, not a branch to merge;
- target/draft coupling originates in common parameter conversion and native/external speculative
  context construction;
- preferred draft contract is `auto|gpu|cpu`; `auto` preserves legacy behavior;
- Buun draft K/V parser already accepts Turbo4 aliases; generated docs are stale;
- task 00-05 alone may edit the prepared clean `/srv/repos/vanwho/llama.cpp-kv-pager` branch worktree
  and Buun; the ordinary llama.cpp `master` checkout remains read-only.

Read: provenance outputs; plan sections 3.3–7, 12.4, 13; prior handoff after each task. Do not load pager
kernel or dynamic policy sources in this cluster.

Exit artifacts: comparable baseline records, `baseline/SALVAGE_MATRIX.md`, and a tested generic/Buun
draft-placement truth table with fork base/tip correspondence.
