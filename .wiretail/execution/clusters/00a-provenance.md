# Cluster 00a — governance and provenance

Tasks: `00-01`, `00-02`.

Purpose: freeze authority and immutable inputs before any experiment or source change. Task 00-01 is
already complete; its handoff authorizes fork-only iteration but no upstream publication.

Carry forward:

- canonical repo is `/srv/repos/vanwho/buun-llama-cpp` on a branch based on synchronized Buun
  `cb703be37`; compared llama.cpp is `67a17c17`;
- actual model/profile/harness metadata must be measured and hashed, not copied from prose;
- fetch is read-only, but upstream movement touching relevant files is a deliberate stop condition;
- never expose GitHub/API/service credentials or mutate model/config/service state.

Read: plan sections 3, 5, 6, 12.4, 13; `handoffs/00-01.md`; task 00-02's benchmark pointers.

Exit artifact: `baseline/PROVENANCE.md` and `baseline/reference.json` sufficient for every later cluster
to identify exact sources, model, hardware, build, profile, and benchmark contract without rescanning.
