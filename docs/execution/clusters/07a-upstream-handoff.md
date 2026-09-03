# Cluster 07a — upstream slices and final evidence

Tasks: `07-01`, `07-02`.

Purpose: convert the proven fork implementation into dependency-ordered review units and leave a
self-consistent operator/evidence package. This cluster prepares; it does not publish upstream.

Carry forward:

- re-fetch and re-read current Buun/llama.cpp contribution instructions and existing #21961/#22569/
  #28115 before classifying any slice;
- expected slice order is generic draft placement, pure page/accounting core, backend residency and host
  seam, generic sparse metadata, Turbo4 CUDA, telemetry/policy, Qwen/server integration, exact, docs;
- each slice records destination, base/tip, dependencies, exact files/hunks, tests, platform, risks,
  compatibility, and range-diff/rebase strategy;
- fork-only commits/pushes are user-authorized for the outer agent; upstream prose, PR creation/replies,
  and merges remain human-owned;
- final claims must point to raw evidence and match actual CLI, limitations, hot pages, host/MTP types,
  quality, and throughput.

Read: plan sections 7, 11.1, 13–18; every handoff, acceptance file, current repository instructions,
and full diff against synchronized bases.

Exit artifacts: `upstream/SLICE_MAP.md`, `evidence/INDEX.md`, current operator docs, validated final state,
and no unstaged/generated/secret/result contamination.
