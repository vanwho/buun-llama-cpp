# Cluster 02a — selected capture and canonical host backing

Tasks: `02-01`, `02-02`.

Purpose: adapt Buun's bounded artifact primitives into live per-page host backing without turning the
pager into a VBR representation tier or using the public state serializer.

Carry forward:

- each complete page payload contains all 16 target layers and K/V sides in opaque device Turbo4 row
  encoding; validate actual geometry;
- every sealed page remains canonical in host RAM even while it is also GPU-resident;
- pageable full backing plus a separately charged bounded pinned ring is the default;
- host keys include model/execution/topology/sequence/page/representation identity and checksums;
- only obsolete/invalid generations may be discarded; valid backing is not an ordinary cache victim;
- full-prefix prompt artifact format remains byte-compatible.

Read: plan sections 5.1–5.3, 8.2–8.4, 9.3, 17; Phase 01 handoffs; VBR artifact capture/adopt/stage,
generation, segment, catalog, and pinned-ring sources.

Exit artifacts: bounded selected-page descriptor/segment seam and budgeted authenticated page catalog,
with fake-provider round trips and corruption/stale-generation coverage.
