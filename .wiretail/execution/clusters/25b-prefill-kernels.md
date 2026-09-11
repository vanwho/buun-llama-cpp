# Cluster 25b-prefill-kernels

Tasks: `25-04`, `25-05`, `25-06`. Model: Luna High.
Revision: speed-first-20260911.

Read tasks/README.md, PHASE25_SPEED_FIRST_STRATEGY.md sections 1–3 and the
sections explicitly named by your packet, then BENCHMARK_PROTOCOL_V6.md.
Reuse this context for the cluster's consecutive tasks; don't import old
phase-14/17/21 acceptance chains or superseded 25-02 handoffs.

Carry only 25-03 timing/host delta contract. Compare dense Turbo reuse/compressed packing vs tiled direct; preserve native positions/transform domain. B and CUDA tile are independent. Winner selected by total packing+kernel+model cost; no custom-kernel preference without evidence.

Record source symbols/ownership decisions, actual supported driver flags,
last successful candidate/config and short raw IDs in the handoff so the next
cluster doesn't need the entire conversation. Models are task recommendations,
not a change to Wiretail tool defaults.
