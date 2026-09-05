# 19-02 Canonical Receipt

Status: `pass` for portable implementation checks; the live production
forced-small-hotset comparison is deferred for hardware isolation.

The pager maps every committed write through `physical_slot` and the
intra-page row. Completion validates page, sequence, slot, and generation
identity. Canonical host capture uses the existing selected-page transfer and
publishes opaque Turbo4 source bytes. Full pages and contiguous tails are
accepted with exact valid-row counts; padding and non-contiguous speculative
holes are rejected. Clean eviction releases only the resident slot and leaves
the host catalog entry available.

The deterministic fixture covered a 17-row tail, 256-row full-page boundary,
32 K/V unit sources, a one-slot clean replacement, same-identity byte rewrite,
stale completion, rollback, and physical-row lookup. The CUDA build and ten
focused executables passed; the pager/residency/policy CTest selection passed
3/3. No live-model or production D2H timing result is claimed.

See `19-02.md` for the compact handoff and exact deferred verification.
