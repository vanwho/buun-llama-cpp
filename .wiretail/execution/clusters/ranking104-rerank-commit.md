# ranking104-rerank-commit

Revision: `hotpath-v10-20260914`.

Tasks104-04–05: fused Turbo4 key-only rerank, bounded staging, authenticated
query commit/replay. Read RANKING104 C/D/memory and immediate handoff. Use the
existing experimental worktree and its committed interface; do not recreate
it or migrate WORK_STATE into it. Keep CPU host pages opaque, GPU MTP full-L,
and frozen target history during generation. No live full-context campaigns.
