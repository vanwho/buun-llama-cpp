# Cluster 15e — speculative rollback and page lifecycle

Tasks: `15-05`.

Purpose: make sequence truncation, speculative accept/reject, cancellation, slot reuse, and checkpoint
restore atomic across canonical host pages, GPU page tables, transfer fences, target state, and draft MTP.

Exit artifact: live 32K selective+MTP rollback regression plus CPU/CUDA fault evidence.
