# Cluster forward100-admission

Revision: `hotpath-v10-20260914`. Amendment: `forward-generation-admission-20260929`.
Tasks: `100-02a`, `100-02b`.

Reuse allocator/turn-policy context across these two repairs. Load current
packet, forward OVERVIEW/TESTING/REPAIR100 and compact predecessor handoff;
inspect only named source symbols. Do not load historical raw logs or the
100-02 campaign packet. First repair budget/ownership, then production batch
preflight/FIFO. CPU encoded-host tests and small CUDA tests suffice here; do
not launch the 27B service in these implementation tasks. High, gpt-6-luna.
No fixed task-count cap; clustering follows subsystem and compatible context.
