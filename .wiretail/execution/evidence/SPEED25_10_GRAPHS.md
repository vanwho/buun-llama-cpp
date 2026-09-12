# SPEED25_10_GRAPHS

Direct paged attention descriptor refresh now separates stable graph storage
from mutable residency contents. A reused shape/address uploads only changed
contiguous page descriptor runs; native positions, masks, and query positions
are copied only when their values differ. Physical remaps remain visible, and
no generation or graph-property change is suppressed.

The CUDA backend now has optional real graph diagnostics controlled by
`GGML_CUDA_GRAPH_DIAGNOSTICS=1`. It records actual capture, instantiate,
executable-update, update-failure, and launch counts plus CPU durations. These
are intentionally separate from llama's backend-neutral admission counters.

CPU llama and focused KV-attention tests passed. The CUDA graph translation unit
compiled with `GGML_CUDA_USE_GRAPHS`; a live graph/server receipt was not run
because the full CUDA server link requires the repository's large template set.
No unobserved graph counts or throughput are reported.
