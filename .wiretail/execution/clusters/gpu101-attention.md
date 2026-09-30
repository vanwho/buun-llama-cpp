# GPU101 attention execution

Revision: `hotpath-v10-20260914`. Amendment: `gpu-execution-101-20260930`.
Tasks: 101-01, 101-02, 101-03. Shared source: CUDA FA dispatch/loaders and
request attribution. Reuse this context while those owners stay relevant.

Read SPEED101, TESTING and current packet plus its compact predecessor handoff.
Lookup named source symbols only. No raw profiler exports, Codex logs, full
historical handoffs, phase100 journals or whole fattn.cu at startup.
Actual GPU execution and end-to-end preparation cost determine the route;
mere route counters are insufficient. No added production CUDA fences.
All target/draft storage stays Turbo4, draft full-L GPU, B1024/U256. Only one
Qwen weights allocation; exact candidate reuse/reload via managed lifecycle.
101-03 leaves a finite, measured compressed batched GPU route for subsequent
graph/recurrent repairs. No full matrix or long occupancy in this cluster.
