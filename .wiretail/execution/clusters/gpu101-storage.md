# GPU101 encoded writes and storage overlap

Revision: `hotpath-v10-20260914`. Amendment: `gpu-execution-101-20260930`.
Tasks: 101-06, 101-07. Shared source: encoded packed append, retained source
leases, backend events and inclusive host-store queues. SPEED101/TESTING/current
packet and immediate predecessor are the only automatic design context.

GPU computes/encodes; host stores identical committed bytes. Byte copies are
not quantized GET_ROWS. Source reuse waits for its exact consumer/copy event;
clean eviction needs no D2H. No per-output historic reselection/H2D. Keep
source/device logic generic and benchmark service paths outside source commits.
