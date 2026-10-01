# GPU101 recurrent parallelism and stable graphs

Revision: `hotpath-v10-20260914`. Amendment: `gpu-execution-101-20260930`.
Tasks: 101-04, 101-04a, 101-05. Shared source: model graph building, GDN snapshots,
hybrid attention reuse and admitted workspace. Load SPEED101/TESTING/current
packet and one preceding compact handoff; lookup named functions on demand.

Prompt chunking must not disable exact MTP rollback snapshots. Mutable native
positions/content are not structural graph keys; allocation owners/shape/
recurrent planes are. Charge peak A/scratch plus overlapping replacement owners.
No CPU live model probes or new huge performance matrices. CUDA numerical
fixtures and one affected representative request are the proof boundary.
