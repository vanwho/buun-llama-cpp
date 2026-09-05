# Cluster 17i — full 256K context capacity and residency

Task `17-18` is the explicit end-to-end capacity gate for the stated product
goal. It must attempt a resolved 262,144-token context with canonical CPU-RAM
Turbo4 target backing, ledger-derived hot Turbo4 pages in VRAM, and fully
GPU-resident context-sized Turbo4 MTP draft KV. A smaller context is only a
diagnostic fallback and cannot pass this gate. Its non-gating speed curve uses
the original three benchmark prompts and the same trial protocol at each
context interval.
